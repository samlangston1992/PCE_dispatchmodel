import pandas as pd
import numpy as np
import math

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from pyomo.environ import *
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')


#define below to run over one day
def battery_optimisation(datetime, spot_price, initial_capacity=0, include_revenue=True, solver: str='glpk'):
    """
    Determine the optimal charge and discharge behavior of a battery based 
    in Victoria. Assuming pure foresight of future spot prices over every 
    half-hour period to maximise the revenue.
    PS: Assuming no degradation to the battery over the timeline and battery cannot
        charge and discharge concurrently.
    ----------
    Parameters
    ----------
    datetime        : a list of time stamp
    spot_price      : a list of spot price of the corresponding time stamp
    initial_capacit : the initial capacity of the battery
    include_revenue : a boolean indicates if return results should include revenue calculation
    solver          : the name of the desire linear programming solver (eg. 'glpk', 'mosek', 'gurobi')

    Returns
    ----------
    A dataframe that contains battery's opening capacity for each half-hour period, spot price
    of each half-hour period and battery's raw power for each half-hour priod
    """
    # Battery's technical specification
    MIN_BATTERY_CAPACITY = 0
    MAX_BATTERY_CAPACITY = 2
    MAX_RAW_POWER = 1
    INITIAL_CAPACITY = initial_capacity # Default initial capacity will assume to be 0
    EFFICIENCY = 0.938
    MLF = 0.991 # Marginal Loss Factor
    MDR = 30 #Minimum discharge revenue
    MAX_DAILY_THROUGHPUT =  MAX_BATTERY_CAPACITY * (EFFICIENCY**2) * 2 * 4
    MAX_YEARLY_THROUGHPUT = MAX_BATTERY_CAPACITY * (EFFICIENCY**2) * 2 * 3 * 365
    MARGINAL_COST = MDR / ((MAX_BATTERY_CAPACITY / MAX_RAW_POWER) * (EFFICIENCY**2) * 2)
    LEAKAGE = 0
    DEG_FACTOR = 0 #applied to throughput 
    #need to define a minimum revenue per throughput



    df = pd.DataFrame({'datetime': datetime, 'spot_price': spot_price}).reset_index(drop=True)
    df['period'] = df.index
    initial_period = 0
    final_period = df.index[-1]
    
    # Define model and solver
    battery = ConcreteModel()
    opt = SolverFactory(solver)

    # defining components of the objective model
    # battery parameters
    battery.Period = Set(initialize=list(df.period), ordered=True)
    battery.Price = Param(initialize=list(df.spot_price), within=Any)

    # battery variables
    battery.Capacity = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Charge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.Discharge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.PositiveDiff = Var(battery.Period, within=NonNegativeReals)
    battery.NegativeDiff = Var(battery.Period, within=NonNegativeReals)
    battery.throughput_daily = Var(battery.Period, within=NonNegativeReals)
    battery.throughput_yearly = Var(battery.Period, within=NonNegativeReals)
    battery.profit = Var(battery.Period, within=Reals)  # Variable for profit in each period

    # model deg impact on max cap
    # - (((battery.Discharge_power[i-1] + battery.Charge_power[i-1]) / (2 * EFFICIENCY)) * DEG_FACTOR)

    # Set constraints for the battery
    # Defining the battery objective (function to be maximise)
    def maximise_profit(battery):
        rev = sum(df.spot_price[i] * (battery.Discharge_power[i] / 2 * EFFICIENCY) * MLF for i in battery.Period)
        cost = sum(df.spot_price[i] * (battery.Charge_power[i] / 2) / MLF for i in battery.Period)
        return rev - cost
    
    #define profit for each period
    def calculate_profit(battery):
        for i in battery.Period:
            revenue = df.spot_price[i] * (battery.Discharge_power[i] / 2 * EFFICIENCY) * MLF
            cost = df.spot_price[i] * (battery.Charge_power[i] / 2) / MLF
            battery.profit[i] = revenue - cost

    # Make sure the battery does not charge above the limit
    def over_charge(battery, i):
        return battery.Charge_power[i] <= (MAX_BATTERY_CAPACITY - battery.Capacity[i]) * 2 / EFFICIENCY

    # Make sure the battery discharge the amount it actually has
    def over_discharge(battery, i):
        return battery.Discharge_power[i] <= battery.Capacity[i] * 2

    # Make sure the battery do not discharge when price are not positive
    def negative_discharge(battery, i):
        # if the spot price is not positive, suppress discharge
        if battery.Price.extract_values_sparse()[None][i] <= 0:
            return battery.Discharge_power[i] == 0

        # otherwise skip the current constraint    
        return Constraint.Skip

    # Defining capacity rule for the battery
    def capacity_constraint(battery, i):
        # Assigning battery's starting capacity at the beginning
        if i == battery.Period.first():
            return battery.Capacity[i] == INITIAL_CAPACITY
        # if not update the capacity normally    
        return battery.Capacity[i] == (battery.Capacity[i-1] - LEAKAGE
                                        + (battery.Charge_power[i-1] / 2 * EFFICIENCY) 
                                        - (battery.Discharge_power[i-1] / 2))
    
    #Defining throughput constraints
    
    def daily_throughput_accumulation_constraint(battery, i):
        if i == battery.Period.first():
            return battery.throughput_daily[i] == (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)
        if i % 48 == 0:
            return battery.throughput_daily[i] == (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)
        else:
            return battery.throughput_daily[i] == battery.throughput_daily[i - 1] + (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)

    def yearly_throughput_accumulation_constraint(battery, i):
        if i == battery.Period.first():
            return battery.throughput_yearly[i] == (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)
        else:
            if i % 17520 == 0:
                return battery.throughput_yearly[i] == battery.throughput_yearly[i - 1] + (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)
            else: 
                return battery.throughput_yearly[i] == battery.throughput_yearly[i - 1] + (battery.Discharge_power[i] + battery.Charge_power[i]) / (2 * EFFICIENCY)

    def daily_throughput_constraint(battery, i):
    # Apply the maximum daily throughput constraint
        return battery.throughput_daily[i] <= MAX_DAILY_THROUGHPUT
    
    def yearly_throughput_constraint(battery, i):
    # Apply the maximum daily throughput constraint
        return battery.throughput_yearly[i] <= MAX_YEARLY_THROUGHPUT
    
    #Define daily profit accumulation
    def daily_profit_accumulation_constraint(battery, i):
        if i == battery.Period.first():
            return battery.profit_daily[i] == battery.profit[i]
        if i % 48 == 0:
            return battery.profit_daily[i] == battery.profit[i]
        else:
            return battery.profit_daily[i] == battery.profit[i] + battery.profit_daily[i-1]
    
    # Set constraint and objective for the battery
    battery.capacity_constraint = Constraint(battery.Period, rule=capacity_constraint)
    battery.over_charge = Constraint(battery.Period, rule=over_charge)
    battery.over_discharge = Constraint(battery.Period, rule=over_discharge)
    battery.negative_discharge = Constraint(battery.Period, rule=negative_discharge)
    battery.daily_throughput_accumulation_constraint = Constraint(battery.Period, rule=daily_throughput_accumulation_constraint)
    battery.daily_throughput_constraint = Constraint(battery.Period, rule=daily_throughput_constraint)
    battery.yearly_throughput_accumulation_constraint = Constraint(battery.Period, rule=yearly_throughput_accumulation_constraint)
    battery.yearly_throughput_constraint = Constraint(battery.Period, rule=yearly_throughput_constraint)
    battery.objective = Objective(rule=maximise_profit, sense=maximize)


    # Maximise the objective
    opt.solve(battery, tee=True)

    #calculate profit for each period
    calculate_profit(battery)

    # unpack results
    charge_power, discharge_power, capacity, spot_price, profit = ([] for i in range(5))
    for i in battery.Period:
        charge_power.append(battery.Charge_power[i].value)
        discharge_power.append(battery.Discharge_power[i].value)
        capacity.append(battery.Capacity[i].value)
        spot_price.append(battery.Price.extract_values_sparse()[None][i])
        profit.append(battery.profit[i].value)

    result = pd.DataFrame({'datetime':datetime, 'spot_price':spot_price, 'charge_power':charge_power,
                           'discharge_power':discharge_power, 'opening_capacity':capacity, 'profit' : profit})
    
    # make sure it does not discharge & charge at the same time
    if not len(result[(result.charge_power != 0) & (result.discharge_power != 0)]) == 0:
        print('Ops! The battery discharges & charges concurrently, the result has been returned')
        return result
    
    # convert columns charge_power & discharge_power to power
    result['power'] = np.where((result.charge_power > 0), 
                                -result.charge_power, 
                                result.discharge_power)
    
    # calculate market dispatch
    result['market_dispatch'] = np.where(result.power < 0,
                                         result.power / 2,
                                         result.power / 2 * EFFICIENCY)
    
    result['throughput'] = result['opening_capacity'].diff().abs().fillna(0)
    
    result = result[['datetime', 'spot_price', 'power', 'market_dispatch', 'opening_capacity', 'throughput', 'profit']]

    final_capacity = result['opening_capacity'].iloc[-1]
    
    return result, final_capacity
    

data = pd.read_csv("vic_test.csv")
data['time'] = pd.to_datetime(data['time'])

#create loop to run above over each day
days = data.groupby(data['time'].dt.date)

# Create an empty DataFrame to store the results
results = pd.DataFrame()

# Initialize initial_capacity
initial_capacity = 0

# Iterate over each day's data and run the battery optimization algorithm
for day, data in days:
    # Run the battery optimization algorithm for each day's data
    result_day, final_capacity = battery_optimisation(data['time'], data['spot_price'], initial_capacity = initial_capacity, include_revenue=True, solver='glpk') #determine initial conditions based on the previous days output
    
    # Append the results for the day to the overall results DataFrame
    initial_capacity = final_capacity
    results = results.append(result_day)

# Reset the index of the results DataFrame
results = results.reset_index(drop=True)

pd.options.display.float_format = '{:.2f}'.format

print("Hello World")
print(results)
