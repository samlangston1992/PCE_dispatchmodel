import pandas as pd
import numpy as np
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from pyomo.environ import *
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

#define battery optimisation function
def battery_optimisation(datetime, spot_price, asset_params, initial_capacity=0, initial_SoH = 2, include_revenue=True, solver: str='glpk'):
    """
    Determine the optimal charge and discharge behavior of a battery based on wholseale power market trading with half-hourly settlement periods.
    Assuming perfect foresight of future power prices over every half-hour period to maximise the revenue.
    
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
    MIN_BATTERY_CAPACITY = asset_params['MIN_BATTERY_CAPACITY']
    MAX_BATTERY_CAPACITY = asset_params['MAX_BATTERY_CAPACITY']
    MAX_RAW_POWER = asset_params['MAX_RAW_POWER']
    DEG_FACTOR = asset_params['DEG_FACTOR']
    INITIAL_CAPACITY = asset_params['INITIAL_CAPACITY']
    EFFICIENCY = asset_params['EFFICIENCY']
    MLF = asset_params['MLF']
    MARGINAL_COST = asset_params['MARGINAL_COST']
    MAX_DAILY_THROUGHPUT = MAX_BATTERY_CAPACITY * EFFICIENCY * 2 * asset_params['DAILY_HARD_CAP'] # redefine for degraded throughput
    MAX_YEARLY_THROUGHPUT = MAX_BATTERY_CAPACITY * EFFICIENCY * 2 * asset_params['SOFT_CAP'] * 365 # redefine for degraded throughput and apply correctly
    SELF_DISCHARGE_RATE = asset_params['SELF_DISCHARGE_RATE']
    
    df = pd.DataFrame({'datetime': datetime, 'spot_price': spot_price}).reset_index(drop=True)
    df['period'] = df.index
    initial_period = 0
    final_period = df.index[-1]
    
    #Initialise SoH 
    SoH = initial_SoH
    
    # Define model and solver
    battery = ConcreteModel()
    opt = SolverFactory(solver)

    # defining components of the objective model
    # battery parameters
    battery.Period = Set(initialize=list(df.period), ordered=True)
    battery.Price = Param(initialize=list(df.spot_price), within=Any)

    # battery variables
    battery.SoH = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Capacity = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, None))
    battery.Charge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.Discharge_power = Var(battery.Period, bounds=(0, MAX_RAW_POWER))
    battery.PositiveDiff = Var(battery.Period, within=NonNegativeReals)
    battery.NegativeDiff = Var(battery.Period, within=NonNegativeReals)
    battery.throughput_daily = Var(battery.Period, within=NonNegativeReals)
    battery.throughput_yearly = Var(battery.Period, within=NonNegativeReals)
    battery.profit_daily = Var(battery.Period, within=Reals)
    battery.profit = Var(battery.Period, within=Reals)  # Variable for profit in each period
      

    # Set constraints for the battery
    # Defining the battery objective (function to be maximise)
    def maximise_profit(battery):
        rev = sum(df.spot_price[i] * (battery.Discharge_power[i] / 2 ) * EFFICIENCY * MLF for i in battery.Period)
        cost = sum(df.spot_price[i] * (battery.Charge_power[i] / 2) / MLF for i in battery.Period)
        return rev - cost
    
    #define profit for each period
    def calculate_profit(battery, i):
        revenue = df.spot_price[i] * (battery.Discharge_power[i] / 2) * EFFICIENCY * MLF
        cost = df.spot_price[i] * (battery.Charge_power[i] / 2) / MLF
        return battery.profit[i] == revenue - cost

    def soh_constraint(battery, i):
        if i == battery.Period.first():
            return battery.SoH[i] == initial_SoH
        else:
            return battery.SoH[i] == battery.SoH[i-1] - (((battery.Discharge_power[i-1] + (battery.Charge_power[i-1] * EFFICIENCY)) / 2) * DEG_FACTOR)
           
    # Set upper bound for Capacity based on SoH
    def capacity_upper_bound_rule(battery, t):
        return battery.Capacity[t] <= battery.SoH[t]
    
    # Make sure the battery does not charge above the limit
    def over_charge(battery, i):
        return battery.Charge_power[i] <= (battery.SoH[i] - battery.Capacity[i]) * 2 / EFFICIENCY

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
        return battery.Capacity[i] == (battery.Capacity[i-1] - SELF_DISCHARGE_RATE
                                        + (battery.Charge_power[i-1] / 2 * EFFICIENCY) 
                                        - (battery.Discharge_power[i-1] / 2))
    
    #Defining throughput constraints
    
    def daily_throughput_accumulation_constraint(battery, i):
        if i == battery.Period.first():
            return battery.throughput_daily[i] == ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2
        if (i-1) % 48 == 0:
            return battery.throughput_daily[i] == ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2
        else:
            return battery.throughput_daily[i] == battery.throughput_daily[i - 1] + ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2

    def yearly_throughput_accumulation_constraint(battery, i): 
        if i == battery.Period.first():
            return battery.throughput_yearly[i] == ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2
        else:
            if (i-1) % 17520 == 0:
                return battery.throughput_yearly[i] == battery.throughput_yearly[i - 1] + ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2
            else: 
                return battery.throughput_yearly[i] == battery.throughput_yearly[i - 1] + ((battery.Discharge_power[i]  * EFFICIENCY) + battery.Charge_power[i]) / 2

    def daily_throughput_constraint(battery, i):
    # Apply the maximum daily throughput constraint
        return battery.throughput_daily[i] <= MAX_DAILY_THROUGHPUT
    
    def yearly_throughput_constraint(battery, i):
    # Apply the maximum daily throughput constraint
        return battery.throughput_yearly[i] <= MAX_YEARLY_THROUGHPUT
    
    #Define daily profit accumulation
    def daily_profit_accumulation_constraint(battery, i): #this only works if the code is executed on chunks of one day
        if i == battery.Period.first():
            return battery.profit_daily[i] == battery.profit[i]
        else:
            return battery.profit_daily[i] == (battery.profit[i] + battery.profit_daily[i - 1])
        
    def daily_profit_constraint(battery, i):
        if i == battery.Period.last():
            return battery.profit_daily[i] >= battery.throughput_daily[i] * MARGINAL_COST
        else:
            return Constraint.Skip 
        
    def charge_discharge_constraint(battery, i): 
        if battery.charge_power[i] != 0:
            battery.discharge_power[i] = 0
    
   
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
    battery.calculate_profit = Constraint(battery.Period, rule = calculate_profit)
    battery.soh_constraint = Constraint(battery.Period, rule=soh_constraint)
    battery.capacity_upper_bound = Constraint(battery.Period, rule=capacity_upper_bound_rule)
    battery.daily_profit_accumulation_constraint = Constraint(battery.Period, rule=daily_profit_accumulation_constraint)
    battery.daily_profit_constraint = Constraint(battery.Period, rule=daily_profit_constraint)
    

    # Maximise the objective
    opt.solve(battery, tee=True)

    
    # unpack results
    charge_power, discharge_power, capacity, throughput_daily, throughput_yearly, spot_price, profit, SoH, profit_daily = ([] for i in range(9))
    for i in battery.Period:
        charge_power.append(battery.Charge_power[i].value)
        discharge_power.append(battery.Discharge_power[i].value)
        capacity.append(battery.Capacity[i].value)
        spot_price.append(battery.Price.extract_values_sparse()[None][i])
        profit.append(battery.profit[i].value)
        SoH.append(battery.SoH[i].value)
        profit_daily.append(battery.profit_daily[i].value)
        throughput_daily.append(battery.throughput_daily[i].value)
        throughput_yearly.append(battery.throughput_yearly[i].value)

    result = pd.DataFrame({'datetime':datetime, 'spot_price':spot_price, 'charge_power':charge_power,
                           'discharge_power':discharge_power, 'throughput_daily' : throughput_daily, 'throughput_yearly' : throughput_yearly,
                           'opening_capacity': capacity, 'profit' : profit, 'profit_daily' : profit_daily, "SoH": SoH})
    
    # make sure it does not discharge & charge at the same time
    #if not len(result[(result.charge_power != 0) & (result.discharge_power != 0)]) == 0:
        #print('Ops! The battery discharges & charges concurrently, the result has been returned')
        #return result
    
    # convert columns charge_power & discharge_power to power
    result['power'] = np.where((result.charge_power > 0), 
                                -result.charge_power, 
                                result.discharge_power)
    
    # calculate market dispatch
    result['market_dispatch'] = np.where(result.power < 0,
                                         result.power / 2,
                                         result.power / 2 * EFFICIENCY)
    
    result['throughput'] = result['market_dispatch'].abs().fillna(0) #Need to be consistent and clear how to define - absolute value of market dispatch?
    
    final_capacity = result['opening_capacity'].iloc[-1] - SELF_DISCHARGE_RATE + ((result['charge_power'].iloc[-1] * EFFICIENCY) / 2) - (result['discharge_power'].iloc[-1] / 2 )

    SoH = result['SoH'].iloc[-1] - (((result['discharge_power'].iloc[-1] + (result['charge_power'].iloc[-1] * EFFICIENCY)) / 2 ) * DEG_FACTOR)

    result = result[['datetime', 'spot_price', 'power', 'market_dispatch', 'opening_capacity', 'throughput', 'throughput_daily', 'throughput_yearly', 'profit', 'profit_daily', 'SoH']]

    return result, final_capacity, SoH
