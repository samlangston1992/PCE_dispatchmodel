import pandas as pd
from datetime import timedelta
from battery_optimisation_function import battery_optimisation
from tqdm import tqdm
from datetime import timedelta

# Define the asset parameters dictionary
asset_params = {
    'MIN_BATTERY_CAPACITY': 0,
    'MAX_BATTERY_CAPACITY': 100,
    'MAX_RAW_POWER': 1,
    'DEG_FACTOR': 0.00005, #0.0000225 need to ratchet
    'INITIAL_CAPACITY': 0,
    'EFFICIENCY': 0.88, #RTE
    'MLF': 1,
    'MARGINAL_COST': 0, #half the desired £/MWh minimum spread 
    'DAILY_HARD_CAP': 4,
    'SOFT_CAP' : 3,
    'SELF_DISCHARGE_RATE': 0
}

#import data, fill blanks and set datetime column
data = pd.read_csv("GB_prices.csv") #2024 = 2022, 2025 = 2021 etc.
data = data.fillna(method='ffill')
data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M', dayfirst=True)

from LCP_API import generate_prices

#data = generate_prices()

#create loop to run above over each day
years = data.groupby(data['time'].dt.year)

# Create empty lists to store the results
result_DA = []
result_spot = []
result_CO = []

def update_deg_factor(asset_params, current_soh):
    # Check the current SoH value or any other relevant condition
    if current_soh <= (asset_params['MAX_BATTERY_CAPACITY'] * 0.903):  # Example condition: Change DEG_FACTOR if SoH falls below 0.8
        asset_params['DEG_FACTOR'] = 0  # Set the new DEG_FACTOR value here #0.000045/2 
    else:
        asset_params['DEG_FACTOR'] = 0  # Set the original DEG_FACTOR value here or any other value you want # 0.00015/2

# Initialize initial_capacity and progress bar  #### add option to tweak DA vols?
initial_capacity = 0
initial_SoH = asset_params['MAX_BATTERY_CAPACITY']
progress_bar = tqdm(years)

# Iterate over each day's data and run the battery optimization algorithm
#need to loop for DA then ID then calulate the difference to create final position then
for year, data in progress_bar:
    # Run the battery optimization algorithm for DA and append to DA list
    result_year_DA, final_capacity_DA, final_SoH_DA, *_ = battery_optimisation(data['time'], data['DA_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk') #determine initial conditions based on the previous days output
    result_DA.append(result_year_DA)

    #run for intraday spot price and append to spot list
    result_year_spot, final_capacity_spot, final_SoH_spot, *_ = battery_optimisation(data['time'], data['spot_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk')
    result_spot.append(result_year_spot)

    #run for CO and append to CO list
    result_year_CO, final_capacity_CO, final_SoH_CO, *_ = battery_optimisation(data['time'], data['CO_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk')
    result_CO.append(result_year_CO)

    #determine next day's capacity and SoH from final capacity   
    initial_capacity = float(final_capacity_CO)
    initial_SoH = float(final_SoH_CO)
 

# Convert the result_DA and result_spot lists to dataframes
result_DA = pd.concat(result_DA)
result_spot = pd.concat(result_spot)
result_CO = pd.concat(result_CO)
    
result_DA = result_DA.rename(columns={'spot_price' : 'DA_price', 'power' : 'power_DA', 'market_dispatch' : 'market_dispatch_DA', 'opening_capacity' : 'opening_capacity_DA', 
                                      'throughput' : 'throughput_DA', 'profit' : 'profit_DA', "SoH": "SoH_DA"}) 

#Rename the Spot columns also
result_spot = result_spot.rename(columns={'power' : 'power_ID', 'market_dispatch' : 'market_dispatch_ID', 'opening_capacity' : 'opening_capacity_ID', 
                                      'throughput' : 'throughput_ID', 'profit' : 'profit_ID', "SoH": "SoH_ID"})

#Rename the CO columns also
result_CO = result_CO.rename(columns={'spot_price' : 'CO_price', 'power' : 'power_CO', 'market_dispatch' : 'market_dispatch_CO', 'opening_capacity' : 'opening_capacity_CO', 
                                      'throughput' : 'throughput_CO', 'profit' : 'profit_CO', "SoH": "SoH_CO"})


# Merge the two dataframes
result_final = pd.concat([result_DA, result_spot, result_CO], axis=1)
result_final = result_final.drop_duplicates(subset='datetime', keep='first')
pd.options.display.float_format = '{:.2f}'.format



# Create the profit column in result_final by calculating the profit from result_DA
# and adding the net difference in market_dispatch multiplied by the spot_price
result_final['profit_ID_net'] = ((result_spot['market_dispatch_ID'] - result_DA['market_dispatch_DA']) * result_spot['spot_price'])

result_final['profit_CO_net'] = ((result_CO['market_dispatch_CO'] - result_spot['market_dispatch_ID']) * result_CO['CO_price'])

result_final["Net_profit"] = result_final["profit_DA"] + result_final['profit_ID_net'] + result_final['profit_CO_net']

result_final = result_final.loc[:, ~result_final.columns.duplicated()]

result_final.to_csv('yearly_100hr.csv', index=False)

print(result_final)