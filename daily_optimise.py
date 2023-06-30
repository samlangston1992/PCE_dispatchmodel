import pandas as pd
from datetime import timedelta
from battery_optimisation_function import battery_optimisation
from tqdm import tqdm
from datetime import timedelta

# Define the asset parameters dictionary
asset_params = {
    'MIN_BATTERY_CAPACITY': 0,
    'MAX_BATTERY_CAPACITY': 2.35,
    'MAX_RAW_POWER': 1,
    'DEG_FACTOR': 0, #0.0000225 need to ratchet
    'INITIAL_CAPACITY': 0,
    'EFFICIENCY': 0.88, #RTE
    'MLF': 1,
    'MARGINAL_COST': 0, #half the desired £/MWh minimum spread 
    'DAILY_HARD_CAP': 4,
    'SELF_DISCHARGE_RATE': 0
}

#import data, fill blanks and set datetime column
data = pd.read_csv("vic_test.csv")
data = data.fillna(method='ffill')
data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M')

from LCP_API import generate_prices

#data = generate_prices()

#create loop to run above over each day
days = data.groupby(data['time'].dt.date)

# Create empty lists to store the results
result_DA = []
result_spot = []
result_CO = []

# Initialize initial_capacity and progress bar  #### add option to tweak DA vols?
initial_capacity = 0
initial_SoH = asset_params['MAX_BATTERY_CAPACITY']
progress_bar = tqdm(days)

# Iterate over each day's data and run the battery optimization algorithm
#need to loop for DA then ID then calulate the difference to create final position then
for day, data in progress_bar:
    # Run the battery optimization algorithm for DA and append to DA list
    result_day_DA, final_capacity_DA, final_SoH_DA = battery_optimisation(data['time'], data['DA_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk') #determine initial conditions based on the previous days output
    result_DA.append(result_day_DA)

    #run for intraday spot price and append to spot list
    result_day_spot, final_capacity_spot, final_SoH_spot = battery_optimisation(data['time'], data['spot_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk')
    result_spot.append(result_day_spot)

    #run for CO and append to CO list
    result_day_CO, final_capacity_CO, final_SoH_CO = battery_optimisation(data['time'], data['CO_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk')
    result_CO.append(result_day_CO)

    #determine next day's capacity and SoH from final capacity   
    initial_capacity = float(final_capacity_CO)
    initial_SoH = float(final_SoH_CO)
    
    #update progress bar
    progress_bar.set_description(f"Optimization performed for day {str(day + timedelta(days=1))}")

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

result_final.to_csv('test.csv', index=False)

print(result_final)

#Add summary table and analysis of result
summary_table = result_final[['profit_DA', 'net_profit', 'throughput']].sum(axis=0)

daily_summary = result_final.groupby(result_final['datetime'].dt.date)[['profit_DA', 'net_profit', 'throughput']].sum()


