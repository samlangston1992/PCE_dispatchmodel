import pandas as pd
from datetime import timedelta
from battery_optimisation_function import battery_optimisation
from tqdm import tqdm
from datetime import timedelta

#import data, fill blanks and set datetime column
data = pd.read_csv("GB_prices.csv")
data = data.fillna(method='ffill')
data['time'] = pd.to_datetime(data['time'])

#create loop to run above over each day
days = data.groupby(data['time'].dt.date)

# Create empty lists to store the results
result_DA = []
result_spot = []

# Initialize initial_capacity and progress bar
initial_capacity = 0
progress_bar = tqdm(days)

# Iterate over each day's data and run the battery optimization algorithm
#need to loop for DA then ID then calulate the difference to create final position then
for day, data in progress_bar:
    # Run the battery optimization algorithm for DA and append to DA list
    result_DA, final_capacity_DA = battery_optimisation(data['time'], data['DA_price'], initial_capacity = initial_capacity, include_revenue=True, solver='glpk') #determine initial conditions based on the previous days output
    result_DA.append(result_DA)

    #run for intraday spot price and append to spot list
    result_spot, final_capacity_spot = battery_optimisation(data['time'], data['spot_price'], initial_capacity = initial_capacity, include_revenue=True, solver='glpk')
    result_spot.append(result_spot)

    #determine next day's capacity from final capacity       
    initial_capacity = final_capacity_spot
    
    #update progress bar
    progress_bar.set_description(f"Optimization performed for day {str(day + timedelta(days=1))}")
    
result_DA = result_DA.rename(columns={'spot_price' : 'DA_price', 'power' : 'power_DA', 'market_dispatch' : 'market_dispatch_DA', 'opening_capacity' : 'opening_capacity_DA', 
                                      'throughput' : 'throughput_DA', 'profit' : 'profit_DA'}) 

# Merge the two dataframes
result_final = pd.concat([result_DA, result_spot], axis=1)
result_final = result_final.drop_duplicates(subset='datetime')
pd.options.display.float_format = '{:.2f}'.format

# Create the profit column in result_final by calculating the profit from result_DA
# and adding the net difference in market_dispatch multiplied by the spot_price
result_final['profit'] = result_DA['profit_DA'] + ((result_spot['market_dispatch'] - result_DA['market_dispatch_DA']) * result_spot['spot_price'])

# Rearrange the columns in the desired order
result_final = result_final[['datetime', 'DA_price', 'spot_price', 'market_dispatch', 'opening_capacity', 'throughput', 'profit']]

print("Hello World")
print(result_final)