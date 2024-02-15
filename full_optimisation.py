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
    'DEG_FACTOR': 0.00007, 
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
data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M', dayfirst=True)

from LCP_API import generate_prices

#data = generate_prices()

#create loop to run above over each day
days = data.groupby(data['time'].dt.date)

# Create empty lists to store the results
result_DA = []
result_spot = []

# Initialize initial_capacity and progress bar  #### add option to tweak DA vols?
initial_capacity = 0
initial_SoH = asset_params['MAX_BATTERY_CAPACITY']
progress_bar = tqdm(days)


result, final_capacity, SoH = battery_optimisation(data['time'], data['CO_price'], asset_params, initial_capacity = initial_capacity, initial_SoH = initial_SoH, include_revenue=True, solver='glpk') #determine initial conditions based on the previous days output



# Create the profit column in result_final by calculating the profit from result_DA
# and adding the net difference in market_dispatch multiplied by the spot_price
result_final['net_profit'] = result_DA['profit_DA'] + ((result_spot['market_dispatch'] - result_DA['market_dispatch_DA']) * result_spot['spot_price'])

result_final = result_final.loc[:, ~result_final.columns.duplicated()]

result_final.to_csv('output3.csv', index=False)

print(result_final)

# Create a Pandas ExcelWriter object to write to the Excel file
with pd.ExcelWriter('test_all_years_1cycle.xlsx') as writer:
    # Write result_final to the first worksheet
    result_final.to_excel(writer, sheet_name='Result_Final', index=False)

    # Write asset_params_df to the second worksheet
    asset_params_df.to_excel(writer, sheet_name='Asset_Params', index=False)

    # Write summary_table to third worksheet
    summary_table.to_excel(writer, sheet_name='Summary', index=True)

    # Write daily_summary to fourth worksheet
    daily_summary.to_excel(writer, sheet_name="Daily Summary", index=True)

#Add summary table and analysis of result
summary_table = result_final[['profit_DA', 'net_profit', 'throughput']].sum(axis=0)

daily_summary = result_final.groupby(result_final['datetime'].dt.date)[['profit_DA', 'net_profit', 'throughput']].sum()

