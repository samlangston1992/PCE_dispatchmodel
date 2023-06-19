import pandas as pd
import matplotlib.pyplot as plt

output = pd.read_csv("output2021.csv")
output['datetime'] = pd.to_datetime(output['datetime'])

plot_date = '2021-01-01'

output_filter = output.loc[output['datetime'].dt.date == pd.to_datetime(plot_date).date()]

plt.plot(output_filter['datetime'], output_filter['market_dispatch'])

plt.xlabel('Datetime')
plt.ylabel('Market Dispatch')
plt.title('Market Dispatch over Time')

# Display the plot
plt.show()