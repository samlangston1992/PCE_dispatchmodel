import pandas as pd
from datetime import timedelta
import requests

def generate_prices():

    #retrieve bearer token
    headers = {'Content-Type': 'application/json','cache-control': 'no-cache'}
    data = '{"Username" : "PulseEnact" , "ApiKey" : "E93fs4&$8HvJ"}'
    bearer = requests.post('https://enactapifd.lcp.uk.com/auth/token', headers=headers, data=data)
    bearerToken = bearer.text


    #Initialise API call using bearer token
    headers = {
        'Authorization': 'Bearer ' + bearerToken
    }

    #Define start and end dates
    Date_from = "2022-01-01T00:00:00Z"
    Date_to = "2023-01-01T00:00:00Z"

    #Request N2EX prices
    request = {
        'SeriesId': 'DayAheadPricesLocalNordpool',
        'CountryId': 'Gb',
        'From': Date_from,
        'To': Date_to
    }

    res_N2EX = requests.post('https://enactapifd.lcp.uk.com/EnactAPI/Series/Data', json=request, headers=headers)
    N2EX = res_N2EX.json()
    N2EX_df = pd.DataFrame(N2EX['data'][1:], columns=N2EX['data'][0])
    N2EX_df.columns = ["Datetime", "Price_N2EX"]


    #Request APX prices
    request = {
        'SeriesId': 'ApxPriceCalculated',
        'CountryId': 'Gb',
        'From': Date_from,
        'To': Date_to
    }

    res_APX = requests.post('https://enactapifd.lcp.uk.com/EnactAPI/Series/Data', json=request, headers=headers)
    APX = res_APX.json()
    APX_df = pd.DataFrame(N2EX['data'][1:], columns=N2EX['data'][0])
    APX_df.columns = ["Datetime", "Price_APX"]


    merge_df = N2EX_df.merge(APX_df, on = "Datetime")

    return merge_df


if __name__ == '__main__':
    merged_prices = generate_prices()


print("Hello World")



