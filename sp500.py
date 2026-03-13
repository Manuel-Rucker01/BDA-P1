import yfinance as yf
import pandas as pd
import os
from datetime import timedelta

TICKER = "^GSPC"  # S&P 500 Index 
FILE_PATH = "datasets/sp500_daily_data.csv"

def update_daily_stock_data():
    # we create the directory if it does not exist
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    # through the API we obtain the ticker object
    ticker_obj = yf.Ticker(TICKER)

    # if the file already exists, we perform an incremental update
    if os.path.exists(FILE_PATH):
        df_existing = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
        
        # we obtain the last date recorded in the file and calculate the start date for downloading new data
        last_date = df_existing.index.max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # we download only the new data
        df_new = ticker_obj.history(start=start_date)
        
        if not df_new.empty:

            df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
            df_combined = pd.concat([df_existing, df_new])
            
            # we remove duplicates
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            
            # we save the updated dataset
            df_combined.to_csv(FILE_PATH)
            print(f"the dataset has been updated with {len(df_new)} new daily records.")
        else:
            print("the dataset is already updated. there is no new data")
            
    # if the file does not exist, we perform an initial load of the historical data
    else:
        # we download the historical data from the last 5 years
        df_initial = ticker_obj.history(period="5y")
        
        if not df_initial.empty:
            df_initial.index = pd.to_datetime(df_initial.index).tz_localize(None)
            # we save the initial dataset
            df_initial.to_csv(FILE_PATH)
            print(f"the initial dataset with {len(df_initial)} historical records has been saved in {FILE_PATH}.")
        else:
            print("there has been an error.")

if __name__ == "__main__":
    update_daily_stock_data()