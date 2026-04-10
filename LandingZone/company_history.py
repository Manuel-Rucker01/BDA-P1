import yfinance as yf
import pandas as pd
import os
from datetime import timedelta

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))
COMPANIES_CSV_PATH = os.path.join(DATASETS_DIR, "nasdaq_companies.csv")
RESULT_FILE_PATH = os.path.join(DATASETS_DIR, "company_history.csv")

# We will use "Symbol" as the column name 
TICKER_COLUMN_NAME = "Symbol"

def update_all_tickers():
    """
    Reads the CSV of companies and collects data for all tickers in a 
    single file, in case the data is already partially collected, 
    it will only update the missing data for each specific company.
    """
    
    # Check if the companies CSV file exists, otherwise we cannot proceed
    if not os.path.exists(COMPANIES_CSV_PATH):
        print(f"[Error]: The file {COMPANIES_CSV_PATH} was not found.")
        return
        
    # Read the whole dataset
    try:
        df_companies = pd.read_csv(COMPANIES_CSV_PATH)
    except Exception as e:
        print(f"[Error]: Error reading {COMPANIES_CSV_PATH}: {e}")
        return

    # Check if the expected ticker column exists
    if TICKER_COLUMN_NAME not in df_companies.columns:
        print(f"[Error]: Column '{TICKER_COLUMN_NAME}' not found. Available columns: {df_companies.columns.tolist()}")
        return
    
    # Get unique tickers and start collecting data
    tickers = df_companies[TICKER_COLUMN_NAME].dropna().unique().tolist()
    print(f"[INFO]: Starting data pipeline for {len(tickers)} companies...")

    # Initialize the result dataframe and ensure directory exists
    os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)
    # If the result file already exists, we will load it and update it
    if os.path.exists(RESULT_FILE_PATH):
        print(f"[INFO]: Loading existing result dataset from {RESULT_FILE_PATH}...")
        df_master = pd.read_csv(RESULT_FILE_PATH, index_col=0, parse_dates=True)
        # Ensure the index has a name so we can reference it when dropping duplicates
        if df_master.index.name is None:
            df_master.index.name = 'Date'
    # If the file does not exist, we will create a new one
    else:
        print(f"[INFO]: No existing result dataset found. A new one will be created.")
        df_master = pd.DataFrame()

    new_data_frames = []

    # Process each ticker
    for ticker in tickers:
        clean_ticker = str(ticker).strip()
        if not clean_ticker:
            continue
        ticker_obj = yf.Ticker(clean_ticker)
        
        # Check if we already have data for this specific company in the master dataframe
        if not df_master.empty and 'Company' in df_master.columns and clean_ticker in df_master['Company'].values:
            # Get the latest date of this ticker and set the start date
            ticker_data = df_master[df_master['Company'] == clean_ticker]
            last_date = ticker_data.index.max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            # try to download only the new data
            try:
                df_new = ticker_obj.history(start=start_date)
                if not df_new.empty:
                    df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
                    df_new['Company'] = clean_ticker
                    new_data_frames.append(df_new)
            except Exception as e:
                print(f"[{clean_ticker}] Error downloading new data: {e}")
                
        # If the ticker is new, we download the full history (1 year)
        else:
            try:
                df_initial = ticker_obj.history(period="1y")
                if not df_initial.empty:
                    df_initial.index = pd.to_datetime(df_initial.index).tz_localize(None)
                    df_initial['Company'] = clean_ticker
                    new_data_frames.append(df_initial)
            except Exception as e:
                print(f"[{clean_ticker}] Error downloading historical data: {e}")

    # Combine and save everything
    if new_data_frames:
        print("\n[INFO]: Merging new data with the master dataset...")
        df_combined_new = pd.concat(new_data_frames)
        
        if not df_master.empty:
            df_final = pd.concat([df_master, df_combined_new])
        else:
            df_final = df_combined_new
            
        # We temporarily convert the Date index into a column so we can drop duplicates 
        df_final = df_final.reset_index()
        # Rename 'index' to 'Date' 
        if 'index' in df_final.columns:
            df_final.rename(columns={'index': 'Date'}, inplace=True)
        df_final = df_final.drop_duplicates(subset=['Date', 'Company'], keep='last')
        df_final = df_final.set_index('Date')
        
        # Save the unified dataset
        df_final.to_csv(RESULT_FILE_PATH)
        print(f"[INFO]: Master dataset saved successfully to {RESULT_FILE_PATH}. Total rows: {len(df_final)}")
    else:
        print(f"[INFO]: No new data to add. Master dataset is up to date.")

if __name__ == "__main__":
    update_all_tickers()