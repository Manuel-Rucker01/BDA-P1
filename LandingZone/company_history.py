import yfinance as yf
import pandas as pd
import os
from datetime import timedelta

# --- Configuration ---
COMPANIES_CSV_PATH = "../datasets/nasdaq_companies.csv"
TICKER_COLUMN_NAME = "Symbol"  
MASTER_FILE_PATH = "../datasets/all_companies_daily_data.csv"

def update_all_tickers():
    """Reads the CSV of companies and updates data for all tickers in a single master file."""
    
    if not os.path.exists(COMPANIES_CSV_PATH):
        print(f"Error: The file {COMPANIES_CSV_PATH} was not found.")
        return
        
    # Read the companies list
    try:
        df_companies = pd.read_csv(COMPANIES_CSV_PATH)
    except Exception as e:
        print(f"Error reading {COMPANIES_CSV_PATH}: {e}")
        return

    if TICKER_COLUMN_NAME not in df_companies.columns:
        print(f"Error: Column '{TICKER_COLUMN_NAME}' not found. Available columns: {df_companies.columns.tolist()}")
        return
        
    tickers = df_companies[TICKER_COLUMN_NAME].dropna().unique().tolist()
    print(f"Starting data pipeline for {len(tickers)} companies...")

    # Initialize the master dataframe and ensure directory exists
    os.makedirs(os.path.dirname(MASTER_FILE_PATH), exist_ok=True)
    
    if os.path.exists(MASTER_FILE_PATH):
        print(f"Loading existing master dataset from {MASTER_FILE_PATH}...")
        df_master = pd.read_csv(MASTER_FILE_PATH, index_col=0, parse_dates=True)
        # Ensure the index has a name so we can reference it when dropping duplicates
        if df_master.index.name is None:
            df_master.index.name = 'Date'
    else:
        print("No existing master dataset found. A new one will be created.")
        df_master = pd.DataFrame()

    new_data_frames = []

    # Process each ticker
    for ticker in tickers:
        clean_ticker = str(ticker).strip()
        if not clean_ticker:
            continue
            
        ticker_obj = yf.Ticker(clean_ticker)
        
        # Check if we already have data for THIS specific company in the master dataframe
        if not df_master.empty and 'Company' in df_master.columns and clean_ticker in df_master['Company'].values:
            # Get the max date specifically for this ticker
            ticker_data = df_master[df_master['Company'] == clean_ticker]
            last_date = ticker_data.index.max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                df_new = ticker_obj.history(start=start_date)
                if not df_new.empty:
                    df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
                    df_new['Company'] = clean_ticker
                    new_data_frames.append(df_new)
                    print(f"[{clean_ticker}] Downloaded {len(df_new)} new records.")
                else:
                    print(f"[{clean_ticker}] Already up to date.")
            except Exception as e:
                print(f"[{clean_ticker}] Error downloading new data: {e}")
                
        # If the ticker is entirely new to our dataset
        else:
            try:
                df_initial = ticker_obj.history(period="1y")
                if not df_initial.empty:
                    df_initial.index = pd.to_datetime(df_initial.index).tz_localize(None)
                    df_initial['Company'] = clean_ticker
                    new_data_frames.append(df_initial)
                    print(f"[{clean_ticker}] Downloaded {len(df_initial)} historical records (New Company).")
                else:
                    print(f"[{clean_ticker}] No data found.")
            except Exception as e:
                print(f"[{clean_ticker}] Error downloading historical data: {e}")

    # Combine and save everything at the very end
    if new_data_frames:
        print("\nMerging new data with the master dataset...")
        df_combined_new = pd.concat(new_data_frames)
        
        if not df_master.empty:
            df_final = pd.concat([df_master, df_combined_new])
        else:
            df_final = df_combined_new
            
        # We temporarily convert the Date index into a column so we can drop duplicates 
        # based on BOTH the Date and the Company column, then put Date back as the index.
        df_final = df_final.reset_index()
        # Rename 'index' to 'Date' if it got named generically during reset
        if 'index' in df_final.columns:
            df_final.rename(columns={'index': 'Date'}, inplace=True)
            
        df_final = df_final.drop_duplicates(subset=['Date', 'Company'], keep='last')
        df_final = df_final.set_index('Date')
        
        # Save the unified dataset
        df_final.to_csv(MASTER_FILE_PATH)
        print(f"Master dataset saved successfully to {MASTER_FILE_PATH}. Total rows: {len(df_final)}")
    else:
        print("\nNo new data to add. Master dataset remains unchanged.")

if __name__ == "__main__":
    update_all_tickers()