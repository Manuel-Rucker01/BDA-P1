"""
Data Collector for Company History

This script reads a list of companies from a CSV file and 
collects historical stock data for each company using the yfinance library.
This collector allows for incremental updates.

"""


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

# --------------------------------------------------------------------------- #
# Alpaca full-window refresh (HISTORY_SOURCE=alpaca)
# --------------------------------------------------------------------------- #
# Output schema identical to the yfinance path (what the Formatted zone reads):
#   Date (index), Open, High, Low, Close, Volume, Dividends, Stock Splits,
#   Company, Capital Gains
# Alpaca SIP bars are split+dividend adjusted (Adjustment.ALL), i.e. the same
# convention as yfinance's auto-adjusted Ticker.history; the corporate-action
# columns are therefore 0 / empty, exactly as yfinance reports on most rows.
#
# The whole [HISTORY_START, HISTORY_END] window is re-downloaded in ONE pass
# (no appending) so adjustment factors are consistent across the series.
# Tickers that Alpaca does not return (delisted / renamed) keep their rows
# from the previous CSV, so downstream training does not silently drop dead
# names (survivorship bias).
#
# Env:
#   HISTORY_SOURCE=alpaca      enable this path (default: yfinance incremental)
#   HISTORY_START=YYYY-MM-DD   window start (default 2024-06-01)
#   HISTORY_END=YYYY-MM-DD     inclusive end (default: last completed close)
#   ACTIVE_ACCOUNT             Alpaca key set used by trading_agent.config
OUTPUT_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Dividends",
                  "Stock Splits", "Company", "Capital Gains"]


def refresh_all_tickers_alpaca(start=None, end=None):
    import sys
    import datetime as dt

    pipeline_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DataAnalysisPipeline2"))
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    from trading_agent import market_data  # noqa: E402  (needs ACTIVE_ACCOUNT set first)

    start = start or os.getenv("HISTORY_START", "2024-06-01")
    end = end or os.getenv("HISTORY_END")
    start_dt = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    if end:
        # inclusive end date -> request up to the following midnight UTC
        end_dt = (dt.datetime.fromisoformat(end) + timedelta(days=1)).replace(tzinfo=dt.timezone.utc)
    else:
        end_dt = market_data._last_completed_close_utc()

    df_companies = pd.read_csv(COMPANIES_CSV_PATH)
    tickers = (df_companies[TICKER_COLUMN_NAME].dropna().astype(str).str.strip())
    tickers = [t for t in dict.fromkeys(tickers) if t]
    print(f"[INFO]: Alpaca SIP refresh for {len(tickers)} tickers, "
          f"{start_dt.date()} -> {end_dt.date()} (exclusive)")

    bars = market_data._alpaca_daily_bars(tickers, start_dt, end_dt, verbose=True)
    bars = bars.drop_duplicates(["ticker", "Date"], keep="first")
    got = set(bars["ticker"])
    print(f"[INFO]: Alpaca returned {len(got)}/{len(tickers)} tickers, {len(bars):,} bars "
          f"({bars['Date'].min()} -> {bars['Date'].max()})")

    df_new = pd.DataFrame({
        "Date": pd.to_datetime(bars["Date"]),
        "Open": bars["Open"].astype(float),
        "High": bars["High"].astype(float),
        "Low": bars["Low"].astype(float),
        "Close": bars["company_close"].astype(float),
        "Volume": bars["company_volume"].round().astype("int64"),
        "Dividends": 0.0,
        "Stock Splits": 0.0,
        "Company": bars["ticker"].astype(str),
        "Capital Gains": float("nan"),
    })

    # Survivorship: keep previous rows of tickers Alpaca no longer serves.
    kept = pd.DataFrame()
    if os.path.exists(RESULT_FILE_PATH):
        old = pd.read_csv(RESULT_FILE_PATH, parse_dates=["Date"])
        old["Company"] = old["Company"].astype(str).str.strip()
        dead = sorted(set(old["Company"]) - got)
        kept = old[old["Company"].isin(dead)]
        print(f"[INFO]: Keeping {len(kept):,} legacy rows for {len(dead)} tickers not served "
              f"by Alpaca (delisted/renamed): {dead[:25]}{' ...' if len(dead) > 25 else ''}")
        kept = kept.reindex(columns=["Date"] + OUTPUT_COLUMNS)

    df_final = pd.concat([df_new, kept], ignore_index=True)
    df_final = (df_final.drop_duplicates(subset=["Date", "Company"], keep="first")
                        .sort_values(["Company", "Date"]))
    df_final["Date"] = df_final["Date"].dt.strftime("%Y-%m-%d")
    df_final = df_final.set_index("Date")[OUTPUT_COLUMNS]
    os.makedirs(os.path.dirname(RESULT_FILE_PATH), exist_ok=True)
    df_final.to_csv(RESULT_FILE_PATH)
    print(f"[INFO]: Saved {RESULT_FILE_PATH}: {len(df_final):,} rows, "
          f"{df_final['Company'].nunique()} tickers, "
          f"{df_final.index.min()} -> {df_final.index.max()}")
    return df_final


if __name__ == "__main__":
    if os.getenv("HISTORY_SOURCE", "yfinance").lower() == "alpaca":
        refresh_all_tickers_alpaca()
    else:
        update_all_tickers()