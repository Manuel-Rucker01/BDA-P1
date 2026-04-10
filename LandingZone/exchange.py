import os
import pandas as pd
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates, RatesNotAvailableError

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))
FINAL_FILE_PATH = os.path.join(DATASETS_DIR, "US_exchange.csv")

# We only want US exchange rates for the last 100 days
BASE_CURRENCY = 'USD'
DAYS_TO_FETCH = 100

os.makedirs(os.path.dirname(FINAL_FILE_PATH), exist_ok=True)
c = CurrencyRates()

# Download the last 100 days of USD exchange rates
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_TO_FETCH)
date_range = pd.date_range(start=start_date, end=end_date, freq='B') 
dataset_rows = []
print(f"[INFO]: Downloading {DAYS_TO_FETCH} days of {BASE_CURRENCY} rates... This might take a moment.")

for single_date in date_range:
    try:
        # Fetch all exchange rates against USD for the given date
        rates = c.get_rates(BASE_CURRENCY, single_date)
        
        # Create a dictionary row with the Date and all the currency rates
        row = {'Date': single_date.strftime("%Y-%m-%d")}
        row.update(rates) 
        dataset_rows.append(row)
    except Exception as e:
        print(f"[ERROR]: Error on {single_date.strftime('%Y-%m-%d')}: {e}")

# Convert to Pandas DataFrame and save
if dataset_rows:
    df = pd.DataFrame(dataset_rows)
    # Save it directly to CSV
    df.to_csv(FINAL_FILE_PATH, index=False)
    print(f"\n[INFO]: Dataset saved successfully to {FINAL_FILE_PATH}")
else:
    print("\n[INFO]: No data was fetched. There might be an error.")