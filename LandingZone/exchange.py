import os
import pandas as pd
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates, RatesNotAvailableError


BASE_CURRENCY = 'USD'
DAYS_TO_FETCH = 100
FINAL_FILE_PATH = "datasets/US_exchange.csv"

os.makedirs(os.path.dirname(FINAL_FILE_PATH), exist_ok=True)
c = CurrencyRates()

end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_TO_FETCH)
date_range = pd.date_range(start=start_date, end=end_date, freq='B') 

dataset_rows = []

print(f"[INFO] Downloading {DAYS_TO_FETCH} days of {BASE_CURRENCY} rates... This might take a moment.")

for single_date in date_range:
    try:
        # Fetch all exchange rates against the base currency for the specific date
        rates = c.get_rates(BASE_CURRENCY, single_date)
        
        # Create a dictionary row with the Date and all the currency rates
        row = {'Date': single_date.strftime("%Y-%m-%d")}
        row.update(rates) 
        dataset_rows.append(row)
        
    except RatesNotAvailableError:
        print(f"[WARNING] API data not available for {single_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"[ERROR] Failed on {single_date.strftime('%Y-%m-%d')}: {e}")

# 5. Convert to Pandas DataFrame and save
if dataset_rows:
    df = pd.DataFrame(dataset_rows)
    
    # Save it directly to CSV
    df.to_csv(FINAL_FILE_PATH, index=False)
    
    print(f"\n[INFO] Success! Dataset saved to {FINAL_FILE_PATH}")
    print(df.head())
else:
    print("\n[INFO] No data was fetched. The API might be down.")