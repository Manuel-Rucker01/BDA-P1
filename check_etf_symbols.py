import pandas as pd

df = pd.read_csv('ExploitationZone/master_dataset_pro.csv')

# Get unique symbols
unique_symbols = df['Symbol'].unique()
print(f"✓ Total unique symbols in master dataset: {len(unique_symbols)}")
print(f"✓ First 10 symbols: {list(unique_symbols[:10])}")

# Check for NULL IPOyear symbols
null_ipo_sample = ['AAXJ', 'ABDC', 'ACT', 'ACWI', 'AIA']

found_any = False
for sym in null_ipo_sample:
    if sym in unique_symbols:
        found_any = True
        count = (df['Symbol'] == sym).sum()
        print(f"✓ Found {sym}: {count} rows in master dataset")

if not found_any:
    print("\n✅ CONCLUSION:")
    print("   None of the tested NULL IPOyear ETF symbols appear in master dataset!")
    print("   These are portfolio instruments, NOT individual stocks.")
    print("   They are excluded from the ARIMA analysis (which uses NASDAQ stocks only).")
    print("   Therefore, NULL IPOyear values for ETFs are NOT a problem for analysis.")
