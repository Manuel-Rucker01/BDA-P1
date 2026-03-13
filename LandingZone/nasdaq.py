import kaggle
import os
import pandas as pd
import tempfile

DATASET = "dhimananubhav/nasdaq-company-list" 
FINAL_FILE_PATH = "datasets/"

# 1. Ensure the final destination directory exists
os.makedirs(os.path.dirname(FINAL_FILE_PATH), exist_ok=True)

# 2. Create a temporary directory context manager
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"[INFO] Downloading dataset '{DATASET}' to a temporary directory...")
    
    # Download and unzip directly into the temp directory
    kaggle.api.dataset_download_files(DATASET, path=temp_dir, unzip=True)

    # 3. Find the CSV file in the temporary folder
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]

    if csv_files:
        csv_path = os.path.join(temp_dir, csv_files[0])
        print(f"[INFO] Found {csv_files[0]}. Loading into Pandas...")
        
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Save it to your final, permanent location
        df.to_csv(FINAL_FILE_PATH+"nasdaq_companies.csv", index=False)
        print(f"[INFO] Dataset successfully saved to {FINAL_FILE_PATH}")
        
    else:
        print("[INFO] No CSV files found in the downloaded dataset.")

# Once the 'with' block ends, temp_dir and the original Kaggle files are automatically deleted!