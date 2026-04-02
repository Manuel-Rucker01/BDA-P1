import kaggle
import os
import pandas as pd
import tempfile

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))
FINAL_CSV_PATH = os.path.join(DATASETS_DIR, "nasdaq_companies.csv")

DATASET = "dhimananubhav/nasdaq-company-list" 

# 1. Ensure the final destination directory exists safely
os.makedirs(DATASETS_DIR, exist_ok=True)

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
        
        # Save it to your final, permanent location using the rock-solid path
        df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"[INFO] Dataset successfully saved to {FINAL_CSV_PATH}")
        
    else:
        print("[INFO] No CSV files found in the downloaded dataset.")
