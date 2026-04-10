"""
Data Collector for NASDAQ Companies

This script downloads the NASDAQ company list from Kaggle and saves it as a CSV file.

"""

import kaggle
import os
import pandas as pd
import tempfile

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))
FINAL_CSV_PATH = os.path.join(DATASETS_DIR, "nasdaq_companies.csv")
os.makedirs(DATASETS_DIR, exist_ok=True)

# Dataset configuration
DATASET = "dhimananubhav/nasdaq-company-list" 

# Create a temporary directory since we will need to unzip the downloaded dataset
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"[INFO]: Downloading dataset '{DATASET}' to a temporary directory...")
    
    # Download and unzip directly into the temp directory
    kaggle.api.dataset_download_files(DATASET, path=temp_dir, unzip=True)

    # Find the CSV 
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]

    if csv_files:
        csv_path = os.path.join(temp_dir, csv_files[0])
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)
        # Save it to a new location in the datasets folder
        df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"[INFO]: Dataset successfully saved to {FINAL_CSV_PATH}")
    else:
        print("[INFO]: No CSV files found in the downloaded dataset.")
