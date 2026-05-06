import os
import shutil
import kaggle
import kagglehub

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))

FORBES_DATASET = "kwongmeiki/forbes-worlds-best-employers-2023"
SP500_DATASET = "samyakrajbayar/s-and-p-500-complete-historical-dataset-50-years"

def setup_directories():
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

def process_forbes_dataset():
    print(f"[INFO]: Downloading Forbes dataset...")
    path = kagglehub.dataset_download(FORBES_DATASET)
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if csv_files:
        source_file = os.path.join(path, csv_files[0])
        target_file = os.path.join(DATASETS_DIR, "forbes_employers.csv")
        shutil.copy2(source_file, target_file)
        print(f"[INFO]: Forbes dataset successfully saved to {target_file}")
    else:
        print(f"[ERROR]: No CSV files found in the downloaded Forbes dataset.")

def process_sp500_dataset():
    print(f"[INFO]: Downloading S&P 500 dataset...")
    path = kagglehub.dataset_download(SP500_DATASET)
    
    # Check subdirectories first
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            
            target_csv = None
            for file_name in csv_files:
                if "company" in file_name.lower() or "info" in file_name.lower() or "constituents" in file_name.lower():
                    target_csv = file_name
                    break
                    
            if not target_csv and csv_files:
                target_csv = csv_files[0]
                
            if target_csv:
                source_file = os.path.join(subdir_path, target_csv)
                target_file = os.path.join(DATASETS_DIR, "sp500_companies.csv")
                shutil.copy2(source_file, target_file)
                print(f"[INFO]: S&P 500 company info successfully saved to {target_file}")
                return
    
    # Fallback: check direct path
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    target_csv = None
    for file_name in csv_files:
        if "company" in file_name.lower() or "info" in file_name.lower() or "constituents" in file_name.lower():
            target_csv = file_name
            break
            
    if not target_csv and csv_files:
        target_csv = csv_files[0]
        
    if target_csv:
        source_file = os.path.join(path, target_csv)
        target_file = os.path.join(DATASETS_DIR, "sp500_companies.csv")
        shutil.copy2(source_file, target_file)
        print(f"[INFO]: S&P 500 company info successfully saved to {target_file}")
    else:
        print(f"[ERROR]: No suitable company info CSV found in the S&P 500 dataset.")

if __name__ == "__main__":
    setup_directories()
    process_forbes_dataset()
    process_sp500_dataset()