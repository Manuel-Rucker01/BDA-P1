#!/usr/bin/env python3
"""
Unified Preprocessing Pipeline Runner
Runs the full end-to-end data engineering pipeline from raw CSVs to structured
DuckDB databases and Semantic Knowledge Graphs (RDF).
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")

# Sequence of preprocessing scripts to run
PIPELINE_STEPS = [
    {
        "name": "Formatted Zone Ingestion",
        "script": os.path.join("FormattedZone", "formatted_zone_pipeline.py"),
        "description": "Ingest raw CSV files into DuckDB using Apache Spark."
    },
    {
        "name": "Trusted Zone Data Quality",
        "script": os.path.join("TrustedZone", "dataQuality.py"),
        "description": "Apply data quality rules / Denial Constraints and resolve HQ countries."
    },
    {
        "name": "Exploitation Zone Data Integration",
        "script": os.path.join("ExploitationZone", "data_integration.py"),
        "description": "Build tabular master dataset in ExploitationZone.duckdb."
    },
    {
        "name": "Macroeconomic & Geopolitical Graph",
        "script": os.path.join("ExploitationZone", "geopolitical_macroeconomic.py"),
        "description": "Fetch macroeconomic indicators and construct macroeconomic_graph.ttl."
    },
    {
        "name": "Financial Knowledge Graph",
        "script": os.path.join("ExploitationZone", "graph_generation.py"),
        "description": "Construct financial_knowledge_graph.ttl combining financial and geopolitical data."
    }
]

REQUIRED_DATASETS = [
    "nasdaq_companies.csv",
    "company_history.csv",
    "US_exchange.csv",
    "sp500_companies.csv",
    "forbes_employers.csv",
    "company_acquisitions.csv"
]

def check_datasets():
    print("=" * 80)
    print("CHECKING RAW DATASETS")
    print("=" * 80)
    
    if not os.path.exists(DATASETS_DIR):
        print(f"[ERROR] Datasets directory not found at: {DATASETS_DIR}")
        return False
        
    missing = []
    for dataset in REQUIRED_DATASETS:
        path = os.path.join(DATASETS_DIR, dataset)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [FOUND] {dataset:<30} ({size_mb:.2f} MB)")
        else:
            missing.append(dataset)
            
    if missing:
        print(f"\n[ERROR] The following required datasets are missing in datasets/: {missing}")
        print("Please run LandingZone/ scripts to download them first, e.g.:")
        print("  python3 LandingZone/nasdaq.py")
        print("  python3 LandingZone/company_history.py")
        print("  python3 LandingZone/exchange.py")
        print("  python3 LandingZone/additional_information.py")
        return False
        
    print("\nAll required datasets are present.")
    return True

def run_step(step):
    name = step["name"]
    script_path = os.path.join(SCRIPT_DIR, step["script"])
    desc = step["description"]
    
    print("\n" + "=" * 80)
    print(f"STAGE: {name.upper()}")
    print(f"Script: {step['script']}")
    print(f"Description: {desc}")
    print("=" * 80)
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found at: {script_path}")
        return False
        
    t0 = time.time()
    try:
        # Run script as subprocess using the same python executable
        # Use SCRIPT_DIR as Cwd to ensure relative paths inside scripts work correctly
        env = os.environ.copy()
        # Set JAVA_HOME check or other environment if needed
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            env=env,
            capture_output=False, # Print straight to console
            text=True
        )
        
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n[SUCCESS] {name} completed in {elapsed:.2f} seconds.")
            return True
        else:
            print(f"\n[FAILED] {name} failed with exit code {result.returncode} after {elapsed:.2f} seconds.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Exception occurred while running {name}: {e}")
        return False

def main():
    print("=" * 80)
    print("BDA END-TO-END PREPROCESSING PIPELINE RUNNER")
    print("=" * 80)
    
    # 1. Verify datasets exist
    if not check_datasets():
        sys.exit(1)
        
    # 2. Run steps in sequence
    start_time = time.time()
    success_steps = 0
    
    for step in PIPELINE_STEPS:
        success = run_step(step)
        if not success:
            print("\n" + "!" * 80)
            print(f"CRITICAL ERROR: Pipeline stopped at step '{step['name']}' due to failure.")
            print("!" * 80)
            sys.exit(1)
        success_steps += 1
        
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("BDA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Successfully executed {success_steps}/{len(PIPELINE_STEPS)} stages.")
    print(f"Total time elapsed: {total_time/60:.2f} minutes ({total_time:.2f} seconds).")
    print("=" * 80)

if __name__ == "__main__":
    main()
