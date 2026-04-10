import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ExploitationZone"))
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")

def train_rf_v5():
    print("--- Train Random Forest ---")
    
    # 1. READ DIRECTLY FROM DUCKDB IN THE EXPLOITATION ZONE
    print(f"[INFO] Reading table 'master_dataset' from: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT * FROM master_dataset").df().dropna()
    conn.close()

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. sort by Symbol and Date to ensure temporal order for each company
    df = df.sort_values(['Symbol', 'Date'])

    # --- FEATURE ENGINEERING (same as MLP for comparability) ---
    df['company_daily_pct'] = df.groupby('Symbol')['company_close'].pct_change()
    df['volume_pressure'] = df['company_volume'] / df.groupby('Symbol')['company_volume'].transform('mean')
    
    df = df.dropna()
    df = df.sort_values('Date')

    # 3. split temporal (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Same features as MLP for a fair comparison.
    features = [
        'Sector', 'Industry', 'MarketCap', 
        'company_daily_pct', 
        'volume_pressure', 
        'eur_rate', 'jpy_rate'
    ]
    
    X_train, y_train = train_df[features], train_df['target_7d_up']
    X_test, y_test = test_df[features], test_df['target_7d_up']

    # 4. Preprocessing: we apply StandardScaler to numerical features and 
    # OneHotEncoder to categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['MarketCap', 'company_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Sector', 'Industry'])
        ])

    # 5. Pipeline with SMOTE and Random Forest: we use SMOTE to handle class 
    # imbalance and Random Forest for classification.
    pipeline = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('resample', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            max_depth=12,      # we limit the depth to prevent overfitting on this small dataset
            min_samples_leaf=5, # we ensure the model learns general patterns
            random_state=42, 
            n_jobs=-1
        ))
    ])

    print(f"Training on {len(X_train)} records...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    print("\n" + "="*45)
    print("      Results of Random Forest Classifier on Test Set      ")
    print("="*45)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline

if __name__ == "__main__":
    train_rf_v5()