import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ExploitationZone"))
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")

def train_optimized_mlp_v5():
    print("--- Optimizing MLP V5 with GridSearch and Temporal Validation ---")
    
    # 1. READ DIRECTLY FROM DUCKDB IN THE EXPLOITATION ZONE
    print(f"[INFO] Reading table 'master_dataset' from: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT * FROM master_dataset").df().dropna()
    conn.close()

    # 2. DATE PARSING AND SORTING
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    # --- FEATURE ENGINEERING ---
    df['company_daily_pct'] = df.groupby('Symbol')['company_close'].pct_change()
    df['volume_pressure'] = df['company_volume'] / df.groupby('Symbol')['company_volume'].transform('mean')
    df = df.dropna().sort_values('Date')

    # Split temporal (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # we include 'company_daily_pct' and 'volume_pressure' as new features, which are common in stock prediction tasks.
    features = ['Sector', 'Industry', 'MarketCap', 'company_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']
    X_train, y_train = train_df[features], train_df['target_7d_up']
    X_test, y_test = test_df[features], test_df['target_7d_up']

    # Preprocessing: we apply StandardScaler to numerical features and 
    # OneHotEncoder to categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['MarketCap', 'company_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Sector', 'Industry'])
        ])

    # Pipeline with SMOTE and MLP
    pipeline = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('resample', SMOTE(random_state=42)),
        ('mlp', MLPClassifier(max_iter=400, early_stopping=True, random_state=42))
    ])

    # Parameters for GridSearch: we try different combinations
    # of hyperparameters to find the best model.
    param_grid = {
        'mlp__hidden_layer_sizes': [(128, 64), (64, 32), (100,)], # Different architectures
        'mlp__activation': ['tanh', 'relu'],                      # Activation functions
        'mlp__alpha': [0.0001, 0.05],                             # Regularization (L2)
        'mlp__learning_rate_init': [0.001, 0.01]                  # Learning rate
    }

    # Temporal cross-validation: to ensure that the model is validated on future data.
    tscv = TimeSeriesSplit(n_splits=3)

    print("Starting GridSearchCV with temporal validation...")
    # We use f1-score as the scoring metric because we want to balance precision and recall in this imbalanced classification problem.
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=tscv, 
        n_jobs=-1, 
        scoring='f1', 
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")

    # Final evaluation on the test set.
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n" + "="*45)
    print("       Results of MLP Classifier on Test Set       ")
    print("="*45)
    print(f"Accuracy Final: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return best_model

if __name__ == "__main__":
    train_optimized_mlp_v5()