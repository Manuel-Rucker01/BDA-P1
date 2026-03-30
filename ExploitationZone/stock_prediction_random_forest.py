import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def train_rf_v5():
    print("--- 🌲 Entrenando Random Forest de Mercado (V5 - Realista) ---")
    df = pd.read_csv("master_dataset_pro.csv").dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. ORDENAR POR SÍMBOLO Y FECHA PARA CALCULAR VARIACIONES
    df = df.sort_values(['Symbol', 'Date'])

    # --- FEATURE ENGINEERING (Mismo que en MLP para ser comparables) ---
    # Usamos el cambio porcentual en lugar del precio fijo
    df['sp500_daily_pct'] = df.groupby('Symbol')['sp500_close'].pct_change()
    df['volume_pressure'] = df['sp500_volume'] / df.groupby('Symbol')['sp500_volume'].transform('mean')
    
    df = df.dropna()
    df = df.sort_values('Date')

    # 2. SPLIT TEMPORAL (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Mismas variables que el modelo MLP
    features = [
        'Sector', 'Industry', 'MarketCap', 
        'sp500_daily_pct', 
        'volume_pressure', 
        'eur_rate', 'jpy_rate'
    ]
    
    X_train, y_train = train_df[features], train_df['target_7d_up']
    X_test, y_test = test_df[features], test_df['target_7d_up']

    # 3. PREPROCESAMIENTO
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['MarketCap', 'sp500_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Sector', 'Industry'])
        ])

    # 4. PIPELINE: Preprocesar -> SMOTE -> Random Forest
    pipeline = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('resample', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            max_depth=12,      # Limitamos profundidad para evitar que memorice de más
            min_samples_leaf=5, # Asegura que el modelo aprenda patrones generales
            random_state=42, 
            n_jobs=-1
        ))
    ])

    print(f"Entrenando con {len(X_train)} registros...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    print("\n" + "="*45)
    print("       RESULTADOS RANDOM FOREST (CIENCIA REAL)")
    print("="*45)
    print(f"Accuracy Realista: {accuracy_score(y_test, y_pred):.4f}")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nInforme:")
    print(classification_report(y_test, y_pred))

    return pipeline

if __name__ == "__main__":
    train_rf_v5()