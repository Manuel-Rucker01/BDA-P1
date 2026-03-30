import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def train_optimized_mlp_v5():
    print("--- 🧠 Optimizando MLP V5 con GridSearch y Validación Temporal ---")
    df = pd.read_csv("master_dataset_pro.csv").dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    # --- FEATURE ENGINEERING ---
    df['sp500_daily_pct'] = df.groupby('Symbol')['sp500_close'].pct_change()
    df['volume_pressure'] = df['sp500_volume'] / df.groupby('Symbol')['sp500_volume'].transform('mean')
    df = df.dropna().sort_values('Date')

    # Split temporal
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    features = ['Sector', 'Industry', 'MarketCap', 'sp500_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']
    X_train, y_train = train_df[features], train_df['target_7d_up']
    X_test, y_test = test_df[features], test_df['target_7d_up']

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['MarketCap', 'sp500_daily_pct', 'volume_pressure', 'eur_rate', 'jpy_rate']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Sector', 'Industry'])
        ])

    # Pipeline con SMOTE y MLP
    pipeline = ImbPipeline(steps=[
        ('prep', preprocessor),
        ('resample', SMOTE(random_state=42)),
        ('mlp', MLPClassifier(max_iter=400, early_stopping=True, random_state=42))
    ])

    # --- ESPACIO DE BÚSQUEDA (Grid) ---
    param_grid = {
        'mlp__hidden_layer_sizes': [(128, 64), (64, 32), (100,)], # Diferentes arquitecturas
        'mlp__activation': ['tanh', 'relu'],                      # Funciones de activación
        'mlp__alpha': [0.0001, 0.05],                             # Regularización (L2)
        'mlp__learning_rate_init': [0.001, 0.01]                  # Velocidad de aprendizaje
    }

    # Validación temporal para no hacer trampas
    tscv = TimeSeriesSplit(n_splits=3)

    print("Iniciando búsqueda... Esto puede tardar varios minutos.")
    # Usamos scoring 'f1' para priorizar la detección de subidas
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=tscv, 
        n_jobs=-1, 
        scoring='f1', 
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ Mejores parámetros: {grid_search.best_params_}")

    # Evaluación final con el TEST
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n" + "="*45)
    print("       RESULTADOS MLP TRAS GRIDSEARCH")
    print("="*45)
    print(f"Accuracy Final: {accuracy_score(y_test, y_pred):.4f}")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nInforme:")
    print(classification_report(y_test, y_pred))

    return best_model

if __name__ == "__main__":
    train_optimized_mlp_v5()