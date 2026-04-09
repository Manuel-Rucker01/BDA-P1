import pandas as pd
import joblib
import os

def menu_sectores():
    sectores = [
        "Technology", "Financial Services", "Healthcare", 
        "Consumer Cyclical", "Communication Services", "Industrial"
    ]
    print("\nSectores disponibles:")
    for i, s in enumerate(sectores, 1):
        print(f"{i}. {s}")
    
    opcion = int(input("\nSelecciona el número del sector (o escribe 1 para Technology): ") or 1)
    return sectores[opcion-1]

def ejecutar_bola_cristal():
    print("="*50)
    print("🔮 SISTEMA DE PREDICCIÓN S&P 500 - BOLA DE CRISTAL v5.0 🔮")
    print("="*50)

    # 1. Intentar cargar el modelo guardado por el script anterior
    archivo_modelo = 'modelo_final_sp500.joblib'
    
    if not os.path.exists(archivo_modelo):
        print(f"❌ ERROR: No se encuentra el archivo '{archivo_modelo}'.")
        print("Asegúrate de ejecutar primero el script de entrenamiento.")
        return

    model = joblib.load(archivo_modelo)
    print("✅ Inteligencia Artificial cargada y lista.")

    # 2. Captura de datos del usuario
    try:
        print("\n--- PASO 1: Contexto de la Empresa ---")
        sector_elegido = menu_sectores()
        mcap = float(input("Capitalización de Mercado (ej: 1000000000 para 1B): ") or 1000000000)

        print("\n--- PASO 2: Estado del Mercado HOY ---")
        pct_hoy = float(input("Variación % del S&P 500 hoy (ej: 0.01 para +1%, -0.005 para -0.5%): "))
        vol_ratio = float(input("Presión de Volumen (1.0 = media, 1.5 = mucho volumen): ") or 1.0)
        eur = float(input("Tipo de cambio EUR/USD (ej: 1.08): "))
        jpy = float(input("Tipo de cambio USD/JPY (ej: 150.5): "))

        # 3. Preparar los datos para el modelo
        # Importante: Usamos los mismos nombres de columnas que en el entrenamiento
        input_data = pd.DataFrame([{
            'Sector': sector_elegido,
            'Industry': 'Software—Infrastructure', # Valor genérico para la industria
            'MarketCap': mcap,
            'sp500_daily_pct': pct_hoy,
            'volume_pressure': vol_ratio,
            'eur_rate': eur,
            'jpy_rate': jpy
        }])

        # 4. Realizar Predicción
        prediccion = model.predict(input_data)[0]
        probabilidad = model.predict_proba(input_data)[0][1]

        # 5. Mostrar Resultado Estético
        print("\n" + "*"*50)
        print("🔍 RESULTADO DEL ANÁLISIS CIENTÍFICO (7 DÍAS VISTA)")
        print("*"*50)
        print(f"Probabilidad de que el S&P 500 suba >0.5%: {probabilidad*100:.2f}%")
        
        if probabilidad > 0.55:
            print("\nVERDICTO: 💹 ALCISTA - El escenario es favorable para invertir.")
        elif probabilidad < 0.45:
            print("\nVERDICTO: 📉 BAJISTA/LATERAL - Riesgo de caída o estancamiento.")
        else:
            print("\nVERDICTO: ⚖️ NEUTRAL - El modelo no ve una tendencia clara.")
        print("*"*50)

    except Exception as e:
        print(f"\n❌ Error en la entrada de datos: {e}")

if __name__ == "__main__":
    ejecutar_bola_cristal()