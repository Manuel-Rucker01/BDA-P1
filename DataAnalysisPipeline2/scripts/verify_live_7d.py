"""
BDA Live 7-Day Out-of-Sample Verification.
This script tests our trained ensemble model's real-world predictive power by:
1. Ingesting yfinance pricing data up to exactly 7 calendar days ago (2026-05-15).
2. Generating features and running inference to predict the 7-day direction (UP/DOWN).
3. Fetching the actual close prices for today (2026-05-22) via yfinance.
4. Evaluating how many predictions the model got correct (Accuracy).
"""

import os
import pickle
import numpy as np
import pandas as pd
import duckdb
import yfinance as yf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")

# --- Helper Technical Indicators (matching pipeline exactly) ---

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, span_fast=12, span_slow=26):
    ema_fast = series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = series.ewm(span=span_slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_macd_signal(macd_series, span_signal=9):
    return macd_series.ewm(span=span_signal, adjust=False).mean()

def load_macro_features(macro_ttl_path: str):
    from rdflib import Graph as RdfGraph, Namespace
    g = RdfGraph()
    g.parse(macro_ttl_path, format="turtle")
    macro_onto = Namespace("http://bda.upc.edu/macro/ontology#")
    macro_ent = Namespace("http://bda.upc.edu/macro/resource/")

    rows = []
    for s in set(g.subjects()):
        if not str(s).startswith(str(macro_ent)):
            continue
        country = str(s).replace(str(macro_ent), "").replace("_", " ")
        gdp = g.value(s, macro_onto.gdpUSD)
        growth = g.value(s, macro_onto.gdpGrowthPercent)
        inflation = g.value(s, macro_onto.inflationPercent)
        trade = g.value(s, macro_onto.tradePercentOfGDP)
        interest = g.value(s, macro_onto.interestRatePercent)
        if gdp is not None or growth is not None or inflation is not None or trade is not None or interest is not None:
            rows.append({
                "country": country,
                "gdp_usd": float(gdp) if gdp is not None else None,
                "gdp_growth_pct": float(growth) if growth is not None else None,
                "inflation_pct": float(inflation) if inflation is not None else None,
                "trade_pct": float(trade) if trade is not None else None,
                "interest_rate_pct": float(interest) if interest is not None else None,
            })
    return pd.DataFrame(rows)

def fetch_company_metadata():
    db_path = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")
    conn = duckdb.connect(db_path, read_only=True)
    df_meta = conn.execute("""
        SELECT Symbol AS ticker, 
               ANY_VALUE(Sector) AS Sector, 
               ANY_VALUE(Industry) AS Industry,
               ANY_VALUE(MarketCap) AS MarketCap,
               ANY_VALUE(eur_rate) AS eur_rate,
               ANY_VALUE(jpy_rate) AS jpy_rate
        FROM master_dataset
        GROUP BY Symbol
    """).df()
    conn.close()
    
    trusted_db = os.path.abspath(os.path.join(EXPLOITATION_DIR, "..", "TrustedZone", "TrustedZone.duckdb"))
    conn_t = duckdb.connect(trusted_db, read_only=True)
    df_country = conn_t.execute("SELECT DISTINCT Symbol AS ticker, country FROM companies").df()
    conn_t.close()
    df_meta = df_meta.merge(df_country, on="ticker", how="left")
    return df_meta

def compute_live_features(live_df, metadata_df, macro_df):
    df = live_df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    
    # Merge static attributes & macroeconomic indicators
    df = df.merge(metadata_df, on="ticker", how="left")
    df = df.merge(macro_df, on="country", how="left")
    df = df.drop(columns=["country"], errors="ignore")
    
    # 1. log_market_cap
    df["log_market_cap"] = np.log(df["MarketCap"].replace(0, 1.0).astype(float))
    
    # 2. daily_return
    df['daily_return'] = df.groupby('ticker')['company_close'].pct_change(1).fillna(0)
    
    # 3. Cumulative returns
    df['return_5d'] = df.groupby('ticker')['company_close'].pct_change(5).fillna(0)
    df['return_10d'] = df.groupby('ticker')['company_close'].pct_change(10).fillna(0)
    df['return_20d'] = df.groupby('ticker')['company_close'].pct_change(20).fillna(0)
    df['return_50d'] = df.groupby('ticker')['company_close'].pct_change(50).fillna(0)
    
    # 4. Moving average ratios
    df['ma5'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['price_vs_ma5'] = (df['company_close'] - df['ma5']) / df['ma5'].replace(0, 1e-9)
    
    df['ma20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['price_vs_ma20'] = (df['company_close'] - df['ma20']) / df['ma20'].replace(0, 1e-9)
    
    # 5. Stochastic-style range
    df['min20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).min())
    df['max20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).max())
    df['stoch_20d'] = (df['company_close'] - df['min20']) / (df['max20'] - df['min20']).replace(0, 1e-9)
    df['stoch_20d'] = df['stoch_20d'].fillna(0)
    
    # 6. Volume ratio
    df['volume_ma5'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['volume_ratio'] = df['company_volume'] / df['volume_ma5'].replace(0, 1e-9)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    
    # 7. Volatilities
    df['rolling_volatility_5d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).std()).fillna(0)
    df['rolling_volatility_10d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(11, min_periods=1).std()).fillna(0)
    df['rolling_volatility_20d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).std()).fillna(0)
    
    # 8. Calendar
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month_of_year'] = pd.to_datetime(df['Date']).dt.month
    
    # 9. Vol adjusted return
    df['vol_adjusted_return'] = df['daily_return'] / df['rolling_volatility_10d'].replace(0, 1e-9)
    df['vol_adjusted_return'] = df['vol_adjusted_return'].fillna(0)
    
    # 10. Volume zscore
    df['vol_mean20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['vol_std20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).std())
    df['volume_zscore_20d'] = (df['company_volume'] - df['vol_mean20']) / df['vol_std20'].replace(0, 1e-9)
    df['volume_zscore_20d'] = df['volume_zscore_20d'].fillna(0)
    
    # Drop rolling intermediates
    df = df.drop(columns=['ma5', 'ma20', 'min20', 'max20', 'volume_ma5', 'vol_mean20', 'vol_std20'])
    
    # 11. Cross-sectional momentum
    df['sector_daily_return'] = df.groupby(['Sector', 'Date'])['daily_return'].transform('mean').fillna(0)
    df['sector_return_5d'] = df.groupby(['Sector', 'Date'])['return_5d'].transform('mean').fillna(0)
    
    # 12. Cross-sectional ranks
    df['rank_daily_return'] = df.groupby('Date')['daily_return'].rank(pct=True).fillna(0.5)
    df['rank_return_5d'] = df.groupby('Date')['return_5d'].rank(pct=True).fillna(0.5)
    df['rank_return_20d'] = df.groupby('Date')['return_20d'].rank(pct=True).fillna(0.5)
    df['rank_volatility'] = df.groupby('Date')['rolling_volatility_10d'].rank(pct=True).fillna(0.5)
    df['rank_volume_ratio'] = df.groupby('Date')['volume_ratio'].rank(pct=True).fillna(0.5)
    
    # 13. Advanced indicators (RSI, MACD, BB Width)
    df['rsi_14'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_rsi(x, 14)).fillna(50)
    df['macd'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_macd(x, 12, 26)).fillna(0)
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: compute_macd_signal(x, 9)).fillna(0)
    
    df['bb_mean'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['bb_std'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    df['bb_width'] = (4 * df['bb_std']) / df['bb_mean'].replace(0, 1e-9)
    df['bb_width'] = df['bb_width'].fillna(0)
    df = df.drop(columns=['bb_mean', 'bb_std'])
    
    # 14. Technical Lags
    for lag in [1, 2, 5]:
        df[f'daily_return_lag_{lag}'] = df.groupby('ticker')['daily_return'].shift(lag).fillna(0)
        df[f'volume_ratio_lag_{lag}'] = df.groupby('ticker')['volume_ratio'].shift(lag).fillna(1.0)
        
    return df

def main():
    print("=" * 80)
    print("BDA 7-DAY OUT-OF-SAMPLE MODEL VERIFIER")
    print("=" * 80)

    # 1. Load best model and structural data
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        return

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    trained_models = model_data["trained_models"]
    mix_models = model_data["mix_models"]
    scaler = model_data["scaler"]
    pca = model_data["pca"]
    tabular_cols = model_data["tabular_cols"]
    pca_cols = model_data["pca_cols"]
    company_embeddings = model_data["company_embeddings"]

    basket = list(company_embeddings.keys())[:20]
    print(f"Loaded ensemble model. Testing prediction for a 20-ticker basket.")
    
    # 2. Fetch price data up to exactly 7 calendar days ago (2026-05-15)
    # Today is May 22, 2026, so May 15 is 7 days ago.
    # We fetch a 65-day history up to May 15, 2026.
    print("\n[Ingestion] Fetching historical pricing up to May 15, 2026...")
    df_list = []
    for ticker in basket:
        ticker_clean = ticker.upper()
        # Fetching price bars from 70 days ago to May 16 to ensure we get the full 60d window up to May 15
        ticker_df = yf.download(ticker_clean, start="2026-03-01", end="2026-05-16", progress=False)
        if not ticker_df.empty:
            ticker_df = ticker_df.reset_index()
            ticker_df["ticker"] = ticker_clean
            ticker_df = ticker_df.rename(columns={
                "Date": "Date",
                "Close": "company_close",
                "Volume": "company_volume",
                "Open": "Open",
                "High": "High",
                "Low": "Low"
            })
            if isinstance(ticker_df.columns, pd.MultiIndex):
                ticker_df.columns = [col[0] for col in ticker_df.columns]
            ticker_df["Date"] = pd.to_datetime(ticker_df["Date"]).dt.strftime('%Y-%m-%d')
            df_list.append(ticker_df)

    if not df_list:
        print("[ERROR] No historical data fetched. Exiting.")
        return

    df_hist = pd.concat(df_list, ignore_index=True)
    
    # 3. Fetch current prices for today (May 22, 2026) to verify the actual outcome
    print("\n[Validation] Fetching current close prices for May 22, 2026...")
    actual_prices = {}
    price_on_may15 = {}
    for ticker in basket:
        live_data = yf.download(ticker, start="2026-05-14", end="2026-05-23", progress=False)
        if not live_data.empty:
            if isinstance(live_data.columns, pd.MultiIndex):
                live_data.columns = [col[0] for col in live_data.columns]
            # Get close price on May 15 and May 22
            live_data = live_data.reset_index()
            live_data["Date"] = pd.to_datetime(live_data["Date"]).dt.strftime('%Y-%m-%d')
            
            p_may15 = live_data[live_data["Date"] == "2026-05-15"]["Close"]
            p_may22 = live_data[live_data["Date"] == "2026-05-22"]["Close"]
            
            if not p_may15.empty and not p_may22.empty:
                price_on_may15[ticker] = float(p_may15.iloc[0])
                actual_prices[ticker] = float(p_may22.iloc[0])
            else:
                # Delisted/no bars
                pass

    # 4. Feature engineering on the historical slice
    print("\n[Processing] Running technical feature engineering on cut dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    
    df_features = compute_live_features(df_hist, metadata_df, macro_df)
    
    # Filter features to exactly May 15, 2026
    latest_df = df_features[df_features["Date"] == "2026-05-15"].copy()
    if latest_df.empty:
        # If May 15 wasn't a trading day for some reasons (should be, Friday), grab the last row per ticker
        latest_df = df_features.sort_values("Date").groupby("ticker").last().reset_index()
        print(f"[WARN] No observations exactly on 2026-05-15; using latest available dates: {latest_df['Date'].unique()}")
    else:
        print(f"Successfully extracted observations for {len(latest_df)} tickers on May 15, 2026.")
        
    # Join structural embeddings
    found_tickers = []
    emb_list = []
    for t in latest_df["ticker"].unique():
        if t in company_embeddings:
            emb_list.append(company_embeddings[t])
            found_tickers.append(t)
            
    latest_df = latest_df[latest_df["ticker"].isin(found_tickers)].copy()
    
    # Project embeddings
    raw_emb = np.array(emb_list)
    reduced_emb = pca.transform(raw_emb)
    emb_df = pd.DataFrame(reduced_emb, columns=pca_cols)
    emb_df["ticker"] = found_tickers
    
    latest_df = latest_df.merge(emb_df, on="ticker", how="inner")
    
    # Extract matrices
    X_tab = latest_df[tabular_cols].fillna(0).values.astype(np.float32)
    X_emb = latest_df[pca_cols].fillna(0).values.astype(np.float32)
    X_full = np.concatenate([X_tab, X_emb], axis=1)
    
    X_full_s = scaler.transform(X_full)
    
    # 5. Predict using Soft-Voting ensemble
    print("\n[Inference] Running Soft-Voting Ensemble on May 15 features...")
    model_probas = []
    for m in mix_models:
        if m in trained_models:
            y_proba = trained_models[m].predict_proba(X_full_s)[:, 1]
            model_probas.append(y_proba)
            
    latest_df["pred_proba"] = np.mean(model_probas, axis=0)
    
    # 6. Evaluate accuracy
    print("\n" + "=" * 90)
    print("VERIFICATION RESULTS: 7-DAY OUT-OF-SAMPLE BACKTEST (May 15 -> May 22)")
    print("=" * 90)
    print(f"{'Ticker':<8} | {'P(up)':<7} | {'Predicted':<10} | {'May 15 Price':<13} | {'May 22 Price':<13} | {'Actual Direction':<16} | {'Outcome':<8}")
    print("-" * 90)
    
    correct = 0
    total = 0
    
    for _, row in latest_df.iterrows():
        ticker = row["ticker"]
        pred_p = row["pred_proba"]
        pred_dir = "UP" if pred_p >= 0.5 else "DOWN"
        
        # Look up actual prices
        if ticker in actual_prices and ticker in price_on_may15:
            p_15 = price_on_may15[ticker]
            p_22 = actual_prices[ticker]
            actual_dir = "UP" if p_22 > p_15 else "DOWN"
            
            is_correct = (pred_dir == actual_dir)
            if is_correct:
                correct += 1
                result_str = "CORRECT"
            else:
                result_str = "WRONG"
                
            total += 1
            print(f"{ticker:<8} | {pred_p:<7.3f} | {pred_dir:<10} | ${p_15:<12.2f} | ${p_22:<12.2f} | {actual_dir:<16} | {result_str:<8}")
            
    print("-" * 90)
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"TOTAL EVALUATED TICKERS: {total}")
        print(f"MODEL OUT-OF-SAMPLE ACCURACY: {accuracy:.2f}% ({correct}/{total} correct)")
    else:
        print("[ERROR] No tickers could be validated due to lack of historical/live price match.")
    print("=" * 90)

if __name__ == "__main__":
    main()
