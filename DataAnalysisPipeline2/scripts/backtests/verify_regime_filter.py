"""
BDA S&P 500 Broad-Market Trend Regime Filter Backtester.
This script compares:
1. Buy & Hold Benchmark
2. Unfiltered Probabilistic Long/Short
3. Regime-Filtered Probabilistic Long/Short (S&P 500 Close > 50-day SMA shuts off shorts)
4. High-Confidence Longs (P >= 0.53) reference
Across 3, 6, 12, and 24-month horizons ending on May 22, 2026.
"""

import os
import pickle
import numpy as np
import pandas as pd
import duckdb
import yfinance as yf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "ExploitationZone"))
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")

# --- Helper Technical Indicators ---

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
    
    df = df.merge(metadata_df, on="ticker", how="left")
    df = df.merge(macro_df, on="country", how="left")
    df = df.drop(columns=["country"], errors="ignore")
    
    df["log_market_cap"] = np.log(df["MarketCap"].replace(0, 1.0).astype(float))
    df['daily_return'] = df.groupby('ticker')['company_close'].pct_change(1).fillna(0)
    
    df['return_5d'] = df.groupby('ticker')['company_close'].pct_change(5).fillna(0)
    df['return_10d'] = df.groupby('ticker')['company_close'].pct_change(10).fillna(0)
    df['return_20d'] = df.groupby('ticker')['company_close'].pct_change(20).fillna(0)
    df['return_50d'] = df.groupby('ticker')['company_close'].pct_change(50).fillna(0)
    
    df['ma5'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['price_vs_ma5'] = (df['company_close'] - df['ma5']) / df['ma5'].replace(0, 1e-9)
    
    df['ma20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['price_vs_ma20'] = (df['company_close'] - df['ma20']) / df['ma20'].replace(0, 1e-9)
    
    df['min20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).min())
    df['max20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).max())
    df['stoch_20d'] = (df['company_close'] - df['min20']) / (df['max20'] - df['min20']).replace(0, 1e-9)
    df['stoch_20d'] = df['stoch_20d'].fillna(0)
    
    df['volume_ma5'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['volume_ratio'] = df['company_volume'] / df['volume_ma5'].replace(0, 1e-9)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    
    df['rolling_volatility_5d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).std()).fillna(0)
    df['rolling_volatility_10d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(11, min_periods=1).std()).fillna(0)
    df['rolling_volatility_20d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).std()).fillna(0)
    
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month_of_year'] = pd.to_datetime(df['Date']).dt.month
    
    df['vol_adjusted_return'] = df['daily_return'] / df['rolling_volatility_10d'].replace(0, 1e-9)
    df['vol_adjusted_return'] = df['vol_adjusted_return'].fillna(0)
    
    df['vol_mean20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['vol_std20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).std())
    df['volume_zscore_20d'] = (df['company_volume'] - df['vol_mean20']) / df['vol_std20'].replace(0, 1e-9)
    df['volume_zscore_20d'] = df['volume_zscore_20d'].fillna(0)
    
    df = df.drop(columns=['ma5', 'ma20', 'min20', 'max20', 'volume_ma5', 'vol_mean20', 'vol_std20'])
    
    df['sector_daily_return'] = df.groupby(['Sector', 'Date'])['daily_return'].transform('mean').fillna(0)
    df['sector_return_5d'] = df.groupby(['Sector', 'Date'])['return_5d'].transform('mean').fillna(0)
    
    df['rank_daily_return'] = df.groupby('Date')['daily_return'].rank(pct=True).fillna(0.5)
    df['rank_return_5d'] = df.groupby('Date')['return_5d'].rank(pct=True).fillna(0.5)
    df['rank_return_20d'] = df.groupby('Date')['return_20d'].rank(pct=True).fillna(0.5)
    df['rank_volatility'] = df.groupby('Date')['rolling_volatility_10d'].rank(pct=True).fillna(0.5)
    df['rank_volume_ratio'] = df.groupby('Date')['volume_ratio'].rank(pct=True).fillna(0.5)
    
    df['rsi_14'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_rsi(x, 14)).fillna(50)
    df['macd'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_macd(x, 12, 26)).fillna(0)
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: compute_macd_signal(x, 9)).fillna(0)
    
    df['bb_mean'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['bb_std'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    df['bb_width'] = (4 * df['bb_std']) / df['bb_mean'].replace(0, 1e-9)
    df['bb_width'] = df['bb_width'].fillna(0)
    df = df.drop(columns=['bb_mean', 'bb_std'])
    
    for lag in [1, 2, 5]:
        df[f'daily_return_lag_{lag}'] = df.groupby('ticker')['daily_return'].shift(lag).fillna(0)
        df[f'volume_ratio_lag_{lag}'] = df.groupby('ticker')['volume_ratio'].shift(lag).fillna(1.0)
        
    return df

def calculate_weights_probabilistic(df_predictions, is_bull=False, target_exposure=1.0, confidence_threshold=0.02):
    df = df_predictions.copy()
    df["raw_weight"] = df["pred_proba"] - 0.5
    
    longs = df[df["raw_weight"] >= confidence_threshold].copy()
    
    if is_bull:
        # Bull Regime: Suppress all short positions (set shorts to empty)
        shorts = pd.DataFrame()
    else:
        # Bear Regime: Allow standard short positions
        shorts = df[df["raw_weight"] <= -confidence_threshold].copy()
        
    if len(longs) == 0 and len(shorts) == 0:
        return {}
        
    selected = pd.concat([longs, shorts])
    total_abs_weight = selected["raw_weight"].abs().sum()
    
    if total_abs_weight == 0:
        return {}
        
    selected["target_weight"] = (selected["raw_weight"] / total_abs_weight) * target_exposure
    return selected.set_index("ticker")["target_weight"].to_dict()

def run_backtest_for_horizon(df_all_feat, df_full, gspc_dict, friday_dates, company_embeddings, scaler, pca, trained_models, mix_models, tabular_cols, pca_cols, start_date, initial_equity=10000.0):
    horizon_fridays = [d for d in friday_dates if d >= start_date]
    if not horizon_fridays:
        return 0.0, 0.0, 0.0, 0.0, 0, 0
    
    portfolio_bh = [initial_equity]
    portfolio_unfiltered = [initial_equity]
    portfolio_filtered = [initial_equity]
    portfolio_high_long = [initial_equity]
    
    bull_weeks = 0
    bear_weeks = 0
    
    for idx, friday in enumerate(horizon_fridays):
        # Determine regime of S&P 500 on this Friday
        # Fallback to the closest date in gspc_dict if exact date isn't found
        if friday in gspc_dict:
            gspc_close = gspc_dict[friday]["Close_GSPC"]
            gspc_sma50 = gspc_dict[friday]["SMA50_GSPC"]
        else:
            # Look backwards up to 3 days to find a close print
            fallback_date = friday
            for lag in range(1, 4):
                test_date = (pd.to_datetime(friday) - pd.Timedelta(days=lag)).strftime('%Y-%m-%d')
                if test_date in gspc_dict:
                    fallback_date = test_date
                    break
            gspc_close = gspc_dict.get(fallback_date, {}).get("Close_GSPC", 0.0)
            gspc_sma50 = gspc_dict.get(fallback_date, {}).get("SMA50_GSPC", 0.0)
            
        is_bull = gspc_close > gspc_sma50
        if is_bull:
            bull_weeks += 1
        else:
            bear_weeks += 1
            
        # Get Friday features for our basket
        friday_obs = df_all_feat[df_all_feat["Date"] == friday].copy()
        found_tickers = [t for t in friday_obs["ticker"].unique() if t in company_embeddings]
        friday_obs = friday_obs[friday_obs["ticker"].isin(found_tickers)].copy()
        
        if friday_obs.empty:
            portfolio_bh.append(portfolio_bh[-1])
            portfolio_unfiltered.append(portfolio_unfiltered[-1])
            portfolio_filtered.append(portfolio_filtered[-1])
            portfolio_high_long.append(portfolio_high_long[-1])
            continue
            
        # KG projection
        emb_list = [company_embeddings[t] for t in found_tickers]
        raw_emb = np.array(emb_list)
        reduced_emb = pca.transform(raw_emb)
        emb_df = pd.DataFrame(reduced_emb, columns=pca_cols)
        emb_df["ticker"] = found_tickers
        
        friday_obs = friday_obs.merge(emb_df, on="ticker", how="inner")
        
        X_tab = friday_obs[tabular_cols].fillna(0).values.astype(np.float32)
        X_emb = friday_obs[pca_cols].fillna(0).values.astype(np.float32)
        X_full = np.concatenate([X_tab, X_emb], axis=1)
        X_full_s = scaler.transform(X_full)
        
        # Convert back to a DataFrame with identical feature names to prevent scikit-learn/LGBM warnings
        X_full_df = pd.DataFrame(X_full_s, columns=tabular_cols + pca_cols)

        # Inference
        import warnings
        model_probas = []
        for m in mix_models:
            if m in trained_models:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    y_proba = trained_models[m].predict_proba(X_full_df)[:, 1]
                model_probas.append(y_proba)
        
        friday_obs["pred_proba"] = np.mean(model_probas, axis=0)
        
        # Get next Friday (or final date May 22, 2026)
        if idx + 1 < len(horizon_fridays):
            next_friday = horizon_fridays[idx + 1]
        else:
            next_friday = "2026-05-22"
            
        curr_prices = friday_obs.set_index("ticker")["company_close"].to_dict()
        next_prices_df = df_full[df_full["Date"] == next_friday].set_index("ticker")["company_close"].to_dict()
        
        ticker_returns = {}
        for t in found_tickers:
            if t in curr_prices and t in next_prices_df:
                p0 = curr_prices[t]
                p1 = next_prices_df[t]
                if p0 > 0:
                    ticker_returns[t] = (p1 - p0) / p0
                    
        # 1. B&H Return
        bh_ret = np.mean(list(ticker_returns.values())) if ticker_returns else 0.0
        
        # 2. Unfiltered Weights & Return
        weights_unfiltered = calculate_weights_probabilistic(friday_obs, is_bull=False, target_exposure=1.0)
        ret_unfiltered = sum(w * ticker_returns.get(t, 0.0) for t, w in weights_unfiltered.items())
        
        # 3. Filtered Weights & Return
        weights_filtered = calculate_weights_probabilistic(friday_obs, is_bull=is_bull, target_exposure=1.0)
        ret_filtered = sum(w * ticker_returns.get(t, 0.0) for t, w in weights_filtered.items())
        
        # 4. High-Confidence Long Return
        high_long_df = friday_obs[friday_obs["pred_proba"] >= 0.53]
        if high_long_df.empty:
            high_long_df = friday_obs.sort_values("pred_proba", ascending=False).head(2)
        high_long_tickers = high_long_df["ticker"].tolist()
        high_long_returns = [ticker_returns[t] for t in high_long_tickers if t in ticker_returns]
        ret_high_long = np.mean(high_long_returns) if high_long_returns else 0.0
        
        # Compound portfolios
        portfolio_bh.append(portfolio_bh[-1] * (1.0 + bh_ret))
        portfolio_unfiltered.append(portfolio_unfiltered[-1] * (1.0 + ret_unfiltered))
        portfolio_filtered.append(portfolio_filtered[-1] * (1.0 + ret_filtered))
        portfolio_high_long.append(portfolio_high_long[-1] * (1.0 + ret_high_long))
        
    return portfolio_bh[-1], portfolio_unfiltered[-1], portfolio_filtered[-1], portfolio_high_long[-1], bull_weeks, bear_weeks

def main():
    print("=" * 110)
    print("BDA BROAD-MARKET TREND REGIME FILTER BACKTESTER (S&P 500 SMA50)")
    print("=" * 110)

    # 1. Load S&P 500 history for trend detection
    print("\n[Ingestion] Fetching S&P 500 (^GSPC) price history from 2024-01-01 to 2026-05-22...")
    gspc_df = yf.download("^GSPC", start="2024-01-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [col[0] for col in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime('%Y-%m-%d')
    gspc_df = gspc_df.sort_values("Date").reset_index(drop=True)
    gspc_df["Close_GSPC"] = gspc_df["Close"]
    gspc_df["SMA50_GSPC"] = gspc_df["Close_GSPC"].rolling(window=50, min_periods=50).mean()
    gspc_trends = gspc_df[["Date", "Close_GSPC", "SMA50_GSPC"]].dropna().copy()
    gspc_dict = gspc_trends.set_index("Date").to_dict(orient="index")
    print(f"Loaded {len(gspc_dict)} S&P 500 trend dates. Latest close: ${gspc_trends.iloc[-1]['Close_GSPC']:,.2f} | SMA50: ${gspc_trends.iloc[-1]['SMA50_GSPC']:,.2f}")

    # 2. Load model
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
    import sys
    ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    if ROOT_PATH not in sys.path:
        sys.path.append(ROOT_PATH)
    try:
        from DataAnalysisPipeline2.trading_agent.config import TICKERS as basket
        print(f"[Config] Successfully loaded {len(basket)} sector-diversified tickers from config.")
    except Exception as e:
        basket = list(company_embeddings.keys())[:20]
        print(f"[Warning] Failed to import from config ({e}). Falling back to alphabetical: {basket}")
    
    print(f"Loaded ensemble. Representative basket has {len(basket)} active symbols.")

    # 3. Fetch asset data (from 2024-03-01 to cover 24-month horizon starts 2024-05-24)
    print("\n[Ingestion] Fetching asset historical prices from 2024-03-01 to 2026-05-22...")
    df_list = []
    for ticker in basket:
        ticker_df = yf.download(ticker, start="2024-03-01", end="2026-05-23", progress=False)
        if not ticker_df.empty:
            ticker_df = ticker_df.reset_index()
            ticker_df["ticker"] = ticker
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
        print("[ERROR] No asset prices fetched. Exiting.")
        return

    df_full = pd.concat(df_list, ignore_index=True)
    
    # Friday rebalance dates
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4) & 
        (dates_df["Date"] >= "2024-05-24") & 
        (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    
    friday_dates = fridays_df["Date"].tolist()
    print(f"Resolved {len(friday_dates)} Friday rebalancing periods.")

    # 4. Precompute features
    print("[Processing] Pre-calculating indicators on full dataset once...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)

    # 5. Run comparisons for horizons
    horizons = {
        "3 Months (Feb 20, 2026)": "2026-02-20",
        "6 Months (Nov 21, 2025)": "2025-11-21",
        "12 Months (May 23, 2025)": "2025-05-23",
        "24 Months (May 24, 2024)": "2024-05-24"
    }

    initial_capital = 10000.0
    print("\n" + "=" * 125)
    print(f"{'Horizon':<24} | {'Regime Breakdown':<18} | {'Strategy Name':<38} | {'Cum Return %':<16} | {'Ending Value ($)':<16}")
    print("=" * 125)

    for label, start_date in horizons.items():
        bh_val, unfil_val, fil_val, high_val, bulls, bears = run_backtest_for_horizon(
            df_all_feat, df_full, gspc_dict, friday_dates, company_embeddings,
            scaler, pca, trained_models, mix_models, tabular_cols, pca_cols,
            start_date, initial_equity=initial_capital
        )
        
        bh_ret = (bh_val - initial_capital) / initial_capital * 100.0
        unfil_ret = (unfil_val - initial_capital) / initial_capital * 100.0
        fil_ret = (fil_val - initial_capital) / initial_capital * 100.0
        high_ret = (high_val - initial_capital) / initial_capital * 100.0
        
        regime_str = f"{bulls} Bull / {bears} Bear"
        
        print(f"{label:<24} | {regime_str:<18} | {'Buy & Hold Basket (Benchmark)':<38} | {bh_ret:>14.2f}% | ${bh_val:>14,.2f}")
        print(f"{'':<24} | {'':<18} | {'Unfiltered Probabilistic Long/Short':<38} | {unfil_ret:>14.2f}% | ${unfil_val:>14,.2f}")
        print(f"{'':<24} | {'':<18} | {'Regime-Filtered Long/Short (SMA50)':<38} | {fil_ret:>14.2f}% | ${fil_val:>14,.2f}")
        print(f"{'':<24} | {'':<18} | {'High-Confidence Longs (Reference)':<38} | {high_ret:>14.2f}% | ${high_val:>14,.2f}")
        print("-" * 125)

if __name__ == "__main__":
    main()
