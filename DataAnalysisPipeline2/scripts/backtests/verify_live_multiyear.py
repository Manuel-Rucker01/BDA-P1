"""
BDA Multi-Year Out-of-Sample Verification for Long Predictions.
This script tests our trained ensemble model's real-world long-only performance by:
1. Ingesting yfinance pricing data from 2022-10-01 to 2026-05-22 (3.5 years).
2. Implementing a look-ahead-bias-free vectorized feature pre-computation.
3. For each Friday in our testing window (2023-01-06 to 2026-05-15):
   a. Running inference on data up to that Friday.
   b. Filtering for Long predictions (P(up) >= 0.5) and High-Confidence Longs (P(up) >= 0.53).
   c. Calculating actual returns over the subsequent week.
4. Printing multi-year returns, win rates, and annualized metrics.
5. Saving a comparison performance chart to ExploitationZone/multiyear_rolling_performance.png.
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
OUTPUT_PLOT_PATH = os.path.join(EXPLOITATION_DIR, "multiyear_rolling_performance.png")

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
    
    trusted_db = os.path.abspath(os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb"))
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
    print("BDA MULTI-YEAR ROLLING LONG PERFORMANCE TESTER")
    print("=" * 80)

    # 1. Load model and metadata
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
    print(f"Loaded ensemble. Selecting representational 20-ticker basket.")

    # 2. Fetch price data over full multi-year window (2022-10-01 to 2026-05-22)
    print("\n[Ingestion] Fetching live multi-year pricing bars (2022-10-01 to 2026-05-22)...")
    df_list = []
    for ticker in basket:
        ticker_df = yf.download(ticker, start="2022-10-01", end="2026-05-23", progress=False)
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
        print("[ERROR] No historical data fetched. Exiting.")
        return

    df_full = pd.concat(df_list, ignore_index=True)
    
    # 3. Resolve rebalance Fridays in testing range (2023-01-06 to 2026-05-15)
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4) & 
        (dates_df["Date"] >= "2023-01-06") & 
        (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    
    friday_dates = fridays_df["Date"].tolist()
    print(f"Found {len(friday_dates)} Friday rebalancing periods over 3.5 years.")

    # 4. Precompute features vectorized on the entire dataset once (eliminating look-ahead and speed lags)
    print("[Processing] Pre-calculating indicators on full dataset once...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    
    # 5. Run rolling simulations
    portfolio_bh = [100.0]
    portfolio_long = [100.0]
    portfolio_high_long = [100.0]
    
    results_rows = []
    
    print("\n[Simulation] Evaluating multi-year week-by-week predictions...")
    for idx, friday in enumerate(friday_dates):
        # Retrieve computed rows for this Friday
        friday_obs = df_all_feat[df_all_feat["Date"] == friday].copy()
        
        # Drop tickers not in embeddings
        found_tickers = [t for t in friday_obs["ticker"].unique() if t in company_embeddings]
        friday_obs = friday_obs[friday_obs["ticker"].isin(found_tickers)].copy()
        
        if friday_obs.empty:
            portfolio_bh.append(portfolio_bh[-1])
            portfolio_long.append(portfolio_long[-1])
            portfolio_high_long.append(portfolio_high_long[-1])
            continue
            
        # Add embedding projection
        emb_list = [company_embeddings[t] for t in found_tickers]
        raw_emb = np.array(emb_list)
        reduced_emb = pca.transform(raw_emb)
        emb_df = pd.DataFrame(reduced_emb, columns=pca_cols)
        emb_df["ticker"] = found_tickers
        
        friday_obs = friday_obs.merge(emb_df, on="ticker", how="inner")
        
        # Prepare inputs
        X_tab = friday_obs[tabular_cols].fillna(0).values.astype(np.float32)
        X_emb = friday_obs[pca_cols].fillna(0).values.astype(np.float32)
        X_full = np.concatenate([X_tab, X_emb], axis=1)
        X_full_s = scaler.transform(X_full)
        
        # Inference
        model_probas = []
        for m in mix_models:
            if m in trained_models:
                y_proba = trained_models[m].predict_proba(X_full_s)[:, 1]
                model_probas.append(y_proba)
        
        friday_obs["pred_proba"] = np.mean(model_probas, axis=0)
        
        # Determine actual returns over the next week
        if idx + 1 < len(friday_dates):
            next_friday = friday_dates[idx + 1]
        else:
            next_friday = "2026-05-22"
            
        curr_prices = friday_obs.set_index("ticker")["company_close"].to_dict()
        
        # Get next Friday's prices
        next_prices_df = df_full[df_full["Date"] == next_friday].set_index("ticker")["company_close"].to_dict()
        
        # Calculate returns
        ticker_returns = {}
        for t in found_tickers:
            if t in curr_prices and t in next_prices_df:
                p0 = curr_prices[t]
                p1 = next_prices_df[t]
                if p0 > 0:
                    ticker_returns[t] = (p1 - p0) / p0
                    
        # 1. B&H Return
        bh_ret = np.mean(list(ticker_returns.values())) if ticker_returns else 0.0
        
        # 2. Long predictions (P(up) >= 0.5)
        long_df = friday_obs[friday_obs["pred_proba"] >= 0.5]
        long_tickers = long_df["ticker"].tolist()
        long_returns = [ticker_returns[t] for t in long_tickers if t in ticker_returns]
        long_ret = np.mean(long_returns) if long_returns else 0.0
        
        # 3. High confidence Long predictions (P(up) >= 0.53)
        high_long_df = friday_obs[friday_obs["pred_proba"] >= 0.53]
        if high_long_df.empty:
            high_long_df = friday_obs.sort_values("pred_proba", ascending=False).head(2)
            
        high_long_tickers = high_long_df["ticker"].tolist()
        high_long_returns = [ticker_returns[t] for t in high_long_tickers if t in ticker_returns]
        high_long_ret = np.mean(high_long_returns) if high_long_returns else 0.0
        
        # Track values (compounded weekly)
        portfolio_bh.append(portfolio_bh[-1] * (1.0 + bh_ret))
        portfolio_long.append(portfolio_long[-1] * (1.0 + long_ret))
        portfolio_high_long.append(portfolio_high_long[-1] * (1.0 + high_long_ret))
        
        results_rows.append({
            "Date": friday,
            "B&H Return": bh_ret * 100.0,
            "Long Return": long_ret * 100.0,
            "High Long Return": high_long_ret * 100.0,
        })
        
    df_results = pd.DataFrame(results_rows)
    
    final_bh = portfolio_bh[-1] - 100.0
    final_long = portfolio_long[-1] - 100.0
    final_high_long = portfolio_high_long[-1] - 100.0
    
    # Annualized Returns
    n_years = len(friday_dates) / 52.0
    ann_bh = (portfolio_bh[-1] / 100.0) ** (1.0 / n_years) - 1.0
    ann_long = (portfolio_long[-1] / 100.0) ** (1.0 / n_years) - 1.0
    ann_high_long = (portfolio_high_long[-1] / 100.0) ** (1.0 / n_years) - 1.0
    
    # Win rates
    win_bh = np.sum(df_results["B&H Return"] > 0) / len(df_results) * 100.0
    win_long = np.sum(df_results["Long Return"] > 0) / len(df_results) * 100.0
    win_high_long = np.sum(df_results["High Long Return"] > 0) / len(df_results) * 100.0
    
    print("\n" + "=" * 90)
    print("3.5-YEAR MULTI-YEAR OUT-OF-SAMPLE BACKTEST SUMMARY (2023 - 2026)")
    print("=" * 90)
    print(f"{'Strategy Name':<28} | {'Cumulative Ret':<15} | {'Annualized Ret':<15} | {'Weekly Win Rate':<15}")
    print("-" * 90)
    print(f"{'Buy & Hold Benchmark':<28} | {final_bh:>13.2f}% | {ann_bh*100:>13.2f}% | {win_bh:>13.2f}%")
    print(f"{'Standard Long-Only (P>=0.5)':<28} | {final_long:>13.2f}% | {ann_long*100:>13.2f}% | {win_long:>13.2f}%")
    print(f"{'High-Confidence Longs':<28} | {final_high_long:>13.2f}% | {ann_high_long*100:>13.2f}% | {win_high_long:>13.2f}%")
    print("=" * 90)
    
    # Generate Matplotlib chart
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11.5, 5.8))
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        plot_dates = [pd.to_datetime(d) for d in friday_dates] + [pd.to_datetime("2026-05-22")]
        
        plt.plot(plot_dates, portfolio_bh, label=f"Buy & Hold Basket (Cum Return: {final_bh:.1f}%)", color="#7f8c8d", linewidth=2.5, linestyle="--")
        plt.plot(plot_dates, portfolio_long, label=f"Standard Long-Only (Cum Return: {final_long:.1f}%)", color="#2ecc71", linewidth=3)
        plt.plot(plot_dates, portfolio_high_long, label=f"High-Confidence Longs (Cum Return: {final_high_long:.1f}%)", color="#1abc9c", linewidth=3.5)
        
        plt.title("3.5-Year Rolling Out-of-Sample Long Strategy Real-World Performance (2023 - 2026)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Rebalance Date", fontsize=11)
        plt.ylabel("Portfolio Performance (Base 100)", fontsize=11)
        plt.yscale("log") # Using log scale due to high compounding returns
        plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()
        
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
        print(f"\nSUCCESS: Performance chart saved to {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"\n[WARNING] Matplotlib error: {e}")

if __name__ == "__main__":
    main()
