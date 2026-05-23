#!/usr/bin/env python3
"""
Verification Backtest Script: SMA50 Baseline vs. Gaussian HMM + Kalman Beta Hedging.
This script compares the performance of the regime-filtered strategy over the last 12 months
(May 23, 2025 - May 22, 2026) using the old SMA50 baseline filter versus the upgraded
NumPy-based Gaussian HMM regime detector combined with recursive Kalman Beta short scaling.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import duckdb
import yfinance as yf

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")

# Ensure trading_agent can be imported
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from trading_agent import config
from trading_agent.bot import GaussianHMM, KalmanBetaFilter, compute_live_features, load_macro_features, fetch_company_metadata

def calculate_metrics(portfolio_values):
    # Weekly returns
    returns = pd.Series(portfolio_values).pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0, 0.0, 0.0
    
    # Cumulative return
    cum_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    
    # Annualized Sharpe (52 weeks/year)
    sharpe = np.sqrt(52) * returns.mean() / returns.std()
    
    # Maximum Drawdown
    running_max = pd.Series(portfolio_values).cummax()
    drawdowns = (portfolio_values - running_max) / running_max * 100
    max_dd = drawdowns.min()
    
    return cum_return, sharpe, max_dd

def main():
    print("=" * 80)
    print("BDA ADVANCED QUANT RISK MITIGATION SYSTEM VERIFICATION BACKTESTER")
    print("=" * 80)

    # 1. Load Model
    print(f"[Model] Loading best ensemble model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    trained_models = model_data["trained_models"]
    mix_models = model_data["mix_models"]
    scaler = model_data["scaler"]
    pca = model_data["pca"]
    tabular_cols = model_data["tabular_cols"]
    pca_cols = model_data["pca_cols"]
    company_embeddings = model_data["company_embeddings"]

    basket = config.HIGH_ALPHA_TICKERS
    print(f"[Config] Stock Basket: High-Alpha Alphabetical Basket ({len(basket)} Small-Caps)")
    print(f"[Config] Target Strategy: Regime-Filtered (Long/Short Rebalancer)")

    # 2. Ingest S&P 500
    print("\n[Ingestion] Fetching S&P 500 (^GSPC) price history...")
    gspc_df = yf.download("^GSPC", start="2024-03-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [col[0] for col in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime('%Y-%m-%d')
    gspc_df = gspc_df.sort_values("Date").reset_index(drop=True)
    
    # S&P 500 simple rolling SMA50
    gspc_df["SMA50_GSPC"] = gspc_df["Close"].rolling(window=50, min_periods=50).mean()
    # S&P 500 daily log returns
    gspc_df["log_ret_GSPC"] = np.log(gspc_df["Close"] / gspc_df["Close"].shift(1))
    
    gspc_dict = gspc_df.set_index("Date").to_dict(orient="index")

    # 3. Ingest Asset Prices
    print(f"\n[Ingestion] Fetching historical pricing for {len(basket)} tickers...")
    df_list = []
    for ticker in basket:
        ticker_df = yf.download(ticker, start="2025-01-01", end="2026-05-23", progress=False)
        if not ticker_df.empty:
            ticker_df = ticker_df.reset_index()
            ticker_df["ticker"] = ticker
            ticker_df = ticker_df.rename(columns={
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

    df_full = pd.concat(df_list, ignore_index=True)

    # 4. Resolve Friday Dates
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4) & 
        (dates_df["Date"] >= "2025-05-23") & 
        (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] Resolved {len(friday_dates)} Friday rebalancing periods.")

    # 5. Precalculate Technical features
    print("[Processing] Computing technical and macroeconomic features...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)

    # 6. Compounding Capital Setup
    initial_equity = 10000.0
    equity_sma = initial_equity
    equity_hmm = initial_equity

    sma_values = [initial_equity]
    hmm_values = [initial_equity]

    # Rebalance loop
    print("\nRunning historical rebalancing simulation...")
    for idx, friday in enumerate(friday_dates):
        # Determine S&P 500 trend regime via SMA50
        gspc_row = gspc_df[gspc_df["Date"] <= friday].iloc[-1]
        gspc_close = float(gspc_row["Close"])
        gspc_sma50 = float(gspc_row["SMA50_GSPC"])
        sma_is_bull = gspc_close > gspc_sma50

        # Determine S&P 500 trend regime via dynamic 2-state Gaussian HMM
        # Train HMM on 250 daily returns leading up to friday
        sp_past = gspc_df[gspc_df["Date"] <= friday].sort_values("Date")
        log_ret_window = sp_past["log_ret_GSPC"].dropna().values[-config.HMM_TRAINING_DAYS:]
        
        hmm = GaussianHMM(n_states=2, max_iter=100)
        hmm.fit(log_ret_window)
        decoded_states = hmm.decode(log_ret_window)
        hmm_is_bull = (decoded_states[-1] == 0)

        # Get features for this Friday
        friday_obs = df_all_feat[df_all_feat["Date"] == friday].copy()
        found_tickers = [t for t in friday_obs["ticker"].unique() if t in company_embeddings]
        friday_obs = friday_obs[friday_obs["ticker"].isin(found_tickers)].copy()

        if friday_obs.empty:
            # Maintained portfolio values on holiday fallback
            sma_values.append(equity_sma)
            hmm_values.append(equity_hmm)
            continue

        # KG embeddings PCA projection
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
        X_full_df = pd.DataFrame(X_full_s, columns=tabular_cols + pca_cols)

        # Ensemble Soft Voting predictions
        model_probas = []
        for m in mix_models:
            if m in trained_models:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    y_proba = trained_models[m].predict_proba(X_full_df)[:, 1]
                model_probas.append(y_proba)
        
        friday_obs["pred_proba"] = np.mean(model_probas, axis=0)

        # Calculate Kalman Betas for each stock dynamically over the rolling 60 days
        kf = KalmanBetaFilter(q_noise=config.KALMAN_Q, r_noise=config.KALMAN_R)
        ticker_betas = {}
        
        # Download S&P 500 returns in that specific window for exact correlation
        past_dates = gspc_df[gspc_df["Date"] <= friday].sort_values("Date")["Date"].tolist()[-60:]
        sp_ret_window = gspc_df[gspc_df["Date"] <= friday].sort_values("Date").set_index("Date")["log_ret_GSPC"].to_dict()
        
        for t in found_tickers:
            t_hist = df_all_feat[(df_all_feat["Date"] <= friday) & (df_all_feat["ticker"] == t)].sort_values("Date").tail(60).copy()
            t_hist["ticker_return"] = t_hist["company_close"].pct_change().fillna(0)
            
            aligned_sp = []
            aligned_stock = []
            for _, row in t_hist.iterrows():
                dt = row["Date"]
                if dt in sp_ret_window:
                    aligned_sp.append(sp_ret_window[dt])
                    aligned_stock.append(row["ticker_return"])
            
            ticker_betas[t] = kf.filter(aligned_sp, aligned_stock)

        # Retrieve closing prices for return calculation
        if idx + 1 < len(friday_dates):
            next_friday = friday_dates[idx + 1]
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

        # --- A. STRATEGY 1: SMA50 Baseline ---
        sma_raw = friday_obs[["ticker", "pred_proba"]].copy()
        sma_raw["raw_weight"] = sma_raw["pred_proba"] - 0.5
        
        sma_longs = sma_raw[sma_raw["raw_weight"] >= 0.02].copy()
        if sma_is_bull:
            sma_shorts = pd.DataFrame()
        else:
            sma_shorts = sma_raw[sma_raw["raw_weight"] <= -0.02].copy()
            
        sma_selected = pd.concat([sma_longs, sma_shorts])
        sma_abs_sum = sma_selected["raw_weight"].abs().sum()
        
        sma_weights = {}
        if sma_abs_sum > 0:
            sma_selected["target_weight"] = (sma_selected["raw_weight"] / sma_abs_sum) * config.TARGET_EXPOSURE
            sma_weights = sma_selected.set_index("ticker")["target_weight"].to_dict()

        # Compute SMA Strategy Weekly Return
        sma_ret = 0.0
        for t, w in sma_weights.items():
            sma_ret += w * ticker_returns.get(t, 0.0)
        
        equity_sma *= (1.0 + sma_ret)
        sma_values.append(equity_sma)

        # --- B. STRATEGY 2: HMM + Kalman Upgraded ---
        hmm_raw = friday_obs[["ticker", "pred_proba"]].copy()
        hmm_raw["kalman_beta"] = hmm_raw["ticker"].map(ticker_betas).fillna(1.0)
        hmm_raw["raw_weight"] = hmm_raw["pred_proba"] - 0.5
        
        # Kalman short scaling
        def scale_short(row):
            w = row["raw_weight"]
            if w < 0:
                beta = row["kalman_beta"]
                return w / max(abs(beta), 0.5)
            return w
        hmm_raw["raw_weight"] = hmm_raw.apply(scale_short, axis=1)
        
        hmm_longs = hmm_raw[hmm_raw["raw_weight"] >= 0.02].copy()
        if hmm_is_bull:
            hmm_shorts = pd.DataFrame()
        else:
            hmm_shorts = hmm_raw[hmm_raw["raw_weight"] <= -0.02].copy()
            
        hmm_selected = pd.concat([hmm_longs, hmm_shorts])
        hmm_abs_sum = hmm_selected["raw_weight"].abs().sum()
        
        hmm_weights = {}
        if hmm_abs_sum > 0:
            hmm_selected["target_weight"] = (hmm_selected["raw_weight"] / hmm_abs_sum) * config.TARGET_EXPOSURE
            hmm_weights = hmm_selected.set_index("ticker")["target_weight"].to_dict()

        # Compute HMM Strategy Weekly Return
        hmm_ret = 0.0
        for t, w in hmm_weights.items():
            hmm_ret += w * ticker_returns.get(t, 0.0)
            
        equity_hmm *= (1.0 + hmm_ret)
        hmm_values.append(equity_hmm)

    # 7. Print Performance Comparison Table
    sma_cum, sma_sharpe, sma_dd = calculate_metrics(sma_values)
    hmm_cum, hmm_sharpe, hmm_dd = calculate_metrics(hmm_values)

    print("\n" + "=" * 80)
    print("PERFORMANCE AUDIT RESULTS (12-MONTH HEAD-TO-HEAD BACKTEST)")
    print("=" * 80)
    print(f"{'Performance Metric':<28} | {'SMA50 Baseline Filter':<22} | {'Gaussian HMM + Kalman Upgraded':<30}")
    print("-" * 80)
    print(f"{'Starting Capital':<28} | ${initial_equity:,.2f}            | ${initial_equity:,.2f}")
    print(f"{'Ending Portfolio Value':<28} | ${equity_sma:,.2f}            | ${equity_hmm:,.2f}")
    print(f"{'Cumulative Return %':<28} | {sma_cum:+.2f}%              | {hmm_cum:+.2f}%")
    print(f"{'Annualized Sharpe Ratio':<28} | {sma_sharpe:.4f}               | {hmm_sharpe:.4f}")
    print(f"{'Maximum Drawdown %':<28} | {sma_dd:.2f}%               | {hmm_dd:.2f}%")
    print("=" * 80)
    print("\n[Verification Summary]")
    if hmm_cum > sma_cum:
        print(f" -> SUCCESS: HMM + Kalman dynamic risk scaling outperformed SMA50 baseline by {hmm_cum - sma_cum:+.2f}% absolute return.")
    else:
        print(" -> HMM + Kalman underperformed SMA50 baseline on raw returns, check risk preservation.")
        
    if hmm_sharpe > sma_sharpe:
        print(f" -> SUCCESS: HMM + Kalman delivered superior risk-adjusted return (Sharpe: {hmm_sharpe:.4f} vs. {sma_sharpe:.4f}).")
    if abs(hmm_dd) < abs(sma_dd):
        print(f" -> SUCCESS: HMM + Kalman successfully compressed maximum drawdown ({hmm_dd:.2f}% vs. {sma_dd:.2f}%).")
    print("=" * 80)

if __name__ == "__main__":
    main()
