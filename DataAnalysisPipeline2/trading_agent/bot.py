"""
Production BDA Trading Bot Core Agent Module.
This module downloads live data, computes technical features, runs PCA-GCN model inference,
determines the S&P 500 market regime, and dispatches optimized differential trades to Alpaca.
"""

import os
import pickle
import numpy as np
import pandas as pd
import duckdb
import yfinance as yf

from . import config
from .operational import (
    TradeLogger, TradeLogRow,
    DrawdownCircuitBreaker,
    PerformanceAttribution,
    build_decision_context, save_feature_snapshot, write_decision_row,
)

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

# --- Gaussian HMM and Kalman Filter Implementations ---

class GaussianHMM:
    def __init__(self, n_states=2, max_iter=100, tol=1e-4):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.start_probs = None
        self.trans_mat = None
        self.means = None
        self.vars = None
        self.latest_probs = None

    def fit(self, x):
        """
        Fits the 2-state Gaussian HMM on observation sequence x using Baum-Welch.
        x: 1D numpy array of S&P 500 daily log returns
        """
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        T = len(x)
        if T < 10:
            self.start_probs = np.array([0.5, 0.5])
            self.trans_mat = np.array([[0.9, 0.1], [0.1, 0.9]])
            self.means = np.array([0.0, 0.0])
            self.vars = np.array([1e-4, 4e-4])
            self.latest_probs = np.array([0.5, 0.5])
            return self

        # Smart Initialization based on median absolute deviations (volatility proxies)
        mad = np.abs(x - np.median(x))
        threshold = np.percentile(mad, 70)
        low_vol_mask = mad <= threshold
        high_vol_mask = mad > threshold

        mean_low = np.mean(x[low_vol_mask]) if np.any(low_vol_mask) else np.mean(x)
        var_low = np.var(x[low_vol_mask]) if np.any(low_vol_mask) else np.var(x)
        
        mean_high = np.mean(x[high_vol_mask]) if np.any(high_vol_mask) else np.mean(x)
        var_high = np.var(x[high_vol_mask]) if np.any(high_vol_mask) else np.var(x) * 4.0

        self.means = np.array([mean_low, mean_high])
        self.vars = np.array([var_low, var_high])
        
        if self.vars[0] > self.vars[1]:
            self.vars[0], self.vars[1] = self.vars[1], self.vars[0]
            self.means[0], self.means[1] = self.means[1], self.means[0]

        self.start_probs = np.array([0.5, 0.5])
        self.trans_mat = np.array([[0.95, 0.05], [0.10, 0.90]])

        prev_log_lik = -np.inf

        for iteration in range(self.max_iter):
            B = np.zeros((T, 2))
            for i in range(2):
                v = max(self.vars[i], 1e-8)
                B[:, i] = (1.0 / np.sqrt(2.0 * np.pi * v)) * np.exp(-0.5 * ((x - self.means[i]) ** 2) / v)
            
            B = np.clip(B, 1e-300, None)

            alpha = np.zeros((T, 2))
            c = np.zeros(T)

            alpha[0] = self.start_probs * B[0]
            c[0] = 1.0 / max(np.sum(alpha[0]), 1e-300)
            alpha[0] *= c[0]

            for t in range(1, T):
                alpha[t] = np.dot(alpha[t-1], self.trans_mat) * B[t]
                c[t] = 1.0 / max(np.sum(alpha[t]), 1e-300)
                alpha[t] *= c[t]

            beta = np.zeros((T, 2))
            beta[T-1] = c[T-1]

            for t in range(T-2, -1, -1):
                beta[t] = np.dot(self.trans_mat, B[t+1] * beta[t+1]) * c[t]

            gamma = alpha * beta
            row_sums = np.sum(gamma, axis=1, keepdims=True)
            gamma = gamma / np.where(row_sums == 0, 1e-300, row_sums)

            xi = np.zeros((T-1, 2, 2))
            for t in range(T-1):
                denom = np.sum(alpha[t] * np.dot(self.trans_mat, B[t+1] * beta[t+1]))
                if denom == 0:
                    denom = 1e-300
                for i in range(2):
                    xi[t, i, :] = alpha[t, i] * self.trans_mat[i, :] * B[t+1] * beta[t+1] / denom

            log_lik = -np.sum(np.log(np.clip(c, 1e-300, None)))

            if np.abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik

            self.start_probs = gamma[0] / max(np.sum(gamma[0]), 1e-300)
            
            sum_xi = np.sum(xi, axis=0)
            sum_gamma = np.sum(gamma[:-1], axis=0, keepdims=True).T
            self.trans_mat = sum_xi / np.where(sum_gamma == 0, 1e-300, sum_gamma)
            self.trans_mat /= np.sum(self.trans_mat, axis=1, keepdims=True)

            sum_gamma_full = np.sum(gamma, axis=0)
            denom_full = np.where(sum_gamma_full == 0, 1e-300, sum_gamma_full)
            
            for i in range(2):
                self.means[i] = np.sum(gamma[:, i] * x) / denom_full[i]
                self.vars[i] = np.sum(gamma[:, i] * ((x - self.means[i]) ** 2)) / denom_full[i]
                self.vars[i] = max(self.vars[i], 1e-8)

        if self.vars[0] > self.vars[1]:
            self.vars = self.vars[::-1]
            self.means = self.means[::-1]
            self.trans_mat = self.trans_mat[::-1, ::-1]
            self.start_probs = self.start_probs[::-1]

        self.latest_probs = gamma[-1]
        return self

    def decode(self, x):
        """
        Computes the most likely sequence of states using the Viterbi algorithm.
        Returns: 1D numpy array of state indices (0 or 1)
        """
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        T = len(x)
        if T == 0:
            return np.array([])

        B = np.zeros((T, 2))
        for i in range(2):
            v = max(self.vars[i], 1e-8)
            B[:, i] = -0.5 * np.log(2.0 * np.pi * v) - 0.5 * ((x - self.means[i]) ** 2) / v

        V = np.zeros((T, 2))
        ptr = np.zeros((T, 2), dtype=int)

        start_p = np.clip(self.start_probs, 1e-300, None)
        trans_p = np.clip(self.trans_mat, 1e-300, None)

        V[0] = np.log(start_p) + B[0]

        for t in range(1, T):
            for j in range(2):
                vals = V[t-1] + np.log(trans_p[:, j])
                ptr[t, j] = np.argmax(vals)
                V[t, j] = B[t, j] + vals[ptr[t, j]]

        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(V[T-1])
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]

        return states


class KalmanBetaFilter:
    def __init__(self, q_noise=1e-4, r_noise=1e-1):
        self.q_noise = q_noise
        self.r_noise = r_noise

    def filter(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 5:
            return 1.0
            
        theta = np.array([0.0, 1.0])
        P = np.eye(2) * 1.0
        Q = np.diag([1e-6, self.q_noise])
        R = self.r_noise

        for t in range(len(x_clean)):
            P_pred = P + Q
            H = np.array([1.0, x_clean[t]])
            y_pred = H[0] * theta[0] + H[1] * theta[1]
            e = y_clean[t] - y_pred
            S = P_pred[0, 0] + 2.0 * x_clean[t] * P_pred[0, 1] + (x_clean[t] ** 2) * P_pred[1, 1] + R
            K = np.array([
                P_pred[0, 0] + x_clean[t] * P_pred[0, 1],
                P_pred[1, 0] + x_clean[t] * P_pred[1, 1]
            ]) / S

            theta = theta + K * e
            KH = np.outer(K, H)
            P = P_pred - KH.dot(P_pred)

        return theta[1]

# --- Semantic and Database Loaders ---

def load_macro_features(macro_ttl_path: str):
    """Read GDP, growth, inflation, trade, and interest rates per country from the macroeconomic graph."""
    if not os.path.exists(macro_ttl_path):
        print(f"[WARNING] Macroeconomic Turtle graph not found at {macro_ttl_path}. Using zero fallbacks.")
        return pd.DataFrame(columns=["country", "gdp_usd", "gdp_growth_pct", "inflation_pct", "trade_pct", "interest_rate_pct"])

    try:
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
    except Exception as e:
        print(f"[WARNING] Failed to parse RDF Macro TTL graph: {e}. Using empty fallbacks.")
        return pd.DataFrame(columns=["country", "gdp_usd", "gdp_growth_pct", "inflation_pct", "trade_pct", "interest_rate_pct"])

def fetch_company_metadata():
    """Fetch static sector, industry, market cap, and country info from historical databases."""
    db_path = os.path.join(config.EXPLOITATION_DIR, "ExploitationZone.duckdb")
    trusted_db = os.path.abspath(os.path.join(config.EXPLOITATION_DIR, "..", "TrustedZone", "TrustedZone.duckdb"))
    
    if not os.path.exists(db_path) or not os.path.exists(trusted_db):
        print("[WARNING] Analytical databases missing. Using default metadata values.")
        rows = [{"ticker": t, "Sector": "Technology", "Industry": "Software", "MarketCap": 5e11, "eur_rate": 1.0, "jpy_rate": 150.0, "country": "United States"} for t in config.TICKERS]
        return pd.DataFrame(rows)

    try:
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
        
        conn_t = duckdb.connect(trusted_db, read_only=True)
        df_country = conn_t.execute("SELECT DISTINCT Symbol AS ticker, country FROM companies").df()
        conn_t.close()
        
        df_meta = df_meta.merge(df_country, on="ticker", how="left")
        return df_meta
    except Exception as e:
        print(f"[WARNING] DuckDB database queries failed: {e}. Falling back to default metadata.")
        rows = [{"ticker": t, "Sector": "Technology", "Industry": "Software", "MarketCap": 5e11, "eur_rate": 1.0, "jpy_rate": 150.0, "country": "United States"} for t in config.TICKERS]
        return pd.DataFrame(rows)

# --- Feature Extraction Pipeline ---

def compute_live_features(live_df, metadata_df, macro_df):
    """Computes technical indicators and joins macro/corporate metadata precisely aligned with training environment."""
    df = live_df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    
    df = df.merge(metadata_df, on="ticker", how="left")
    df = df.merge(macro_df, on="country", how="left")
    df = df.drop(columns=["country"], errors="ignore")
    
    # Fill in critical defaults
    df["eur_rate"] = df["eur_rate"].fillna(1.0)
    df["jpy_rate"] = df["jpy_rate"].fillna(155.0)
    df["Sector"] = df["Sector"].fillna("Technology")
    df["MarketCap"] = df["MarketCap"].fillna(5e10)
    df["gdp_usd"] = df["gdp_usd"].fillna(2.7e13)
    df["gdp_growth_pct"] = df["gdp_growth_pct"].fillna(2.5)
    df["inflation_pct"] = df["inflation_pct"].fillna(3.1)
    df["trade_pct"] = df["trade_pct"].fillna(25.0)
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(5.25)
    
    df["log_market_cap"] = np.log(df["MarketCap"].replace(0, 1.0).astype(float))
    df['daily_return'] = df.groupby('ticker')['company_close'].pct_change(1).fillna(0)
    
    # Price returns over multiple horizons
    df['return_5d'] = df.groupby('ticker')['company_close'].pct_change(5).fillna(0)
    df['return_10d'] = df.groupby('ticker')['company_close'].pct_change(10).fillna(0)
    df['return_20d'] = df.groupby('ticker')['company_close'].pct_change(20).fillna(0)
    df['return_50d'] = df.groupby('ticker')['company_close'].pct_change(50).fillna(0)
    
    # Simple Moving Averages and ratios
    df['ma5'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['price_vs_ma5'] = (df['company_close'] - df['ma5']) / df['ma5'].replace(0, 1e-9)
    
    df['ma20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['price_vs_ma20'] = (df['company_close'] - df['ma20']) / df['ma20'].replace(0, 1e-9)
    
    # Stochastic range indicator
    df['min20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).min())
    df['max20'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).max())
    df['stoch_20d'] = (df['company_close'] - df['min20']) / (df['max20'] - df['min20']).replace(0, 1e-9)
    df['stoch_20d'] = df['stoch_20d'].fillna(0)
    
    # Volume dynamics
    df['volume_ma5'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df['volume_ratio'] = df['company_volume'] / df['volume_ma5'].replace(0, 1e-9)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    
    # Volatility bounds
    df['rolling_volatility_5d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(6, min_periods=1).std()).fillna(0)
    df['rolling_volatility_10d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(11, min_periods=1).std()).fillna(0)
    df['rolling_volatility_20d'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(21, min_periods=1).std()).fillna(0)
    
    # Square-Root Market Impact metrics
    df['dollar_volume'] = df['company_close'] * df['company_volume']
    df['adv_usd'] = df.groupby('ticker')['dollar_volume'].transform(lambda x: x.rolling(21, min_periods=1).mean()).fillna(1e6)
    df['return_volatility_20d'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(21, min_periods=1).std()).fillna(0.01)
    
    # Calendrical features
    df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month_of_year'] = pd.to_datetime(df['Date']).dt.month
    
    # Volatility adjusted return
    df['vol_adjusted_return'] = df['daily_return'] / df['rolling_volatility_10d'].replace(0, 1e-9)
    df['vol_adjusted_return'] = df['vol_adjusted_return'].fillna(0)
    
    # Volume Z-score
    df['vol_mean20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df['vol_std20'] = df.groupby('ticker')['company_volume'].transform(lambda x: x.rolling(21, min_periods=1).std())
    df['volume_zscore_20d'] = (df['company_volume'] - df['vol_mean20']) / df['vol_std20'].replace(0, 1e-9)
    df['volume_zscore_20d'] = df['volume_zscore_20d'].fillna(0)
    
    df = df.drop(columns=['ma5', 'ma20', 'min20', 'max20', 'volume_ma5', 'vol_mean20', 'vol_std20'])
    
    # Sector performance features
    df['sector_daily_return'] = df.groupby(['Sector', 'Date'])['daily_return'].transform('mean').fillna(0)
    df['sector_return_5d'] = df.groupby(['Sector', 'Date'])['return_5d'].transform('mean').fillna(0)
    
    # Cross-sectional market rankings
    df['rank_daily_return'] = df.groupby('Date')['daily_return'].rank(pct=True).fillna(0.5)
    df['rank_return_5d'] = df.groupby('Date')['return_5d'].rank(pct=True).fillna(0.5)
    df['rank_return_20d'] = df.groupby('Date')['return_20d'].rank(pct=True).fillna(0.5)
    df['rank_volatility'] = df.groupby('Date')['rolling_volatility_10d'].rank(pct=True).fillna(0.5)
    df['rank_volume_ratio'] = df.groupby('Date')['volume_ratio'].rank(pct=True).fillna(0.5)
    
    # Momentum Oscillators & Technical metrics
    df['rsi_14'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_rsi(x, 14)).fillna(50)
    df['macd'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_macd(x, 12, 26)).fillna(0)
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: compute_macd_signal(x, 9)).fillna(0)
    
    df['bb_mean'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['bb_std'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    df['bb_width'] = (4 * df['bb_std']) / df['bb_mean'].replace(0, 1e-9)
    df['bb_width'] = df['bb_width'].fillna(0)
    df = df.drop(columns=['bb_mean', 'bb_std'])
    
    # Time-series historical lags
    for lag in [1, 2, 5]:
        df[f'daily_return_lag_{lag}'] = df.groupby('ticker')['daily_return'].shift(lag).fillna(0)
        df[f'volume_ratio_lag_{lag}'] = df.groupby('ticker')['volume_ratio'].shift(lag).fillna(1.0)
        
    return df

# --- Core Trading Agent Class ---

class BDATradingAgent:
    def __init__(self):
        self.model_data = None
        self.trained_models = {}
        self.mix_models = []
        self.scaler = None
        self.pca = None
        self.tabular_cols = []
        self.pca_cols = []
        self.company_embeddings = {}
        self.hmm_probs = np.array([0.5, 0.5])
        self.hmm_state = 0
        self.kalman_betas = {}

        # Operational hardening — logger, drawdown circuit breaker, attribution.
        # These all persist to ./agent_logs/.
        self.trade_logger = TradeLogger()
        self.circuit_breaker = DrawdownCircuitBreaker(
            threshold_pct=getattr(config, "DRAWDOWN_LIMIT_PCT", 5.0)
        )
        self.attribution = PerformanceAttribution()

        # Filled in by run_inference()/execute_alpaca_rebalance() so we have a
        # full audit trail of "what did the bot see when it traded?"
        self.last_decision_ctx = None
        self.last_predictions_df = None     # ticker, pred_rank, pred_proba, kalman_beta, ...
        self.last_feature_matrix = None     # scaled X used to call .predict()

    def load_model(self):
        """Loads serialized ensemble classifier models and scaling metrics from best_model.pkl."""
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Baked ensemble model file not found at: {config.MODEL_PATH}. "
                                    f"Please run model training first.")

        print(f"[Agent] Loading best ensemble model from {config.MODEL_PATH}...")
        with open(config.MODEL_PATH, "rb") as f:
            self.model_data = pickle.load(f)

        self.trained_models = self.model_data["trained_models"]
        self.mix_models = self.model_data["mix_models"]
        self.scaler = self.model_data["scaler"]
        self.pca = self.model_data["pca"]
        self.tabular_cols = self.model_data["tabular_cols"]
        self.pca_cols = self.model_data["pca_cols"]
        self.company_embeddings = self.model_data["company_embeddings"]
        print(f"[Agent] Model successfully loaded. Base models: {self.mix_models}")

    def check_market_regime(self, force_regime=None):
        """
        Determines the current S&P 500 trend regime using a 2-state Gaussian HMM.
        Returns is_bull = True if the latest decoded state is State 0 (Low Volatility Bull), else False.
        """
        if force_regime:
            is_bull = force_regime.lower() == "bull"
            print(f"[Agent] Regime forced by operator: {'BULL (Shorts Disabled)' if is_bull else 'BEAR (Shorts Enabled)'}")
            self.hmm_state = 0 if is_bull else 1
            self.hmm_probs = np.array([1.0, 0.0]) if is_bull else np.array([0.0, 1.0])
            return is_bull

        if not config.REGIME_FILTER_ENABLED:
            print("[Agent] S&P 500 Regime Filter is disabled. Defaulting to BEAR (Shorts Enabled) mode.")
            self.hmm_state = 1
            self.hmm_probs = np.array([0.0, 1.0])
            return False

        print(f"[Agent] Fetching S&P 500 ({config.SP500_INDEX}) data to train Gaussian HMM...")
        try:
            sp_df = yf.download(config.SP500_INDEX, period="380d", progress=False)
            if sp_df.empty:
                raise ValueError("Downloaded DataFrame is empty.")
            
            if isinstance(sp_df.columns, pd.MultiIndex):
                sp_df.columns = [col[0] for col in sp_df.columns]
                
            sp_df = sp_df.reset_index()
            sp_df = sp_df.sort_values("Date").reset_index(drop=True)
            
            sp_df["log_return"] = np.log(sp_df["Close"] / sp_df["Close"].shift(1))
            log_returns = sp_df["log_return"].dropna().values
            
            training_window = log_returns[-config.HMM_TRAINING_DAYS:]
            
            hmm = GaussianHMM(n_states=2, max_iter=100)
            hmm.fit(training_window)
            
            decoded_states = hmm.decode(training_window)
            latest_state = decoded_states[-1]
            
            self.hmm_state = latest_state
            self.hmm_probs = hmm.latest_probs
            
            is_bull = (latest_state == 0)
            
            print(f"[Agent] S&P 500 Decoded State: State {latest_state} ({'BULL' if is_bull else 'BEAR'})")
            print(f"[Agent] Regime Probabilities -> BULL (State 0): {self.hmm_probs[0]*100:.2f}% | BEAR (State 1): {self.hmm_probs[1]*100:.2f}%")
            return is_bull
            
        except Exception as e:
            print(f"[WARNING] Failed to decode market regime using HMM: {e}. Defaulting to BEAR (Shorts Enabled) mode.")
            self.hmm_state = 1
            self.hmm_probs = np.array([0.0, 1.0])
            return False

    def fetch_live_data(self):
        """Downloads live historical bars for our 20 high-alpha tickers."""
        print(f"[Agent] Downloading 60 days of historical daily bars for {len(config.TICKERS)} tickers...")
        df_list = []
        for ticker in config.TICKERS:
            try:
                ticker_df = yf.download(ticker, period="60d", progress=False)
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
            except Exception as e:
                print(f"[WARNING] Failed to download data for {ticker}: {e}")

        if not df_list:
            raise RuntimeError("No historical bars could be fetched for any tickers.")
        
        return pd.concat(df_list, ignore_index=True)

    def run_inference(self, price_history_df):
        """Performs feature calculations, PCA embedding projections, and Soft-Voting ensemble inference."""
        metadata_df = fetch_company_metadata()
        macro_df = load_macro_features(config.MACRO_KG_PATH)

        print("[Agent] Computing technical and macroeconomic features...")
        df_features = compute_live_features(price_history_df, metadata_df, macro_df)

        # Retrieve the single latest day of trading for live predictions
        latest_df = df_features.sort_values("Date").groupby("ticker").last().reset_index()

        found_tickers = []
        emb_list = []
        for t in latest_df["ticker"].unique():
            if t in self.company_embeddings:
                emb_list.append(self.company_embeddings[t])
                found_tickers.append(t)

        latest_df = latest_df[latest_df["ticker"].isin(found_tickers)].copy()
        if latest_df.empty:
            raise RuntimeError("No tickers in live price data matched GCN structural embeddings.")

        # Project RotatE structural embeddings through PCA
        raw_emb = np.array(emb_list)
        reduced_emb = self.pca.transform(raw_emb)
        emb_df = pd.DataFrame(reduced_emb, columns=self.pca_cols)
        emb_df["ticker"] = found_tickers

        latest_df = latest_df.merge(emb_df, on="ticker", how="inner")

        # Scale features using standard scaling
        X_tab = latest_df[self.tabular_cols].fillna(0).values.astype(np.float32)
        X_emb = latest_df[self.pca_cols].fillna(0).values.astype(np.float32)
        X_full = np.concatenate([X_tab, X_emb], axis=1)
        X_full_s = self.scaler.transform(X_full)

        # Convert back to a DataFrame with identical feature names to prevent scikit-learn/LGBM warnings
        X_full_df = pd.DataFrame(X_full_s, columns=self.tabular_cols + self.pca_cols)

        # ── Ensemble inference (regressor interface) ──────────────────────
        # The new best_model.pkl ships REGRESSORS trained against the 30-day
        # cross-sectional rank target in [0, 1].  We:
        #   1. average the continuous predictions across the base ensemble,
        #   2. re-rank the average across the live universe so that the
        #      output column "pred_rank" is again a clean [0, 1] cross-
        #      sectional rank.  This makes the downstream `>= threshold`
        #      logic in calculate_target_weights still meaningful.
        # The legacy "pred_proba" column is kept as an alias so existing
        # consumers (trade logger, backtests pre-migration) don't break.
        import warnings
        model_preds = []
        for m in self.mix_models:
            if m in self.trained_models:
                est = self.trained_models[m]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    if hasattr(est, "predict_proba"):  # legacy classifier path
                        y = est.predict_proba(X_full_df)[:, 1]
                    else:                              # regressor (current)
                        y = est.predict(X_full_df)
                model_preds.append(np.asarray(y, dtype=float))

        if not model_preds:
            raise RuntimeError("[Agent] No mix_models produced predictions.")

        # Soft-vote: rank each base model's predictions on the live universe,
        # then average the ranks.  This is robust to scale shifts across the
        # different regressor families (xgb/lgb/cat/rf each produce values in
        # their own range — averaging raw outputs would over-weight the
        # widest-range model).
        rank_matrix = np.column_stack(
            [pd.Series(p).rank(pct=True).values for p in model_preds]
        )
        avg_rank = pd.Series(rank_matrix.mean(axis=1)).rank(pct=True).values
        latest_df["pred_rank"] = avg_rank
        # Backwards-compatible alias for log + downstream consumers
        latest_df["pred_proba"] = avg_rank

        # 1. Download S&P 500 returns for Kalman Beta
        print("[Agent] Fetching S&P 500 history for Kalman Beta calculations...")
        try:
            sp_kf_df = yf.download(config.SP500_INDEX, period="100d", progress=False)
            if isinstance(sp_kf_df.columns, pd.MultiIndex):
                sp_kf_df.columns = [col[0] for col in sp_kf_df.columns]
            sp_kf_df = sp_kf_df.reset_index().sort_values("Date").reset_index(drop=True)
            sp_kf_df["Date"] = pd.to_datetime(sp_kf_df["Date"]).dt.strftime('%Y-%m-%d')
            sp_kf_df["sp_return"] = sp_kf_df["Close"].pct_change().fillna(0)
            sp_returns_dict = sp_kf_df.set_index("Date")["sp_return"].to_dict()
        except Exception as e:
            print(f"[WARNING] Failed to fetch S&P 500 for Kalman Filter: {e}.")
            sp_returns_dict = {}

        # 2. Run Kalman Filter for each ticker
        self.kalman_betas = {}
        kf = KalmanBetaFilter(q_noise=config.KALMAN_Q, r_noise=config.KALMAN_R)
        
        for ticker in latest_df["ticker"].unique():
            ticker_hist = df_features[df_features["ticker"] == ticker].sort_values("Date").copy()
            ticker_hist["ticker_return"] = ticker_hist["company_close"].pct_change().fillna(0)
            
            aligned_sp = []
            aligned_stock = []
            for _, row in ticker_hist.iterrows():
                dt = row["Date"]
                if dt in sp_returns_dict:
                    aligned_sp.append(sp_returns_dict[dt])
                    aligned_stock.append(row["ticker_return"])
                    
            beta_val = kf.filter(aligned_sp, aligned_stock)
            self.kalman_betas[ticker] = beta_val
            
        latest_df["kalman_beta"] = latest_df["ticker"].map(self.kalman_betas).fillna(1.0)

        # --- Build & persist the reproducibility tag for this decision ---
        try:
            self.last_decision_ctx = build_decision_context(
                model_path=config.MODEL_PATH,
                feature_df=X_full_df,
                universe=list(latest_df["ticker"].unique()),
                hmm_state=self.hmm_state,
                hmm_probs=self.hmm_probs,
                decision_date=str(latest_df["Date"].max()),
            )
            snapshot_path = save_feature_snapshot(
                self.last_decision_ctx.decision_id, X_full_df)
            write_decision_row(
                self.last_decision_ctx,
                extras={"snapshot_path": snapshot_path,
                        "n_tickers": int(len(latest_df))},
            )
            print(f"[Agent] decision_id = {self.last_decision_ctx.decision_id} "
                  f"(snapshot: {os.path.basename(snapshot_path)})")
        except Exception as e:
            print(f"[WARN] Could not record decision context: {e}")

        # Save the predictions and feature matrix for later attribution / replay
        self.last_predictions_df = latest_df.copy()
        self.last_feature_matrix = X_full_df.copy()

        return latest_df[["ticker", "company_close", "pred_rank",
                          "pred_proba", "kalman_beta"]].sort_values(
            "pred_rank", ascending=False)

    def calculate_target_weights(self, predictions_df, is_bull, strategy="high_confidence"):
        """
        Computes optimal target portfolio weights from cross-sectional 30d
        rank predictions in [0, 1].

        Strategies:
          - high_confidence : long the top-ranked tickers (pred_rank >= CONFIDENCE_THRESHOLD).
                              Threshold is a *rank percentile* now, not a probability —
                              0.53 means roughly the top 47% of the live universe.
          - regime_filtered : long/short tilt with raw_weight = pred_rank − 0.5,
                              shorts disabled in bull regimes via the HMM gate.
        """
        print(f"[Agent] Computing target allocations using '{strategy}' strategy "
              f"(pred_rank threshold = {config.CONFIDENCE_THRESHOLD})")

        target_weights = {t: 0.0 for t in config.TICKERS}

        # Use pred_rank as the canonical signal column; fall back to pred_proba
        # for any caller still using the legacy column name.
        if "pred_rank" not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df["pred_rank"] = predictions_df["pred_proba"]

        if strategy == "high_confidence":
            high_longs = predictions_df[
                predictions_df["pred_rank"] >= config.CONFIDENCE_THRESHOLD
            ].copy()

            if high_longs.empty:
                print(f"[Agent] No tickers above rank {config.CONFIDENCE_THRESHOLD}. "
                      f"Falling back to top 2 by rank.")
                high_longs = predictions_df.head(2).copy()

            # Weights proportional to predicted rank (top ranks get more capital)
            sum_rank = high_longs["pred_rank"].sum()
            if sum_rank > 0:
                for _, row in high_longs.iterrows():
                    target_weights[row["ticker"]] = (
                        row["pred_rank"] / sum_rank
                    ) * config.TARGET_EXPOSURE

        elif strategy == "regime_filtered":
            predictions_df = predictions_df.copy()
            # Centre at 0.5 so a uniform random ranker produces zero weights
            predictions_df["raw_weight"] = predictions_df["pred_rank"] - 0.5
            
            # Scale shorts using Kalman Beta
            def scale_short(row):
                w = row["raw_weight"]
                if w < 0:
                    beta = row.get("kalman_beta", 1.0)
                    if pd.isna(beta):
                        beta = 1.0
                    return w / max(abs(beta), 0.5)
                return w
            
            predictions_df["raw_weight"] = predictions_df.apply(scale_short, axis=1)
            
            # Apply confidence threshold triggers
            longs = predictions_df[predictions_df["raw_weight"] >= 0.02].copy()
            if is_bull:
                shorts = pd.DataFrame() # Suppress shorts in bull market
            else:
                shorts = predictions_df[predictions_df["raw_weight"] <= -0.02].copy()
                
            # Upgraded Short Easy-to-Borrow (ETB) Checks
            if not shorts.empty and getattr(config, "ALPACA_CHECK_BORROWABILITY", True):
                has_credentials = len(config.ALPACA_API_KEY) > 0 and len(config.ALPACA_SECRET_KEY) > 0
                if has_credentials:
                    print("[Agent] Checking short candidate borrowability via Alpaca Asset API...")
                    try:
                        from alpaca.trading.client import TradingClient
                        tc = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER_TRADING)
                        
                        etb_tickers = []
                        for _, row in shorts.iterrows():
                            ticker = row["ticker"]
                            try:
                                asset = tc.get_asset(ticker)
                                if asset.shortable and asset.easy_to_borrow:
                                    etb_tickers.append(ticker)
                                    print(f"  -> {ticker}: Easy-To-Borrow (ETB) verified.")
                                else:
                                    print(f"  -> [REJECTED] {ticker}: NOT Easy-To-Borrow or shortable. Setting weight to 0.0.")
                            except Exception as ex:
                                print(f"  -> [WARNING] Failed to query borrowability for {ticker}: {ex}. Setting weight to 0.0 for safety.")
                        
                        shorts = shorts[shorts["ticker"].isin(etb_tickers)].copy()
                    except Exception as e:
                        print(f"[WARNING] Alpaca Asset API query failed: {e}. Defaulting to safe no-short mode.")
                        shorts = pd.DataFrame()
                else:
                    print("[Agent] Alpaca credentials missing. Simulating dynamic ETB proxy (excluding alphabetical small-caps starting with 'AA' to emulate limited borrow pool)...")
                    etb_tickers = []
                    for _, row in shorts.iterrows():
                        ticker = row["ticker"]
                        if ticker.startswith("AA") and ticker not in ["AAPL"]:
                            print(f"  -> [MOCK REJECTED] {ticker}: NOT Easy-To-Borrow (mock). Setting weight to 0.0.")
                        else:
                            etb_tickers.append(ticker)
                    shorts = shorts[shorts["ticker"].isin(etb_tickers)].copy()

            if longs.empty and shorts.empty:
                print("[Agent] No candidates qualified for exposure. Holding 100% Cash.")
                return target_weights
                
            selected = pd.concat([longs, shorts])
            total_abs_weight = selected["raw_weight"].abs().sum()
            
            if total_abs_weight > 0:
                selected["target_weight"] = (selected["raw_weight"] / total_abs_weight) * config.TARGET_EXPOSURE
                for _, row in selected.iterrows():
                    target_weights[row["ticker"]] = row["target_weight"]
        else:
            raise ValueError(f"Unknown strategy code: {strategy}")

        return target_weights

    def _trade_log_row(self, *, ticker, action, target_w, delta_pct,
                       qty, notional, ref_price, dry_run, side="flat", notes=""):
        """Build a TradeLogRow stamped with this rebalance's decision_id and
        the signal decomposition for `ticker`."""
        pred_row = None
        if self.last_predictions_df is not None:
            sel = self.last_predictions_df[self.last_predictions_df["ticker"] == ticker]
            if not sel.empty:
                pred_row = sel.iloc[0]
        ml = float(pred_row.get("pred_rank",
                                pred_row.get("pred_proba", float("nan")))) \
            if pred_row is not None else float("nan")
        beta = float(pred_row.get("kalman_beta", 1.0)) if pred_row is not None else float("nan")
        ctx = self.last_decision_ctx
        return TradeLogRow(
            decision_id=ctx.decision_id if ctx else "no_ctx",
            decision_date=ctx.decision_date if ctx else "",
            ts=pd.Timestamp.utcnow().isoformat(timespec="seconds"),
            ticker=ticker, action=action,
            target_weight=float(target_w),
            delta_weight=float(delta_pct),
            intended_qty=int(qty), intended_notional_usd=float(notional),
            ref_price=float(ref_price),
            raw_score=ml,
            ml_signal=ml,
            hmm_state=int(self.hmm_state),
            hmm_prob_bull=float(self.hmm_probs[0]) if len(self.hmm_probs) > 0 else float("nan"),
            kalman_beta=beta, side=side, dry_run=bool(dry_run), notes=notes,
        )

    def execute_alpaca_rebalance(self, target_weights, prices_df=None, dry_run=True):
        """
        Executes real-world portfolio rebalancing on Alpaca.
        Leverages the Differential Portfolio Rebalancing Optimizer to save 30-58% in volume fees.

        Operational hardening:
          * Drawdown circuit breaker: refuses to trade if a -5% peak-to-trough
            stop has been hit (manual-reset halt file under ./agent_logs/).
          * Per-trade structured log: every order writes a row to
            ./agent_logs/trades.csv with the full signal decomposition.
        """
        print("\n" + "=" * 80)
        print("DIFFERENTIAL PORTFOLIO REBALANCING OPTIMIZER")
        print("=" * 80)

        # ── Drawdown circuit breaker ──────────────────────────────────────
        # Skip the check entirely in dry-run mode without credentials (no real
        # equity to track); otherwise pull current equity from Alpaca first.
        has_credentials = len(config.ALPACA_API_KEY) > 0 and len(config.ALPACA_SECRET_KEY) > 0
        if self.circuit_breaker.is_halted():
            print("[HALT] DrawdownCircuitBreaker is tripped — refusing to submit any orders.")
            print(f"[HALT] Inspect {self.circuit_breaker.history_path} and remove the halt file manually after review.")
            print("=" * 80)
            return

        if has_credentials and not dry_run:
            try:
                from alpaca.trading.client import TradingClient
                _tc = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                                    paper=config.ALPACA_PAPER_TRADING)
                _acc = _tc.get_account()
                _eq = float(_acc.portfolio_value)
                status = self.circuit_breaker.update_and_check(_eq)
                print(f"[CircuitBreaker] equity=${_eq:,.2f}  peak=${status['peak_equity']:,.2f}  "
                      f"drawdown={status['drawdown_pct']:.2f}%  threshold={status['threshold_pct']:.2f}%")
                if status["halted"]:
                    print("[HALT] Drawdown threshold breached — bot just halted.  Aborting rebalance.")
                    print("=" * 80)
                    return
            except Exception as e:
                print(f"[CircuitBreaker] [WARN] Could not fetch live equity: {e}.  Proceeding with rebalance.")

        if dry_run or not has_credentials:
            if not has_credentials:
                print("[MOCK NOTE] Alpaca API credentials not set in environmental variables.")
                print("[MOCK NOTE] Running in interactive local dry-run simulation mode.\n")

            simulated_equity = 10000.0
            print(f"Starting simulated account equity: ${simulated_equity:,.2f}")
            
            # Mock portfolio simulation (holding small previous short and long exposures)
            mock_holdings = {
                'AAME': -300.0,
                'AAOI': -350.0,
                'AAPL': 1000.0,
                'ABCB': 800.0,
            }
            
            print("Current Simulated Holdings (Previous Week):")
            for ticker, val in mock_holdings.items():
                side = "LONG" if val > 0 else "SHORT"
                print(f"  -> {ticker:<5} : ${abs(val):>8,.2f} ({abs(val)/simulated_equity*100:.2f}% {side})")
            print(f"  -> CASH  : ${simulated_equity - sum(mock_holdings.values()):>8,.2f}")

            # Rebalancing logic
            naive_traded_volume = 0.0
            differential_traded_volume = 0.0
            
            trades = []
            for ticker in config.TICKERS:
                curr_val = mock_holdings.get(ticker, 0.0)
                target_w = target_weights.get(ticker, 0.0)
                target_val = target_w * simulated_equity
                
                delta_val = target_val - curr_val
                naive_traded_volume += abs(target_val) + abs(curr_val)
                differential_traded_volume += abs(delta_val)
                
                if abs(delta_val) >= config.MIN_ORDER_VALUE:
                    if curr_val == 0.0 and target_val != 0.0:
                        side_str = "long" if target_val > 0 else "short"
                        action = "BUY (Long)" if target_val > 0 else "SELL (Short)"
                        reason = f"Establish NEW {side_str} position"
                    elif target_val == 0.0 and curr_val != 0.0:
                        side_str = "flat"
                        action = "SELL (Cover)" if curr_val < 0 else "SELL (Liquidate)"
                        reason = "LIQUIDATE position entirely"
                    else:
                        side_str = "long" if target_val > 0 else ("short" if target_val < 0 else "flat")
                        action = "BUY" if delta_val > 0 else "SELL"
                        reason = "Adjust existing target exposure"

                    trades.append({
                        "ticker": ticker,
                        "curr": curr_val / simulated_equity * 100.0,
                        "tgt": target_w * 100.0,
                        "delta": delta_val / simulated_equity * 100.0,
                        "action": action,
                        "trade_usd": abs(delta_val),
                        "reason": reason
                    })

                    # Per-trade structured log (dry-run side)
                    try:
                        ref_p = 1.0
                        if prices_df is not None:
                            pr = prices_df[prices_df["ticker"] == ticker]
                            if not pr.empty:
                                ref_p = float(pr.sort_values("Date").iloc[-1]["company_close"])
                        self.trade_logger.log(self._trade_log_row(
                            ticker=ticker, action=action, target_w=target_w,
                            delta_pct=delta_val / simulated_equity,
                            qty=int(abs(delta_val) / max(ref_p, 1e-6)),
                            notional=abs(delta_val), ref_price=ref_p,
                            dry_run=True, side=side_str, notes=reason,
                        ))
                    except Exception as e:
                        print(f"  -> [WARN] trade logger failed for {ticker}: {e}")
                    
            trades.sort(key=lambda x: x["delta"])  # SELLs first
            
            print("\nOptimized Action Schedule:")
            print("-" * 115)
            print(f"{'Ticker':<8} | {'Current %':<10} | {'Target %':<10} | {'Delta %':<10} | {'Action':<15} | {'Trade USD':<12} | {'Reason'}")
            print("-" * 115)
            for t in trades:
                print(f"{t['ticker']:<8} | {t['curr']:>8.2f}% | {t['tgt']:>8.2f}% | {t['delta']:>8.2f}% | {t['action']:<15} | ${t['trade_usd']:>10.2f} | {t['reason']}")
            print("-" * 115)
            
            volume_saved = abs(naive_traded_volume - differential_traded_volume)
            pct_saved = (volume_saved / max(naive_traded_volume, 1e-9)) * 100
            
            print(f"Naive 'Liquidate-All' Traded Volume      : ${naive_traded_volume:,.2f}")
            print(f"Differential Rebalance Traded Volume    : ${differential_traded_volume:,.2f}")
            print(f"TRADED VOLUME ELIMINATED (SAVINGS)      : ${volume_saved:,.2f} ({pct_saved:.2f}% reduction)")
            print(f"ESTIMATED TRANSACTION COST SAVED (0.2%): ${volume_saved * 0.0020:,.2f}")
            print("=" * 80)
            return

        # Alpaca live trading execution
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER_TRADING)
            account = client.get_account()
            equity = float(account.portfolio_value)
            print(f"[Broker] Connected to Alpaca. Endpoint: {config.ALPACA_URL}")
            print(f"[Broker] Net Liquidity / Account Equity: ${equity:,.2f}")

            # Fetch active positions
            positions = client.get_all_positions()
            current_holdings = {pos.symbol: float(pos.market_value) for pos in positions}
            
            print("[Broker] Current portfolio positions fetched:")
            for symbol, val in current_holdings.items():
                print(f"  -> Active: {symbol:<5} | Market Value: ${val:>8,.2f} ({val/equity*100:.2f}%)")

            # Differential trades list
            trades_to_execute = []
            for ticker in config.TICKERS:
                curr_val = current_holdings.get(ticker, 0.0)
                target_w = target_weights.get(ticker, 0.0)
                target_val = target_w * equity
                
                delta_val = target_val - curr_val
                if abs(delta_val) >= config.MIN_ORDER_VALUE:
                    trades_to_execute.append({
                        "ticker": ticker,
                        "delta": delta_val,
                        "target_w": target_w
                    })

            # Sort so negative trades (Sells / covers) execute first
            trades_to_execute.sort(key=lambda x: x["delta"])

            print(f"\n[Broker] Dispatching {len(trades_to_execute)} optimized differential orders...")
            for trade in trades_to_execute:
                ticker = trade["ticker"]
                delta = trade["delta"]
                target_w = trade["target_w"]

                # Get latest price of the stock to calculate whole shares
                ticker_price = 1.0
                if prices_df is not None:
                    ticker_rows = prices_df[prices_df["ticker"] == ticker]
                    if not ticker_rows.empty:
                        ticker_price = float(ticker_rows.sort_values("Date").iloc[-1]["company_close"])
                    else:
                        try:
                            ticker_price = float(yf.download(ticker, period="1d", progress=False).iloc[-1]["Close"])
                        except Exception:
                            ticker_price = 1.0
                else:
                    try:
                        ticker_price = float(yf.download(ticker, period="1d", progress=False).iloc[-1]["Close"])
                    except Exception:
                        ticker_price = 1.0

                if delta < 0:
                    # Sell
                    side = OrderSide.SELL
                    side_str = "flat" if target_w == 0.0 else ("short" if target_w < 0 else "long")
                    if target_w == 0.0:
                        print(f"  -> [ORDER] Liquidating 100% of {ticker} (Value: ${abs(delta):,.2f})")
                        client.close_position(ticker)
                        action = "SELL (Liquidate)"
                        qty_logged = 0
                    else:
                        qty = int(abs(delta) / ticker_price)
                        action = "SELL"
                        qty_logged = qty
                        if qty > 0:
                            print(f"  -> [ORDER] Selling {qty} shares of {ticker} to reduce exposure (Value: ${qty * ticker_price:,.2f}, price: ${ticker_price:.2f})")
                            order = MarketOrderRequest(symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY)
                            client.submit_order(order)
                        else:
                            print(f"  -> [skip] Order value for {ticker} is too small for a whole share (Value: ${abs(delta):,.2f}).")
                else:
                    # Buy
                    side = OrderSide.BUY
                    side_str = "long" if target_w > 0 else ("short" if target_w < 0 else "flat")
                    qty = int(delta / ticker_price)
                    action = "BUY"
                    qty_logged = qty
                    if qty > 0:
                        print(f"  -> [ORDER] Buying {qty} shares of {ticker} to establish/increase exposure (Value: ${qty * ticker_price:,.2f}, price: ${ticker_price:.2f})")
                        order = MarketOrderRequest(symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY)
                        client.submit_order(order)
                    else:
                        print(f"  -> [skip] Order value for {ticker} is too small for a whole share (Value: ${delta:,.2f}).")

                # Per-trade structured log (live side, after submit attempt)
                try:
                    self.trade_logger.log(self._trade_log_row(
                        ticker=ticker, action=action, target_w=target_w,
                        delta_pct=delta / max(equity, 1e-6),
                        qty=int(qty_logged),
                        notional=abs(delta), ref_price=ticker_price,
                        dry_run=False, side=side_str,
                    ))
                except Exception as e:
                    print(f"  -> [WARN] trade logger failed for {ticker}: {e}")

            print("[Broker] Portfolio rebalance completed successfully.")
            print("=" * 80)
        except Exception as e:
            print(f"\n[ERROR] Alpaca Broker Execution failure: {e}")
            print("=" * 80)

    def record_attribution(self, *, weights_prev, weights_curr,
                            realised_returns, benchmark_return,
                            period_start, period_end):
        """Decompose realised PnL between the previous and current rebalance
        into HMM / ML / Kalman / residual buckets and append to
        ./agent_logs/attribution.csv.

        Caller responsibilities:
          - weights_prev: {ticker: weight} as of the prior rebalance
          - weights_curr: {ticker: weight} just decided
          - realised_returns: {ticker: realised return over the window}
          - benchmark_return: realised S&P 500 return over the window
        """
        ctx = self.last_decision_ctx
        decision_id = ctx.decision_id if ctx else "no_ctx"
        try:
            snap = self.attribution.attribute(
                decision_id=decision_id,
                period_start=str(period_start),
                period_end=str(period_end),
                weights_prev=weights_prev,
                weights_curr=weights_curr,
                realised_returns=realised_returns,
                benchmark_return=float(benchmark_return),
                kalman_betas=dict(self.kalman_betas),
                hmm_state=self.hmm_state,
            )
            print(f"[Attribution] regime={snap.pnl_regime_pct:+.3f}% "
                  f"ml={snap.pnl_ml_pct:+.3f}% kalman={snap.pnl_kalman_pct:+.3f}% "
                  f"residual={snap.pnl_residual_pct:+.3f}%  "
                  f"realised={snap.realised_return_pct:+.3f}% "
                  f"(decision_id={decision_id})")
        except Exception as e:
            print(f"[WARN] attribution failed: {e}")
