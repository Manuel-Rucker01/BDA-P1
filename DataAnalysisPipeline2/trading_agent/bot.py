"""
Production BDA Trading Bot Core Agent Module.
This module downloads live data, computes technical features, runs PCA-GCN model inference,
determines the S&P 500 market regime, and dispatches optimized differential trades to Alpaca.
"""

import os
import pickle
import sys
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

_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PIPELINE_DIR not in sys.path:
    sys.path.append(_PIPELINE_DIR)

from features.macro_provider import get_macro_features_for_date, load_static_macro_features

# --- Helper Technical Indicators ---

# ── Pickle compat shim for MLP regressor ─────────────────────────────────
# best_model.pkl serialises a TorchMLPRegressor that lives in
# DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py. When the
# bake-off ran as `python kg_embeddings_classifier.py`, the class was
# pickled under __main__.TorchMLPRegressor — so this loader has to
# re-publish it on __main__ for pickle.load to resolve.
try:
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _rel in ('..', '../..', '../scripts', '../../scripts'):
        _cand = _os.path.abspath(_os.path.join(_here, _rel))
        if _os.path.exists(_os.path.join(_cand, 'kg_embeddings_classifier.py')):
            if _cand not in _sys.path:
                _sys.path.insert(0, _cand)
            break
    from kg_embeddings_classifier import TorchMLPRegressor as _TorchMLPRegressor
    _sys.modules['__main__'].TorchMLPRegressor = _TorchMLPRegressor
except Exception as _e:
    print(f'[shim] could not pre-register TorchMLPRegressor: {_e}')
# ──────────────────────────────────────────────────────────────────────────


def inverse_volatility_weights(tickers, vols, target_exposure=1.0,
                               max_weight=0.25, vol_floor=1e-3, n_cap_iters=4):
    """Risk-parity-lite intra-basket sizing: w_i proportional to 1/vol_i.

    Higher-volatility names receive less capital so a single volatile name
    cannot dominate portfolio drawdown. A per-name cap (`max_weight`) is
    enforced by iteratively clipping and redistributing the excess across the
    uncapped names, so the result respects the cap and still sums to
    `target_exposure`.

    Parameters
    ----------
    tickers : sequence of symbols
    vols    : sequence of realized volatilities aligned with `tickers`
    Returns
    -------
    dict ticker -> weight (summing to ~target_exposure)
    """
    import numpy as _np
    tickers = list(tickers)
    if not tickers:
        return {}
    v = _np.maximum(_np.asarray(vols, dtype=float), vol_floor)
    w = 1.0 / v
    w = w / w.sum()
    # Enforce the per-name cap by FREEZING names at the cap and redistributing
    # the remaining budget only among strictly-free names. Freezing prevents
    # the ping-pong where an already-capped name gets re-inflated by the next
    # redistribution. If the cap makes full exposure infeasible
    # (n_names * max_weight < 1) the book is deliberately left partially in
    # cash rather than breaching the cap.
    cap = max_weight
    capped = _np.zeros(len(w), dtype=bool)
    for _ in range(n_cap_iters):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        w[over] = cap
        capped |= over
        free = ~capped
        remaining = 1.0 - float(w[capped].sum())
        if not free.any() or remaining <= 0:
            w[free] = 0.0
            break
        w[free] = w[free] / w[free].sum() * remaining
    w = w * target_exposure
    return {t: float(wi) for t, wi in zip(tickers, w)}


def portfolio_exposure_summary(weights):
    """Return long/short/gross/net exposure for a signed weight mapping."""
    vals = np.asarray(list(weights.values()), dtype=float) if weights else np.asarray([], dtype=float)
    if vals.size == 0:
        return {"long": 0.0, "short": 0.0, "gross": 0.0, "net": 0.0}
    long_exposure = float(vals[vals > 0].sum())
    short_exposure = float(-vals[vals < 0].sum())
    return {
        "long": long_exposure,
        "short": short_exposure,
        "gross": long_exposure + short_exposure,
        "net": float(vals.sum()),
    }


def enforce_portfolio_constraints(weights, *, max_gross, max_net, max_short,
                                  max_long, max_position, tol=1e-10):
    """Clip and scale signed weights to satisfy portfolio exposure guardrails.

    The transformation only reduces exposure: per-name clips first, side caps
    second, then gross/net caps. This keeps rank-based allocations intact within
    each side while ensuring the final book cannot exceed configured limits.
    """
    if not weights:
        return {}

    capped = {
        t: float(np.clip(w, -max_position, max_position))
        for t, w in weights.items()
    }

    for _ in range(4):
        summary = portfolio_exposure_summary(capped)
        long_exp = summary["long"]
        short_exp = summary["short"]

        if long_exp > max_long + tol and long_exp > 0:
            scale = max_long / long_exp
            capped = {t: (w * scale if w > 0 else w) for t, w in capped.items()}
        if short_exp > max_short + tol and short_exp > 0:
            scale = max_short / short_exp
            capped = {t: (w * scale if w < 0 else w) for t, w in capped.items()}

        summary = portfolio_exposure_summary(capped)
        if summary["gross"] > max_gross + tol and summary["gross"] > 0:
            scale = max_gross / summary["gross"]
            capped = {t: w * scale for t, w in capped.items()}

        summary = portfolio_exposure_summary(capped)
        net = summary["net"]
        if net > max_net + tol and summary["long"] > 0:
            allowed_long = max(max_net + summary["short"], 0.0)
            scale = min(1.0, allowed_long / summary["long"])
            capped = {t: (w * scale if w > 0 else w) for t, w in capped.items()}
        elif net < -max_net - tol and summary["short"] > 0:
            allowed_short = max(max_net + summary["long"], 0.0)
            scale = min(1.0, allowed_short / summary["short"])
            capped = {t: (w * scale if w < 0 else w) for t, w in capped.items()}

    return {t: (0.0 if abs(w) < tol else float(w)) for t, w in capped.items()}


def assert_portfolio_invariants(weights, *, max_gross, max_net, max_short,
                                max_long, max_position, tol=1e-8):
    """Raise if a signed weight mapping breaches configured exposure limits."""
    summary = portfolio_exposure_summary(weights)
    max_abs_name = max((abs(float(w)) for w in weights.values()), default=0.0)
    checks = {
        "gross": summary["gross"] <= max_gross + tol,
        "net": abs(summary["net"]) <= max_net + tol,
        "short": summary["short"] <= max_short + tol,
        "long": summary["long"] <= max_long + tol,
        "position": max_abs_name <= max_position + tol,
    }
    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise AssertionError(
            f"Portfolio invariant breach ({failed}): "
            f"long={summary['long']:.6f}, short={summary['short']:.6f}, "
            f"gross={summary['gross']:.6f}, net={summary['net']:.6f}, "
            f"max_abs_position={max_abs_name:.6f}"
        )
    return summary


def build_regime_filtered_weights(predictions_df, *, is_bull,
                                  target_exposure=1.0,
                                  confidence_threshold=0.02,
                                  max_gross=1.0,
                                  max_net=1.0,
                                  max_short=0.30,
                                  max_long=1.0,
                                  max_position=0.25,
                                  apply_kalman_short_scaling=True):
    """Build regime-filtered long/short weights with explicit invariants."""
    if predictions_df is None or predictions_df.empty:
        return {}, {"long": 0.0, "short": 0.0, "gross": 0.0, "net": 0.0}

    df = predictions_df.copy()
    if "pred_rank" not in df.columns:
        df["pred_rank"] = df["pred_proba"]

    # Centre at 0.5 so a uniform random ranker produces zero weights.
    df["raw_weight"] = df["pred_rank"].astype(float) - 0.5

    if apply_kalman_short_scaling and "kalman_beta" in df.columns:
        def scale_short(row):
            w = row["raw_weight"]
            if w < 0:
                beta = row.get("kalman_beta", 1.0)
                if pd.isna(beta):
                    beta = 1.0
                return w / max(abs(beta), 0.5)
            return w
        df["raw_weight"] = df.apply(scale_short, axis=1)

    longs = df[df["raw_weight"] >= confidence_threshold].copy()
    shorts = pd.DataFrame() if is_bull else df[df["raw_weight"] <= -confidence_threshold].copy()

    if longs.empty and shorts.empty:
        return {}, {"long": 0.0, "short": 0.0, "gross": 0.0, "net": 0.0}

    weights = {}
    long_budget = min(float(target_exposure), float(max_long), float(max_gross))
    short_budget = 0.0 if is_bull else min(float(max_short), float(max_gross))

    if not longs.empty and long_budget > 0:
        long_sum = float(longs["raw_weight"].sum())
        if long_sum > 0:
            longs["target_weight"] = (longs["raw_weight"] / long_sum) * long_budget
            weights.update(longs.set_index("ticker")["target_weight"].to_dict())

    if not shorts.empty and short_budget > 0:
        short_sum = float(shorts["raw_weight"].abs().sum())
        if short_sum > 0:
            shorts["target_weight"] = (shorts["raw_weight"] / short_sum) * short_budget
            weights.update(shorts.set_index("ticker")["target_weight"].to_dict())

    weights = enforce_portfolio_constraints(
        weights,
        max_gross=float(max_gross),
        max_net=float(max_net),
        max_short=float(max_short),
        max_long=float(max_long),
        max_position=float(max_position),
    )
    summary = assert_portfolio_invariants(
        weights,
        max_gross=float(max_gross),
        max_net=float(max_net),
        max_short=float(max_short),
        max_long=float(max_long),
        max_position=float(max_position),
    )
    return weights, summary


def select_top_k_with_sector_cap(gate_sorted_df, top_k, max_sector_frac,
                                  sector_col="Sector"):
    """Pick the top-K names from a rank-sorted candidate frame while capping how
    many may come from any single sector, so the book spans several sectors
    instead of clustering in one.

    The cap is a per-sector COUNT: max_per_sector = max(1, round(top_k *
    max_sector_frac)). Names are admitted in descending rank order; a name is
    skipped only if its sector is already full. If sector caps make it
    impossible to reach K (too few sectors represented), the shortfall is then
    filled by the highest-ranked remaining names ignoring the cap, so the book
    always holds K names when K candidates exist.

    Falls back to a plain head(top_k) when the cap is disabled or the sector
    column is missing.
    """
    if (max_sector_frac is None or max_sector_frac >= 1.0
            or sector_col not in gate_sorted_df.columns):
        return gate_sorted_df.head(top_k)
    max_per_sector = max(1, int(round(top_k * max_sector_frac)))
    chosen, counts = [], {}
    for idx, row in gate_sorted_df.iterrows():
        sec = row.get(sector_col, "UNKNOWN")
        if sec is None or (isinstance(sec, float) and pd.isna(sec)):
            sec = "UNKNOWN"
        if counts.get(sec, 0) < max_per_sector:
            chosen.append(idx)
            counts[sec] = counts.get(sec, 0) + 1
        if len(chosen) >= top_k:
            break
    if len(chosen) < top_k:  # under-filled: relax cap, fill by rank
        chosen_set = set(chosen)
        for idx in gate_sorted_df.index:
            if idx not in chosen_set:
                chosen.append(idx)
                if len(chosen) >= top_k:
                    break
    return gate_sorted_df.loc[chosen]


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
    """Load static macro features through the shared offline provider."""
    return load_static_macro_features(macro_ttl_path)

def fetch_company_metadata():
    """Fetch static sector, industry, market cap, and country info from historical databases."""
    db_path = os.path.join(config.EXPLOITATION_DIR, "ExploitationZone.duckdb")

    if not os.path.exists(db_path):
        print("[WARNING] Analytical database missing. Using default metadata values.")
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
        
        # Country lookup now lives in the Exploitation Zone (materialised by
        # graph_generation.py), not the Trusted Zone — read it from the same DB.
        conn_t = duckdb.connect(db_path, read_only=True)
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
    # Current macro policy is a static offline TTL snapshot shared with training
    # and backtests; it is not a point-in-time release feed.
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


def validate_model_artifact_schema(model_data):
    """Validate the minimal best_model.pkl contract before live inference.

    Older pickles do not carry the new metadata/manifest block, so this helper
    only requires the operational keys needed to reproduce the feature matrix.
    It raises ValueError with explicit missing/invalid keys instead of allowing
    downstream KeyError or silent feature misalignment.
    """
    if not isinstance(model_data, dict):
        raise ValueError("Model artifact must be a dict loaded from best_model.pkl.")

    required_keys = {
        "trained_models", "mix_models", "scaler", "pca",
        "tabular_cols", "pca_cols", "company_embeddings",
    }
    missing = sorted(required_keys - set(model_data.keys()))
    if missing:
        raise ValueError(f"Model artifact missing required keys: {missing}")

    if not isinstance(model_data["trained_models"], dict) or not model_data["trained_models"]:
        raise ValueError("Model artifact key 'trained_models' must be a non-empty dict.")
    column_seq_types = (list, tuple, pd.Index)

    if not isinstance(model_data["mix_models"], (list, tuple)) or len(model_data["mix_models"]) == 0:
        raise ValueError("Model artifact key 'mix_models' must be a non-empty list/tuple.")
    if not isinstance(model_data["tabular_cols"], column_seq_types) or len(model_data["tabular_cols"]) == 0:
        raise ValueError("Model artifact key 'tabular_cols' must be a non-empty list/tuple.")
    if not isinstance(model_data["pca_cols"], column_seq_types) or len(model_data["pca_cols"]) == 0:
        raise ValueError("Model artifact key 'pca_cols' must be a non-empty list/tuple.")
    if not isinstance(model_data["company_embeddings"], dict) or not model_data["company_embeddings"]:
        raise ValueError("Model artifact key 'company_embeddings' must be a non-empty dict.")
    if not hasattr(model_data["pca"], "transform"):
        raise ValueError("Model artifact key 'pca' must provide a transform(...) method.")
    if not bool(model_data.get("cs_z_standardize", False)) and not hasattr(model_data["scaler"], "transform"):
        raise ValueError("Model artifact key 'scaler' must provide a transform(...) method.")

    missing_models = sorted(set(model_data["mix_models"]) - set(model_data["trained_models"].keys()))
    if missing_models:
        raise ValueError(f"Model artifact mix_models not present in trained_models: {missing_models}")

    feature_cols = list(model_data["tabular_cols"]) + list(model_data["pca_cols"])
    duplicate_cols = sorted({c for c in feature_cols if feature_cols.count(c) > 1})
    if duplicate_cols:
        raise ValueError(f"Model artifact has duplicate feature columns: {duplicate_cols}")


def validate_required_columns(df, required_cols, context):
    """Fail fast when an inference DataFrame does not match the trained schema."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        available = sorted(map(str, df.columns))
        raise ValueError(
            f"{context} schema mismatch: missing required columns {missing}. "
            f"Available columns: {available}"
        )


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
        self.model_metadata = {}
        self.cs_z_standardize = False
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

        validate_model_artifact_schema(self.model_data)
        self.trained_models = self.model_data["trained_models"]
        self.mix_models = list(self.model_data["mix_models"])
        self.scaler = self.model_data["scaler"]
        self.pca = self.model_data["pca"]
        self.tabular_cols = list(self.model_data["tabular_cols"])
        self.pca_cols = list(self.model_data["pca_cols"])
        self.company_embeddings = self.model_data["company_embeddings"]
        self.model_metadata = (
            self.model_data.get("metadata")
            or self.model_data.get("manifest")
            or {}
        )
        # Part-4 P3: when CS-Z standardisation was used at training time,
        # apply the same transform at inference (per-date z-score on the
        # live cross-section). Falls back to global scaler otherwise.
        self.cs_z_standardize = bool(self.model_data.get("cs_z_standardize", False))
        if self.model_metadata:
            expected_cols = self.model_metadata.get("feature_columns")
            current_cols = self.tabular_cols + self.pca_cols
            if expected_cols is not None and list(expected_cols) != current_cols:
                raise ValueError(
                    "Model artifact metadata feature_columns does not match "
                    "tabular_cols + pca_cols."
                )
        print(f"[Agent] Model successfully loaded. Base models: {self.mix_models}  "
              f"cs_z_standardize={self.cs_z_standardize}  "
              f"schema_version={self.model_metadata.get('artifact_schema_version', 'legacy')}")

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
        """Downloads live historical bars for the active ticker universe.

        For small baskets (<= 30 tickers) we keep the per-ticker loop so error
        reporting stays granular. For the full-universe mode (~1,890 tickers)
        we use a single batched `yf.download(...)` call which is dramatically
        faster (Yahoo parallelises internally) and only logs the names that
        actually failed.
        """
        n = len(config.TICKERS)
        print(f"[Agent] Downloading 60 days of historical daily bars for {n} tickers...")

        if n <= 30:
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

        # ── Full-universe batched path ──────────────────────────────────────
        # yfinance returns a MultiIndex (field, ticker) DataFrame in batch
        # mode. We pivot it into long form with the same columns the per-
        # ticker path produces.
        batch = yf.download(
            tickers=" ".join(config.TICKERS),
            period="60d",
            progress=False,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )
        if batch is None or batch.empty:
            raise RuntimeError("Batched yfinance download returned no rows.")

        long_rows = []
        failures = []
        for ticker in config.TICKERS:
            try:
                if isinstance(batch.columns, pd.MultiIndex):
                    if ticker not in batch.columns.get_level_values(0):
                        failures.append(ticker)
                        continue
                    sub = batch[ticker].dropna(how="all").reset_index()
                else:
                    sub = batch.dropna(how="all").reset_index()
                if sub.empty:
                    failures.append(ticker)
                    continue
                sub = sub.rename(columns={
                    "Close": "company_close",
                    "Volume": "company_volume",
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                })
                sub["ticker"] = ticker
                sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
                long_rows.append(sub)
            except Exception as e:
                failures.append(ticker)
        if failures:
            print(f"[Agent] yfinance returned no data for {len(failures)} tickers "
                  f"(first 10: {failures[:10]}). Continuing with the rest.")
        if not long_rows:
            raise RuntimeError("Batched yfinance download produced no usable rows.")
        return pd.concat(long_rows, ignore_index=True)

    def run_inference(self, price_history_df):
        """Performs feature calculations, PCA embedding projections, and Soft-Voting ensemble inference."""
        validate_required_columns(
            price_history_df,
            ["ticker", "Date", "company_close", "company_volume"],
            "price_history_df",
        )
        metadata_df = fetch_company_metadata()
        latest_as_of = price_history_df["Date"].max()
        macro_df = get_macro_features_for_date(latest_as_of, config.MACRO_KG_PATH)

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

        # ── Point-in-time news features (only if the loaded model expects them) ──
        # A news-trained model carries news_* columns in self.tabular_cols. We
        # compute the SAME as-of features live (articles strictly before today)
        # via the canonical provider, so live inference matches training. Names
        # with no recent news get 0 (no signal). Fails soft to zeros.
        news_cols = [c for c in self.tabular_cols if c.startswith("news_")]
        if news_cols:
            for c in news_cols:
                if c not in latest_df.columns:
                    latest_df[c] = 0.0
            try:
                try:
                    from . import news_sentiment as _ns
                except Exception:
                    import news_sentiment as _ns
                as_of = pd.to_datetime(latest_df["Date"]).max()
                nf = _ns.compute_asof_news_features(
                    latest_df["ticker"].unique().tolist(), [as_of])
                if nf is not None and not nf.empty:
                    nf = nf.drop(columns=["Date"]).set_index("ticker")
                    for c in news_cols:
                        if c in nf.columns:
                            latest_df[c] = (latest_df["ticker"].map(nf[c])
                                            .astype(float).fillna(0.0).values)
                    cov = int((latest_df.get("news_count_7d", 0) > 0).sum())
                    print(f"[News-feat] PIT news features for {len(news_cols)} cols; "
                          f"{cov}/{len(latest_df)} names have coverage (as-of {as_of.date()}).")
                else:
                    print("[News-feat] no news returned; news features = 0.")
            except Exception as e:
                print(f"[News-feat] failed ({e}); news features = 0.")
            for c in news_cols:
                latest_df[c] = latest_df[c].fillna(0.0)

        # Scale features using standard scaling
        validate_required_columns(latest_df, self.tabular_cols, "live tabular feature frame")
        validate_required_columns(latest_df, self.pca_cols, "live embedding feature frame")
        X_tab = latest_df[self.tabular_cols].fillna(0).values.astype(np.float32)
        X_emb = latest_df[self.pca_cols].fillna(0).values.astype(np.float32)
        X_full = np.concatenate([X_tab, X_emb], axis=1)
        if self.cs_z_standardize:
            # Single-date cross-section z-score (mirrors training preprocessing)
            mean = X_full.mean(axis=0, keepdims=True)
            std = X_full.std(axis=0, keepdims=True) + 1e-8
            X_full_s = np.clip((X_full - mean) / std, -6.0, 6.0).astype(np.float32)
        else:
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

        if "return_volatility_20d" not in latest_df.columns:
            latest_df["return_volatility_20d"] = 0.01
        if "Sector" not in latest_df.columns:
            latest_df["Sector"] = "UNKNOWN"
        return latest_df[["ticker", "company_close", "pred_rank",
                          "pred_proba", "kalman_beta",
                          "return_volatility_20d", "Sector"]].sort_values(
            "pred_rank", ascending=False)

    def apply_news_sentiment_tilt(self, predictions_df):
        """LIVE-ONLY: tilt the model's predicted rank by real-time news
        sentiment, then re-rank. The model decides the base ordering; recent
        news nudges it. This runs ONLY in the live path — it is never part of
        the backtests or the structural embeddings (no PIT news archive exists
        to backtest it without look-ahead bias).

            tilted = pred_rank + λ · sentiment       (sentiment ∈ [-1, 1])
            pred_rank ← percentile-rank(tilted)

        Fails soft: if news/sentiment is unavailable the predictions are
        returned unchanged.
        """
        if not getattr(config, "USE_NEWS_SENTIMENT", False):
            return predictions_df
        try:
            from . import news_sentiment
        except Exception:
            import news_sentiment  # script-mode fallback
        try:
            tickers = predictions_df["ticker"].tolist()
            agg = news_sentiment.get_live_sentiment(
                tickers, config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                lookback_days=getattr(config, "NEWS_LOOKBACK_DAYS", 7),
            )
            if not agg:
                print("[News] no sentiment available — predictions unchanged.")
                predictions_df["news_sentiment"] = 0.0
                predictions_df["news_n"] = 0
                return predictions_df

            df = predictions_df.copy()
            df["news_sentiment"] = df["ticker"].map(
                lambda t: agg.get(t, {}).get("sentiment", 0.0)).astype(float)
            df["news_n"] = df["ticker"].map(
                lambda t: agg.get(t, {}).get("n", 0)).astype(int)

            lam = float(getattr(config, "NEWS_TILT_LAMBDA", 0.10))
            tilted = df["pred_rank"].astype(float) + lam * df["news_sentiment"]
            df["pred_rank"] = pd.Series(tilted).rank(pct=True).values
            df["pred_proba"] = df["pred_rank"]
            n_tilted = int((df["news_n"] > 0).sum())
            print(f"[News] applied sentiment tilt (λ={lam}) to {n_tilted} names "
                  f"with news coverage; re-ranked {len(df)} predictions.")
            return df.sort_values("pred_rank", ascending=False)
        except Exception as e:
            print(f"[News] sentiment tilt failed ({e}); predictions unchanged.")
            return predictions_df

    def calculate_target_weights(self, predictions_df, is_bull, strategy="high_confidence"):
        """
        Computes optimal target portfolio weights from cross-sectional 30d
        rank predictions in [0, 1].

        Strategies:
          - high_confidence : long the top-ranked tickers (pred_rank >= CONFIDENCE_THRESHOLD).
                              Threshold is a *rank percentile* now, not a probability —
                              0.53 means roughly the top 47% of the live universe.
          - top_k           : the concentrated full-universe mode. Filter to the
                              top TOP_PCT_THRESHOLD percent of pred_rank, cap at
                              TOP_K_HOLDINGS names, equal-weight (or pred_rank
                              proportional if EQUAL_WEIGHT_TOP_K=False).
          - regime_filtered : long/short tilt with raw_weight = pred_rank − 0.5,
                              shorts disabled in bull regimes via the HMM gate,
                              then clipped/scaled to configured exposure limits.
        """
        print(f"[Agent] Computing target allocations using '{strategy}' strategy "
              f"(pred_rank threshold = {config.CONFIDENCE_THRESHOLD})")

        target_weights = {t: 0.0 for t in config.TICKERS}

        # Use pred_rank as the canonical signal column; fall back to pred_proba
        # for any caller still using the legacy column name.
        if "pred_rank" not in predictions_df.columns:
            predictions_df = predictions_df.copy()
            predictions_df["pred_rank"] = predictions_df["pred_proba"]

        if strategy == "top_k":
            # Concentrated full-universe mode: top-pct gate → top-K cap → equal weight.
            n = len(predictions_df)
            pct_cutoff = 1.0 - (config.TOP_PCT_THRESHOLD / 100.0)
            gate = predictions_df[predictions_df["pred_rank"] >= pct_cutoff].copy()
            gate = gate.sort_values("pred_rank", ascending=False)
            max_sec = getattr(config, "MAX_SECTOR_WEIGHT", 1.0)
            holdings = select_top_k_with_sector_cap(
                gate, config.TOP_K_HOLDINGS, max_sec).copy()
            if holdings.empty:
                print(f"[Agent] No tickers above the top {config.TOP_PCT_THRESHOLD}% gate. "
                      f"Falling back to top {config.TOP_K_HOLDINGS} by rank.")
                fallback = predictions_df.sort_values("pred_rank", ascending=False)
                holdings = select_top_k_with_sector_cap(
                    fallback, config.TOP_K_HOLDINGS, max_sec).copy()
            scheme = getattr(config, "WEIGHTING_SCHEME", "inverse_vol")
            n_sectors = holdings["Sector"].nunique() if "Sector" in holdings.columns else 0
            print(f"[Agent] top_k mode: universe={n} | "
                  f"above-top-{config.TOP_PCT_THRESHOLD}% gate={len(gate)} | "
                  f"held={len(holdings)} across {n_sectors} sectors "
                  f"(<= {max(1, round(config.TOP_K_HOLDINGS * max_sec))}/sector) | "
                  f"weighting={scheme}")

            if scheme == "inverse_vol" and "return_volatility_20d" in holdings.columns:
                # Alpha layer already chose the names; size them by 1/vol so
                # high-volatility names take less capital and don't dominate
                # the concentrated book's drawdown.
                w = inverse_volatility_weights(
                    holdings["ticker"].tolist(),
                    holdings["return_volatility_20d"].tolist(),
                    target_exposure=config.TARGET_EXPOSURE,
                    max_weight=getattr(config, "MAX_POSITION_WEIGHT", 0.25),
                    vol_floor=getattr(config, "VOL_FLOOR", 1e-3),
                )
                target_weights.update(w)
            elif scheme == "pred_rank":
                sum_rank = holdings["pred_rank"].sum()
                if sum_rank > 0:
                    for _, row in holdings.iterrows():
                        target_weights[row["ticker"]] = (
                            row["pred_rank"] / sum_rank
                        ) * config.TARGET_EXPOSURE
            else:  # "equal" (or inverse_vol fallback when vol column absent)
                w_each = config.TARGET_EXPOSURE / max(len(holdings), 1)
                for _, row in holdings.iterrows():
                    target_weights[row["ticker"]] = w_each

        elif strategy == "high_confidence":
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
            predictions_df["raw_weight"] = predictions_df["pred_rank"].astype(float) - 0.5

            # Scale shorts using Kalman Beta before signal gating/borrow checks.
            def scale_short(row):
                w = row["raw_weight"]
                if w < 0:
                    beta = row.get("kalman_beta", 1.0)
                    if pd.isna(beta):
                        beta = 1.0
                    return w / max(abs(beta), 0.5)
                return w
            
            predictions_df["raw_weight"] = predictions_df.apply(scale_short, axis=1)
            
            # Apply confidence threshold triggers; bull regimes carry no shorts.
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

            # Size long and short books separately, then clip/scale the final
            # signed book so gross/net/side/per-name invariants always hold.
            selected = pd.concat([longs, shorts])
            constrained_weights, exposure = build_regime_filtered_weights(
                selected,
                is_bull=is_bull,
                target_exposure=getattr(config, "TARGET_EXPOSURE", 1.0),
                confidence_threshold=0.02,
                max_gross=getattr(config, "MAX_GROSS_EXPOSURE", 1.0),
                max_net=getattr(config, "MAX_NET_EXPOSURE", 1.0),
                max_short=getattr(config, "MAX_SHORT_EXPOSURE", 0.30),
                max_long=getattr(config, "MAX_LONG_EXPOSURE", 1.0),
                max_position=getattr(config, "MAX_POSITION_WEIGHT", 0.25),
                apply_kalman_short_scaling=True,
            )
            if not constrained_weights:
                print("[Agent] Exposure constraints left no tradable allocation. Holding 100% Cash.")
                return target_weights
            target_weights.update(constrained_weights)
            print("[Agent] regime_filtered exposure: "
                  f"long={exposure['long']:.2f} short={exposure['short']:.2f} "
                  f"gross={exposure['gross']:.2f} net={exposure['net']:.2f}")
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
