"""Canonical portfolio helpers for backtest scripts.

This module is the lightweight consolidation point for selection, sizing, and
basic exposure/cost accounting used by canonical backtests.  Where possible it
delegates to the production trading agent helpers so the backtest path follows
the same top-k and inverse-volatility semantics as live allocation.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

try:
    from trading_agent import config
except Exception:
    config = None

try:
    from trading_agent.bot import (
        inverse_volatility_weights as _agent_inverse_volatility_weights,
        select_top_k_with_sector_cap as _agent_select_top_k_with_sector_cap,
    )
except Exception:
    _agent_inverse_volatility_weights = None
    _agent_select_top_k_with_sector_cap = None


def _cfg(name: str, default):
    if config is None:
        return default
    return getattr(config, name, default)


def signal_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Return the canonical score column, preferring pred_rank over pred_proba."""
    candidates = [preferred] if preferred else []
    candidates.extend(["pred_rank", "pred_proba"])
    for col in candidates:
        if col and col in df.columns:
            return col
    raise KeyError("pred_rank or pred_proba column is required for canonical selection")


def select_top_k(
    predictions_df: pd.DataFrame,
    pct_threshold: float | None = None,
    top_k: int | None = None,
    max_sector_frac: float | None = None,
    signal_col: str | None = None,
    sector_col: str = "Sector",
) -> pd.DataFrame:
    """Apply the production top-pct gate, top-k cap, and sector cap.

    The live bot ranks on ``pred_rank``; older backtests may still emit
    ``pred_proba``.  This helper accepts either and delegates sector capping to
    ``trading_agent.bot.select_top_k_with_sector_cap`` when available.
    """
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=getattr(predictions_df, "columns", []))

    score_col = signal_column(predictions_df, signal_col)
    pct_threshold = float(
        pct_threshold
        if pct_threshold is not None
        else os.environ.get("BACKTEST_TOP_PCT", _cfg("TOP_PCT_THRESHOLD", 5.0))
    )
    top_k = int(
        top_k
        if top_k is not None
        else os.environ.get("BACKTEST_TOP_K", _cfg("TOP_K_HOLDINGS", 20))
    )
    max_sector_frac = float(
        max_sector_frac
        if max_sector_frac is not None
        else os.environ.get("BACKTEST_MAX_SECTOR", _cfg("MAX_SECTOR_WEIGHT", 1.0))
    )

    cutoff = 1.0 - (pct_threshold / 100.0)
    ranked = predictions_df.sort_values(score_col, ascending=False)
    gated = ranked[ranked[score_col] >= cutoff].copy()
    if gated.empty:
        gated = ranked.copy()

    if _agent_select_top_k_with_sector_cap is not None:
        selected = _agent_select_top_k_with_sector_cap(
            gated, top_k, max_sector_frac, sector_col=sector_col
        )
    elif max_sector_frac is None or max_sector_frac >= 1.0 or sector_col not in gated.columns:
        selected = gated.head(top_k)
    else:
        max_per_sector = max(1, int(round(top_k * max_sector_frac)))
        chosen = []
        counts: dict[object, int] = {}
        for idx, row in gated.iterrows():
            sector = row.get(sector_col, "UNKNOWN")
            if pd.isna(sector):
                sector = "UNKNOWN"
            if counts.get(sector, 0) < max_per_sector:
                chosen.append(idx)
                counts[sector] = counts.get(sector, 0) + 1
            if len(chosen) >= top_k:
                break
        if len(chosen) < top_k:
            chosen_set = set(chosen)
            for idx in gated.index:
                if idx not in chosen_set:
                    chosen.append(idx)
                if len(chosen) >= top_k:
                    break
        selected = gated.loc[chosen]

    return selected.copy()


def inverse_volatility_weights_from_frame(
    selected_df: pd.DataFrame,
    target_exposure: float = 1.0,
    weighting_scheme: str | None = None,
    vol_col: str = "return_volatility_20d",
) -> dict[str, float]:
    """Return canonical long weights for a selected basket.

    ``weighting_scheme`` follows the production config.  Non-inverse-vol schemes
    and missing volatility data fall back to equal weights.
    """
    if selected_df is None or selected_df.empty:
        return {}

    tickers = selected_df["ticker"].tolist()
    scheme = weighting_scheme or _cfg("WEIGHTING_SCHEME", "inverse_vol")
    if scheme != "inverse_vol" or vol_col not in selected_df.columns:
        w_each = float(target_exposure) / max(len(tickers), 1)
        return {ticker: w_each for ticker in tickers}

    vols = selected_df[vol_col].to_numpy(dtype=float)
    if _agent_inverse_volatility_weights is not None:
        return _agent_inverse_volatility_weights(
            tickers,
            vols,
            target_exposure=target_exposure,
            max_weight=_cfg("MAX_POSITION_WEIGHT", 0.25),
            vol_floor=_cfg("VOL_FLOOR", 1e-3),
        )

    vols = np.maximum(vols, _cfg("VOL_FLOOR", 1e-3))
    raw = 1.0 / vols
    weights = raw / raw.sum()
    cap = _cfg("MAX_POSITION_WEIGHT", 0.25)
    capped = np.zeros(len(weights), dtype=bool)
    for _ in range(4):
        over = (weights > cap + 1e-12) & ~capped
        if not over.any():
            break
        weights[over] = cap
        capped |= over
        free = ~capped
        remaining = 1.0 - float(weights[capped].sum())
        if not free.any() or remaining <= 0:
            weights[free] = 0.0
            break
        weights[free] = weights[free] / weights[free].sum() * remaining
    weights = weights * target_exposure
    return {ticker: float(weight) for ticker, weight in zip(tickers, weights)}


def weighted_return(
    weights: Mapping[str, float],
    ticker_returns: Mapping[str, float],
) -> float:
    """Compute a signed portfolio return from ticker weights and realized returns."""
    return float(sum(weight * ticker_returns.get(ticker, 0.0) for ticker, weight in weights.items()))


def exposure_metrics(
    weights: Mapping[str, float],
    previous_weights: Mapping[str, float] | None = None,
    transaction_cost_bps: float = 0.0,
) -> dict[str, float]:
    """Summarize gross/net exposure, turnover, and simple proportional costs."""
    weights = dict(weights or {})
    previous_weights = dict(previous_weights or {})
    tickers = set(weights) | set(previous_weights)
    turnover = sum(abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in tickers)
    gross = sum(abs(weight) for weight in weights.values())
    net = sum(weights.values())
    long_exposure = sum(weight for weight in weights.values() if weight > 0)
    short_exposure = -sum(weight for weight in weights.values() if weight < 0)
    cost = turnover * (float(transaction_cost_bps) / 10000.0)
    return {
        "gross_exposure": float(gross),
        "net_exposure": float(net),
        "long_exposure": float(long_exposure),
        "short_exposure": float(short_exposure),
        "turnover": float(turnover),
        "transaction_cost": float(cost),
    }
