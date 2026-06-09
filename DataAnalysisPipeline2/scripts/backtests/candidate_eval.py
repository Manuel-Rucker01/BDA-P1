"""Shared helpers for the CANDIDATE-vs-PRODUCTION out-of-sample comparison.

The canonical engine builds ``friday_obs`` features via
``trading_agent.bot.compute_live_features`` which emits the 46 PRODUCTION
columns but NOT the 8 extra price-factor columns the CANDIDATE (candv1) model
was trained with:

    return_60d, return_120d, rank_return_60d, rank_return_120d,
    reversal_5d, rank_reversal_5d, mom_vol_adj, rank_mom_vol_adj

If the candidate model is run without computing these, ``reindex`` fills them
with 0 and the candidate predicts garbage. ``augment_extra_factors`` computes
exactly the missing extra columns (those that are in the model's tabular_cols
but absent from ``friday_obs``) from the price history, matching the training
recipe (NaN -> 0; rank_* = per-Friday cross-sectional PERCENT_RANK).

This module is import-only and never mutates models on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# The 8 extra price-factor columns the candidate adds on top of the 46
# production tabular columns. base -> how to derive; rank_* are cross-sectional.
EXTRA_BASE_FACTORS = ("return_60d", "return_120d", "reversal_5d", "mom_vol_adj")
EXTRA_RANK_FACTORS = {
    "rank_return_60d": "return_60d",
    "rank_return_120d": "return_120d",
    "rank_reversal_5d": "reversal_5d",
    "rank_mom_vol_adj": "mom_vol_adj",
}
EXTRA_FACTOR_COLS = tuple(EXTRA_BASE_FACTORS) + tuple(EXTRA_RANK_FACTORS)


def _base_factor_series(friday_obs: pd.DataFrame, df_all_feat: pd.DataFrame,
                        friday: str, col: str) -> pd.Series:
    """Compute a base extra-factor for the tickers present in friday_obs,
    as of `friday`, from the per-ticker close history in df_all_feat.

    Training filled NaN (insufficient history) with 0, so we mirror that.
    Returns a Series indexed by ticker.
    """
    tickers = friday_obs["ticker"].tolist()

    if col in ("return_60d", "return_120d"):
        lag = 60 if col == "return_60d" else 120
        out = {}
        for t in tickers:
            hist = df_all_feat[(df_all_feat["ticker"] == t) &
                               (df_all_feat["Date"] <= friday)].sort_values("Date")
            closes = hist["company_close"].to_numpy(dtype=float)
            if len(closes) > lag and closes[-1 - lag] > 0:
                out[t] = closes[-1] / closes[-1 - lag] - 1.0
            else:
                out[t] = 0.0  # NaN early in ~2y history -> 0 (matches training)
        return pd.Series(out)

    if col == "reversal_5d":
        # reversal_5d = -return_5d. return_5d already exists in the feature
        # output; fall back to computing it from closes if missing.
        if "return_5d" in friday_obs.columns:
            s = -friday_obs.set_index("ticker")["return_5d"].astype(float)
            return s.fillna(0.0)
        out = {}
        for t in tickers:
            hist = df_all_feat[(df_all_feat["ticker"] == t) &
                               (df_all_feat["Date"] <= friday)].sort_values("Date")
            closes = hist["company_close"].to_numpy(dtype=float)
            if len(closes) > 5 and closes[-6] > 0:
                out[t] = -(closes[-1] / closes[-6] - 1.0)
            else:
                out[t] = 0.0
        return pd.Series(out)

    if col == "mom_vol_adj":
        # mom_vol_adj = return_20d / (rolling_volatility_20d + 1e-9).
        # Both columns exist in compute_live_features output.
        df = friday_obs.set_index("ticker")
        r20 = df["return_20d"].astype(float) if "return_20d" in df.columns else pd.Series(0.0, index=df.index)
        v20 = df["rolling_volatility_20d"].astype(float) if "rolling_volatility_20d" in df.columns else pd.Series(0.0, index=df.index)
        s = r20 / (v20 + 1e-9)
        return s.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    raise ValueError(f"unknown base factor: {col}")


def augment_extra_factors(friday_obs: pd.DataFrame, df_all_feat: pd.DataFrame,
                          friday: str, tabular_cols) -> pd.DataFrame:
    """Add ONLY the extra factor columns that are required by ``tabular_cols``
    but missing from ``friday_obs``.

    For the PRODUCTION model (no extra factors in tabular_cols) this is a no-op.
    For the CANDIDATE model it computes the 8 momentum/reversal factors and
    their per-Friday cross-sectional percent-ranks. Returns a copy.
    """
    needed = [c for c in EXTRA_FACTOR_COLS
              if c in set(tabular_cols) and c not in friday_obs.columns]
    if not needed:
        return friday_obs

    out = friday_obs.copy()

    # Base factors first (ranks depend on them).
    for col in EXTRA_BASE_FACTORS:
        if col in needed or any(EXTRA_RANK_FACTORS.get(r) == col
                                for r in needed if r in EXTRA_RANK_FACTORS):
            s = _base_factor_series(out, df_all_feat, friday, col)
            out[col] = out["ticker"].map(s).astype(float).fillna(0.0)

    # Cross-sectional percent-ranks (per this Friday's names).
    for rank_col in EXTRA_RANK_FACTORS:
        if rank_col in needed:
            base = EXTRA_RANK_FACTORS[rank_col]
            out[rank_col] = out[base].rank(pct=True).fillna(0.5)

    return out


def sanity_check_extra_factors(friday_obs: pd.DataFrame, tabular_cols,
                               label: str = "") -> bool:
    """Assert the computed extra-factor columns are non-degenerate and PRINT a
    one-line check. Returns True iff every required extra base factor present
    has cross-sectional variance (not all-zero / not constant).

    If the candidate's extra factors are all zero the comparison is invalid.
    """
    present = [c for c in EXTRA_FACTOR_COLS if c in set(tabular_cols)]
    if not present:
        return True  # production model: nothing to check
    n = len(friday_obs)
    stats = []
    ok = True
    for c in EXTRA_BASE_FACTORS:
        if c not in set(tabular_cols) or c not in friday_obs.columns:
            continue
        vals = friday_obs[c].to_numpy(dtype=float)
        std = float(np.nanstd(vals))
        nonzero = int(np.count_nonzero(vals))
        col_ok = (nonzero > 0) and (std > 0) and (n >= 2)
        ok = ok and col_ok
        stats.append(f"{c}:std={std:.4f},nz={nonzero}/{n}")
    flag = "OK" if ok else "DEGENERATE(all-zero/constant!)"
    print(f"    [factor-sanity{(' '+label) if label else ''}] {flag} | " + " | ".join(stats))
    return ok
