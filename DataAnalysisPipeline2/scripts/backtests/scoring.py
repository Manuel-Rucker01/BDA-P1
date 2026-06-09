"""Track E1 — no-retrain cross-sectional scoring transforms.

Pure functions that re-shape a per-date cross-section's model score WITHOUT
retraining the model. Each transform operates only on the CURRENT date's rows
(no look-ahead / no use of any other date) and returns a NEW score that has
been re-ranked cross-sectionally to [0, 1] (percentile rank), so the downstream
``select_top_k`` sort is always consistent regardless of the transform applied.

Transforms
----------
* ``sector_neutral`` — rank the score WITHIN each sector, then blend with the
  global rank. Defends against the diagnostics finding that cross-sectional IC
  is sector-inconsistent (positive in some sectors, negative in others), so a
  global top-K over-allocates to sectors where the signal is wrong.
* ``beta_adjusted`` — penalise high-beta names so the book is not just a
  leveraged market bet.
* ``vol_adjusted`` — penalise extreme-volatility names.

All of these are backtest-only research transforms; they do not touch any live
trading code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from common import signal_column
except Exception:  # pragma: no cover - allow direct import outside the pkg path
    import os
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    from common import signal_column


TRANSFORM_NAMES = ("baseline", "sector_neutral", "beta_adjusted", "vol_adjusted")


def _pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]. Ties averaged; NaNs -> 0.5."""
    s = pd.Series(np.asarray(s, dtype=float))
    r = s.rank(pct=True, method="average")
    # A single-row cross-section ranks to 1.0; map to a neutral 0.5 so it does
    # not dominate. (Degenerate case; rarely hit on a 20-name basket.)
    if len(r) == 1:
        return pd.Series([0.5], index=r.index)
    return r.fillna(0.5)


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; constant / empty input -> all zeros."""
    s = pd.Series(np.asarray(s, dtype=float))
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def sector_neutral_score(
    df: pd.DataFrame,
    score_col: str,
    sector_col: str = "Sector",
    blend: float = 0.0,
    min_sector_size: int = 2,
) -> pd.Series:
    """Sector-neutral re-rank of ``score_col``.

    Rank ``score_col`` WITHIN each sector (percentile), then blend with the
    global percentile rank::

        blended = (1 - blend) * within_sector_rank + blend * global_rank

    ``blend=0`` -> fully sector-neutral; ``blend=1`` -> unchanged global rank.
    Names in tiny (< ``min_sector_size``) or unknown/NaN sectors fall back to
    the global rank. The blended score is re-ranked to [0, 1] and returned with
    ``df``'s index.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    global_rank = _pct_rank(df[score_col])

    if sector_col not in df.columns:
        return _pct_rank(global_rank)

    within = global_rank.copy()
    sectors = df[sector_col]
    for sector, idx in df.groupby(sectors).groups.items():
        idx = list(idx)
        if (
            sector is None
            or (isinstance(sector, float) and np.isnan(sector))
            or str(sector).upper() in {"UNKNOWN", "NAN", ""}
            or len(idx) < min_sector_size
        ):
            # Tiny / unknown sector -> keep the global rank for these names.
            within.loc[idx] = global_rank.loc[idx]
        else:
            within.loc[idx] = _pct_rank(df.loc[idx, score_col]).values

    blend = float(blend)
    blended = (1.0 - blend) * within + blend * global_rank
    return _pct_rank(blended)


def beta_adjusted_score(
    df: pd.DataFrame,
    score_col: str,
    beta_col: str = "kalman_beta",
    lam: float = 0.5,
) -> pd.Series:
    """Beta-penalised re-rank: ``adj = global_rank(score) - lam * zscore(beta)``.

    Penalises high-beta names so the book is not simply a leveraged market bet.
    Falls back to the plain global rank if ``beta_col`` is missing. Returns a
    re-ranked [0, 1] Series on ``df``'s index.
    """
    df = df.reset_index(drop=True)
    global_rank = _pct_rank(df[score_col])
    if beta_col not in df.columns:
        return _pct_rank(global_rank)
    adj = global_rank - float(lam) * _zscore(df[beta_col])
    return _pct_rank(adj)


def vol_adjusted_score(
    df: pd.DataFrame,
    score_col: str,
    vol_col: str = "return_volatility_20d",
    lam: float = 0.5,
) -> pd.Series:
    """Volatility-penalised re-rank: ``adj = global_rank(score) - lam * zscore(vol)``.

    Penalises extreme-volatility names. Falls back to the plain global rank if
    ``vol_col`` is missing. Returns a re-ranked [0, 1] Series on ``df``'s index.
    """
    df = df.reset_index(drop=True)
    global_rank = _pct_rank(df[score_col])
    if vol_col not in df.columns:
        return _pct_rank(global_rank)
    adj = global_rank - float(lam) * _zscore(df[vol_col])
    return _pct_rank(adj)


def apply_score_transform(
    df: pd.DataFrame,
    name: str,
    score_col: str | None = None,
    sector_col: str = "Sector",
    beta_col: str = "kalman_beta",
    vol_col: str = "return_volatility_20d",
    blend: float = 0.0,
    lam: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Dispatcher: return a COPY of ``df`` with ``score_col`` replaced by the
    transformed [0, 1] cross-sectional rank (original kept as ``score_col_raw``).

    ``name`` in {baseline, sector_neutral, beta_adjusted, vol_adjusted}.
    ``baseline`` is a passthrough (re-ranked global, so downstream sort is
    identical to the raw ranking). Robust to missing sector/beta/vol columns:
    falls back to the global rank and prints a one-line note.
    """
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"unknown transform '{name}'; expected one of {TRANSFORM_NAMES}")

    out = df.copy().reset_index(drop=True)
    if score_col is None:
        score_col = signal_column(out)

    raw_col = score_col + "_raw"
    if raw_col not in out.columns:
        out[raw_col] = out[score_col].to_numpy()

    if name == "baseline":
        new_score = _pct_rank(out[score_col])
    elif name == "sector_neutral":
        if sector_col not in out.columns and verbose:
            print(f"[scoring] sector_neutral: '{sector_col}' missing -> global-rank fallback.")
        new_score = sector_neutral_score(out, score_col, sector_col=sector_col, blend=blend)
    elif name == "beta_adjusted":
        if beta_col not in out.columns and verbose:
            print(f"[scoring] beta_adjusted: '{beta_col}' missing -> global-rank fallback.")
        new_score = beta_adjusted_score(out, score_col, beta_col=beta_col, lam=lam)
    elif name == "vol_adjusted":
        if vol_col not in out.columns and verbose:
            print(f"[scoring] vol_adjusted: '{vol_col}' missing -> global-rank fallback.")
        new_score = vol_adjusted_score(out, score_col, vol_col=vol_col, lam=lam)

    out[score_col] = np.asarray(new_score, dtype=float)
    return out
