"""Benchmark-aware active-overlay (enhanced-indexing) portfolio construction.

This module builds a long-only portfolio as the sum of two sleeves:

    portfolio = passive benchmark sleeve  +  active alpha-tilt sleeve

* The **benchmark sleeve** is an equal-weight holding of the full investable
  universe present in ``predictions_df`` (every ticker with a finite score).
  It receives ``benchmark_frac`` of the total exposure.

* The **active sleeve** selects the top names via the canonical
  ``select_top_k`` gate and sizes them with inverse-volatility weighting via
  ``inverse_volatility_weights_from_frame``.  It receives ``active_frac`` of
  the total exposure and expresses the alpha tilt on top of the index.

The combined weights sum to ``target_exposure * (benchmark_frac + active_frac)``
(i.e. ``target_exposure`` when the two fractions sum to 1.0), respect a hard
per-name cap ``max_total_weight`` (enforced via water-filling redistribution),
and are long-only.

All selection / sizing / accounting reuses the canonical helpers in
``common`` so the overlay path follows the same semantics as the rest of the
backtest stack.  No model or market data is loaded here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Mirror common.py: make ``trading_agent`` / ``common`` importable whether this
# module is imported as a package member or run as a loose script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from common import (  # noqa: E402  (path setup must precede import)
    exposure_metrics,
    inverse_volatility_weights_from_frame,
    select_top_k,
    signal_column,
)

__all__ = [
    "benchmark_overlay_weights",
    "overlay_strategy_grid",
    "exposure_metrics",
]


def _water_fill_cap(weights: dict[str, float], cap: float, total: float) -> dict[str, float]:
    """Cap each weight at ``cap`` and redistribute the excess to uncapped names.

    Mirrors the cap loop inside ``inverse_volatility_weights_from_frame``:
    iteratively clamp names exceeding ``cap`` and proportionally re-spread the
    freed mass over the remaining (uncapped) names so the dict keeps summing to
    ``total``.  If the cap is too tight to hold ``total`` (cap * N < total) the
    result saturates at ``cap`` for every name.
    """
    if not weights or total <= 0:
        return {k: 0.0 for k in weights}
    names = list(weights.keys())
    w = np.array([max(0.0, float(weights[k])) for k in names], dtype=float)
    s = w.sum()
    if s <= 0:
        return {k: 0.0 for k in names}
    w = w / s * total  # normalise to the target total before capping

    cap = float(cap)
    capped = np.zeros(len(w), dtype=bool)
    # A few passes are enough; redistribution can push a previously-fine name
    # over the cap, so we re-check until stable or iteration budget exhausted.
    for _ in range(len(w) + 4):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        w[over] = cap
        capped |= over
        free = ~capped
        remaining = total - float(w[capped].sum())
        if not free.any() or remaining <= 0:
            w[free] = 0.0
            break
        free_sum = float(w[free].sum())
        if free_sum <= 0:
            # Spread evenly if the free names carried no mass.
            w[free] = remaining / int(free.sum())
        else:
            w[free] = w[free] / free_sum * remaining
    return {k: float(v) for k, v in zip(names, w)}


def benchmark_overlay_weights(
    predictions_df: pd.DataFrame,
    *,
    benchmark_frac: float = 0.80,
    active_frac: float = 0.20,
    top_k: int | None = None,
    pct_threshold: float | None = None,
    max_sector_frac: float | None = None,
    max_total_weight: float = 0.25,
    max_active_weight: float | None = None,
    target_exposure: float = 1.0,
    vol_col: str = "return_volatility_20d",
    sector_col: str = "Sector",
    signal_col: str | None = None,
) -> dict[str, float]:
    """Construct active-overlay (enhanced-indexing) long-only portfolio weights.

    Returns a ``{ticker: weight}`` mapping whose values sum to
    ``target_exposure * (benchmark_frac + active_frac)`` (== ``target_exposure``
    when the fractions sum to 1.0).  See the module docstring for semantics.

    Parameters mirror the production selection / sizing knobs; ``top_k``,
    ``pct_threshold``, ``max_sector_frac``, ``signal_col`` and ``sector_col``
    are forwarded to ``select_top_k`` for the active sleeve.
    """
    if predictions_df is None or len(predictions_df) == 0:
        return {}

    benchmark_frac = float(benchmark_frac)
    active_frac = float(active_frac)
    target_exposure = float(target_exposure)
    total_target = target_exposure * (benchmark_frac + active_frac)
    if total_target <= 0:
        return {}

    # --- Investable universe: every ticker with a finite score. ----------
    score_col = signal_column(predictions_df, signal_col)
    universe = predictions_df[
        predictions_df[score_col].apply(lambda v: np.isfinite(v))
    ].copy()
    # De-duplicate on ticker (keep the first / best-ranked occurrence) so the
    # equal-weight benchmark counts each name once.
    universe = universe.drop_duplicates(subset="ticker", keep="first")
    if universe.empty:
        return {}

    bench_tickers = universe["ticker"].tolist()
    n = len(bench_tickers)

    # --- (1) Passive benchmark sleeve: equal weight over the universe. ----
    bench_each = (benchmark_frac * target_exposure) / n if benchmark_frac > 0 else 0.0
    combined: dict[str, float] = {t: bench_each for t in bench_tickers}

    # --- (2) Active alpha-tilt sleeve. ------------------------------------
    if active_frac > 0:
        selected = select_top_k(
            universe,
            pct_threshold=pct_threshold,
            top_k=top_k,
            max_sector_frac=max_sector_frac,
            signal_col=signal_col,
            sector_col=sector_col,
        )
        if selected is not None and not selected.empty:
            # Intra-sleeve weights sum to 1.0 (inverse-vol, equal-weight
            # fallback when the vol column is absent).
            intra = inverse_volatility_weights_from_frame(
                selected, target_exposure=1.0, vol_col=vol_col
            )
            active_total = active_frac * target_exposure
            active = {t: w * active_total for t, w in intra.items()}

            # Optional per-name cap inside the active sleeve, renormalised to
            # keep the sleeve summing to ``active_total`` (water-filling).
            if max_active_weight is not None:
                active = _water_fill_cap(
                    active, max_active_weight * target_exposure, active_total
                )

            for t, w in active.items():
                combined[t] = combined.get(t, 0.0) + w

    # --- (4) Hard per-name cap on the COMBINED weights + renormalisation. -
    combined = _water_fill_cap(combined, max_total_weight * target_exposure, total_target)

    # --- (5) Long-only safety: drop any numerical negatives. --------------
    combined = {t: w for t, w in combined.items() if w > 0}
    return combined


def overlay_strategy_grid() -> dict[str, dict]:
    """Canonical overlay strategy configs: ``name -> benchmark_overlay_weights`` kwargs.

    Backtests can iterate this grid and splat each kwargs dict into
    ``benchmark_overlay_weights``.
    """
    return {
        "benchmark_overlay_90_10": {"benchmark_frac": 0.90, "active_frac": 0.10},
        "benchmark_overlay_80_20": {"benchmark_frac": 0.80, "active_frac": 0.20},
        "benchmark_overlay_70_30": {"benchmark_frac": 0.70, "active_frac": 0.30},
    }
