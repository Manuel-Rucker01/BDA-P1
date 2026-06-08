"""Unit tests for the benchmark active-overlay portfolio constructor.

No network, no model: all fixtures are small synthetic prediction frames.
Tests assert the construction invariants documented in ``overlay.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from overlay import (  # noqa: E402
    benchmark_overlay_weights,
    exposure_metrics,
    overlay_strategy_grid,
)

EPS = 1e-6


def make_predictions(n=12, with_vol=True, with_sector=True, equal_scores=False):
    """Synthetic cross-section ranked by descending pred_rank."""
    tickers = [f"T{i:02d}" for i in range(n)]
    if equal_scores:
        scores = np.full(n, 0.5)
    else:
        # Decreasing scores in [~0.05, ~0.99] so the top names are unambiguous.
        scores = np.linspace(0.99, 0.05, n)
    data = {"ticker": tickers, "pred_rank": scores}
    if with_vol:
        # Varying vols so inverse-vol does something non-trivial.
        data["return_volatility_20d"] = np.linspace(0.10, 0.40, n)
    if with_sector:
        data["Sector"] = [["Tech", "Energy", "Health"][i % 3] for i in range(n)]
    return pd.DataFrame(data)


# --- Invariant 1: weights sum to target_exposure * (bf + af). -------------

def test_weights_sum_full_exposure():
    df = make_predictions()
    w = benchmark_overlay_weights(df, benchmark_frac=0.80, active_frac=0.20)
    assert abs(sum(w.values()) - 1.0) < EPS


def test_weights_sum_partial_exposure():
    df = make_predictions()
    w = benchmark_overlay_weights(df, benchmark_frac=0.90, active_frac=0.10)
    assert abs(sum(w.values()) - 1.0) < EPS

    # 0.9/0.1 fractions but with explicit target_exposure scaling.
    w2 = benchmark_overlay_weights(
        df, benchmark_frac=0.6, active_frac=0.3, target_exposure=1.0
    )
    assert abs(sum(w2.values()) - 0.9) < EPS


def test_target_exposure_scaling():
    df = make_predictions()
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.8, active_frac=0.2, target_exposure=2.0
    )
    assert abs(sum(w.values()) - 2.0) < EPS
    assert all(v >= 0 for v in w.values())


# --- Invariant 2: no single name exceeds max_total_weight. ----------------

def test_max_total_weight_cap():
    df = make_predictions(n=8)
    cap = 0.20
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.5, active_frac=0.5, max_total_weight=cap, top_k=3
    )
    assert all(v <= cap + EPS for v in w.values())
    assert abs(sum(w.values()) - 1.0) < EPS


# --- Invariant 3: active sleeve total and per-name active cap. -------------

def test_active_sleeve_total_and_cap():
    df = make_predictions(n=12)
    benchmark_frac, active_frac = 0.70, 0.30
    max_active = 0.10
    w_with = benchmark_overlay_weights(
        df,
        benchmark_frac=benchmark_frac,
        active_frac=active_frac,
        top_k=6,
        pct_threshold=100.0,  # gate the whole universe so top_k governs
        max_active_weight=max_active,
        max_total_weight=1.0,  # don't let the final cap mask the active cap
    )
    # Active contribution per name = combined - benchmark-only weight.
    n = len(df)
    bench_each = benchmark_frac / n
    active_contrib = {t: w_with[t] - bench_each for t in w_with}
    # Each active name's sleeve weight <= max_active (+eps).
    assert all(v <= max_active + 1e-6 for v in active_contrib.values())
    # Active sleeve sums to active_frac.
    assert abs(sum(active_contrib.values()) - active_frac) < 1e-6


def test_active_sleeve_total_no_cap():
    df = make_predictions(n=12)
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.8, active_frac=0.2, top_k=5, pct_threshold=100.0,
        max_total_weight=1.0,
    )
    n = len(df)
    bench_each = 0.8 / n
    active_total = sum(max(0.0, v - bench_each) for v in w.values())
    assert abs(active_total - 0.2) < 1e-6


# --- Invariant 4: long-only. ----------------------------------------------

def test_long_only():
    df = make_predictions()
    w = benchmark_overlay_weights(df, benchmark_frac=0.8, active_frac=0.2)
    assert all(v >= 0 for v in w.values())


# --- Invariant 5: tilt direction (top names get more total weight). -------

def test_tilt_direction():
    df = make_predictions(n=12, with_vol=False)  # equal-weight active sleeve
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.7, active_frac=0.3, top_k=4, pct_threshold=100.0,
        max_total_weight=1.0,
    )
    top_names = df.sort_values("pred_rank", ascending=False)["ticker"].head(4).tolist()
    bottom_names = df.sort_values("pred_rank", ascending=False)["ticker"].tail(4).tolist()
    top_w = sum(w[t] for t in top_names)
    bottom_w = sum(w[t] for t in bottom_names)
    assert top_w > bottom_w


# --- Invariant 6: turnover via exposure_metrics (hand-checked). -----------

def test_turnover_known_value():
    prev = {"A": 0.5, "B": 0.5}
    curr = {"A": 0.7, "C": 0.3}
    m = exposure_metrics(curr, previous_weights=prev, transaction_cost_bps=10.0)
    # |0.7-0.5| + |0.0-0.5| + |0.3-0.0| = 0.2 + 0.5 + 0.3 = 1.0
    assert abs(m["turnover"] - 1.0) < EPS
    assert abs(m["gross_exposure"] - 1.0) < EPS
    assert abs(m["net_exposure"] - 1.0) < EPS
    # cost = turnover * bps/10000 = 1.0 * 0.001 = 0.001
    assert abs(m["transaction_cost"] - 0.001) < EPS


# --- Invariant 7: edge cases. ---------------------------------------------

def test_empty_df():
    assert benchmark_overlay_weights(pd.DataFrame()) == {}
    empty = pd.DataFrame(columns=["ticker", "pred_rank"])
    assert benchmark_overlay_weights(empty) == {}


def test_single_name():
    df = pd.DataFrame({"ticker": ["ONLY"], "pred_rank": [0.9]})
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.8, active_frac=0.2, target_exposure=1.0,
        max_total_weight=1.0,
    )
    assert set(w.keys()) == {"ONLY"}
    assert abs(w["ONLY"] - 1.0) < EPS


def test_equal_scores():
    df = make_predictions(equal_scores=True)
    w = benchmark_overlay_weights(df, benchmark_frac=0.8, active_frac=0.2, top_k=4)
    assert abs(sum(w.values()) - 1.0) < EPS
    assert all(v >= 0 for v in w.values())


def test_missing_vol_equal_weight_active():
    df = make_predictions(with_vol=False)
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.7, active_frac=0.3, top_k=3, pct_threshold=100.0,
        max_total_weight=1.0,
    )
    assert abs(sum(w.values()) - 1.0) < EPS
    # Equal-weight active sleeve => the 3 selected top names share active mass
    # equally; their active contributions should be ~equal.
    n = len(df)
    bench_each = 0.7 / n
    top3 = df.sort_values("pred_rank", ascending=False)["ticker"].head(3).tolist()
    contribs = [w[t] - bench_each for t in top3]
    assert max(contribs) - min(contribs) < 1e-6


def test_missing_sector_no_cap():
    df = make_predictions(with_sector=False)
    w = benchmark_overlay_weights(
        df, benchmark_frac=0.8, active_frac=0.2, top_k=4, max_sector_frac=0.5
    )
    assert abs(sum(w.values()) - 1.0) < EPS


# --- Invariant 8: sector active cap limits names from one sector. ---------

def test_sector_active_cap():
    # 9 names; top of the ranking is dominated by 'Tech'. With a sector cap,
    # the active sleeve must not load up only on Tech names.
    tickers = [f"S{i}" for i in range(9)]
    scores = np.linspace(0.99, 0.10, 9)
    # First 5 (highest scored) are Tech; rest spread out.
    sectors = ["Tech", "Tech", "Tech", "Tech", "Tech", "Energy", "Health", "Energy", "Health"]
    vols = np.linspace(0.1, 0.3, 9)
    df = pd.DataFrame(
        {"ticker": tickers, "pred_rank": scores, "Sector": sectors,
         "return_volatility_20d": vols}
    )
    top_k = 4
    max_sector_frac = 0.5  # max_per_sector = round(4*0.5) = 2 Tech names
    w = benchmark_overlay_weights(
        df,
        benchmark_frac=0.7,
        active_frac=0.3,
        top_k=top_k,
        pct_threshold=100.0,
        max_sector_frac=max_sector_frac,
        max_total_weight=1.0,
    )
    n = len(df)
    bench_each = 0.7 / n
    # Names with active contribution above benchmark are the selected ones.
    active_names = [t for t in w if w[t] - bench_each > 1e-9]
    sector_of = dict(zip(tickers, sectors))
    tech_active = [t for t in active_names if sector_of[t] == "Tech"]
    max_per_sector = max(1, round(top_k * max_sector_frac))
    assert len(tech_active) <= max_per_sector


# --- Strategy grid contract. ----------------------------------------------

def test_strategy_grid():
    grid = overlay_strategy_grid()
    assert set(grid) == {
        "benchmark_overlay_90_10",
        "benchmark_overlay_80_20",
        "benchmark_overlay_70_30",
    }
    assert grid["benchmark_overlay_80_20"] == {"benchmark_frac": 0.80, "active_frac": 0.20}
    # Each config must produce valid weights summing to 1.0.
    df = make_predictions()
    for name, kwargs in grid.items():
        w = benchmark_overlay_weights(df, **kwargs, top_k=5)
        assert abs(sum(w.values()) - 1.0) < EPS, name
        assert all(v >= 0 for v in w.values()), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
