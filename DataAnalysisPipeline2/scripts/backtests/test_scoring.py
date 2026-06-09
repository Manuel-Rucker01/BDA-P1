"""Unit tests for the Track E1 scoring transforms (no network, no model)."""

import numpy as np
import pandas as pd

from scoring import (
    apply_score_transform,
    beta_adjusted_score,
    sector_neutral_score,
    vol_adjusted_score,
)


def _valid_rank(s):
    arr = np.asarray(s, dtype=float)
    return np.all(arr >= -1e-9) and np.all(arr <= 1.0 + 1e-9)


def _frame():
    # Two sectors. Tech holds the three HIGHEST global scores; Energy holds the
    # three LOWEST. E_best is the best WITHIN Energy but has a low GLOBAL score,
    # so under full sector-neutrality (blend=0) it must rank near the top, while
    # under pure global ranking (blend=1) it sits near the bottom.
    return pd.DataFrame({
        "ticker": ["T_hi", "T_mid", "T_low", "E_best", "E_lo", "E_mid"],
        "pred_proba": [0.95, 0.75, 0.60, 0.40, 0.10, 0.25],
        "Sector": ["Tech", "Tech", "Tech", "Energy", "Energy", "Energy"],
        "kalman_beta": [2.0, 1.0, 0.5, 1.5, 0.8, 1.1],
        "return_volatility_20d": [0.05, 0.02, 0.01, 0.08, 0.015, 0.03],
    })


def test_outputs_are_valid_ranks():
    df = _frame()
    for name in ("baseline", "sector_neutral", "beta_adjusted", "vol_adjusted"):
        out = apply_score_transform(df, name, score_col="pred_proba", verbose=False)
        assert _valid_rank(out["pred_proba"]), f"{name} produced out-of-range ranks"
        assert "pred_proba_raw" in out.columns
        # raw preserved
        assert np.allclose(out["pred_proba_raw"], df["pred_proba"])


def test_baseline_is_identity_ordering():
    df = _frame()
    out = apply_score_transform(df, "baseline", score_col="pred_proba", verbose=False)
    # Re-ranked baseline must preserve the original ordering of names.
    order_raw = df.sort_values("pred_proba")["ticker"].tolist()
    order_new = out.sort_values("pred_proba")["ticker"].tolist()
    assert order_raw == order_new


def test_sector_neutral_blend0_internal_ranking():
    df = _frame()
    s = sector_neutral_score(df, "pred_proba", sector_col="Sector", blend=0.0)
    out = df.assign(score=np.asarray(s))
    sc = out.set_index("ticker")["score"]
    # E_best is the best within Energy (low global score); under full sector-
    # neutrality it must outrank the WORSE Tech names.
    assert sc["E_best"] > sc["T_low"]
    # The top-of-sector name from each sector should land near the top.
    top2 = out.sort_values("score", ascending=False)["ticker"].head(2).tolist()
    assert "T_hi" in top2 and "E_best" in top2


def test_sector_neutral_blend_moves_toward_global():
    df = _frame()
    s0 = sector_neutral_score(df, "pred_proba", blend=0.0)
    s1 = sector_neutral_score(df, "pred_proba", blend=1.0)
    out = df.assign(s0=np.asarray(s0), s1=np.asarray(s1)).set_index("ticker")
    # blend=1 -> pure global rank: E_best (low global score) sits at/below midpoint.
    assert out.loc["E_best", "s1"] <= 0.5
    # blend=0 -> E_best promoted (best in its sector) vs blend=1.
    assert out.loc["E_best", "s0"] > out.loc["E_best", "s1"]


def test_beta_adjusted_demotes_highest_beta():
    df = _frame()
    base = apply_score_transform(df, "baseline", score_col="pred_proba", verbose=False)
    adj = beta_adjusted_score(df, "pred_proba", beta_col="kalman_beta", lam=0.8)
    b = base.assign(adj=np.asarray(adj)).set_index("ticker")
    # T_hi has the highest beta (2.0); its rank should drop vs baseline.
    assert b.loc["T_hi", "adj"] < b.loc["T_hi", "pred_proba"]


def test_vol_adjusted_demotes_highest_vol():
    df = _frame()
    base = apply_score_transform(df, "baseline", score_col="pred_proba", verbose=False)
    adj = vol_adjusted_score(df, "pred_proba", vol_col="return_volatility_20d", lam=0.8)
    b = base.assign(adj=np.asarray(adj)).set_index("ticker")
    # E_best has the highest vol (0.08); its rank should drop vs baseline.
    assert b.loc["E_best", "adj"] < b.loc["E_best", "pred_proba"]


def test_robust_to_missing_columns():
    df = _frame().drop(columns=["Sector", "kalman_beta", "return_volatility_20d"])
    base = apply_score_transform(df, "baseline", score_col="pred_proba", verbose=False)
    for name in ("sector_neutral", "beta_adjusted", "vol_adjusted"):
        out = apply_score_transform(df, name, score_col="pred_proba", verbose=False)
        assert _valid_rank(out["pred_proba"])
        # With the keyed column absent, all three fall back to the global rank.
        assert np.allclose(
            np.asarray(out["pred_proba"]), np.asarray(base["pred_proba"])
        )
