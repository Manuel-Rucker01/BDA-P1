"""Property tests for the shared feature module (feature_version 2)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.features import core

DATES = pd.bdate_range("2025-01-01", periods=140).strftime("%Y-%m-%d").tolist()


def _series(seed, start=50.0, n=len(DATES)):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, 0.02, n)))


def _panel(n_tickers=12, penny=("PNY",), short=("NEW",), stale=("OLD",)):
    rows = []
    for i in range(n_tickers):
        t = f"T{i:02d}"
        px, vol = _series(i), 1e5 * (1 + np.random.default_rng(100 + i).random(len(DATES)))
        rows += [{"ticker": t, "Date": d, "company_close": p, "company_volume": v}
                 for d, p, v in zip(DATES, px, vol)]
    for t in penny:   # always below $1
        rows += [{"ticker": t, "Date": d, "company_close": p, "company_volume": 1e6}
                 for d, p in zip(DATES, _series(7, start=0.4))]
    for t in short:   # listed recently: fewer than MIN_HISTORY bars
        rows += [{"ticker": t, "Date": d, "company_close": p, "company_volume": 1e5}
                 for d, p in zip(DATES[-30:], _series(8, n=30))]
    for t in stale:   # last bar 20 sessions before the end
        rows += [{"ticker": t, "Date": d, "company_close": p, "company_volume": 1e5}
                 for d, p in zip(DATES[:-20], _series(9, n=len(DATES) - 20))]
    return pd.DataFrame(rows)


META = pd.DataFrame({"ticker": [f"T{i:02d}" for i in range(12)],
                     "Sector": ["Technology"] * 6 + ["Finance"] * 6})


def test_point_in_time_no_lookahead():
    panel = _panel()
    cut = DATES[110]
    full = core.compute_features(panel, META)
    trunc = core.compute_features(panel[panel["Date"] <= cut], META)
    a = full[full["Date"] <= cut].set_index(["ticker", "Date"])[core.FEATURE_COLUMNS].sort_index()
    b = trunc.set_index(["ticker", "Date"])[core.FEATURE_COLUMNS].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_scale_invariance():
    panel = _panel()
    scaled = panel.copy()
    m = scaled["ticker"] == "T03"
    scaled.loc[m, "company_close"] *= 10.0  # e.g. a different split-adjustment basis
    f1 = core.compute_features(panel, META).set_index(["ticker", "Date"])
    f2 = core.compute_features(scaled, META).set_index(["ticker", "Date"])
    cols = [c for c in core.FEATURE_COLUMNS if c != "log_dollar_volume_20d"]  # level by design
    pd.testing.assert_frame_equal(f1.loc["T03", cols], f2.loc["T03", cols], atol=1e-9)


def test_eligibility_and_cross_section():
    f = core.compute_features(_panel(), META)
    last = f[f["Date"] == DATES[-1]].set_index("ticker")
    assert not last.loc["PNY", "eligible"]                 # penny stock
    assert not last.loc["NEW", "eligible"]                 # not enough history
    assert last.loc["T00", "eligible"]
    assert np.isnan(last.loc["PNY", "rank_return_5d"])     # excluded from ranks
    el = last[last["eligible"]]
    assert el["rank_return_5d"].between(0, 1).all()
    assert el["rank_return_5d"].max() == 1.0               # ranks among eligible only
    tech = el[el["Sector"] == "Technology"]
    assert np.allclose(tech["sector_daily_return"], tech["daily_return"].mean())


def test_decision_rows_drop_stale_and_ineligible():
    rows = core.decision_rows(core.compute_features(_panel(), META))
    assert set(rows["ticker"]) == {f"T{i:02d}" for i in range(12)}  # no PNY, NEW, OLD
    assert (rows["Date"] == DATES[-1]).all()


def test_robust_zscore():
    df = pd.DataFrame({"Date": ["d1"] * 100 + ["d2"] * 100,
                       "const": [1.08] * 200,                 # e.g. an FX rate
                       "x": list(range(100)) + list(range(100))})
    df.loc[0, "x"] = 1e6                                      # outlier on d1
    z = core.cross_sectional_zscore(df, ["const", "x"])
    assert (z["const"] == 0).all()                            # no float noise
    assert z["x"].abs().max() <= 6.0
    d1, d2 = z.iloc[:100]["x"], z.iloc[100:]["x"]
    assert abs(d1.iloc[1:].mean() - d2.iloc[1:].mean()) < 0.1  # outlier didn't drag the rest
    assert np.isclose(d2.mean(), 0, atol=1e-6) and np.isclose(d2.std(ddof=0), 1, atol=1e-3)


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        core.compute_features(pd.DataFrame({"ticker": ["A"], "Date": ["2025-01-01"]}))
