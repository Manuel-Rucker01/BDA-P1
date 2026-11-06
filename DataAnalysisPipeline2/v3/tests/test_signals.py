"""Round-4 signal blocks: SUE arithmetic and PIT, residual momentum vs a direct
regression, point-in-time behaviour, joint hot-deck imputation."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import dataset as ds
from DataAnalysisPipeline2.v3 import signals as sg
from DataAnalysisPipeline2.v3.tests.synthetic import make_panel


def _asof(cik=1, n=12):
    ends = pd.date_range("2020-03-31", periods=n, freq="QE")
    ttm = 100 + np.cumsum(np.r_[0, [5, -3, 8, 2, 6, -4, 9, 1, 7, 3, -2][: n - 1]])
    rows = []
    for k, (e, t) in enumerate(zip(ends, ttm)):
        rows.append({"cik": cik, "asof_date": e + pd.Timedelta(days=40), "last_filed": e + pd.Timedelta(days=39),
                     "net_income_end": e, "net_income_ttm": float(t), "net_income_ttm_prev": float(t) - 10,
                     "revenue_ttm": 1000.0 + k, "revenue_ttm_prev": 990.0, "gross_profit_ttm": 400.0 + k,
                     "gross_profit_ttm_prev": 390.0, "cfo_ttm": 120.0, "cfo_ttm_prev": 110.0,
                     "assets_end": e, "assets": 2000.0 * (1.02 ** k), "liabilities": 1000.0, "equity": 1000.0,
                     "shares": 1e2, "wdil_yoy": 0.0})
    return pd.DataFrame(rows), ttm


def test_sue_is_scaled_seasonal_change_and_known_at_filing():
    fa, ttm = _asof()
    ev = sg.sue_events(fa)
    d = np.diff(ttm)                                  # TTM(E_k) - TTM(E_{k-1}) = latest quarter's YoY change
    k = 6
    hist = d[max(0, k - 1 - 7): k]
    assert np.isclose(ev["sue"].iloc[k], d[k - 1] / hist.std(ddof=1))
    assert ev["sue"].iloc[:4].isna().all()            # needs >= 4 changes
    assert (ev["report_date"] == ev["asof_date"]).all()
    # asset growth vs the period end ~1 year earlier
    assert np.isclose(ev["asset_growth"].iloc[8], 1.02 ** 4 - 1)
    assert np.isnan(ev["asset_growth"].iloc[2])


def test_extra_events_joined_pit_and_imputed_jointly():
    panel, meta = make_panel(n_per_sector=3, n_sessions=330, start="2023-01-02")
    fa, _ = _asof(n=12)
    fa["asof_date"] = pd.date_range("2023-02-01", periods=12, freq="MS")      # inside the panel
    cik_map = pd.DataFrame({"ticker": ["S0_0"], "cik": [1]})
    block = ds.FundamentalsBlock(fa, cik_map, extra_events=sg.sue_events(fa))
    assert set(sg.SUE_COLUMNS) <= set(block.columns)
    f = ds.compute_v3_features(panel, meta, blocks=[block])
    s = f[f["ticker"] == "S0_0"].set_index("Date")
    assert s.loc[s.index < "2023-02-01", "sue"].isna().all()
    x = f[f["Date"] == f["Date"].max()].set_index("ticker")
    assert np.isclose(x.loc["S0_0", "days_since_report"],
                      (pd.Timestamp(f["Date"].max()) - fa["asof_date"].max()).days)
    imp = x[x["eligible"] & x["fund_imputed"]]
    cols = [c for c in block.columns]
    assert len(imp) and np.allclose(imp[cols].to_numpy(float), x.loc[["S0_0"] * len(imp), cols].to_numpy(float),
                                    equal_nan=True)                       # one donor -> identical vectors


def _long(n_sessions=600, seed=1):
    panel, meta = make_panel(n_per_sector=5, n_sessions=n_sessions, seed=seed, start="2022-01-03")
    f = ds.compute_v3_features(panel, meta)
    return f, ds.session_calendar(f["Date"])


def test_residual_momentum_matches_direct_regression():
    f, cal = _long()
    rm = sg.residual_momentum(f, cal)
    L = f.assign(y=f["daily_return"].clip(*sg.RET_CLIP))
    el = L[L["eligible"] & L["y"].notna()]
    mkt = el.groupby("Date")["y"].mean()
    sec = el.groupby(["Sector", "Date"])["y"].mean()
    t = "S1_2"
    g = L[L["ticker"] == t].reset_index()
    i = len(g) - 1
    w = g.iloc[i - sg.REG_WINDOW + 1: i + 1]
    w = w[w["y"].notna() & w["Date"].isin(mkt.index)]
    X = np.column_stack([np.ones(len(w)), w["Date"].map(mkt), [sec.get((s, d), mkt[d]) for s, d in zip(w["Sector"], w["Date"])]])
    beta, *_ = np.linalg.lstsq(X, w["y"].to_numpy(), rcond=None)
    resid = w["y"].to_numpy() - X @ beta
    sig = resid.std(ddof=0)
    sub = g.iloc[i - 251: i - 20]
    sub = sub[sub["y"].notna() & sub["Date"].isin(mkt.index)]
    Xs = np.column_stack([sub["Date"].map(mkt), [sec.get((s, d), mkt[d]) for s, d in zip(sub["Sector"], sub["Date"])]])
    e = sub["y"].to_numpy() - Xs @ beta[1:]
    expect = e.sum() / (sig * np.sqrt(len(sub)))
    got = rm.loc[g.loc[i, "index"], "resmom_12_1"]
    assert np.isclose(got, expect, rtol=1e-6)
    assert np.isclose(rm.loc[g.loc[i, "index"], "idio_vol"], sig * np.sqrt(252), rtol=1e-6)


def test_residual_momentum_point_in_time():
    f, cal = _long()
    cut = cal[500]
    a = sg.residual_momentum(f, cal)
    ft = f[f["Date"] <= cut]
    b = sg.residual_momentum(ft, ds.session_calendar(ft["Date"]))
    cols = sg.RESMOM_COLUMNS + ["beta_mkt", "idio_vol"]
    pd.testing.assert_frame_equal(a.loc[ft.index, cols], b[cols], check_exact=False, atol=1e-9)
    assert a["resmom_12_1"].notna().sum() > 0


def test_lvq_composite_ranks():
    df = pd.DataFrame({"Date": "d", "eligible": [True, True, True, False], "beta_mkt": [0.5, 1.0, 1.5, 0.1],
                       "idio_vol": [0.1, 0.2, 0.3, 0.0], "gpa": [0.3, 0.2, 0.1, 1.0]})
    c = sg.lvq_composite(df)
    assert c.iloc[0] > c.iloc[1] > c.iloc[2] and np.isnan(c.iloc[3])
