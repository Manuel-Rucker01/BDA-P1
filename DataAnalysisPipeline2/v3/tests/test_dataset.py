"""v3 dataset: PIT features, labels (incl. delistings), sectors, chunked == live."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.features import core
from DataAnalysisPipeline2.v3 import dataset as ds
from DataAnalysisPipeline2.v3.tests.synthetic import make_panel

FEATS = ds.FEATURE_COLUMNS


@pytest.fixture(scope="module")
def synth():
    return make_panel()


@pytest.fixture(scope="module")
def full(synth):
    panel, meta = synth
    return ds.compute_v3_features(panel, meta, labels=True)


# ------------------------------------------------------------------ labels
def _toy():
    cal = pd.bdate_range("2025-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    rows = [{"ticker": "A", "Date": d, "company_close": 10.0 + i} for i, d in enumerate(cal)]
    rows += [{"ticker": "D", "Date": d, "company_close": 10.0 * (0.5 if i == 3 else 1.0)}
             for i, d in enumerate(cal[:4])]                 # delisted after session 3
    rows += [{"ticker": "H", "Date": d, "company_close": 10.0 + i}
             for i, d in enumerate(cal) if i not in (4, 5)]   # halted on sessions 4-5
    return pd.DataFrame(rows), np.array(cal)


def test_forward_returns_basic_delisting_halt_and_end():
    df, cal = _toy()
    fr = ds.forward_returns(df, cal, horizons=(2,))["fwd_ret_2"]
    a = fr[df["ticker"] == "A"].to_numpy()
    assert np.allclose(a[:8], [(12 + i) / (10 + i) - 1 for i in range(8)])
    assert np.isnan(a[8:]).all()                      # t+2 beyond the data
    d = fr[df["ticker"] == "D"].to_numpy()
    # D's last close (session 3) is 5.0: sessions 1, 2, 3 -> return to last close
    assert np.isclose(d[1], 5.0 / 10.0 - 1) and np.isclose(d[2], -0.5) and np.isclose(d[3], 0.0)
    h = fr[df["ticker"] == "H"].reset_index(drop=True)
    # H at session 3 -> t+2 = session 5 (halted) -> last close at/before = session 3
    assert np.isclose(h[3], 0.0)


def test_forward_returns_winsorized():
    cal = np.array(pd.bdate_range("2025-01-01", periods=4).strftime("%Y-%m-%d"))
    df = pd.DataFrame({"ticker": "X", "Date": cal, "company_close": [1.0, 100.0, 0.001, 0.001]})
    fr = ds.forward_returns(df, cal, horizons=(1,))["fwd_ret_1"].to_numpy()
    assert fr[0] == ds.LABEL_CLIP[1] and fr[1] == ds.LABEL_CLIP[0]


def test_labels_ignore_prices_after_horizon(synth):
    panel, meta = synth
    cal = ds.session_calendar(panel["Date"])
    t = cal[100]
    p2 = panel.copy()
    late = p2["Date"] > cal[100 + 30]
    p2.loc[late, "company_close"] *= 7.0
    at = (panel["Date"] == t).to_numpy()
    a = ds.forward_returns(panel, cal)[at]
    b = ds.forward_returns(p2, cal)[at]
    pd.testing.assert_frame_equal(a, b)
    # ... but the 30-session label does use session t+30
    p3 = panel.copy()
    p3.loc[p3["Date"] == cal[130], "company_close"] *= 2.0
    c = ds.forward_returns(p3, cal)[at]
    assert not np.allclose(a["fwd_ret_30"], c["fwd_ret_30"])


# ------------------------------------------------------------- features PIT
def test_features_point_in_time(synth, full):
    panel, meta = synth
    cal = ds.session_calendar(panel["Date"])
    cut = cal[250]
    trunc = ds.compute_v3_features(panel[panel["Date"] <= cut], meta)
    cols = FEATS + ["eligible", "Sector"]
    a = full[full["Date"] <= cut].set_index(["ticker", "Date"])[cols].sort_index()
    b = trunc.set_index(["ticker", "Date"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10)


def test_eligibility_rule(full):
    last = full[full["Date"] == full["Date"].max()].set_index("ticker")
    assert not last.loc["PNY", "eligible"]           # raw close < $3
    assert not last.loc["NEW", "eligible"]           # < 120 bars
    assert not last.loc["ILQ", "eligible"]           # ADV < $2M
    assert last.loc["S0_0", "eligible"] and last.loc["UNK", "eligible"]
    assert "DEL" not in last.index
    first = full.groupby("ticker")["n_bars"].min()
    assert (first == 1).all()
    assert not full.loc[full["n_bars"] < 120, "eligible"].any()


def test_cross_section_over_eligible_population(full):
    d = full["Date"].max()
    x = full[full["Date"] == d]
    assert x.loc[~x["eligible"], "rank_return_5d"].isna().all()
    el = x[x["eligible"]]
    assert el["rank_return_5d"].max() == 1.0
    assert np.isclose(el["rank_return_5d"].min(), 1.0 / len(el))


def test_proxy_sector_and_unknowns(full):
    late = full[full["Date"] >= full["Date"].max()[:7]]
    unk = late[late["ticker"] == "UNK"]
    assert (unk["Sector"] == "Energy").all() and (unk["sector_source"] == "proxy").all()
    s0 = late[late["ticker"] == "S0_0"]
    assert (s0["sector_source"] == "meta").all() and (s0["Sector"] == "Technology").all()
    # first month: no 120-session window yet -> Unknown
    first = full[full["Date"] <= full["Date"].min()[:7] + "-31"]
    assert (first.loc[first["ticker"] == "UNK", "Sector"] == ds.UNKNOWN).all()


def test_targets_are_ranks_over_eligible(full):
    d = sorted(full["Date"].unique())[200]
    x = full[full["Date"] == d]
    assert x.loc[~x["eligible"], "target_raw_30"].isna().all()
    el = x[x["eligible"]]
    assert el["target_raw_30"].notna().all() and el["target_raw_30"].max() == 1.0
    r = el["fwd_ret_30"].rank(pct=True)
    assert np.allclose(el["target_raw_30"], r)
    dm = el["fwd_ret_30"] - el.groupby("Sector")["fwd_ret_30"].transform("mean")
    assert np.allclose(el["target_sn_30"], dm.rank(pct=True))


def test_weekly_and_monthly_dates():
    cal = pd.bdate_range("2025-01-01", "2025-03-10").strftime("%Y-%m-%d").tolist()
    cal.remove("2025-01-17")                                   # a Friday holiday
    w = ds.weekly_decision_dates(cal, start="2025-01-01")
    assert "2025-01-16" in w and "2025-01-10" in w and w[-1] == "2025-03-10"
    assert ds.first_sessions_of_month(cal) == ["2025-01-01", "2025-02-03", "2025-03-03"]


# ------------------------------------------------- chunked offline == live
def test_chunked_builder_matches_self_contained(tmp_path, synth, full):
    panel, meta = synth
    info = ds.build_dataset(prices=panel, meta=meta, out_dir=str(tmp_path), chunk_size=4, log=lambda *a: None)
    off = ds.load_daily(out_dir=str(tmp_path))
    assert info["feature_version"] == 3
    cols = FEATS + ["fwd_ret_21", "fwd_ret_30", "target_raw_30", "target_sn_21"]
    a = off.set_index(["ticker", "Date"]).sort_index()
    b = full.set_index(["ticker", "Date"]).sort_index()
    pd.testing.assert_frame_equal(a[cols].astype(float), b[cols].astype(float), check_exact=False,
                                  rtol=1e-5, atol=1e-6)
    assert (a["eligible"] == b["eligible"]).all() and (a["Sector"] == b["Sector"]).all()


def test_core_population_argument():
    panel, meta = make_panel(n_per_sector=4, n_sessions=150)
    base = core.compute_features(panel, meta)
    pop = panel["ticker"].str.startswith("S0")
    alt = core.compute_features(panel, meta, population=pop)
    d = alt["Date"].max()
    x = alt[alt["Date"] == d]
    assert set(x.loc[x["eligible"], "ticker"]) == {f"S0_{i}" for i in range(4)}
    assert x.loc[~x["eligible"], "rank_return_5d"].isna().all()
    # time-series features do not depend on the population
    pd.testing.assert_frame_equal(base[core.TS_COLUMNS], alt[core.TS_COLUMNS])
    # a column name works too
    alt2 = core.compute_features(panel.assign(p=pop), meta, population="p")
    pd.testing.assert_series_equal(alt["rank_return_5d"], alt2["rank_return_5d"])


# ------------------------------------------------- fundamentals block / EDGAR sectors
def _edgar_fixture():
    asof = pd.DataFrame([{"cik": c, "asof_date": d, "last_filed": f, "revenue_ttm": 1e9 * c,
                          "revenue_ttm_prev": 8e8, "net_income_ttm": 1e8 * c, "net_income_ttm_prev": 5e7,
                          "gross_profit_ttm": 4e8, "cfo_ttm": 1.2e8, "assets": 2e9, "liabilities": 1e9,
                          "equity": 1e9, "shares": 1e8, "wdil_yoy": 0.01}
                         for c, d, f in [(1, "2024-06-03", "2024-05-31"), (2, "2024-03-01", "2024-02-29")]])
    cik_map = pd.DataFrame({"ticker": ["S0_0", "S1_0"], "cik": [1, 2]})
    sectors = pd.DataFrame({"cik": [1, 2], "ff12": [6, 11], "sector_name": ["Business Equipment", "Finance"]})
    return asof, cik_map, sectors


def test_fundamentals_block_pit_on_filing_date():
    panel, meta = make_panel(n_per_sector=3, n_sessions=200)
    asof, cik_map, _ = _edgar_fixture()
    block = ds.FundamentalsBlock(asof, cik_map, impute=False)
    f = ds.compute_v3_features(panel, meta, blocks=[block])
    s = f[f["ticker"] == "S0_0"].set_index("Date")
    assert s.loc[s.index < "2024-06-03", "ep"].isna().all()          # filed 05-31, usable from 06-03
    after = s[s.index >= "2024-06-03"]
    raw = panel[(panel["ticker"] == "S0_0") & (panel["Date"] == after.index[0])]["close_raw"].iloc[0]
    assert np.isclose(after["ep"].iloc[0], 1e8 / (raw * 1e8))         # E/P on the raw close
    assert f.loc[f["ticker"] == "S2_0", "ep"].isna().all()             # unmapped, no imputation
    assert ds.feature_columns([block])[-len(block.columns):] == block.columns


def test_hot_deck_makes_missingness_uninformative():
    panel, meta = make_panel(n_per_sector=3, n_sessions=200)
    asof, cik_map, _ = _edgar_fixture()
    block = ds.FundamentalsBlock(asof, cik_map)
    f = ds.compute_v3_features(panel, meta, blocks=[block])
    x = f[f["Date"] == f["Date"].max()].set_index("ticker")
    real = ["S0_0", "S1_0"]
    assert not x.loc[real, "fund_imputed"].any()
    el = x[x["eligible"]]
    imp = el[~el.index.isin(real)]
    assert len(imp) and imp["fund_imputed"].all()
    donors = x.loc[real, block.columns]
    for _, row in imp[block.columns].iterrows():   # complete vectors copied from same-date donors
        assert any(np.allclose(row.to_numpy(float), donors.loc[k].to_numpy(float), equal_nan=True) for k in real)
    assert f.loc[f["Date"] < "2024-03-01", "ep"].isna().all()         # no donor yet -> stays NaN
    f2 = ds.compute_v3_features(panel, meta, blocks=[block])
    pd.testing.assert_frame_equal(f[block.columns], f2[block.columns])   # deterministic


def test_edgar_sector_map_resolves_segments_and_drops_none():
    _, cik_map, sectors = _edgar_fixture()
    m = ds.edgar_sector_map(["S0_0", "S1_0~1", "S2_0"], "2024-06-03", cik_map, sectors)
    assert dict(zip(m["ticker"], m["Sector"])) == {"S0_0": "Business Equipment", "S1_0~1": "Finance"}


