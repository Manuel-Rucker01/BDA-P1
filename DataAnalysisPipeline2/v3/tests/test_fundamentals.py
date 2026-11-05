"""Offline point-in-time tests for v3.fundamentals / v3.edgar (synthetic companyfacts, no network).

Run from DataAnalysisPipeline2/:  python -m pytest v3/tests/test_fundamentals.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # DataAnalysisPipeline2/

from v3 import edgar  # noqa: E402
from v3.fundamentals import (_FlowState, _ord, build_company_asof, fundamental_features,  # noqa: E402
                             next_business_day, resolve_cik)


# --------------------------------------------------------------------------- helpers
def F(end, val, filed, start=None, form="10-Q", accn=None):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn or f"acc-{filed}-{form}"}
    if start:
        d["start"] = start
    return d


def cf(cik=1, gaap=None, dei=None, unit_shares=()):
    facts = {"us-gaap": {}, "dei": {}}
    for concept, rows in (gaap or {}).items():
        unit = "shares" if concept in unit_shares else "USD"
        facts["us-gaap"][concept] = {"units": {unit: rows}}
    for concept, rows in (dei or {}).items():
        facts["dei"][concept] = {"units": {"shares": rows}}
    return {"cik": cik, "entityName": "TEST", "facts": facts}


def asof(df, d):
    """Row of the event table as known on date d (or None)."""
    d = pd.Timestamp(d)
    sub = df[df["asof_date"] <= d]
    return None if sub.empty else sub.iloc[-1]


# Revenue: calendar FY. 2022 quarters 90/100/105/105 (FY 400), 2023 quarters 110/120/130/140 (FY 500).
# 2022-Q2/Q3 and 2023-Q2/Q3 are reported ONLY as YTD (6M / 9M); Q4 is only implicit in the FY.
REV = [
    F("2022-03-31", 90, "2022-05-04", "2022-01-01"),
    F("2022-06-30", 190, "2022-08-03", "2022-01-01"),
    F("2022-09-30", 295, "2022-11-02", "2022-01-01"),
    F("2022-12-31", 400, "2023-02-15", "2022-01-01", form="10-K"),     # Wed -> avail Thu 02-16
    F("2023-03-31", 110, "2023-05-05", "2023-01-01"),                  # Fri -> avail Mon 05-08
    F("2022-03-31", 90, "2023-05-05", "2022-01-01"),                   # comparative
    F("2023-06-30", 230, "2023-08-02", "2023-01-01"),
    F("2023-09-30", 360, "2023-11-01", "2023-01-01"),
    F("2023-12-31", 500, "2024-02-14", "2023-01-01", form="10-K"),
]


# --------------------------------------------------------------------------- calendar
def test_next_business_day():
    assert next_business_day("2023-02-15") == pd.Timestamp("2023-02-16")   # Wed -> Thu
    assert next_business_day("2023-05-05") == pd.Timestamp("2023-05-08")   # Fri -> Mon
    assert next_business_day("2023-05-06") == pd.Timestamp("2023-05-08")   # Sat -> Mon
    assert next_business_day("2023-07-03") == pd.Timestamp("2023-07-05")   # July 4th holiday
    assert next_business_day("2024-03-28") == pd.Timestamp("2024-04-01")   # Good Friday


# --------------------------------------------------------------------------- PIT visibility
def test_value_never_visible_before_filed_plus_1bd():
    df = build_company_asof(cf(gaap={"Revenues": REV}))
    # FY2022 10-K filed Wed 2023-02-15: not on 02-15, visible on Thu 02-16
    r = asof(df, "2023-02-15")                 # only Q1..Q3-2022 known: no TTM formable yet
    assert pd.isna(r["revenue_ttm"]) and pd.isna(r["revenue_end"])
    r = asof(df, "2023-02-16")
    assert r["revenue_end"] == pd.Timestamp("2022-12-31") and r["revenue_ttm"] == 400
    # Q1-2023 10-Q filed Fri 2023-05-05: invisible on Fri, Sat, Sun; visible Mon
    for d in ("2023-05-05", "2023-05-06", "2023-05-07"):
        assert asof(df, d)["revenue_end"] == pd.Timestamp("2022-12-31")
    assert asof(df, "2023-05-08")["revenue_end"] == pd.Timestamp("2023-03-31")
    # every event row is strictly after its triggering filing date
    assert (df["asof_date"] > df["last_filed"]).all()
    assert (df["days_since_filing"] >= 1).all()
    assert asof(df, "2022-05-04") is None   # nothing before the first filing is public


# --------------------------------------------------------------------------- YTD -> quarters, Q4, TTM
def _flow(facts):
    st = _FlowState()
    for s, e, v in facts:
        st.update(0, _ord(s), _ord(e), v)
    return st


def test_ytd_differencing_and_q4():
    st = _flow([(f["start"], f["end"], f["val"]) for f in REV])
    q = lambda e: st.quarter(_ord(e))  # noqa: E731
    assert q("2022-03-31") == 90
    assert q("2022-06-30") == 100      # 6M - Q1
    assert q("2022-09-30") == 105      # 9M - 6M
    assert q("2022-12-31") == 105      # FY - 9M
    assert q("2023-06-30") == 120
    assert q("2023-09-30") == 130
    assert q("2023-12-31") == 140


def test_q4_from_fy_minus_three_quarters():
    st = _flow([("2023-01-01", "2023-03-31", 10), ("2023-04-01", "2023-06-30", 20),
                ("2023-07-01", "2023-09-30", 30), ("2023-01-01", "2023-12-31", 100)])
    assert st.quarter(_ord("2023-12-31")) == 40
    # 4 chained discrete quarters also give a TTM without any FY fact
    st2 = _flow([("2023-01-01", "2023-03-31", 10), ("2023-04-01", "2023-06-30", 20),
                 ("2023-07-01", "2023-09-30", 30), ("2023-10-01", "2023-12-31", 40)])
    assert st2.ttm(_ord("2023-12-31")) == 100


def test_ttm_math_and_yoy():
    df = build_company_asof(cf(gaap={"Revenues": REV}))
    r = asof(df, "2023-05-08")                                  # TTM to Q1-23 = 110+105+105+100
    assert r["revenue_ttm"] == pytest.approx(420)
    assert r["revenue_ttm_prev"] is None or np.isnan(r["revenue_ttm_prev"])  # 2021 unknown
    r = asof(df, "2023-08-03")                                  # TTM to Q2-23 = 120+110+105+105
    assert r["revenue_ttm"] == pytest.approx(440)
    r = asof(df, "2023-11-02")                                  # TTM to Q3-23 = 130+120+110+105
    assert r["revenue_ttm"] == pytest.approx(465)
    r = asof(df, "2024-02-15")                                  # FY fact used directly
    assert r["revenue_ttm"] == 500 and r["revenue_ttm_prev"] == 400


def test_52_53_week_fiscal_year():
    # 53-week year with a 14-week Q4 (98 days) still differences correctly
    st = _flow([("2022-01-02", "2022-04-02", 10), ("2022-01-02", "2022-07-02", 30),
                ("2022-01-02", "2022-10-01", 60), ("2022-01-02", "2023-01-07", 110)])
    assert st.quarter(_ord("2023-01-07")) == 50
    assert st.ttm(_ord("2023-01-07")) == 110


# --------------------------------------------------------------------------- restatements
def test_restatement_only_from_its_own_filed_date():
    ni = [F("2022-12-31", 50, "2023-02-15", "2022-01-01", form="10-K", accn="orig"),
          F("2022-12-31", 40, "2023-06-01", "2022-01-01", form="10-K/A", accn="amend")]
    df = build_company_asof(cf(gaap={"NetIncomeLoss": ni}))
    assert asof(df, "2023-02-16")["net_income_ttm"] == 50
    assert asof(df, "2023-06-01")["net_income_ttm"] == 50     # amendment filed that day: not yet usable
    assert asof(df, "2023-06-02")["net_income_ttm"] == 40
    assert asof(df, "2023-06-02")["last_form"] == "10-K/A"
    # history is not rewritten: the row valid in March still says 50
    assert df.loc[df.asof_date == pd.Timestamp("2023-02-16"), "net_income_ttm"].item() == 50


def test_restated_comparative_in_later_10k():
    # FY2022 originally 400; FY2023 10-K restates 2022 comparative to 380 -> prev TTM changes then
    rev = [F("2022-12-31", 400, "2023-02-15", "2022-01-01", form="10-K"),
           F("2023-12-31", 500, "2024-02-14", "2023-01-01", form="10-K"),
           F("2022-12-31", 380, "2024-02-14", "2022-01-01", form="10-K")]
    df = build_company_asof(cf(gaap={"Revenues": rev}))
    assert asof(df, "2023-12-29")["revenue_ttm"] == 400
    r = asof(df, "2024-02-15")
    assert r["revenue_ttm"] == 500 and r["revenue_ttm_prev"] == 380


# --------------------------------------------------------------------------- fallback concepts
def test_fallback_concepts():
    doc = cf(
        gaap={
            # ASC 606 switch: SalesRevenueNet for FY2017, RevenueFromContract... for FY2018
            "SalesRevenueNet": [F("2017-12-31", 300, "2018-02-20", "2017-01-01", form="10-K")],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                F("2018-12-31", 330, "2019-02-20", "2018-01-01", form="10-K")],
            # Revenues preferred when both report the same period
            "Revenues": [F("2018-12-31", 333, "2019-02-20", "2018-01-01", form="10-K")],
            "CostOfRevenue": [F("2017-12-31", 200, "2018-02-20", "2017-01-01", form="10-K"),
                              F("2018-12-31", 210, "2019-02-20", "2018-01-01", form="10-K")],
            "Assets": [F("2018-12-31", 1000, "2019-02-20", form="10-K")],
            "LiabilitiesAndStockholdersEquity": [F("2018-12-31", 1000, "2019-02-20", form="10-K")],
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                F("2018-12-31", 350, "2019-02-20", form="10-K")],
            "ProfitLoss": [F("2018-12-31", 25, "2019-02-20", "2018-01-01", form="10-K")],
            "CommonStockSharesOutstanding": [F("2018-12-31", 77, "2019-02-20", form="10-K")],
        },
        unit_shares={"CommonStockSharesOutstanding"},
    )
    df = build_company_asof(doc)
    r = asof(df, "2018-03-01")
    assert r["revenue_ttm"] == 300                       # SalesRevenueNet
    assert r["gross_profit_ttm"] == 100 and r["gp_source"] == "revenue-cost"
    r = asof(df, "2019-03-01")
    assert r["revenue_ttm"] == 333                       # Revenues wins over RevenueFromContract...
    assert r["revenue_ttm_prev"] == 300                  # stitched across the concept switch
    assert r["gross_profit_ttm"] == 123
    assert r["gross_profit_ttm_prev"] == 100
    assert r["liabilities"] == 650                       # L&SE - total equity
    assert r["equity"] == 350                            # equity incl. NCI fallback
    assert r["net_income_ttm"] == 25                     # ProfitLoss fallback
    assert r["shares"] == 77                             # us-gaap shares fallback (no dei)


def test_dei_shares_preferred_and_stale_fallback():
    dei = {"EntityCommonStockSharesOutstanding": [F("2019-02-10", 100, "2019-02-20", form="10-K")]}
    gaap = {"CommonStockSharesOutstanding": [F("2018-12-31", 90, "2019-02-20", form="10-K"),
                                             F("2020-12-31", 95, "2021-02-20", form="10-K")]}
    df = build_company_asof(cf(gaap=gaap, dei=dei, unit_shares={"CommonStockSharesOutstanding"}))
    assert asof(df, "2019-03-01")["shares"] == 100       # dei preferred
    assert asof(df, "2021-03-01")["shares"] == 95        # dei stale by >120d -> fallback


# --------------------------------------------------------------------------- features / as-of join
def _feature_inputs():
    gaap = {
        "Revenues": REV,
        "NetIncomeLoss": [F("2022-12-31", 40, "2023-02-15", "2022-01-01", form="10-K"),
                          F("2023-12-31", 60, "2024-02-14", "2023-01-01", form="10-K")],
        "NetCashProvidedByUsedInOperatingActivities": [
            F("2022-12-31", 30, "2023-02-15", "2022-01-01", form="10-K")],
        "GrossProfit": [F("2022-12-31", 200, "2023-02-15", "2022-01-01", form="10-K")],
        "Assets": [F("2022-12-31", 1000, "2023-02-15", form="10-K")],
        "Liabilities": [F("2022-12-31", 600, "2023-02-15", form="10-K")],
        "StockholdersEquity": [F("2022-12-31", 400, "2023-02-15", form="10-K")],
    }
    dei = {"EntityCommonStockSharesOutstanding": [
        F("2023-02-10", 100, "2023-02-15", form="10-K"),
        F("2023-05-01", 200, "2023-05-05", form="10-Q")]}
    fund = build_company_asof(cf(cik=42, gaap=gaap, dei=dei))
    dates = pd.bdate_range("2023-02-10", "2023-05-12")
    prices = pd.DataFrame({"ticker": "ABC", "Date": dates.strftime("%Y-%m-%d"), "company_close": 10.0})
    cmap = pd.DataFrame({"cik": [42], "ticker": ["ABC"], "valid_from": [pd.Timestamp("2000-01-01")],
                         "valid_to": [pd.NaT]})
    sectors = pd.DataFrame({"cik": [42], "ff12": [6], "sector_name": ["Business Equipment"]})
    return prices, fund, cmap, sectors


def test_asof_join_never_uses_future_filings():
    prices, fund, cmap, sectors = _feature_inputs()
    out = fundamental_features(prices, fund, cmap, sectors)
    assert len(out) == len(prices) and (out.index == prices.index).all()
    d = pd.to_datetime(out["Date"])
    before = out[d <= "2023-02-15"]
    assert before[["mcap", "ep", "bp", "roa"]].isna().all().all()    # 10-K filed 02-15 not yet usable
    r = out[d == "2023-02-16"].iloc[0]
    assert r["mcap"] == 1000 and r["ep"] == pytest.approx(0.04) and r["bp"] == pytest.approx(0.4)
    assert r["roa"] == pytest.approx(0.04) and r["roe"] == pytest.approx(0.1)
    assert r["gpa"] == pytest.approx(0.2) and r["leverage"] == pytest.approx(0.6)
    assert r["accruals"] == pytest.approx(0.01) and r["sp"] == pytest.approx(0.4)
    assert r["fund_staleness_days"] == 1 and r["sector"] == "Business Equipment" and r["ff12"] == 6
    # staleness grows with calendar time until the next filing
    assert out.loc[d == "2023-03-01", "fund_staleness_days"].item() == 14
    # brute-force check: for every row, features equal those computed from filings filed < Date only
    for _, row in out.iterrows():
        known = fund[fund["asof_date"] <= row["Date"]]
        if known.empty:
            assert np.isnan(row["mcap"])
        else:
            np.testing.assert_allclose(row["mcap"], 10.0 * known.iloc[-1]["shares"])


def test_market_cap_uses_shares_known_at_date():
    prices, fund, cmap, sectors = _feature_inputs()
    out = fundamental_features(prices, fund, cmap, sectors)
    d = pd.to_datetime(out["Date"])
    assert out.loc[d == "2023-05-05", "mcap"].item() == 1000   # 10-Q with 200 shares filed that day
    assert out.loc[d == "2023-05-08", "mcap"].item() == 2000
    assert out.loc[d == "2023-05-08", "log_mcap"].item() == pytest.approx(np.log(2000))


def test_split_factor_and_negative_equity():
    prices, fund, cmap, sectors = _feature_inputs()
    prices["split_factor"] = 2.0          # adjusted close x factor = raw close
    out = fundamental_features(prices, fund, cmap, sectors, split_factor_col="split_factor")
    d = pd.to_datetime(out["Date"])
    assert out.loc[d == "2023-02-16", "mcap"].item() == 2000
    fund2 = fund.copy()
    fund2["equity"] = -5.0
    out2 = fundamental_features(prices, fund2, cmap, sectors)
    assert out2["roe"].isna().all()


def test_growth_and_net_issuance():
    gaap = {
        "Revenues": [F("2022-12-31", 400, "2023-02-15", "2022-01-01", form="10-K"),
                     F("2023-12-31", 500, "2024-02-14", "2023-01-01", form="10-K")],
        "NetIncomeLoss": [F("2022-12-31", 40, "2023-02-15", "2022-01-01", form="10-K"),
                          F("2023-12-31", 60, "2024-02-14", "2023-01-01", form="10-K")],
        "Assets": [F("2023-12-31", 1000, "2024-02-14", form="10-K")],
        # weighted diluted shares: FY2023 = 110 and the split-restated FY2022 comparative = 100
        "WeightedAverageNumberOfDilutedSharesOutstanding": [
            F("2022-12-31", 50, "2023-02-15", "2022-01-01", form="10-K"),       # pre 2:1 split
            F("2023-12-31", 110, "2024-02-14", "2023-01-01", form="10-K"),
            F("2022-12-31", 100, "2024-02-14", "2022-01-01", form="10-K")],     # restated comparative
    }
    fund = build_company_asof(cf(cik=7, gaap=gaap,
                                 unit_shares={"WeightedAverageNumberOfDilutedSharesOutstanding"}))
    prices = pd.DataFrame({"ticker": ["XYZ"], "Date": ["2024-02-15"], "company_close": [1.0]})
    cmap = pd.DataFrame({"cik": [7], "ticker": ["XYZ"]})
    out = fundamental_features(prices, fund, cmap, None).iloc[0]
    assert out["sales_growth_yoy"] == pytest.approx(0.25)
    assert out["ni_growth_scaled"] == pytest.approx(0.02)
    assert out["net_issuance_12m"] == pytest.approx(0.10)       # split-robust, not +120%


# --------------------------------------------------------------------------- mapping / sectors / UA
def test_resolve_cik_reused_ticker_and_class_suffix():
    cmap = pd.DataFrame({
        "cik": [1, 2, 3],
        "ticker": ["OLD", "OLD", "BRK-B"],
        "valid_from": pd.to_datetime(["1995-01-01", "2001-01-01", "1996-01-01"]),
        "valid_to": pd.to_datetime(["2019-06-30", None, None]),
    })
    keys = pd.DataFrame({"tkey": ["OLD", "OLD", "BRK.B", "NOPE"],
                         "Date": pd.to_datetime(["2018-01-02", "2020-01-02", "2020-01-02", "2020-01-02"])})
    got = resolve_cik(keys, cmap).tolist()
    assert got[:3] == [1, 2, 3] and pd.isna(got[3])


def test_build_cik_map_and_sectors():
    subs = [
        {"cik": 10, "name": "ACME CORP /DE/", "entity_type": "operating", "sic": "3571",
         "sic_description": "Electronic Computers", "tickers": ["ACME"], "exchanges": ["Nasdaq"],
         "former_names": [{"name": "ACME WIDGETS INC", "from": "2001-01-01T00:00:00.000Z",
                           "to": "2010-01-01T00:00:00.000Z"}],
         "first_filing": "2001-02-01", "last_filing": "2026-10-01", "last_periodic_filing": "2026-10-01",
         "n_periodic_recent": 20},
        {"cik": 11, "name": "GONE BANCORP", "entity_type": "operating", "sic": "6022",
         "sic_description": "State commercial banks", "tickers": ["ACME"], "exchanges": ["NYSE"],
         "former_names": [], "first_filing": "1996-01-01", "last_filing": "2015-03-01",
         "last_periodic_filing": "2015-03-01", "n_periodic_recent": 10},
        {"cik": 12, "name": "DELISTED CO", "entity_type": "operating", "sic": "",
         "sic_description": "", "tickers": [], "exchanges": [], "former_names": [],
         "first_filing": "1999-01-01", "last_filing": "2012-01-01", "last_periodic_filing": "2012-01-01",
         "n_periodic_recent": 5},
    ]
    tex = {"fields": ["cik", "name", "ticker", "exchange"], "data": [[10, "Acme", "ACME", "Nasdaq"]]}
    cm = edgar.build_cik_map(subs, tex)
    assert set(cm.cik) == {10, 11} and cm["ticker_ambiguous"].all()
    assert pd.isna(cm.loc[cm.cik == 10, "valid_to"].item())
    assert cm.loc[cm.cik == 11, "valid_to"].item() == pd.Timestamp("2015-03-01") + pd.Timedelta(days=120)
    sec = edgar.build_sectors(subs).set_index("cik")
    assert sec.loc[10, "ff12_code"] == "BusEq" and sec.loc[11, "ff12_code"] == "Money"
    assert pd.isna(sec.loc[12, "ff12"])                      # delisted CIK still present, SIC unknown
    names = edgar.build_cik_names(subs)
    assets = pd.DataFrame({"ticker": ["ACME", "DLST"], "name": ["Acme Corp", "Delisted Co."]})
    m = edgar.match_assets_to_cik(assets, cm, names)
    assert m["cik"].tolist() == [10, 12]
    assert m["map_method"].tolist() == ["ticker+name", "name"]


@pytest.mark.parametrize("sic,ff", [(100, 1), (2834, 10), (3711, 2), (1311, 4), (2810, 5),
                                    (7372, 6), (4813, 7), (4911, 8), (5812, 9), (6021, 11),
                                    (3714, 2), (3825, 6), (1000, 12), (4950, 12), (9999, 12)])
def test_ff12(sic, ff):
    assert edgar.sic_to_ff12(sic) == ff


def test_norm_ticker():
    assert edgar.norm_ticker("brk-b") == "BRK.B" == edgar.norm_ticker("BRK.B") == edgar.norm_ticker("BRK/B")


def test_missing_user_agent_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "_AGENT_ENV", tmp_path / "nope.env")
    with pytest.raises(edgar.SECUserAgentMissing, match="SEC_USER_AGENT"):
        edgar.download(edgar.TICKERS_EXCHANGE_URL, tmp_path / "x.json")
    with pytest.raises(edgar.SECUserAgentMissing):
        edgar.download_all()
    monkeypatch.setenv("SEC_USER_AGENT", "no-contact")
    with pytest.raises(edgar.SECUserAgentMissing, match="e-mail"):
        edgar.sec_user_agent()
