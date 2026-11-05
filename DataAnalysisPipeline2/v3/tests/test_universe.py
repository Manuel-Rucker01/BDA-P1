"""Universe: classification rules, symbol-reuse detection, PIT eligibility."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import universe as U


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BDA_DATA_V3", str(tmp_path))
    U._SEG_CACHE.clear()


# ----------------------------------------------------------- classification
@pytest.mark.parametrize("symbol,name,exchange,expected", [
    ("AAPL", "Apple Inc. Common Stock", "NASDAQ", "common"),
    ("GOOGL", "Alphabet Inc. Class A Common Stock", "NASDAQ", "common"),       # 5th letter L, but common
    ("BRK.B", "Berkshire Hathaway Inc. Class B", "NYSE", "common"),
    ("BABA", "Alibaba Group Holding Limited American Depositary Shares", "NYSE", "common"),  # ADRs stay
    ("RLX", "RLX Technology Inc. American Depositary Shares, each representing the right to receive one "
            "Class A ordinary share", "NYSE", "common"),
    ("FRT", "Federal Realty Investment Trust Common Shares of Beneficial Interest", "NYSE", "common"),
    ("EPD", "Enterprise Products Partners L.P. Common Units", "NYSE", "common"),  # MLP units stay
    ("WT", "WisdomTree, Inc.", "NYSE", "common"),                               # ETF sponsor company
    ("CVI", "CVR ENERGY, INC.", "NYSE", "common"),
    ("PFBC", "Preferred Bank Common Stock", "NASDAQ", "common"),
    ("APO", "Apollo Global Management, Inc.", "NYSE", "common"),
    ("QQQ", "Invesco QQQ Trust, Series 1", "NASDAQ", "etf"),
    ("TQQQ", "ProShares UltraPro QQQ", "NASDAQ", "etf"),
    ("IUSG", "iShares Core S&P U.S. Growth ETF", "NASDAQ", "etf"),
    ("NVG", "Nuveen AMT-Free Municipal Credit Income Fund", "NYSE", "fund"),
    ("BME", "Blackrock Health Sciences Trust", "NYSE", "fund"),
    ("PBT", "Permian Basin Royalty Trust", "NYSE", "trust"),
    ("OBTC", "Osprey Bitcoin Trust Common Units of Beneficial Interest", "NASDAQ", "trust"),
    ("DSX.WS", "Diana Shipping Inc. Warrants", "NYSE", "warrant"),
    ("ABCDW", "Something Acquisition Corp. Warrant", "NASDAQ", "warrant"),
    ("ABCDW", "Something Corp Common Stock", "NASDAQ", "warrant"),             # symbol W wins
    ("PACE.U", "TPG Pace Units, each consisting of one share and one-fifth of one warrant", "NYSE", "unit"),
    ("ABCDU", "Foo Acquisition Corp Units", "NASDAQ", "unit"),
    ("GIX.RT", "GigCapital2, Inc. Rights", "NYSE", "right"),
    ("ABCDR", "Foo Acquisition Corp Right", "NASDAQ", "right"),
    ("WFC.PRA", "Wells Fargo & Company", "NYSE", "preferred"),
    ("BPYPN", "Brookfield Property Partners L.P. 5.750% Class A Cumulative Redeemable Perpetual Preferred Units",
     "NASDAQ", "preferred"),
    ("OXLCO", "Oxford Lane Capital Corp. 6.00% Series 2029 Term Preferred Shares", "NASDAQ", "preferred"),
    ("NEWTO", "NewtekOne, Inc. 8.50% Fixed Rate Senior Notes due 2031", "NASDAQ", "note"),
    ("ELC", "Entergy Louisiana, LLC Collateral Trust Mortgage Bonds, 4.875 % Series due September 1, 2066",
     "NYSE", "note"),
    ("723ESC018", "ESC something", "NASDAQ", "other"),
    ("IP.WI", "International Paper Company", "NYSE", "other"),
    ("SPY", "SPDR S&P 500 ETF Trust", "ARCA", "non_listed_exchange"),
    ("AAPL", "Apple Inc.", "OTC", "non_listed_exchange"),
])
def test_classify_name(symbol, name, exchange, expected):
    assert U.classify_name(symbol, name, exchange)[0] == expected


def test_classify_symbol_and_spac_flag():
    assert U.classify_symbol("ANCB_DELISTED", "NASDAQ") == (None, "sym:plain")
    assert U.classify_symbol("ABERX", None)[0] == "fund"         # 5th letter X: mutual fund
    assert U.classify_symbol("PRE.PRH")[0] == "preferred"
    assert U.is_spac_like("Cohen Circle Acquisition Corp. II Class A Ordinary Shares")
    assert not U.is_spac_like("Apple Inc. Common Stock")


def test_max_abs_corr_finds_etf_twin():
    rng = np.random.default_rng(1)
    idx = pd.RangeIndex(120)
    mkt = rng.normal(0, 0.02, 120)
    refs = pd.DataFrame({"SPY": mkt, "XLE": rng.normal(0, 0.02, 120)}, index=idx)
    tgt = pd.DataFrame({"DEADETF": -2 * mkt + rng.normal(0, 0.001, 120),        # inverse leveraged twin
                        "STOCK": 0.5 * mkt + rng.normal(0, 0.03, 120),
                        "SHORT": np.r_[mkt[:10], [np.nan] * 110]}, index=idx)
    r = U.max_abs_corr(tgt, refs).set_index("symbol")
    assert r.loc["DEADETF", "etf_corr"] > 0.99 and r.loc["DEADETF", "etf_match"] == "SPY"
    assert r.loc["STOCK", "etf_corr"] < 0.6
    assert np.isnan(r.loc["SHORT", "etf_corr"])  # < 26 overlapping weeks: untested


# ------------------------------------------------------------- symbol reuse
def _px(sym, dates, close, vol=1e6):
    return pd.DataFrame({"ticker": sym, "Date": list(dates), "company_close": close,
                         "company_volume": vol})


def _cls(rows):
    return pd.DataFrame(rows, columns=["symbol", "fetch_symbol", "status", "source", "cls"])


def test_reuse_split_on_long_gap_even_with_zero_volume_filler():
    d1 = pd.bdate_range("2016-01-04", periods=300).strftime("%Y-%m-%d")
    filler = pd.bdate_range("2017-03-01", "2018-06-29").strftime("%Y-%m-%d")      # no trades
    d2 = pd.bdate_range("2018-07-02", periods=200).strftime("%Y-%m-%d")
    px = pd.concat([_px("APC", d1, 45.0), _px("APC", filler, 50.0, vol=0.0), _px("APC", d2, 13.0),
                    _px("MSFT", pd.bdate_range("2016-01-04", d2[-1]).strftime("%Y-%m-%d"), 100.0)],
                   ignore_index=True)
    cls = _cls([("APC", "APC", "inactive", "alpaca_assets", "common"),
                ("APC", "APC", "active", "alpaca_assets", "common"),
                ("MSFT", "MSFT", "active", "alpaca_assets", "common")])
    seg = U.detect_symbol_reuse(px, cls).set_index("ticker")
    assert set(seg.index) == {"APC~1", "APC", "MSFT"}
    assert seg.loc["APC~1", "end"] == d1[-1] and seg.loc["APC", "start"] == d2[0]
    assert seg.loc["APC", "split_reason"] == "gap" and seg.loc["APC", "gap_before_days"] > 90
    out = U.apply_segments(px)
    assert set(out["ticker"]) >= {"APC~1", "APC"}
    assert not ((out["ticker"] == "APC") & (out["Date"] < d2[0])).any()
    assert not out.loc[out["ticker"].str.startswith("APC"), "Date"].isin(filler).any()  # filler dropped
    assert (out["ticker"] == "MSFT").sum() == (px["ticker"] == "MSFT").sum()            # untouched


def test_reuse_multi_security_short_gap_and_price_break():
    d = pd.bdate_range("2020-01-01", periods=100).strftime("%Y-%m-%d")
    # single security: a 10-day halt is NOT a new listing
    halt = pd.concat([_px("HLT", d[:50], 10.0), _px("HLT", d[60:], 10.0)])
    # multi security, 10-day gap -> new listing
    multi_gap = pd.concat([_px("MGP", d[:50], 10.0), _px("MGP", d[60:], 10.0)])
    # multi security, no gap but price level jump x10 (AMTD-like) -> split at the jump
    brk = _px("BRK", d, np.r_[np.full(50, 1000.0), np.full(50, 12.0)])
    cls = _cls([("HLT", "HLT", "active", "alpaca_assets", "common"),
                ("MGP", "MGP", "active", "alpaca_assets", "common"),
                ("MGP_DELISTED", "MGP", "inactive", "alpaca_assets", "common"),
                ("BRK", "BRK", "active", "alpaca_assets", "common"),
                ("BRK", "BRK", "inactive", "alpaca_assets", "common")])
    seg = U.detect_symbol_reuse(pd.concat([halt, multi_gap, brk]), cls)
    n = seg.groupby("symbol")["n_seg"].first()
    assert n["HLT"] == 1 and n["MGP"] == 2 and n["BRK"] == 2
    assert seg.loc[seg["ticker"] == "BRK", "split_reason"].item() == "price_break"
    assert seg.loc[seg["ticker"] == "BRK~1", "end"].item() == d[49]


def test_reuse_segment_class_follows_active_vs_inactive_security():
    d1 = pd.bdate_range("2016-01-04", periods=200).strftime("%Y-%m-%d")
    d2 = pd.bdate_range("2023-01-02", periods=200).strftime("%Y-%m-%d")
    px = pd.concat([_px("FB", d1, 100.0), _px("FB", d2, 25.0)])
    cls = _cls([("FB", "FB", "inactive", "alpaca_assets", "common"),
                ("FB", "FB", "active", "alpaca_assets", "etf")])
    seg = U.detect_symbol_reuse(px, cls).set_index("ticker")
    assert seg.loc["FB~1", "cls"] == "common" and seg.loc["FB~1", "keep"]
    assert seg.loc["FB", "cls"] == "etf" and not seg.loc["FB", "keep"]
    out = U.apply_segments(px)
    assert set(out["ticker"]) == {"FB~1"} and len(out) == 200


# --------------------------------------------------------------- eligibility
def _panel(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n).strftime("%Y-%m-%d")
    rows = []
    for t, px0, vol in (("BIG", 50.0, 1e6), ("PENNY", 1.0, 1e7), ("THIN", 50.0, 1e3), ("NEW", 50.0, 1e6)):
        px = px0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        df = pd.DataFrame({"ticker": t, "Date": dates, "company_close": px, "company_volume": vol,
                           "close_raw": px})
        rows.append(df.iloc[150:] if t == "NEW" else df)
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed)  # shuffled on purpose


def test_eligibility_rule():
    p = _panel()
    det = U.eligibility_details(p)
    e = U.eligible(p)
    assert e.index.equals(p.index)
    assert not e[p["ticker"].isin(["PENNY", "THIN", "NEW"])].any()  # price / ADV / history
    big = det[p["ticker"] == "BIG"].sort_values("Date")
    assert not big["eligible"].iloc[:119].any() and big["eligible"].iloc[119:].mean() > 0.9
    assert big["adv_usd"].iloc[:19].isna().all()


def test_eligibility_uses_raw_close_for_price_floor():
    p = _panel()
    p = p[p["ticker"] == "BIG"].copy()
    p["company_close"] = p["close_raw"] / 40.0     # adjusted for a later 40:1 split: ~$1.3
    p["company_volume"] *= 40.0
    assert U.eligible(p).sum() > 0                 # PIT price ~$50 passes the $3 floor
    assert U.eligible(p.drop(columns="close_raw")).sum() == 0


def test_eligibility_is_point_in_time():
    """Appending future rows (even absurd ones) never changes past eligibility."""
    p = _panel(n=200)
    base = U.eligibility_details(p)
    future_dates = pd.bdate_range("2016-10-24", periods=40).strftime("%Y-%m-%d")
    assert future_dates[0] > p["Date"].max()
    fut = pd.concat([pd.DataFrame({"ticker": t, "Date": future_dates, "company_close": c,
                                   "company_volume": v, "close_raw": c})
                     for t, c, v in (("BIG", 0.01, 0.0), ("PENNY", 500.0, 1e9), ("THIN", 99.0, 1e12),
                                     ("NEWCO", 10.0, 1e7))])
    ext = U.eligibility_details(pd.concat([p, fut], ignore_index=False).reset_index(drop=True))
    m = ext.iloc[: len(p)]
    pd.testing.assert_frame_equal(m.reset_index(drop=True), base.reset_index(drop=True))
