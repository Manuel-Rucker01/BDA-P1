"""Round 10 earnings-surprise blocks: SUE from companyfacts, announcement
events from submissions, EAR, point-in-time rules and no-lookahead checks
(synthetic, no network)."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import altdata_r10 as a10
from DataAnalysisPipeline2.v3 import gate


# --------------------------------------------------------------------------- #
# Event timing
# --------------------------------------------------------------------------- #
CAL = np.array([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-06-02", "2025-08-29")
                if d.strftime("%Y-%m-%d") != "2025-07-04"])


def _pos(d):
    return int(np.searchsorted(CAL, d))


def test_event_day0_session_rules():
    acc = pd.Series(pd.to_datetime([
        "2025-07-01T11:00:00Z",   # 07:00 EDT pre-market, Tue -> Tue
        "2025-07-01T19:59:00Z",   # 15:59 EDT -> same day
        "2025-07-01T20:00:00Z",   # 16:00 EDT -> next session
        "2025-07-03T21:00:00Z",   # Thu 17:00 EDT, Fri 07-04 holiday -> Mon 07-07
        "2025-07-05T15:00:00Z",   # Saturday -> Monday
        "2025-07-02T02:00:00Z",   # Tue 22:00 EDT (UTC already Wed) -> Wed
    ], utc=True))
    got = [CAL[p] for p in a10.event_day0(acc, CAL)]
    assert got == ["2025-07-01", "2025-07-01", "2025-07-02", "2025-07-07", "2025-07-07", "2025-07-02"]


def _filings(rows):
    f = pd.DataFrame(rows, columns=["cik", "accn", "form", "filing_date", "acceptance", "items"])
    f["acceptance"] = pd.to_datetime(f["acceptance"], utc=True)
    return f


def test_announcement_events_fallback_and_filters():
    f = _filings([
        (1, "a1", "8-K", "2025-06-10", "2025-06-10T20:30:00Z", "2.02,9.01"),
        (1, "a2", "10-Q", "2025-06-20", "2025-06-20T12:00:00Z", ""),       # 8-K 2.02 10 days before -> dropped
        (1, "a3", "8-K", "2025-07-15", "2025-07-15T12:00:00Z", "5.02"),    # not an earnings 8-K
        (2, "b1", "10-Q", "2025-07-10", "2025-07-10T12:00:00Z", ""),       # no 8-K 2.02 -> fallback
        (2, "b2", "8-K", "2025-08-12", "2025-08-12T12:00:00Z", "2.02"),
        (3, "c1", "8-K/A", "2025-07-10", "2025-07-10T12:00:00Z", "2.02"),  # amendment ignored
        (4, "d1", "8-K", "2025-07-10", "2025-07-10T12:00:00Z", "12.02"),   # item 12.02 is not 2.02
    ])
    ev = a10.announcement_events(f, CAL)
    assert list(ev["cik"]) == [1, 2, 2]
    assert [CAL[p] for p in ev["day0"]] == ["2025-06-11", "2025-07-10", "2025-08-12"]
    assert list(ev["src10q"]) == [0, 1, 0]
    assert list(ev["avail_pos"] - ev["day0"]) == [2, 2, 2]
    assert pd.isna(ev["prev_day0"].iloc[1]) and ev["prev_day0"].iloc[2] == ev["day0"].iloc[1]


def test_fallback_uses_only_past_8k():
    # 10-Q first, the 8-K 2.02 after it: the 10-Q stays an event (no lookahead)
    f = _filings([(1, "q", "10-Q", "2025-07-01", "2025-07-01T12:00:00Z", ""),
                  (1, "k", "8-K", "2025-07-08", "2025-07-08T12:00:00Z", "2.02")])
    ev = a10.announcement_events(f, CAL)
    assert list(ev["src10q"]) == [1, 0]


# --------------------------------------------------------------------------- #
# EAR values and weekly join
# --------------------------------------------------------------------------- #
def _prices(T, seed=0):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.01, (T, 2)), axis=0)) * 10
    vol = rng.uniform(1e5, 2e5, (T, 2))
    mkt = np.cumsum(rng.normal(0, 0.005, T))
    return close, vol, mkt


def test_ear_values_definition():
    T = len(CAL)
    close, vol, mkt = _prices(T)
    p = len(CAL) - 2
    pairs = pd.DataFrame({"ticker": ["X", "Y", "Z"], "day0": [p, p, 1]})
    out = a10.ear_values(pairs, close, vol, {"X": 0, "Y": 1}, mkt)
    exp = np.log(close[p + 1, 0] / close[p - 2, 0]) - (mkt[p + 1] - mkt[p - 2])
    assert np.isclose(out["ear"].iloc[0], exp)
    exp_v = np.log(vol[p - 1:p + 2, 1].mean() / vol[p - 60:p - 10, 1].mean())
    assert np.isclose(out["ear_abvol"].iloc[1], exp_v)
    assert out.iloc[2].isna().all()                       # unknown ticker / no history


def test_ear_weekly_availability_and_no_lookahead():
    T = len(CAL)
    close, vol, mkt = _prices(T, seed=1)
    ev = pd.DataFrame({"cik": [7, 7], "day0": [_pos("2025-07-15"), _pos("2025-08-12")], "src10q": [0, 1]})
    ev["avail_pos"] = ev["day0"] + 2
    ev["prev_day0"] = ev["day0"].shift(1)
    dates = ["2025-07-15", "2025-07-16", "2025-07-17", "2025-08-13", "2025-08-14", "2025-08-29"]
    rows = pd.DataFrame({"ticker": "X", "cik": 7.0, "Date": dates})
    out = a10.ear_weekly(rows, ev, CAL, close, vol, {"X": 0}, mkt)
    assert out.iloc[:2].isna().all().all()                # day0, day0+1: not yet usable
    p1 = _pos("2025-07-15")
    assert np.isclose(out["ear"].iloc[2], np.log(close[p1 + 1, 0] / close[p1 - 2, 0]) - (mkt[p1 + 1] - mkt[p1 - 2]))
    assert out["ear_src10q"].iloc[2] == 0 and out["ear_days"].iloc[2] == 2
    assert np.isclose(out["ear"].iloc[3], out["ear"].iloc[2])   # held until the next event is usable
    assert out["ear_src10q"].iloc[4] == 1 and np.isclose(out["ear_prev"].iloc[4], out["ear"].iloc[2])
    # no lookahead: prices / volume after each row's date never change its value
    for k, d in enumerate(dates):
        c2, v2 = close.copy(), vol.copy()
        c2[_pos(d) + 1:] *= 3.0
        v2[_pos(d) + 1:] *= 5.0
        o2 = a10.ear_weekly(rows.iloc[[k]], ev, CAL, c2, v2, {"X": 0}, mkt)
        pd.testing.assert_frame_equal(o2.reset_index(drop=True), out.iloc[[k]].reset_index(drop=True))
    # rows without a CIK stay missing (hot-decked later)
    o3 = a10.ear_weekly(rows.assign(cik=np.nan), ev, CAL, close, vol, {"X": 0}, mkt)
    assert o3.isna().all().all()


# --------------------------------------------------------------------------- #
# SUE from companyfacts
# --------------------------------------------------------------------------- #
def _q_ends(years):
    out = []
    for y in years:
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            out.append(pd.Timestamp(y, m, d))
    return out


def _cf(eps_by_q, ni_only=False, filed_lag=35):
    """Synthetic companyfacts: calendar fiscal year, 10-Q for Q1-Q3 (3-month
    facts + the prior-year comparative), 10-K with the FY fact only (Q4 must be
    derived)."""
    facts = {}

    def add(concept, unit, start, end, val, form, filed):
        facts.setdefault(concept, {"units": {unit: []}})["units"][unit].append(
            {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "val": val, "form": form,
             "filed": filed.strftime("%Y-%m-%d"), "accn": f"{form}-{filed:%Y%m%d}"})
    ends = sorted(eps_by_q)
    for e in ends:
        filed = e + pd.Timedelta(days=filed_lag if e.month != 12 else 60)
        start_q = (e - pd.offsets.QuarterBegin(startingMonth=1)).normalize() if e.month != 1 else e
        start_q = pd.Timestamp(e.year, e.month - 2, 1)
        prior = pd.Timestamp(e.year - 1, e.month, e.day)
        if e.month != 12:
            for ee, ss in ((e, start_q), (prior, pd.Timestamp(prior.year, prior.month - 2, 1))):
                if ee in eps_by_q:
                    if ni_only:
                        add("NetIncomeLoss", "USD", ss, ee, eps_by_q[ee] * 100.0, "10-Q", filed)
                        add("WeightedAverageNumberOfDilutedSharesOutstanding", "shares", ss, ee, 100.0, "10-Q", filed)
                    else:
                        add("EarningsPerShareDiluted", "USD/shares", ss, ee, eps_by_q[ee], "10-Q", filed)
        else:
            fy = [pd.Timestamp(e.year, m, d) for m, d in ((3, 31), (6, 30), (9, 30), (12, 31))]
            if all(x in eps_by_q for x in fy):
                tot = sum(eps_by_q[x] for x in fy)
                if ni_only:
                    add("NetIncomeLoss", "USD", pd.Timestamp(e.year, 1, 1), e, tot * 100.0, "10-K", filed)
                    add("WeightedAverageNumberOfDilutedSharesOutstanding", "shares", pd.Timestamp(e.year, 1, 1), e,
                        100.0, "10-K", filed)
                else:
                    add("EarningsPerShareDiluted", "USD/shares", pd.Timestamp(e.year, 1, 1), e, tot, "10-K", filed)
    return {"cik": 42, "facts": {"us-gaap": facts}}


def _eps(seed=3):
    rng = np.random.default_rng(seed)
    ends = _q_ends(range(2015, 2021))
    return {e: round(1.0 + 0.05 * k + rng.normal(0, 0.2), 4) for k, e in enumerate(ends)}


def _expected(eps):
    ends = sorted(eps)
    d = {e: eps[e] - eps[ends[k - 4]] for k, e in enumerate(ends) if k >= 4}
    out = {}
    for k, e in enumerate(ends):
        if e not in d:
            continue
        prev = [d[x] for x in ends[max(0, k - 8):k] if x in d]
        out[e] = d[e] / np.std(prev, ddof=1) if len(prev) >= 4 else np.nan
    return out


def test_build_company_sue_values_and_q4():
    eps = _eps()
    ev = a10.build_company_sue(_cf(eps), 42)
    exp = _expected(eps)
    got = dict(zip(ev["qend"], ev["sue_eps"]))
    for e, v in exp.items():
        if np.isfinite(v):
            assert np.isclose(got[e], v, atol=1e-9), e
    q4 = ev[ev["qend"] == pd.Timestamp("2019-12-31")].iloc[0]
    assert np.isclose(q4["eps_q"], eps[pd.Timestamp("2019-12-31")], atol=1e-9)     # FY - Q1..Q3
    # availability: next business day after the filing date
    r = ev[ev["qend"] == pd.Timestamp("2019-06-30")].iloc[0]
    assert r["asof_date"] == pd.Timestamp("2019-08-05")                            # filed Sun 08-04 -> Mon
    # streak / change consistent with the d signs and SUE path
    s = ev.set_index("qend")
    assert np.isclose(s.loc["2020-06-30", "sue_eps_chg"], s.loc["2020-06-30", "sue_eps"] - s.loc["2020-03-31", "sue_eps"])
    assert abs(s["sue_streak"]).max() <= a10.STREAK_CAP


def test_build_company_sue_ni_fallback_matches_eps():
    eps = _eps(seed=5)
    a = a10.build_company_sue(_cf(eps), 42)
    b = a10.build_company_sue(_cf(eps, ni_only=True), 42)
    assert (b["eps_src"] == 1).all() and (a["eps_src"] == 0).all()
    np.testing.assert_allclose(a["sue_eps"].to_numpy(), b["sue_eps"].to_numpy(), atol=1e-9)


def test_build_company_sue_no_lookahead():
    cf = _cf(_eps(seed=7))
    full = a10.build_company_sue(cf, 42)
    for cut in ("2018-05-10", "2019-03-15", "2020-08-20"):
        cutd = pd.Timestamp(cut)
        trunc = {"cik": 42, "facts": {"us-gaap": {
            k: {"units": {u: [f for f in fs if pd.Timestamp(f["filed"]) < cutd] for u, fs in v["units"].items()}}
            for k, v in cf["facts"]["us-gaap"].items()}}}
        part = a10.build_company_sue(trunc, 42)
        known = full[full["asof_date"] <= cutd].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), known)


def test_sue_weekly_join_point_in_time():
    sue = pd.DataFrame({"cik": [5, 5], "asof_date": pd.to_datetime(["2020-02-03", "2020-05-04"]),
                        "sue_eps": [1.0, -2.0], "sue_eps_chg": [np.nan, -3.0], "sue_streak": [1.0, -1.0],
                        "surge": [0.5, 0.1]})
    rows = pd.DataFrame({"cik": [5.0, 5.0, 5.0, np.nan], "Date": ["2020-01-31", "2020-02-03", "2020-05-08", "2020-05-08"]})
    out = a10.sue_weekly(rows, sue)
    assert out.iloc[0].isna().all() and out.iloc[3].isna().all()
    assert out["sue_eps"].iloc[1] == 1.0 and out["sue_days"].iloc[1] == 0
    assert out["sue_eps"].iloc[2] == -2.0 and out["sue_days"].iloc[2] == 4


def test_stale_events_are_missing():
    sue = pd.DataFrame({"cik": [5], "asof_date": pd.to_datetime(["2020-02-03"]), "sue_eps": [1.0],
                        "sue_eps_chg": [0.0], "sue_streak": [1.0], "surge": [0.5]})
    rows = pd.DataFrame({"cik": [5.0, 5.0], "Date": ["2021-03-09", "2021-03-10"]})   # 400 / 401 days
    out = a10.sue_weekly(rows, sue)
    assert out["sue_eps"].iloc[0] == 1.0 and out.iloc[1].isna().all()


def test_round10_preregistration():
    assert gate.N_TRIALS_R10 == 98 and gate.R10_BASE == "sn30_altall_hd"
    ic = {"sn30_altall_hd": 0.07, "sn30_r10_sue": 0.0751, "sn30_r10_ear": 0.0749, "sn30_r10_both": 0.08}
    icm = {"sn30_altall_hd": 0.06, "sn30_r10_sue": 0.0626, "sn30_r10_ear": 0.07, "sn30_r10_both": 0.062}
    assert gate.r10_bookable(ic, icm) == ["sn30_r10_sue"]
    assert gate.choose_config_r10({}) is None
    perf = {"sn30_r10_sue": {"ir_spy": 0.1, "max_dd": -0.4}, "sn30_r10_ear": {"ir_spy": -0.1, "max_dd": -0.3}}
    assert gate.choose_config_r10(perf) == "sn30_r10_ear"
