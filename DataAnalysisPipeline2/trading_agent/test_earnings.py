"""Round-11 earnings blackout in the live optimized_v3 path: optimizer ub_cap ==
round11.optimize_ext, the event snapshot, the window / expected dates, the
staleness skip and the error fallback. Synthetic data, no network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot, config, earnings, optimized
from DataAnalysisPipeline2.v3 import earnlive, gate
from DataAnalysisPipeline2.v3 import optimizer as op
from DataAnalysisPipeline2.v3 import round11 as r11
from DataAnalysisPipeline2.v3.tests.live_synthetic import make_live_bars_and_fund

TARGET = 0.98


# --------------------------------------------------------------------------- #
# Optimizer: ub_cap
# --------------------------------------------------------------------------- #
def _problem(n=80, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.hstack([rng.normal(size=(n, k)), np.eye(3)[rng.integers(0, 3, n)]])
    F = np.diag(np.r_[rng.uniform(1e-5, 4e-5, k), np.full(3, 2e-5)])
    tick = np.array([f"T{i:03d}" for i in range(n)])
    return dict(tickers=tick, scores=rng.normal(size=n), X=X, F_daily=F, D_daily=rng.uniform(1e-4, 4e-4, n),
                mcap=rng.lognormal(22, 1.2, n), sector=rng.integers(0, 3, n),
                factors=[f"s{j}" for j in range(k)] + ["ind:a", "ind:b", "ind:c"],
                half_spread=rng.uniform(5e-4, 2e-3, n), sigma_daily=rng.uniform(0.01, 0.03, n),
                adv=rng.lognormal(17, 1, n), ic=0.05)


SPEC = {**gate.R5_COMMON, **gate.R5_CONFIGS["mv_abs_all"]}


def test_ub_cap_none_is_backward_compatible():
    a = _problem()
    w0, _ = op.optimize_portfolio(SPEC, aum=1e6, **a)
    w1, _ = op.optimize_portfolio(SPEC, aum=1e6, ub_cap=None, **a)
    assert w0 == w1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ub_cap_reproduces_round11_optimize_ext(seed):
    a = _problem(seed=seed)
    w0, _ = op.optimize_portfolio(SPEC, aum=1e6, **a)
    tick = a["tickers"]
    cur = np.array([w0.get(t, 0.0) for t in tick])
    mask = np.zeros(len(tick), bool)
    held = np.flatnonzero(cur > 0)
    mask[held[::2]] = True                  # half the held names capped at half their weight
    mask[np.flatnonzero(cur == 0)[:10]] = True   # some unheld names -> 0
    cap = np.where(mask, cur * 0.5, np.inf)
    ref, dref = r11.optimize_ext(SPEC, aum=1e6, w_prev=w0, ub_cap=cap, **a)
    got, dgot = op.optimize_portfolio(SPEC, aum=1e6, w_prev=w0, ub_cap=cap, **a)
    got_d, _ = op.optimize_portfolio(SPEC, aum=1e6, w_prev=w0,
                                     ub_cap={t: c for t, c in zip(tick, cap) if np.isfinite(c)}, **a)
    assert got == ref and got_d == ref
    assert dgot["status"] == dref["status"] == "solved"
    v = np.array([got.get(t, 0.0) for t in tick])
    assert np.all(v <= np.where(np.isfinite(cap), cap, 1.0) + 1e-12)
    assert np.all(v[mask & (cur == 0)] == 0.0)
    assert v.sum() == pytest.approx(1.0, abs=2e-3)


def test_ub_cap_shape_is_checked():
    a = _problem()
    with pytest.raises(ValueError):
        op.optimize_portfolio(SPEC, aum=1e6, ub_cap=np.ones(3), **a)


# --------------------------------------------------------------------------- #
# Events, expected dates, window
# --------------------------------------------------------------------------- #
def _raw():
    rows = [(1, "2026-01-28 21:05:00"), (1, "2026-01-29 13:00:00"),   # same quarter (2nd dropped by the 60-d rule)
            (1, "2026-04-29 21:00:00"), (1, "2026-07-29 21:00:00"), (1, "2026-08-10 12:00:00"),
            (2, "2026-02-12 11:00:00"), (2, "2026-05-13 11:00:00"), (2, "2026-08-12 11:00:00"),
            (3, "2026-09-30 03:30:00")]                               # 03:30 UTC = 23:30 NY on 09-29
    return pd.DataFrame({"cik": [c for c, _ in rows], "acceptance": pd.to_datetime([a for _, a in rows], utc=True),
                         "src10q": 0})


def test_compact_snapshot_gives_the_same_expected_dates():
    raw = _raw()
    full = r11.retained_events(raw)
    assert list(full.loc[full["cik"] == 1, "adate"]) == ["2026-01-28", "2026-04-29", "2026-07-29"]
    assert full.loc[full["cik"] == 3, "adate"].item() == "2026-09-29"
    small = earnings.compact(raw, keep=2)
    assert small.groupby("cik").size().max() == 2
    for d in ("2026-08-15", "2026-09-30", "2026-10-30"):
        assert r11.expected_dates(full, d).equals(r11.expected_dates(small, d))
    exp = r11.expected_dates(full, "2026-10-30")
    assert exp[1] == "2026-10-28" and exp[2] == "2026-11-11" and exp[3] == "2026-12-29"
    assert 3 not in r11.expected_dates(full, "2026-09-28").index      # point in time


def test_blackout_caps_window_and_caps():
    raw = _raw()
    kept = r11.retained_events(raw)
    cal = earnings.session_calendar("2026-10-30")
    cik_of = {"AAA": 1, "BBB": 2, "CCC": 3}
    w_cur = {"AAA": 0.02, "BBB": 0.0}
    caps, info = earnings.blackout_caps(["AAA", "BBB", "CCC", "DDD"], "2026-10-30", kept, cik_of, w_cur,
                                        TARGET, calendar=cal, today="2026-11-02")
    assert info["rebalance_session"] == "2026-11-02"
    assert info["window"] == ["2026-10-23", "2026-11-09"]     # R - 10 d .. 5th session after R
    assert info["names"] == ["AAA"] and caps == {"AAA": pytest.approx(0.02 / TARGET)}
    assert info["blackout_held_w"] == pytest.approx(0.02) and info["n_blackout_held"] == 1
    # a later rebalance (BBB expected 2026-11-11 falls in the window, not held -> cap 0)
    caps, info = earnings.blackout_caps(["AAA", "BBB", "CCC"], "2026-11-05", kept, cik_of, w_cur, TARGET,
                                        calendar=earnings.session_calendar("2026-11-05"))
    assert info["rebalance_session"] == "2026-11-06" and set(caps) == {"AAA", "BBB"} and caps["BBB"] == 0.0


def test_window_matches_round11_with_the_backtest_calendar():
    cal = earnings.session_calendar("2026-07-01")
    assert r11.blackout_window("2026-07-01", cal) == ("2026-06-21", "2026-07-09")   # skips July 3 (holiday)


def test_rebalance_session_uses_the_run_date():
    cal = earnings.session_calendar("2026-06-26")
    assert earnings.rebalance_session("2026-06-26", cal) == "2026-06-29"
    assert earnings.rebalance_session("2026-06-26", cal, today="2026-07-01") == "2026-07-01"
    assert earnings.rebalance_session("2026-11-06", earnings.session_calendar("2026-11-06"),
                                      today="2026-11-08") == "2026-11-09"


# --------------------------------------------------------------------------- #
# Snapshot + staleness
# --------------------------------------------------------------------------- #
def test_snapshot_roundtrip_and_staleness(tmp_path):
    man = earnings.build_snapshot(str(tmp_path), raw=_raw(), as_of="2026-10-31", today="2026-11-01")
    assert man["as_of"] == "2026-10-31" and man["rows"] == len(earnings.read_snapshot(str(tmp_path))["kept"])
    src = earnings.resolve_events("2026-11-10", 14, str(tmp_path),
                                  [("snapshot", lambda: earnings.read_snapshot(str(tmp_path)))])
    assert src["age_days"] == 10
    with pytest.raises(earnings.EarningsUnavailable, match="stale"):
        earnings.resolve_events("2026-11-20", 14, str(tmp_path),
                                [("snapshot", lambda: earnings.read_snapshot(str(tmp_path)))])
    old = {"kept": earnings.compact(_raw()), "as_of": pd.Timestamp("2026-09-01"), "source": "store"}
    src = earnings.resolve_events("2026-11-10", 14, str(tmp_path),
                                  [("store", lambda: old), ("snapshot", lambda: earnings.read_snapshot(str(tmp_path)))])
    assert src["source"] == "snapshot"
    with pytest.raises(earnings.EarningsUnavailable, match="missing"):
        earnings.resolve_events("2026-11-10", 14, str(tmp_path / "nope"),
                                [("snapshot", lambda: earnings.read_snapshot(str(tmp_path / "nope")))])


def test_committed_earnings_snapshot_is_small_and_has_a_manifest():
    p = os.path.join(earnings.SNAPSHOT_DIR, earnings.EVENTS_FILE)
    assert os.path.exists(p) and os.path.getsize(p) < 1e6
    man = json.load(open(os.path.join(earnings.SNAPSHOT_DIR, earnings.MANIFEST_FILE)))
    assert pd.Timestamp(man["as_of"]) >= pd.Timestamp("2026-11-01")
    kept = pd.read_parquet(p)
    assert list(kept.columns) == ["cik", "adate"] and kept.groupby("cik").size().max() <= earnings.KEEP_PER_CIK


def test_earnlive_parsers():
    idx = ("Description: x\n\nCIK|Company Name|Form Type|Date Filed|Filename\n----\n"
           "320193|APPLE INC|8-K|20261106|edgar/data/320193/0000320193-26-000101.txt\n"
           "320193|APPLE INC|8-K/A|20261106|edgar/data/320193/0000320193-26-000102.txt\n"
           "789019|MICROSOFT|4|20261106|edgar/data/789019/0000789019-26-000001.txt\n")
    df = earnlive.parse_index_8k(idx)
    assert df["cik"].tolist() == [320193] and df["accession"].tolist() == ["0000320193-26-000101"]
    doc = {"filings": {"recent": {"form": ["8-K", "8-K", "10-Q", "4"],
                                  "accessionNumber": ["a1", "a2", "a3", "a4"],
                                  "filingDate": ["2026-11-06"] * 4,
                                  "acceptanceDateTime": ["2026-11-06T21:01:02.000Z"] * 4,
                                  "items": ["2.02,9.01", "8.01", "", ""]}}}
    f = earnlive.parse_submissions_json(doc, 320193)
    assert f["accn"].tolist() == ["a1", "a2", "a3"]
    raw = earnlive.to_raw(f)
    assert len(raw) == 1 and raw["cik"].item() == 320193 and raw["src10q"].item() == 0


# --------------------------------------------------------------------------- #
# optimized_book + bot wiring
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("fund"))
    bars, meta = make_live_bars_and_fund(d, n=200, n_sessions=330, seed=3)
    rng = np.random.default_rng(7)
    tick = sorted(bars["ticker"].unique())
    preds = pd.DataFrame({"ticker": tick, "pred_rank": rng.uniform(size=len(tick)),
                          "Sector": [meta["sector"].get(t, "Other") for t in tick],
                          "return_volatility_20d": rng.uniform(0.01, 0.04, len(tick)),
                          "kalman_beta": 1.0, "company_close": 50.0, "pred_proba": 0.5})
    today = (meta["last_filed"] + pd.Timedelta(days=5)).date()
    return bars, meta, d, preds, today


def _book(live, **kw):
    bars, _, d, preds, today = live
    args = dict(aum=1e6, ic=0.07, target_exposure=TARGET, candidates=[("test", d)], today=today,
                min_universe=50, min_names=10, log=lambda *a: None)
    args.update(kw)
    return optimized.optimized_book(preds, bars, **args)


def test_optimized_book_applies_caps(live):
    w0, _ = _book(live)
    held = sorted(w0, key=lambda t: -w0[t])
    w_prev = {t: w0[t] for t in held}
    blocked = held[:5]
    new = [t for t in live[3]["ticker"] if t not in w0][:20]
    w_prev_small = {t: w0[t] * 0.5 for t in held}

    def cap_fn(tickers, meta):
        assert "decision_date" in meta and "fund_dir" in meta
        return ({t: w_prev_small[t] / TARGET for t in blocked} | {t: 0.0 for t in new},
                {"applied": True, "n_blackout": len(blocked) + len(new)})
    w, diag = _book(live, w_prev=w_prev_small, cap_fn=cap_fn)
    bl = diag["earn_blackout"]
    assert bl["applied"] and bl["n_blackout"] == 25
    for t in blocked:
        assert w.get(t, 0.0) <= w_prev_small[t] + 1e-12
    assert not set(new) & set(w)
    assert sum(w.values()) == pytest.approx(TARGET, abs=3e-3)
    assert bl["book_w_blackout"] == pytest.approx(sum(w.get(t, 0.0) for t in blocked))


@pytest.mark.parametrize("how", ["raises", "stale"])
def test_optimized_book_ignores_a_failing_blackout(live, how):
    w0, _ = _book(live)

    def cap_fn(tickers, meta):
        if how == "raises":
            raise KeyError("boom")
        raise earnings.EarningsUnavailable("no usable earnings events: snapshot: stale")
    w, diag = _book(live, cap_fn=cap_fn)
    assert w == w0
    assert diag["earn_blackout"]["applied"] is False and diag["earn_blackout"]["skipped"]


def test_optimized_book_drops_caps_that_break_the_book(live):
    w0, _ = _book(live)
    w, diag = _book(live, cap_fn=lambda t, m: ({x: 0.0 for x in t}, {"applied": True}))  # everything capped at 0
    assert w == w0
    assert diag["earn_blackout"]["applied"] is False and "rejected" in diag["earn_blackout"]["skipped"]


@pytest.fixture
def agent_cfg(live, monkeypatch):
    bars, _, d, preds, _ = live
    monkeypatch.setattr(config, "TICKERS", preds["ticker"].tolist())
    for k, v in dict(V3_K=30, V3_MAX_PER_SECTOR=5, V3_MAX_WEIGHT=0.06, V3_TARGET_EXPOSURE=TARGET,
                     OPT_AUM=1e6, OPT_IC=0.07, OPT_IC_ALTALL=0.07, OPT_TARGET_EXPOSURE=TARGET,
                     OPT_MIN_WEIGHT=0.0025, OPT_FUND_DIR=d, OPT_FUND_MAX_AGE_DAYS=100_000, OPT_MIN_UNIVERSE=50,
                     OPT_MIN_NAMES=10, ALPACA_API_KEY="", ALPACA_SECRET_KEY="", EARN_BLACKOUT=True,
                     EARN_MAX_AGE_DAYS=14).items():
        monkeypatch.setattr(config, k, v, raising=False)
    agent = object.__new__(bot.BDATradingAgent)
    agent.last_bars = bars
    agent.signal = "altall"
    return agent, preds, d


def _events_snapshot(tmp_path, d, decision, ciks):
    """Snapshot in which ``ciks`` reported 91 days before ``decision`` (expected
    on the decision date -> blacked out)."""
    acc = pd.Timestamp(decision) - pd.Timedelta(days=91) + pd.Timedelta(hours=21)
    raw = pd.DataFrame({"cik": list(ciks), "acceptance": pd.Timestamp(acc, tz="UTC"), "src10q": 0})
    earnings.build_snapshot(str(tmp_path), raw=raw, as_of=decision)
    return str(tmp_path)


def test_bot_applies_the_blackout_and_logs_it(agent_cfg, monkeypatch, tmp_path, capsys):
    agent, preds, d = agent_cfg
    decision = str(pd.Timestamp(agent.last_bars["Date"].max()).date())
    cm = pd.read_parquet(os.path.join(d, "cik_map.parquet"))
    monkeypatch.setattr(config, "EARN_BLACKOUT", False)
    w_free = agent.calculate_target_weights(preds, None, strategy="optimized_v3")
    monkeypatch.setattr(config, "EARN_BLACKOUT", True)
    top = sorted((t for t, x in w_free.items() if x and t in set(cm["ticker"])), key=lambda t: -w_free[t])[:4]
    ciks = cm.loc[cm["ticker"].isin(top), "cik"].astype("int64").tolist()
    snap = _events_snapshot(tmp_path, d, decision, ciks)
    monkeypatch.setattr(config, "ALT_SNAPSHOT_DIR", snap)
    monkeypatch.setattr(earnings, "_store_exists", lambda: False)
    real = earnings.live_caps
    monkeypatch.setattr(earnings, "live_caps", lambda *a, **k: real(*a, **{**k, "today": decision}))
    w = agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    c = agent.last_construction
    assert c["construction"] == "optimized_v3" and c["earn_blackout"]["applied"]
    assert set(top) <= set(c["earn_blackout"]["names"])
    assert all(not w.get(t) for t in top)            # not held (w_prev = {}) -> cap 0
    assert "earn_blackout:" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["stale", "error", "disabled", "production"])
def test_bot_skips_the_blackout_without_failing(agent_cfg, monkeypatch, tmp_path, case):
    agent, preds, d = agent_cfg
    decision = str(pd.Timestamp(agent.last_bars["Date"].max()).date())
    monkeypatch.setattr(earnings, "_store_exists", lambda: False)
    if case == "stale":
        old = str((pd.Timestamp(decision) - pd.Timedelta(days=30)).date())
        monkeypatch.setattr(config, "ALT_SNAPSHOT_DIR", _events_snapshot(tmp_path, d, old, [1000]))
    elif case == "error":
        monkeypatch.setattr(earnings, "live_caps", lambda *a, **k: 1 / 0)
    elif case == "disabled":
        monkeypatch.setattr(config, "EARN_BLACKOUT", False)
    else:
        agent.signal = "production"
    agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    c = agent.last_construction
    assert c["construction"] == "optimized_v3"
    assert c["earn_blackout"]["applied"] is False and c["earn_blackout"]["skipped"]
    if case == "stale":
        assert "stale" in c["earn_blackout"]["skipped"]
