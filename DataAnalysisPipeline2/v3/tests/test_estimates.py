"""Analyst-estimates adapter: disabled without a key, FMP parsing, PIT features (no network)."""
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import estimates as es


class _Resp:
    def __init__(self, data, code=200):
        self.data, self.status_code = data, code

    def json(self):
        return self.data


class _Session:
    def __init__(self, by_symbol):
        self.by_symbol, self.calls = by_symbol, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        sym = url.rstrip("/").split("/")[-1]
        return _Resp(*self.by_symbol[sym]) if sym in self.by_symbol else _Resp([], 404)


def test_disabled_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(es, "_env_key", lambda name="FMP_API_KEY": None)
    p = es.FMPEstimates(log=lambda *a: None)
    assert not p.enabled
    assert p.current_consensus(["AAPL"]).empty
    assert es.SnapshotStore(str(tmp_path)).record(p, ["AAPL"], "2026-01-02") == 0


def test_fmp_parsing_and_failed_symbols():
    sess = _Session({"AAA": ([{"date": "2026-03-31", "estimatedEpsAvg": 1.2, "estimatedEpsHigh": 1.5,
                               "estimatedEpsLow": 1.0, "numberAnalystsEstimatedEps": 7}],)})
    p = es.FMPEstimates(api_key="test", session=sess, log=lambda *a: None)
    df = p.current_consensus(["AAA", "BBB"])
    assert list(df["ticker"]) == ["AAA"] and df["n_analysts"].iloc[0] == 7
    assert sess.calls[0][1]["period"] == "quarter"


def _snaps():
    rows = []
    for asof, eps in [("2026-01-02", 1.00), ("2026-02-03", 1.10), ("2026-04-06", 1.32)]:
        rows.append({"ticker": "AAA", "asof_date": asof, "period_end": "2026-06-30", "eps_avg": eps,
                     "eps_high": eps + 0.2, "eps_low": eps - 0.2, "n_analysts": 5})
        rows.append({"ticker": "AAA", "asof_date": asof, "period_end": "2026-03-31", "eps_avg": 0.9,
                     "eps_high": 1.0, "eps_low": 0.8, "n_analysts": 6})
    return pd.DataFrame(rows)


def test_features_are_point_in_time_and_revisions():
    keys = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "ZZZ"],
                         "Date": ["2025-12-31", "2026-02-10", "2026-04-10", "2026-04-10"]})
    f = es.estimate_features(_snaps(), keys)
    assert f.iloc[0].isna().all() and f.iloc[3].isna().all()      # before any snapshot / unknown name
    # 2026-02-10: latest snapshot 02-03 -> next period after it = 2026-03-31 (eps 0.9; unchanged vs 01-02)
    assert np.isclose(f.loc[1, "est_eps"], 0.9) and f.loc[1, "est_rev_1m"] == 0.0 and np.isnan(f.loc[1, "est_rev_3m"])
    # 2026-04-10: snapshot 04-06 -> next period 2026-06-30; 1m-ago snapshot = 02-03 (1.10), 3m-ago = 01-02 (1.00)
    assert np.isclose(f.loc[2, "est_eps"], 1.32)
    assert np.isclose(f.loc[2, "est_rev_1m"], (1.32 - 1.10) / 1.10)
    assert np.isclose(f.loc[2, "est_rev_3m"], (1.32 - 1.00) / 1.00)
    assert np.isclose(f.loc[2, "est_dispersion"], 0.4 / 1.32) and f.loc[2, "est_n_analysts"] == 5


def test_snapshot_store_roundtrip(tmp_path):
    sess = _Session({"AAA": ([{"date": "2026-06-30", "estimatedEpsAvg": 1.0, "estimatedEpsHigh": 1.1,
                               "estimatedEpsLow": 0.9, "numberAnalystsEstimatedEps": 3}],)})
    p = es.FMPEstimates(api_key="test", session=sess, log=lambda *a: None)
    st = es.SnapshotStore(str(tmp_path))
    assert st.record(p, ["AAA"], "2026-05-01") == 1
    snap = st.load()
    assert list(snap.columns) == es.SNAPSHOT_COLUMNS and snap["asof_date"].iloc[0] == "2026-05-01"
    f = es.estimate_features(snap, pd.DataFrame({"ticker": ["AAA"], "Date": ["2026-05-04"]}))
    assert np.isclose(f.loc[0, "est_eps"], 1.0)


# --------------------------------------------------------------------------- #
# Yahoo provider (recorded fixture, no network), storage, PIT, workflow
# --------------------------------------------------------------------------- #
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "yahoo_earnings_trend.json")
WORKFLOW = os.path.join(es.REPO_ROOT, ".github", "workflows", "estimates_snapshot.yml")


def _fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def test_parse_yahoo_trend_fixture():
    fx = _fixture()
    rows = es.parse_yahoo_trend("MSFT", fx["MSFT"])
    assert [r["period"] for r in rows] == ["0q", "+1q", "0y", "+1y"]
    raw0 = fx["MSFT"][0]
    r0 = rows[0]
    assert r0["period_end"] == raw0["endDate"] and r0["source"] == "yahoo" and r0["provider_symbol"] == "MSFT"
    assert r0["eps_avg"] == raw0["earningsEstimate"]["avg"]["raw"]
    assert r0["n_analysts"] == raw0["earningsEstimate"]["numberOfAnalysts"]["raw"]
    assert r0["eps_trend_30d"] == raw0["epsTrend"]["30daysAgo"]["raw"]
    assert r0["eps_trend_90d"] == raw0["epsTrend"]["90daysAgo"]["raw"]
    assert r0["eps_up_30d"] == raw0["epsRevisions"]["upLast30days"]["raw"]
    assert r0["eps_down_7d"] == raw0["epsRevisions"]["downLast7Days"]["raw"]
    assert r0["revenue_avg"] == raw0["revenueEstimate"]["avg"]["raw"]
    assert r0["eps_currency"] == "USD"
    # a period no revenue analyst covers: Yahoo's 0 placeholders become NaN, the row stays (EPS present)
    brk = {r["period"]: r for r in es.parse_yahoo_trend("BRK.B", fx["BRK-B"])}
    assert brk["0q"]["provider_symbol"] == "BRK-B"
    assert fx["BRK-B"][0]["revenueEstimate"]["numberOfAnalysts"]["raw"] == 0
    assert np.isnan(brk["0q"]["revenue_avg"]) and not np.isnan(brk["0q"]["eps_avg"])
    # stale shell (2011 endDate, empty fields, zero revenue placeholders) -> no rows
    assert es.parse_yahoo_trend("AAME", fx["AAME"]) == []
    assert es.parse_yahoo_trend("X", []) == [] and es.parse_yahoo_trend("X", None) == []


def test_yahoo_provider_retries_and_tolerates_failures():
    fx, calls = _fixture(), []

    def fetch(sym):
        calls.append(sym)
        if sym == "FLAKY" and calls.count("FLAKY") < 2:
            raise ConnectionError("reset")
        if sym == "DEAD":
            raise RuntimeError("boom")
        return {"MSFT": fx["MSFT"], "BRK-B": fx["BRK-B"], "FLAKY": fx["MSFT"]}.get(sym, [])

    p = es.YahooEstimates(rate_per_s=0, workers=2, retries=2, backoff_s=0, fetch=fetch, log=lambda *a: None)
    df = p.current_consensus(["MSFT", "BRK.B", "FLAKY", "DEAD", "NONE"])
    assert set(df["ticker"]) == {"MSFT", "BRK.B", "FLAKY"}
    assert "BRK-B" in calls and calls.count("DEAD") == 3 and calls.count("FLAKY") == 2
    assert p.stats["requested"] == 5 and p.stats["ok"] == 3 and p.stats["empty"] == 1 and p.stats["error"] == 1
    assert df["eps_avg"].dtype == float


def test_yahoo_provider_deadline():
    p = es.YahooEstimates(rate_per_s=0, backoff_s=0, fetch=lambda s: [], deadline=0.0, log=lambda *a: None)
    assert p.current_consensus(["A", "B"]).empty and p.stats["skipped_deadline"] == 2


def test_next_weekday_and_finalize():
    assert es.next_weekday("2026-11-06") == "2026-11-09"     # Fri -> Mon
    assert es.next_weekday("2026-11-04") == "2026-11-05"
    cur = pd.DataFrame(es.parse_yahoo_trend("MSFT", _fixture()["MSFT"]))
    snap = es.finalize_snapshot(cur, "2026-11-06")
    assert list(snap.columns) == es.SNAPSHOT_COLUMNS
    assert set(snap["available_date"]) == {"2026-11-09"} and set(snap["asof_date"]) == {"2026-11-06"}


def test_daily_layout_and_legacy_compat(tmp_path):
    cur = pd.DataFrame(es.parse_yahoo_trend("MSFT", _fixture()["MSFT"]))
    p = es.write_daily(es.finalize_snapshot(cur, "2026-11-06"), str(tmp_path), "2026-11-06")
    assert p.endswith(os.path.join("estimates", "2026", "2026-11-06.parquet"))
    _snaps().query("asof_date == '2026-01-02'").to_parquet(tmp_path / "snapshot_2026-01-02.parquet", index=False)
    (tmp_path / "estimates" / "2026" / "notes.parquet").write_bytes(b"")   # ignored: not a dated file
    df = es.load_dir(str(tmp_path))
    assert list(df.columns) == es.SNAPSHOT_COLUMNS and len(df) == 4 + 2
    legacy = df[df["source"].isna()]
    assert len(legacy) == 2 and legacy["available_date"].isna().all() and legacy["eps_trend_30d"].isna().all()


def _ysnap(asof, period, period_end, eps, trend=None, up=(0, 0), down=(0, 0), n=10, ticker="AAA"):
    trend = trend or {}
    return {"ticker": ticker, "asof_date": asof, "available_date": es.next_weekday(asof), "source": "yahoo",
            "period": period, "period_end": period_end, "eps_avg": eps, "eps_high": eps + 0.1,
            "eps_low": eps - 0.1, "n_analysts": n, "eps_trend_current": eps,
            "eps_trend_7d": trend.get("7d"), "eps_trend_30d": trend.get("30d"), "eps_trend_90d": trend.get("90d"),
            "eps_up_7d": up[0], "eps_up_30d": up[1], "eps_down_7d": down[0], "eps_down_30d": down[1]}


def test_yahoo_features_pit_vendor_and_own_revisions():
    rows = [
        # 2026-08-03: +1q = fiscal quarter ending 12-31; FY also ends 12-31 (same date, other horizon)
        _ysnap("2026-08-03", "0q", "2026-09-30", 1.00),
        _ysnap("2026-08-03", "+1q", "2026-12-31", 2.00),
        _ysnap("2026-08-03", "0y", "2026-12-31", 5.00),
        # 2026-10-02: still before the Q3 report
        _ysnap("2026-10-02", "0q", "2026-09-30", 1.05),
        _ysnap("2026-10-02", "+1q", "2026-12-31", 2.10),
        _ysnap("2026-10-02", "0y", "2026-12-31", 5.50),
        # 2026-11-06 (Fri): after the report, 0q rolled to the 12-31 quarter
        _ysnap("2026-11-06", "0q", "2026-12-31", 2.20, trend={"7d": 2.0, "30d": 2.2, "90d": 1.1},
               up=(3, 6), down=(1, 2), n=8),
        _ysnap("2026-11-06", "0y", "2026-12-31", 6.00, trend={"30d": 5.0, "90d": 6.0}, up=(0, 1), down=(0, 5), n=8),
    ]
    snaps = pd.DataFrame(rows)
    keys = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "AAA"],
                         "Date": ["2026-11-06", "2026-11-09", "2026-08-03", "2026-12-31"]})
    f = es.estimate_features(snaps, keys)
    assert set(f.columns) == set(es.ALL_FEATURE_COLUMNS)
    # Friday's snapshot is only available on Monday; on Fri the latest usable one (10-02) is > 14 days old
    assert f.iloc[0].isna().all()
    # a snapshot is never usable on its own asof date
    assert f.iloc[2].isna().all()
    # max_age: 12-31 is 55 days after the last snapshot
    assert f.iloc[3].isna().all()
    m = f.iloc[1]
    assert np.isclose(m["est_eps"], 2.20) and m["est_n_analysts"] == 8 and m["est_age_days"] == 3
    assert np.isclose(m["est_dispersion"], 0.2 / 2.2)
    # own revisions: same fiscal quarter (12-31), quoted as +1q earlier -> 0q rollover handled
    assert np.isclose(m["est_rev_1m"], (2.20 - 2.10) / 2.10)
    assert np.isclose(m["est_rev_3m"], (2.20 - 2.00) / 2.00)
    # FY revisions use the annual horizon only (not the quarterly rows with the same period_end)
    assert np.isclose(m["est_rev_fy_1m"], (6.0 - 5.5) / 5.5) and np.isclose(m["est_rev_fy_3m"], (6.0 - 5.0) / 5.0)
    # vendor-reported (one snapshot)
    assert np.isclose(m["estv_rev_7d"], 0.2 / 2.0) and np.isclose(m["estv_rev_30d"], 0.0)
    assert np.isclose(m["estv_rev_90d"], (2.2 - 1.1) / 1.1)
    assert np.isclose(m["estv_rev_fy_30d"], 0.2) and np.isclose(m["estv_rev_fy_90d"], 0.0)
    assert np.isclose(m["estv_breadth_7d"], 2 / 8) and np.isclose(m["estv_breadth_30d"], 4 / 8)
    assert np.isclose(m["estv_breadth_fy_30d"], -4 / 8)
    # no age limit -> Friday uses the 10-02 snapshot (0q 09-30; 1m-older = 08-03)
    g = es.estimate_features(snaps, keys.iloc[[0]], max_age_days=None)
    assert np.isclose(g.iloc[0]["est_eps"], 1.05) and np.isclose(g.iloc[0]["est_rev_1m"], 0.05)
    assert np.isnan(g.iloc[0]["est_rev_3m"]) and np.isnan(g.iloc[0]["estv_rev_7d"])
    # rev_tolerance_days caps how much older than the lookback the comparison snapshot may be
    h = es.estimate_features(snaps, keys.iloc[[0]], max_age_days=None, rev_tolerance_days=10)
    assert np.isnan(h.iloc[0]["est_rev_1m"])


def test_features_ignore_future_snapshots_exhaustively():
    """Each row's features equal those computed from snapshots truncated to
    available_date <= Date (no look-ahead)."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-01-02", "2026-06-30")
    rows = []
    for d in days[::3]:
        for t in ("AAA", "BBB"):
            pe = "2026-03-31" if d < pd.Timestamp("2026-04-20") else "2026-06-30"
            a = d.date().isoformat()
            rows.append(_ysnap(a, "0q", pe, float(1 + rng.normal(0, .1)), trend={"7d": 1.0, "30d": 1.0, "90d": 1.0},
                               ticker=t))
            rows.append(_ysnap(a, "+1q", "2026-06-30", float(1.5 + rng.normal(0, .1)), ticker=t))
            rows.append(_ysnap(a, "0y", "2026-12-31", float(4 + rng.normal(0, .1)), ticker=t))
    snaps = pd.DataFrame(rows)
    keys = pd.DataFrame({"ticker": ["AAA", "BBB"] * 30,
                         "Date": [d.date().isoformat() for d in days[::4][:30] for _ in (0, 1)]})
    full = es.estimate_features(snaps, keys)
    for i, d in enumerate(keys["Date"]):
        part = es.estimate_features(snaps[snaps["available_date"] <= d], keys.iloc[[i]])
        pd.testing.assert_series_equal(full.iloc[i], part.iloc[0], check_names=False)
    assert full["est_rev_1m"].notna().any() and full["est_rev_3m"].notna().any()


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_load_branch_snapshots(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    cfg = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    _git("init", "-q", "-b", "main", cwd=origin)
    (origin / "x.txt").write_text("main")
    _git("add", "x.txt", cwd=origin)
    _git(*cfg, "commit", "-qm", "main", cwd=origin)
    _git("checkout", "-q", "--orphan", "data/estimates", cwd=origin)
    _git("rm", "-rfq", ".", cwd=origin)
    cur = pd.DataFrame(es.parse_yahoo_trend("MSFT", _fixture()["MSFT"]))
    for asof in ("2026-11-05", "2026-11-06"):
        es.write_daily(es.finalize_snapshot(cur, asof), str(origin), asof)
    _git("add", "estimates", cwd=origin)
    _git(*cfg, "commit", "-qm", "snaps", cwd=origin)
    _git("checkout", "-q", "main", cwd=origin)
    clone = tmp_path / "clone"
    _git("clone", "-q", "--branch", "main", str(origin), str(clone), cwd=tmp_path)
    df = es.load_branch_snapshots(repo=str(clone), fetch=True)
    assert sorted(df["asof_date"].unique()) == ["2026-11-05", "2026-11-06"] and len(df) == 8
    assert list(df.columns) == es.SNAPSHOT_COLUMNS
    assert set(es.load_branch_snapshots(repo=str(clone), fetch=False, since="2026-11-06")["asof_date"]) \
        == {"2026-11-06"}
    with pytest.raises(FileNotFoundError):
        es.load_branch_snapshots(repo=str(clone), branch="nope", fetch=False)


def test_snapshot_universe(tmp_path):
    uf = tmp_path / "u.txt"
    uf.write_text("# header\nAAA\nBRK.B\n\nbad-sym\n")
    pdir = tmp_path / "preds" / "monthly"
    pdir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["OLD"]}).to_csv(pdir / "2026-10-01.csv.gz", index=False)
    pd.DataFrame({"ticker": ["CCC", "AAA"]}).to_csv(pdir / "2026-11-05.csv.gz", index=False)
    assert es.snapshot_universe(str(uf), str(tmp_path / "preds" / "*" / "*.csv.gz")) == ["AAA", "BRK.B", "CCC"]


def test_committed_universe_file_is_sane():
    u = es.snapshot_universe(es.UNIVERSE_FILE, "/nonexistent/*.csv.gz")
    assert len(u) > 1000 and "AAPL" in u and len(set(u)) == len(u)


def test_workflow_yaml_sanity():
    yaml = pytest.importorskip("yaml")
    with open(WORKFLOW) as f:
        text = f.read()
    wf = yaml.safe_load(text)
    on = wf.get("on", wf.get(True))                     # PyYAML parses a bare `on` key as True
    cron = on["schedule"][0]["cron"].split()
    assert cron[4] == "1-5" and int(cron[1]) >= 21       # weekdays, after the US close (UTC)
    assert "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "write"}
    job = wf["jobs"]["snapshot"]
    assert job["timeout-minutes"] < 60
    assert job["env"]["DATA_BRANCH"] == es.DATA_BRANCH
    run = "\n".join(s.get("run", "") for s in job["steps"])
    assert "DataAnalysisPipeline2.v3.estimates snapshot" in run and "--orphan" in run
    assert "yfinance==1.2.0" in run
    assert "secrets." not in text                        # no secrets needed
    assert int(re.search(r"--max-minutes (\d+)", run).group(1)) < job["timeout-minutes"] - 5
