"""Live-model monitoring (WS5): offline tests on synthetic data, no network."""
import gzip
import json
import types

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import monitoring as mon
from DataAnalysisPipeline2.trading_agent import scheduled_rebalance as sr

DATES = pd.bdate_range("2026-01-02", "2026-04-30").strftime("%Y-%m-%d").tolist()
D1, D2, D3 = "2026-01-30", "2026-02-27", "2026-03-31"


def _panel(n=50, seed=0, signal=1.0):
    """Synthetic prices whose per-period return is ``signal * true_rank`` + noise.

    Returns (prices, true_score) where true_score is a [0, 1] rank per ticker."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    score = pd.Series(np.linspace(0, 1, n), index=tickers)
    rows = []
    for t in tickers:
        daily = 0.002 * signal * (score[t] - 0.5) + rng.normal(0, 0.0005, len(DATES))
        px = 100 * np.cumprod(1 + daily)
        rows += [{"ticker": t, "Date": d, "company_close": p} for d, p in zip(DATES, px)]
    return pd.DataFrame(rows), score


def _bench(drift=0.0005):
    return pd.DataFrame({"Date": DATES, "Close": 400 * np.cumprod(np.full(len(DATES), 1 + drift))})


def _record(tmp_path, date, score, weights, model_id="m1", account="monthly"):
    preds = pd.DataFrame({"ticker": score.index, "pred_rank": score.values})
    return mon.record_predictions(preds, weights, date, account, model_id, base_dir=tmp_path)


# --------------------------------------------------------------------------- #
# record_predictions
# --------------------------------------------------------------------------- #
def test_record_predictions_format_and_determinism(tmp_path):
    preds = pd.DataFrame({"ticker": ["MSFT", "AAPL", "XOM"], "pred_rank": [0.9, 0.123456789, 0.1],
                          "company_close": [1, 2, 3]})
    w = {"MSFT": 0.5, "AAPL": 0.25, "ZZZ": 0.25}  # ZZZ held but not scored
    p = mon.record_predictions(preds, w, "2026-11-04", "Monthly", "abc", base_dir=tmp_path)
    assert p == tmp_path / "predictions" / "monthly" / "2026-11-04.csv.gz"
    df = pd.read_csv(p, compression="gzip")
    assert list(df.columns) == ["ticker", "pred_rank", "weight", "model_id"]
    assert df["ticker"].tolist() == ["AAPL", "MSFT", "XOM", "ZZZ"]  # sorted
    assert df.set_index("ticker")["weight"].to_dict() == {"AAPL": 0.25, "MSFT": 0.5, "XOM": 0.0, "ZZZ": 0.25}
    assert df.loc[df.ticker == "AAPL", "pred_rank"].item() == pytest.approx(0.123457)
    assert np.isnan(df.loc[df.ticker == "ZZZ", "pred_rank"].item())
    assert set(df["model_id"]) == {"abc"}
    first = p.read_bytes()
    mon.record_predictions(preds, w, "2026-11-04", "monthly", "abc", base_dir=tmp_path)
    assert p.read_bytes() == first  # byte-identical (gzip mtime fixed), overwrite in place
    with gzip.open(p, "rt") as f:
        assert f.readline().strip() == "ticker,pred_rank,weight,model_id"


def test_record_predictions_requires_columns(tmp_path):
    with pytest.raises(ValueError):
        mon.record_predictions(pd.DataFrame({"ticker": ["A"]}), {}, "2026-01-01", "x", "m", base_dir=tmp_path)


def test_model_id_from_manifest_or_hash(tmp_path):
    assert mon.model_id_from({"model_id": "v2final"}) == "v2final"
    f = tmp_path / "m.pkl"
    f.write_bytes(b"model")
    assert mon.model_id_from({}, str(f)).startswith("sha256:")
    assert mon.model_id_from(None, str(tmp_path / "missing.pkl")) == "unknown"


# --------------------------------------------------------------------------- #
# realized_metrics
# --------------------------------------------------------------------------- #
def test_ic_and_spread_on_perfect_signal(tmp_path):
    prices, score = _panel(signal=1.0)
    held = {t: 0.1 for t in score.index[-10:]}  # top 10 names
    f = _record(tmp_path, D1, score, held)
    m = mon.realized_metrics(f, prices, _bench(), next_decision_date=D2)
    assert m["status"] == "closed" and m["end_date"] == D2
    assert m["n_eligible"] == 50 and m["n_held"] == 10
    assert m["ic"] > 0.95
    assert m["spread_d10_d1"] > 0
    assert m["book_return"] > m["ew_return"]  # holding the top decile beats EW
    assert m["book_coverage"] == pytest.approx(1.0)
    # book return is exactly the weighted sum of realised name returns
    sub = prices.set_index(["ticker", "Date"])["company_close"]
    r = {t: sub[(t, D2)] / sub[(t, D1)] - 1 for t in held}
    assert m["book_return"] == pytest.approx(sum(0.1 * v for v in r.values()))
    allr = [sub[(t, D2)] / sub[(t, D1)] - 1 for t in score.index]
    assert m["ew_return"] == pytest.approx(np.mean(allr))
    b = _bench().set_index("Date")["Close"]
    assert m["spy_return"] == pytest.approx(b[D2] / b[D1] - 1)
    assert m["excess_vs_spy"] == pytest.approx(m["book_return"] - m["spy_return"])


def test_ic_negative_when_signal_inverted(tmp_path):
    prices, score = _panel(signal=-1.0)
    f = _record(tmp_path, D1, score, {})
    m = mon.realized_metrics(f, prices, None, next_decision_date=D2)
    assert m["ic"] < -0.95 and m["spread_d10_d1"] < 0
    assert np.isnan(m["book_return"]) and np.isnan(m["spy_return"])


def test_spread_matches_manual_deciles(tmp_path):
    rng = np.random.default_rng(3)
    tickers = [f"S{i:02d}" for i in range(40)]
    rets = pd.Series(rng.normal(0, 0.05, 40), index=tickers)
    prices = pd.concat([
        pd.DataFrame({"ticker": tickers, "Date": D1, "company_close": 100.0}),
        pd.DataFrame({"ticker": tickers, "Date": D2, "company_close": 100.0 * (1 + rets.values)}),
    ])
    score = pd.Series(np.arange(40) / 39, index=tickers)
    f = _record(tmp_path, D1, score, {})
    m = mon.realized_metrics(f, prices, None, next_decision_date=D2)
    assert m["spread_d10_d1"] == pytest.approx(rets.iloc[-4:].mean() - rets.iloc[:4].mean())
    from scipy.stats import spearmanr
    assert m["ic"] == pytest.approx(spearmanr(score, rets)[0])


def test_no_lookahead_period_boundaries(tmp_path):
    """Only close(D) (base) and bars in (D, E] are read: perturbing prices
    before D or after E must not change anything."""
    prices, score = _panel()
    f = _record(tmp_path, D2, score, {t: 0.05 for t in score.index[:20]})
    base = mon.realized_metrics(f, prices, _bench(), next_decision_date=D3)

    tampered = prices.copy()
    before = tampered["Date"] < D2
    after = tampered["Date"] > D3
    tampered.loc[before | after, "company_close"] *= np.random.default_rng(9).uniform(0.2, 5, (before | after).sum())
    b = _bench()
    b.loc[(b["Date"] < D2) | (b["Date"] > D3), "Close"] *= 3
    again = mon.realized_metrics(f, tampered, b, next_decision_date=D3)
    for k in ("ic", "spread_d10_d1", "book_return", "ew_return", "spy_return", "end_date"):
        assert again[k] == pytest.approx(base[k]) if isinstance(base[k], float) else again[k] == base[k]

    # Changing a bar INSIDE the period does change the result (sanity).
    inside = prices.copy()
    inside.loc[(inside["Date"] == D3) & (inside["ticker"] == score.index[0]), "company_close"] *= 2
    assert mon.realized_metrics(f, inside, None, next_decision_date=D3)["book_return"] != \
        pytest.approx(base["book_return"])


def test_open_period_uses_latest_bar_not_beyond_today(tmp_path):
    prices, score = _panel()
    f = _record(tmp_path, D3, score, {})
    m = mon.realized_metrics(f, prices, None, today="2026-04-15")
    assert m["status"] == "open" and m["end_date"] == "2026-04-15"
    assert mon.realized_metrics(f, prices, None, today=D3) is None  # nothing after D yet


def test_delisted_name_uses_last_bar_and_missing_base_is_excluded(tmp_path):
    prices, score = _panel()
    gone = score.index[0]
    prices = prices[~((prices["ticker"] == gone) & (prices["Date"] > "2026-02-10"))]
    newbie = score.index[1]
    prices = prices[~((prices["ticker"] == newbie) & (prices["Date"] <= D1))]  # no base close
    f = _record(tmp_path, D1, score, {gone: 0.5, newbie: 0.5})
    m = mon.realized_metrics(f, prices, None, next_decision_date=D2)
    assert m["n_eligible"] == 49
    assert m["book_coverage"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def test_psi_math():
    assert mon.psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    e, a = np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.1, 0.2, 0.3, 0.4])
    assert mon.psi(e, a) == pytest.approx(float(np.sum((a - e) * np.log(a / e))))


def test_feature_profile_and_drift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({f"f{i}": rng.normal(0, 1, 20000) for i in range(5)})
    prof = mon.build_feature_profile(train, n_bins=10)
    spec = prof["features"]["f0"]
    assert len(spec["edges"]) == 9 and len(spec["fractions"]) == 10
    assert sum(spec["fractions"]) == pytest.approx(1.0)
    assert all(f == pytest.approx(0.1, abs=0.01) for f in spec["fractions"])
    json.dumps(prof)  # manifest-serialisable

    live = pd.DataFrame({f"f{i}": rng.normal(0, 1, 2000) for i in range(5)})
    live["f3"] = live["f3"] + 1.5
    live["f4"] = live["f4"] * 3
    d = mon.feature_drift(live, {"feature_profile": prof})  # accepts the manifest too
    assert d["available"]
    assert d["psi"]["f0"] < 0.05 and d["psi"]["f3"] > 0.25 and d["psi"]["f4"] > 0.25
    assert set(d["high"]) == {"f3", "f4"} and d["n_high"] == 2


def test_drift_skips_gracefully_without_profile():
    d = mon.feature_drift(pd.DataFrame({"a": [1.0]}), {"feature_version": 2})
    assert d["available"] is False and "feature_profile" in d["reason"]
    assert mon.feature_drift(pd.DataFrame({"a": [1.0]}), None)["available"] is False


def test_profile_handles_constant_and_missing_features():
    train = pd.DataFrame({"c": [0.0] * 100, "x": np.arange(100.0)})
    prof = mon.build_feature_profile(train)
    assert prof["features"]["c"]["edges"] == [0.0]
    d = mon.feature_drift(pd.DataFrame({"c": [0.0] * 10}), prof)
    assert d["psi"]["c"] == pytest.approx(0.0) and d["missing"] == ["x"]


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def _p(ic, book=0.01, ew=0.01, status="closed"):
    return {"status": status, "ic": ic, "book_return": book, "ew_return": ew}


def test_alert_on_negative_rolling_ic():
    assert mon.evaluate_alerts([_p(0.05), _p(-0.02), _p(-0.03), _p(-0.01)])[0]["rule"] == "ic"
    assert mon.evaluate_alerts([_p(0.05), _p(-0.02), _p(0.01)]) == []  # mean > 0
    assert mon.evaluate_alerts([_p(-0.05), _p(-0.02)]) == []          # too few periods
    # the open (partial) period is not counted
    assert mon.evaluate_alerts([_p(0.05), _p(0.04), _p(0.03), _p(-0.9, status="open")]) == []


def test_alert_on_feature_drift():
    drift = {"available": True, "psi": {"a": 0.3, "b": 0.26, "c": 0.5, "d": 0.1}}
    assert [a["rule"] for a in mon.evaluate_alerts([], drift)] == ["drift"]
    drift["psi"]["c"] = 0.25  # threshold is strict (> 0.25)
    assert mon.evaluate_alerts([], drift) == []
    assert mon.evaluate_alerts([], {"available": False, "psi": {}}) == []


def test_alert_on_underperformance_vs_equal_weight():
    under = [_p(0.01, book=-0.03, ew=0.01)] * 3  # ~ -12 pp compounded
    assert [a["rule"] for a in mon.evaluate_alerts(under)] == ["underperformance"]
    mild = [_p(0.01, book=-0.02, ew=0.01)] * 3   # ~ -9 pp
    assert mon.evaluate_alerts(mild) == []


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fetchers(prices, calls):
    def pf(tickers, start):
        calls.append(("prices", start, len(tickers)))
        return prices
    def bf(start):
        return _bench()
    return pf, bf


def test_update_report_idempotent_and_keeps_closed_periods(tmp_path):
    prices, score = _panel()
    for d in (D1, D2, D3):
        _record(tmp_path, d, score, {t: 0.1 for t in score.index[-10:]})
    mon.record_drift({"available": False, "reason": "no feature_profile", "psi": {}}, D3, "monthly",
                     base_dir=tmp_path)
    calls = []
    pf, bf = _fetchers(prices, calls)
    s1 = mon.update_monitoring_report("monthly", base_dir=tmp_path, prices_fetcher=pf,
                                      benchmark_fetcher=bf, today="2026-04-15")
    csv = pd.read_csv(tmp_path / "monitoring.csv")
    assert len(csv) == 3 and list(csv["status"]) == ["closed", "closed", "open"]
    assert s1["n_periods"] == 3 and s1["n_recomputed"] == 3
    assert s1["alerts"] == [] and s1["rolling_ic_3"] is None  # only 2 closed periods
    md = (tmp_path / "monitoring.md").read_text()
    assert "Account `monthly`" in md and "Feature drift" in md and "skipped" in md

    s2 = mon.update_monitoring_report("monthly", base_dir=tmp_path, prices_fetcher=pf,
                                      benchmark_fetcher=bf, today="2026-04-20")
    csv2 = pd.read_csv(tmp_path / "monitoring.csv")
    assert len(csv2) == 3                       # no duplicated rows
    assert s2["n_recomputed"] == 1              # only the open period is refetched
    assert calls[-1][1] > D2                    # ...from just before its decision date
    assert csv2.iloc[:2].equals(csv.iloc[:2])   # closed rows untouched
    assert csv2.iloc[2]["end_date"] == "2026-04-20"

    # A new decision closes the previously open period.
    _record(tmp_path, "2026-04-20", score, {})
    mon.update_monitoring_report("monthly", base_dir=tmp_path, prices_fetcher=pf,
                                 benchmark_fetcher=bf, today="2026-04-30")
    csv3 = pd.read_csv(tmp_path / "monitoring.csv")
    assert list(csv3["status"]) == ["closed", "closed", "closed", "open"]
    assert csv3.iloc[2]["end_date"] == "2026-04-20"


def test_update_report_preserves_other_accounts_and_raises_alerts(tmp_path):
    prices, score = _panel(signal=-1.0)  # inverted model -> negative IC
    for d in (D1, D2, D3, "2026-04-20"):
        _record(tmp_path, d, score, {}, account="weekly")
    _record(tmp_path, D1, score, {}, account="monthly")
    pf, bf = _fetchers(prices, [])
    mon.update_monitoring_report("monthly", base_dir=tmp_path, prices_fetcher=pf,
                                 benchmark_fetcher=bf, today="2026-04-30")
    s = mon.update_monitoring_report("weekly", base_dir=tmp_path, prices_fetcher=pf,
                                     benchmark_fetcher=bf, today="2026-04-30")
    csv = pd.read_csv(tmp_path / "monitoring.csv")
    assert sorted(csv["account"].value_counts().items()) == [("monthly", 1), ("weekly", 4)]
    assert [a["rule"] for a in s["alerts"]] == ["ic"]
    md = (tmp_path / "monitoring.md").read_text()
    assert "Account `monthly`" in md and "Account `weekly`" in md and "ALERT [ic]" in md


def test_update_report_with_no_predictions(tmp_path):
    s = mon.update_monitoring_report("monthly", base_dir=tmp_path,
                                     prices_fetcher=lambda *a: pytest.fail("no fetch expected"),
                                     benchmark_fetcher=lambda *a: pytest.fail("no fetch expected"))
    assert s["n_periods"] == 0 and s["alerts"] == []
    assert (tmp_path / "monitoring.md").exists()


# --------------------------------------------------------------------------- #
# Scheduler integration (pure helpers)
# --------------------------------------------------------------------------- #
def test_scheduler_monitor_gating_and_exit_codes():
    a = types.SimpleNamespace(monitor=False, execute=True)
    assert sr.should_monitor(a, {"decision": "trade", "result": {}})
    assert not sr.should_monitor(a, {"decision": "skip"})
    assert not sr.should_monitor(types.SimpleNamespace(monitor=False, execute=False),
                                 {"decision": "trade"})  # check-only "would trade"
    assert sr.should_monitor(types.SimpleNamespace(monitor=True, execute=False), {})
    alerts = [{"rule": "ic", "message": "x"}]
    assert sr.monitor_exit_code(0, alerts, False) == 0
    assert sr.monitor_exit_code(0, alerts, True) == sr.EXIT_ALERT
    assert sr.monitor_exit_code(2, alerts, True) == 2
    assert sr.monitor_exit_code(0, [], True) == 0


def test_run_monitoring_never_raises():
    def boom(account):
        raise RuntimeError("no data")
    s = sr.run_monitoring("monthly", updater=boom)
    assert s["alerts"] == [] and "no data" in s["error"]


# --------------------------------------------------------------------------- #
# run.py hook
# --------------------------------------------------------------------------- #
def test_run_hook_records_predictions_and_drift(tmp_path, monkeypatch):
    from DataAnalysisPipeline2.trading_agent import run
    monkeypatch.setenv("BDA_OPS_DIR", str(tmp_path))
    monkeypatch.setenv("ACTIVE_ACCOUNT", "monthly")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f": rng.normal(0, 1, 30)})
    agent = types.SimpleNamespace(
        model_metadata={"model_id": "v2final", "feature_profile": mon.build_feature_profile(X)},
        model_data={},
        last_predictions_df=pd.DataFrame({"ticker": [f"T{i}" for i in range(30)],
                                          "pred_rank": np.linspace(0, 1, 30)}),
        last_feature_matrix=X)
    prices_df = pd.DataFrame({"Date": ["2026-11-03", "2026-11-04"]})
    out = run._record_for_monitoring(agent, None, {"T29": 0.5}, prices_df)
    assert out["model_id"] == "v2final"
    f = tmp_path / "predictions" / "monthly" / "2026-11-04.csv.gz"
    assert f.exists() and pd.read_csv(f).set_index("ticker").loc["T29", "weight"] == 0.5
    drift = json.loads((tmp_path / "predictions" / "monthly" / "2026-11-04.drift.json").read_text())
    assert drift["available"] and drift["psi"]["f"] == pytest.approx(0.0, abs=1e-6)


def test_run_hook_never_raises(monkeypatch, tmp_path):
    from DataAnalysisPipeline2.trading_agent import run
    monkeypatch.setenv("BDA_OPS_DIR", str(tmp_path))
    agent = types.SimpleNamespace(model_metadata={}, model_data=None,
                                  last_predictions_df=None, last_feature_matrix=None)
    out = run._record_for_monitoring(agent, None, {}, pd.DataFrame({"Date": []}))
    assert "predictions_file" not in out


def test_feature_profile_null_calibration_suppresses_single_date_artefacts():
    """A per-date constant feature (like sector_return_5d after z-scoring) has a
    huge single-date PSI with no drift; the calibrated null level absorbs it,
    while a genuinely shifted feature is still flagged."""
    import numpy as np
    import pandas as pd
    from DataAnalysisPipeline2.trading_agent import monitoring as mon
    rng = np.random.default_rng(0)
    n_d, n = 60, 300
    dates = np.repeat(np.arange(n_d), n)
    sector = rng.integers(0, 10, n_d * n)
    day_sector = rng.normal(size=(n_d, 10))
    train = pd.DataFrame({"sec": day_sector[dates, sector], "x": rng.normal(size=n_d * n)})
    prof = mon.build_feature_profile(train, dates=dates)
    assert prof["features"]["sec"]["psi_null_q"] > 0.25
    live_sector = rng.integers(0, 10, n)
    live = pd.DataFrame({"sec": rng.normal(size=10)[live_sector], "x": rng.normal(loc=1.5, size=n)})
    d = mon.feature_drift(live, prof)
    assert d["psi"]["sec"] > 0.25            # raw PSI would have flagged it
    assert "sec" not in d["high"] and "x" in d["high"]
    alerts = mon.evaluate_alerts([], d, max_drift_features=1)
    assert any(a["rule"] == "drift" for a in alerts)
    assert "sec" not in [a for a in alerts if a["rule"] == "drift"][0]["message"]
