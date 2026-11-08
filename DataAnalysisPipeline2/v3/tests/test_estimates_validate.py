"""Vendor-revision validation (``python -m DataAnalysisPipeline2.v3.estimates validate``):
Yahoo's "7/30/90 days ago" consensus vs our own snapshot of that day (synthetic, no network)."""
import json

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import estimates as es


def _snap(asof, period, period_end, eps, ago=None, ticker="AAA"):
    ago = ago or {}
    return {"ticker": ticker, "asof_date": asof, "available_date": es.next_weekday(asof), "source": "yahoo",
            "period": period, "period_end": period_end, "eps_avg": eps, "eps_high": eps + 0.1,
            "eps_low": eps - 0.1, "n_analysts": 10, "eps_trend_current": eps,
            "eps_trend_7d": ago.get("7d"), "eps_trend_30d": ago.get("30d"), "eps_trend_90d": ago.get("90d")}


def test_insufficient_history():
    snaps = pd.DataFrame([_snap("2026-11-02", "0q", "2026-12-31", 1.0),
                          _snap("2026-11-06", "0q", "2026-12-31", 1.0, ago={"7d": 1.0})])
    rep = es.validate_vendor_revisions(snaps)
    assert rep["span_days"] == 4 and rep["snapshot_days"] == 2
    assert all(v["status"] == "insufficient history" for v in rep["lags"].values())
    assert es.format_validation(rep).splitlines()[0] == "insufficient history (4 days)"
    empty = es.validate_vendor_revisions(pd.DataFrame(columns=es.SNAPSHOT_COLUMNS))
    assert es.format_validation(empty).splitlines()[0] == "insufficient history (0 days)"


def _rollover_snaps():
    return pd.DataFrame([
        # 2026-10-01 (Thu): the 12-31 quarter is still AAA's +1q
        _snap("2026-10-01", "0q", "2026-09-30", 1.00),
        _snap("2026-10-01", "+1q", "2026-12-31", 2.00),
        _snap("2026-10-01", "0y", "2026-12-31", 5.00),
        _snap("2026-10-01", "0q", "2026-12-31", 3.00, ticker="BBB"),
        _snap("2026-10-30", "0q", "2026-12-31", 2.10),
        _snap("2026-10-30", "0q", "2026-12-31", 3.30, ticker="BBB"),
        # 2026-11-06 (Fri): AAA's 0q rolled to the 12-31 quarter
        _snap("2026-11-06", "0q", "2026-12-31", 2.20, ago={"7d": 2.10, "30d": 2.0}),
        _snap("2026-11-06", "0y", "2026-12-31", 6.00, ago={"30d": 5.0}),
        _snap("2026-11-06", "0q", "2026-12-31", 3.40, ago={"7d": 3.36, "30d": 3.0}, ticker="BBB"),
    ])


def test_match_rate_bias_and_window():
    snaps = _rollover_snaps()
    rep = es.validate_vendor_revisions(snaps, lags=(7, 30, 90), abs_tol=0.005, rel_tol=0.01, window_days=3)
    assert rep["span_days"] == 36
    l7 = rep["lags"][7]
    # AAA: 2.10 vs our 2.10 on 10-30 (exact date); BBB: 3.36 vs 3.30 (1.8% > 1%) -> no match
    assert l7["n_vendor_rows"] == 2 and l7["n_pairs"] == 2 and l7["exact_date_share"] == 1.0
    assert l7["match_rate"] == 0.5
    assert np.isclose(l7["bias_mean"], 0.03) and np.isclose(l7["rel_bias_median"], (0 + 0.06 / 3.30) / 2)
    assert l7["by_period"]["0q"]["n"] == 2
    # 30d: target 10-07; our nearest snapshot (10-01) is 6 days away -> outside +/-3 d -> unpaired
    assert rep["lags"][30]["n_vendor_rows"] == 3 and rep["lags"][30]["n_pairs"] == 0
    assert rep["lags"][90]["status"] == "insufficient history"
    txt = es.format_validation(rep)
    assert "insufficient history" in txt and "match 50.0%" in txt


def test_rollover_matched_on_period_end_and_horizon():
    wide = es.validate_vendor_revisions(_rollover_snaps(), lags=(30,), window_days=7)["lags"][30]
    # AAA 0q(12-31) 30 days ago was quoted as +1q (2.00); the annual row pairs with the 0y row (5.00),
    # not with the quarterly row sharing its period_end
    assert wide["n_pairs"] == 3 and wide["match_rate"] == 1.0 and wide["exact_date_share"] == 0.0
    assert set(wide["by_period"]) == {"0q", "0y"} and wide["by_period"]["0y"]["n"] == 1
    assert wide["bias_mean"] == 0.0


def test_daily_history_rates():
    days = pd.bdate_range("2026-01-05", "2026-06-30")
    base = {"AAA": 0.0, "BBB": 1.0, "CCC": -2.0}
    truth = {(t, d): 1.0 + 0.001 * (d - days[0]).days + b for d in days for t, b in base.items()}
    rows = []
    for d in days:
        for t in base:
            ago = {f"{L}d": truth[(t, d - pd.Timedelta(days=L))] * 1.002      # +0.2% vendor bias
                   for L in (7, 30, 90) if (t, d - pd.Timedelta(days=L)) in truth}
            rows.append(_snap(d.date().isoformat(), "0q", "2026-12-31", truth[(t, d)], ago=ago, ticker=t))
    snaps = pd.DataFrame(rows)
    rep = es.validate_vendor_revisions(snaps)
    for L in (7, 30, 90):
        x = rep["lags"][L]
        assert x["status"] == "ok" and x["pair_coverage"] == 1.0 and x["exact_date_share"] == 1.0
        assert x["match_rate"] == 1.0 and np.isclose(x["rel_bias_median"], 0.002, atol=1e-6)
        assert x["n_tickers"] == 3
    bad = snaps.assign(eps_trend_7d=snaps["eps_trend_7d"] * 1.05)
    assert es.validate_vendor_revisions(bad, lags=(7,))["lags"][7]["match_rate"] == 0.0


def test_cli_local_dir(tmp_path, capsys):
    snaps = pd.DataFrame([_snap("2026-10-30", "0q", "2026-12-31", 2.10),
                          _snap("2026-11-06", "0q", "2026-12-31", 2.20, ago={"7d": 2.10})])
    for asof, g in snaps.groupby("asof_date"):
        es.write_daily(g, str(tmp_path), asof)
    out = tmp_path / "rep.json"
    assert es.main(["validate", "--dir", str(tmp_path), "--json", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "7d ago" in txt and "match 100.0%" in txt
    assert json.load(open(out))["lags"]["7"]["n_pairs"] == 1
