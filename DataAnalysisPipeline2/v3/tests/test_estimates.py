"""Analyst-estimates adapter: disabled without a key, FMP parsing, PIT features (no network)."""
import numpy as np
import pandas as pd

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
