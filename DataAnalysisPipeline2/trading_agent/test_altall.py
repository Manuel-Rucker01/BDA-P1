"""BDA_SIGNAL=altall (Round-8 sn30_altall_hd): live==offline parity of the full
model input, snapshot freshness, artefact contract and every fallback to the
production model."""
import json
import os
import pickle
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from DataAnalysisPipeline2.trading_agent import altall, bot, config, optimized, run
from DataAnalysisPipeline2.v3 import altdata as ad
from DataAnalysisPipeline2.v3 import altlive
from DataAnalysisPipeline2.v3 import dataset as ds
from DataAnalysisPipeline2.v3 import fundamentals as fu
from DataAnalysisPipeline2.v3._alpaca import data_dir

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTEFACT = os.path.join(ROOT, "ExploitationZone", "model_v3_altall.pkl")
ALT_COLS = list(altlive.MODEL_COLUMNS)
FEATS = list(ds.FEATURE_COLUMNS) + list(fu.FEATURE_COLUMNS) + ALT_COLS


def _real_artefact():
    if not os.path.exists(ARTEFACT):
        return None
    with open(ARTEFACT, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            return None
    return pickle.load(open(ARTEFACT, "rb"))


def _final_frame_path():
    from DataAnalysisPipeline2.v3 import final_altall
    return final_altall.frame_path()


@pytest.fixture(autouse=True)
def _logs(tmp_path):
    from DataAnalysisPipeline2.trading_agent import operational
    old = operational.AGENT_LOG_DIR
    operational.set_log_dir(str(tmp_path / "agent_logs"))
    yield
    operational.set_log_dir(old)


HAVE_LOCAL = all(os.path.exists(p) for p in (
    os.path.join(data_dir(), "v3model", "weekly_dataset_r8_final.parquet"),
    os.path.join(data_dir("fundamentals"), "fundamentals_asof.parquet"),
    os.path.join(data_dir("sec_insider"), "events.parquet"),
    os.path.join(data_dir("finra_si"), "si.parquet")))


# --------------------------------------------------------------------------- #
# Live == offline (real data; skipped without the local store)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_LOCAL, reason="local v3 store / final altall frame not available")
@pytest.mark.parametrize("date", ["2025-06-27", "2026-06-26"])
def test_live_model_input_equals_offline_training_row(date):
    """The bot's model input (altall.model_input on a 420-day price window,
    fundamentals, insider events, FINRA) equals the final training frame's
    rows on a historical decision date: same eligible universe, alt-data and
    fundamentals columns exact, core features exact up to float32 storage /
    EMA warm-up on a handful of cells; predictions rank-identical."""
    live, off, info = altall.offline_vs_live(date)
    assert list(live["ticker"]) == list(off["ticker"]) and len(live) > 2500
    cols = [c for c in live.columns if c != "ticker"]
    assert cols == FEATS
    exact = ALT_COLS + list(fu.FEATURE_COLUMNS)
    d = (live[cols].astype(float) - off[cols].astype(float)).abs()
    assert float(d[exact].max().max()) < 1e-4
    assert float((d > 1e-3).to_numpy().mean()) < 1e-4          # core: a few EMA / tie cells at most
    assert float(d.max().max()) < 0.1
    art = _real_artefact()
    if art is not None:
        def pred(X):
            rs = [pd.Series(np.asarray(art["trained_models"][m].predict(X[art["tabular_cols"]]))).rank(pct=True)
                  for m in art["mix_models"]]
            return pd.concat(rs, axis=1).mean(axis=1).rank(pct=True)
        pl, po = pred(live), pred(off)
        assert pl.corr(po, method="spearman") > 0.9999
        top_l = set(live["ticker"][pl.nlargest(100).index])
        top_o = set(off["ticker"][po.nlargest(100).index])
        assert len(top_l & top_o) >= 97


@pytest.mark.skipif(not os.path.exists(os.path.join(altall.SNAPSHOT_DIR, "manifest.json")) or not HAVE_LOCAL,
                    reason="snapshot or local caches not available")
def test_snapshot_is_lossless_for_its_window():
    """Insider / SI features from the committed snapshot equal those from the
    full local caches on a date inside the snapshot window."""
    from DataAnalysisPipeline2.v3 import form4
    snap = altall.read_snapshot()
    d = str((snap["insider_as_of"] - pd.Timedelta(days=7)).date())
    full = form4.load_events()
    ciks = set(snap["events"]["issuer_cik"].dropna().astype("int64"))
    a = ad.insider_features(snap["events"], [d]).set_index("issuer_cik").sort_index()
    b = ad.insider_features(full[full["issuer_cik"].isin(ciks)], [d]).set_index("issuer_cik").sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
    rows = pd.DataFrame({"ticker": ["AAPL", "MSFT", "TSLA", "GME"], "Date": [d] * 4,
                         "first_date": ["2016-01-04"] * 4, "shares": [1e9] * 4})
    pd.testing.assert_frame_equal(ad.si_features(snap["si"], rows), ad.si_features(ad.load_si(), rows))


# --------------------------------------------------------------------------- #
# Snapshot freshness
# --------------------------------------------------------------------------- #
def _write_snapshot(path, insider_as_of="2026-11-06", settlements=("2026-09-30", "2026-10-15")):
    os.makedirs(path, exist_ok=True)
    ev = pd.DataFrame({"accession": ["a1", "a2"], "issuer_cik": pd.array([1, 2], dtype="Int64"),
                       "filing_date": pd.to_datetime(["2026-10-01", "2026-10-20"]),
                       "owner_cik": pd.array([10, 11], dtype="Int64"), "is_od": [True, False],
                       "code": ["P", "S"], "shares": [100.0, 50.0], "price": [10.0, 20.0],
                       "value": [1000.0, 1000.0], "avail_date": pd.to_datetime(["2026-10-02", "2026-10-21"])})
    si = pd.DataFrame({"symbol": ["AAA"] * len(settlements), "settlement_date": pd.to_datetime(list(settlements)),
                       "si": 1e6, "si_prev": 9e5, "si_dtc": 2.0})
    si["avail_date"] = ad.next_bday(si["settlement_date"], 8).to_numpy()
    ev.to_parquet(os.path.join(path, altall.EVENTS_FILE))
    si.to_parquet(os.path.join(path, altall.SI_FILE))
    json.dump({"insider_as_of": insider_as_of, "si_last_settlement": settlements[-1]},
              open(os.path.join(path, "manifest.json"), "w"))


def test_expected_si_floor():
    # 2026-10-15 settlement is public 8 NYSE days later (2026-10-27)
    assert altall.expected_si_floor("2026-10-26") == pd.Timestamp("2026-09-23")
    assert altall.expected_si_floor("2026-10-27") == pd.Timestamp("2026-10-09")
    assert altall.expected_si_floor("2026-11-07") == pd.Timestamp("2026-10-09")


def test_snapshot_fresh_and_stale(tmp_path):
    _write_snapshot(tmp_path)
    src = altall.read_snapshot(str(tmp_path))
    ok = altall.check_fresh(src, "2026-11-09", 14)
    assert ok["insider_age_days"] == 3 and ok["si_last_settlement"] == "2026-10-15"
    with pytest.raises(altall.AltDataUnavailable, match="insider feed"):
        altall.check_fresh(src, "2026-11-30", 14)
    # a FINRA file that is already public is missing -> stale
    _write_snapshot(tmp_path / "s2", settlements=("2026-09-15", "2026-09-30"))
    with pytest.raises(altall.AltDataUnavailable, match="FINRA"):
        altall.check_fresh(altall.read_snapshot(str(tmp_path / "s2")), "2026-11-09", 14)


def test_snapshot_missing_or_unreadable(tmp_path):
    with pytest.raises(altall.AltDataUnavailable, match="missing"):
        altall.read_snapshot(str(tmp_path))
    _write_snapshot(tmp_path)
    open(tmp_path / altall.SI_FILE, "w").write("not parquet")
    with pytest.raises(altall.AltDataUnavailable, match="unreadable"):
        altall.read_snapshot(str(tmp_path))


def test_load_sources_stale_without_network(tmp_path, monkeypatch):
    """No SEC_USER_AGENT and a FINRA refresh that finds nothing -> stale -> raise."""
    _write_snapshot(tmp_path, insider_as_of="2026-10-01")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(altall, "refresh_si", lambda si, d, log=print: si)
    with pytest.raises(altall.AltDataUnavailable, match="insider feed"):
        altall.load_sources("2026-11-09", snapshot_dir=str(tmp_path), log=lambda *a: None)


def test_refresh_si_appends_public_file(tmp_path):
    _write_snapshot(tmp_path, settlements=("2026-09-15", "2026-09-30"))
    si = altall.read_snapshot(str(tmp_path))["si"]
    body = ('"accountingYearMonthNumber"|"symbolCode"|"issueName"|"marketClassCode"|"currentShortPositionQuantity"|'
            '"previousShortPositionQuantity"|"settlementDate"|"daysToCoverQuantity"\n'
            '"20261015"|"AAA"|"A"|"NNM"|"1200000"|"1000000"|"2026-10-15"|"3.0"\n')

    class S:
        def get(self, url, timeout=60):
            ok = url.endswith("shrt20261015.csv")
            return types.SimpleNamespace(status_code=200 if ok else 404, text=body if ok else "")
    out = altall.refresh_si(si, "2026-11-09", log=lambda *a: None, session=S())
    assert pd.Timestamp(out["settlement_date"].max()) == pd.Timestamp("2026-10-15")
    assert len(out) == len(si) + 1


# --------------------------------------------------------------------------- #
# Artefact contract
# --------------------------------------------------------------------------- #
def _manifest(cols, blocks):
    return {"feature_version": 3, "feature_columns": list(cols), "label_horizon": 30, "eligibility_rule": {},
            "zscore_method": "x", "training_range": ["a", "b"], "feature_blocks": blocks,
            "model_id": "v3_test_altall"}


def _tiny_altall(path, blocks=("core", "fundamentals", "insider_hd", "si")):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATS))), columns=FEATS)
    m = LinearRegression().fit(X, rng.normal(size=200))
    art = {"trained_models": {"A": m}, "mix_models": ["A"], "tabular_cols": FEATS, "pca_cols": [],
           "company_embeddings": {}, "scaler": None, "pca": None, "cs_z_standardize": True,
           "manifest": _manifest(FEATS, list(blocks))}
    pickle.dump(art, open(path, "wb"))
    return art


def _tiny_production(path):
    rng = np.random.default_rng(1)
    cols = list(ds.FEATURE_COLUMNS)
    X = pd.DataFrame(rng.normal(size=(200, len(cols))), columns=cols)
    art = {"trained_models": {"A": LinearRegression().fit(X, rng.normal(size=200))}, "mix_models": ["A"],
           "tabular_cols": cols, "pca_cols": [], "company_embeddings": {}, "scaler": None, "pca": None,
           "cs_z_standardize": True, "manifest": {**_manifest(cols, ["core"]), "model_id": "v3_test_prod"}}
    pickle.dump(art, open(path, "wb"))


def test_schema_accepts_declared_blocks_only(tmp_path):
    art = _tiny_altall(tmp_path / "a.pkl")
    bot.validate_model_artifact_schema(art)
    undeclared = dict(art, manifest=_manifest(FEATS, ["core"]))
    with pytest.raises(ValueError, match="columns the v3 feature code does not produce"):
        bot.validate_model_artifact_schema(undeclared)


def test_real_artefact_contract():
    art = _real_artefact()
    if art is None:
        pytest.skip("model_v3_altall.pkl not present (LFS not pulled)")
    from DataAnalysisPipeline2.v3 import final_altall
    bot.validate_model_artifact_schema(art)
    man = art["manifest"]
    assert art["tabular_cols"] == final_altall.feature_columns() == FEATS
    assert man["signal"] == "sn30_altall_hd" and man["target"] == "target_sn_30"
    assert set(man["feature_profile"]["features"]) == set(FEATS)
    assert man["training_range"][1] >= "2026-09-01" and man["evaluation"]["n_trials"] == 92
    assert man["opt_ic"]["trailing_ic"] == config.OPT_IC_ALTALL


# --------------------------------------------------------------------------- #
# Fallback to the production model
# --------------------------------------------------------------------------- #
@pytest.fixture
def paths(tmp_path, monkeypatch):
    prod = tmp_path / "prod.pkl"
    _tiny_production(prod)
    monkeypatch.setattr(config, "MODEL_PATH", str(prod))
    monkeypatch.setattr(config, "SIGNAL", "altall")
    monkeypatch.setattr(config, "ALT_REFRESH", False)
    return tmp_path


def test_altall_missing_artefact_falls_back(paths, monkeypatch):
    monkeypatch.setattr(config, "ALTALL_MODEL_PATH", str(paths / "nope.pkl"))
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.signal == "production" and a.model_metadata["model_id"] == "v3_test_prod"
    assert "FileNotFoundError" in a.signal_fallback_reason
    assert a.signal_info()["signal_requested"] == "altall"


def test_altall_lfs_pointer_falls_back(paths, monkeypatch):
    p = paths / "ptr.pkl"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    monkeypatch.setattr(config, "ALTALL_MODEL_PATH", str(p))
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.signal == "production" and "LFS pointer" in a.signal_fallback_reason


def test_altall_stale_snapshot_falls_back(paths, monkeypatch):
    _tiny_altall(paths / "alt.pkl")
    monkeypatch.setattr(config, "ALTALL_MODEL_PATH", str(paths / "alt.pkl"))
    _write_snapshot(paths / "snap", insider_as_of="2020-01-01")
    monkeypatch.setattr(config, "ALT_SNAPSHOT_DIR", str(paths / "snap"))
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.signal == "production" and "stale alt data" in a.signal_fallback_reason


def test_altall_loads_when_inputs_fresh(paths, monkeypatch):
    _tiny_altall(paths / "alt.pkl")
    monkeypatch.setattr(config, "ALTALL_MODEL_PATH", str(paths / "alt.pkl"))
    today = pd.Timestamp.today().normalize()
    _write_snapshot(paths / "snap", insider_as_of=str(today.date()),
                    settlements=(str(altall.expected_si_floor(today).date()),))
    monkeypatch.setattr(config, "ALT_SNAPSHOT_DIR", str(paths / "snap"))
    monkeypatch.setattr(optimized, "resolve_fund_dir", lambda *a, **k: {"path": "x", "source": "test",
                                                                       "as_of": "2026-11-05", "age_days": 1})
    monkeypatch.setattr(altall, "read_fundamentals", lambda p: {"cik_map": pd.DataFrame({"cik": [1, 2]}),
                                                                "fund_asof": None, "sectors": None})
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.signal == "altall" and a.signal_fallback_reason is None
    assert a.tabular_cols == FEATS and a.model_path.endswith("alt.pkl")
    # then a failure after load: use_production swaps the artefact
    a.use_production("boom")
    assert a.signal == "production" and a.model_path == str(paths / "prod.pkl")
    assert a.signal_info()["signal_fallback_reason"] == "boom"


def test_altall_missing_fundamental_columns_falls_back(paths, monkeypatch):
    _tiny_altall(paths / "alt.pkl")
    monkeypatch.setattr(config, "ALTALL_MODEL_PATH", str(paths / "alt.pkl"))
    fd = paths / "fund"
    fd.mkdir()
    pd.DataFrame({"cik": [1], "asof_date": ["2026-11-01"], "last_filed": ["2026-11-01"]}).to_parquet(
        fd / "fundamentals_asof.parquet")
    pd.DataFrame({"ticker": ["A"], "cik": [1]}).to_parquet(fd / "cik_map.parquet")
    pd.DataFrame({"cik": [1], "ff12": [1], "sector_name": ["X"]}).to_parquet(fd / "sectors.parquet")
    monkeypatch.setattr(optimized, "resolve_fund_dir", lambda *a, **k: {"path": str(fd), "source": "test",
                                                                       "as_of": "2026-11-01", "age_days": 1})
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.signal == "production" and "lack" in a.signal_fallback_reason


class _FakeAgent:
    def __init__(self, signal="altall"):
        self.signal = signal
        self.last_construction = None
        self.calls = []

    def use_production(self, reason):
        self.signal, self.reason = "production", reason


def test_run_falls_back_to_production_on_inference_error(monkeypatch):
    agent, args = _FakeAgent(), types.SimpleNamespace(strategy="optimized_v3")

    def score(a, args_, universe):
        a.calls.append(a.signal)
        if a.signal == "altall":
            raise RuntimeError("feature failure")
        a.last_construction = {"construction": "optimized_v3"}
        return (None, "px", "preds", {"AAA": 0.5})
    monkeypatch.setattr(config, "TICKERS", ["X"])
    out = run.run_with_signal_fallback(agent, args, "full", ["REQ"], score=score)
    assert out[3] == {"AAA": 0.5} and agent.calls == ["altall", "production"]
    assert "feature failure" in agent.reason and config.TICKERS == ["REQ"]


def test_run_falls_back_when_optimizer_did_not_build_altall_book():
    agent, args = _FakeAgent(), types.SimpleNamespace(strategy="optimized_v3")

    def score(a, args_, universe):
        a.calls.append(a.signal)
        a.last_construction = ({"construction": "long_only_v3", "fallback_reason": "solver"}
                               if a.signal == "altall" else {"construction": "long_only_v3"})
        return (None, None, None, {a.signal: 1.0})
    out = run.run_with_signal_fallback(agent, args, "full", [], score=score)
    assert agent.calls == ["altall", "production"] and out[3] == {"production": 1.0}
    assert "solver" in agent.reason


def test_run_production_errors_propagate():
    agent = _FakeAgent("production")

    def score(a, args_, universe):
        raise RuntimeError("outage")
    with pytest.raises(RuntimeError, match="outage"):
        run.run_with_signal_fallback(agent, types.SimpleNamespace(strategy="optimized_v3"), "full", [],
                                     score=score)


@pytest.mark.parametrize("signal,expected", [("altall", "OPT_IC_ALTALL"), ("production", "OPT_IC")])
def test_optimized_v3_uses_the_signal_ic(signal, expected, monkeypatch):
    seen = {}

    def fake_book(preds, bars, **kw):
        seen["ic"] = kw["ic"]
        raise optimized.OptimizerUnavailable("stop here")
    monkeypatch.setattr(optimized, "optimized_book", fake_book)
    a = bot.BDATradingAgent()
    a.signal = signal
    a.last_bars = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(a, "_current_weights", lambda: {})
    monkeypatch.setattr(a, "_long_only_v3_book", lambda p: ({}, {}))
    _, diag = a._optimized_v3_or_fallback(pd.DataFrame({"ticker": ["A"], "pred_rank": [0.5]}))
    assert seen["ic"] == getattr(config, expected) and diag["construction"] == "long_only_v3"
    assert config.OPT_IC_ALTALL == pytest.approx(0.0534)
