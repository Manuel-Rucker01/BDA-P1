"""End-to-end live inference through the feature_version-3 path (no network).

The live agent sees only the last LIVE_HISTORY_DAYS_V3 calendar days of bars;
the offline dataset is built (chunked) from the full history. Features,
eligibility, sectors, the z-scored model input and the predictions must agree.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from DataAnalysisPipeline2.trading_agent import bot, config, market_data, operational
from DataAnalysisPipeline2.v3 import artefact, dataset as ds, train as tr
from DataAnalysisPipeline2.v3.tests.synthetic import make_panel

FEATS = ds.FEATURE_COLUMNS


@pytest.fixture(scope="module")
def synth():
    return make_panel(n_per_sector=10, n_sessions=520, seed=3, start="2023-01-02")


@pytest.fixture(scope="module")
def offline(synth, tmp_path_factory):
    panel, meta = synth
    out = tmp_path_factory.mktemp("v3ds")
    ds.build_dataset(prices=panel, meta=meta, out_dir=str(out), chunk_size=5, log=lambda *a: None)
    return ds.load_daily(out_dir=str(out))


def _live_window(panel):
    last = pd.Timestamp(panel["Date"].max())
    start = (last - pd.Timedelta(days=config.LIVE_HISTORY_DAYS_V3)).strftime("%Y-%m-%d")
    return panel[panel["Date"] > start].drop(columns=[]).reset_index(drop=True)


@pytest.fixture
def v3_agent(tmp_path, monkeypatch, synth):
    panel, meta = synth
    old = operational.AGENT_LOG_DIR
    operational.set_log_dir(str(tmp_path))
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, len(FEATS))), columns=FEATS)
    m1 = LinearRegression().fit(X, X["return_20d"] * -0.1 + rng.normal(size=300))
    m2 = LinearRegression().fit(X, X["ret_vol_20d"] * -0.1 + rng.normal(size=300))
    a = bot.BDATradingAgent()
    a.trained_models, a.mix_models = {"A": m1, "B": m2}, ["A", "B"]
    a.tabular_cols, a.pca_cols, a.company_embeddings = list(FEATS), [], {}
    a.cs_z_standardize, a.feature_version = True, 3
    monkeypatch.setattr(bot, "fetch_company_metadata", lambda: meta.copy())
    dates = sorted(panel["Date"].unique())
    spx = pd.DataFrame({"Date": dates, "Close": 5000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))})
    monkeypatch.setattr(market_data, "fetch_benchmark_closes", lambda *a, **k: spx)
    yield a
    operational.set_log_dir(old)


def test_live_window_is_shorter_than_offline_history(synth):
    panel, _ = synth
    live = _live_window(panel)
    assert live["Date"].nunique() < panel["Date"].nunique() - 200
    assert live["Date"].nunique() >= 250


def test_v3_live_features_equal_offline_dataset(v3_agent, synth, offline):
    panel, _ = synth
    out = v3_agent.run_inference(_live_window(panel))
    d = panel["Date"].max()
    off = offline[(offline["Date"] == d) & offline["eligible"]].set_index("ticker").sort_index()
    live = v3_agent.last_predictions_df.set_index("ticker").sort_index()
    # same universe: no PNY (raw < $3), NEW (< 120 bars), ILQ (ADV), DEL (stale)
    assert list(live.index) == list(off.index)
    assert not {"PNY", "NEW", "ILQ", "DEL"} & set(live.index)
    assert (live["Sector"] == off["Sector"]).all() and live.loc["UNK", "sector_source"] == "proxy"
    pd.testing.assert_frame_equal(live[FEATS].astype(float), off[FEATS].astype(float),
                                  check_exact=False, rtol=1e-5, atol=1e-6)
    # identical model input (z-scored over the eligible cross-section) ...
    Xoff = ds.zscore(off.reset_index())[FEATS]
    Xlive = v3_agent.last_feature_matrix.set_axis(v3_agent.last_predictions_df["ticker"])
    Xoff.index = off.index
    pd.testing.assert_frame_equal(Xlive.sort_index()[FEATS].astype(float), Xoff.astype(float),
                                  check_exact=False, atol=1e-4)
    # ... and identical predictions
    off_pred = tr.predict_ensemble(v3_agent.trained_models, Xoff.assign(Date=d), FEATS)
    live_pred = out.set_index("ticker").sort_index()["pred_rank"].to_numpy()
    assert np.allclose(off_pred, live_pred)
    assert out["pred_rank"].between(0, 1).all()
    assert set(out.columns) >= {"ticker", "company_close", "pred_rank", "kalman_beta",
                                "return_volatility_20d", "Sector"}


def test_v3_artifact_roundtrip_through_load_model(tmp_path, monkeypatch, offline):
    """A real (tiny) build_artifact pickle loads through the manifest gate."""
    cal = ds.session_calendar(offline["Date"])
    dates = ds.weekly_decision_dates(cal, start=cal[150])
    f = offline[offline["Date"].isin(dates) & offline["eligible"]].reset_index(drop=True)
    f = pd.concat([f.drop(columns=FEATS), ds.zscore(f)], axis=1)
    art = artefact.build_artifact(f, "target_raw_30", cal, variant="base", construction="lo50",
                                  models=("lgbm",), log=lambda *a: None)
    pkl, js = artefact.save_artifact(art, out_dir=str(tmp_path))
    monkeypatch.setattr(config, "MODEL_PATH", pkl)
    monkeypatch.setattr(config, "SIGNAL", "production")   # test the MODEL_PATH artefact, not altall
    a = bot.BDATradingAgent()
    a.load_model()
    assert a.feature_version == 3 and a.pca_cols == [] and a.tabular_cols == FEATS
    man = art["manifest"]
    assert man["label_horizon"] == 30 and man["feature_profile"]["features"]
    assert man["training_range"][1] <= cal[-31]


def test_v3_schema_rejects_kg_or_mismatched_columns():
    good = {"trained_models": {"A": object()}, "mix_models": ["A"], "tabular_cols": list(FEATS),
            "pca_cols": [], "cs_z_standardize": True,
            "manifest": {"feature_version": 3, "feature_columns": list(FEATS), "label_horizon": 30,
                         "eligibility_rule": {}, "zscore_method": "x", "training_range": ["a", "b"]}}
    bot.validate_model_artifact_schema(good)
    bad = dict(good, pca_cols=["kg_pc0"])
    with pytest.raises(ValueError):
        bot.validate_model_artifact_schema(bad)
    bad = dict(good, manifest=dict(good["manifest"], feature_columns=FEATS[:-1]))
    with pytest.raises(ValueError):
        bot.validate_model_artifact_schema(bad)
    bad = dict(good, tabular_cols=FEATS + ["kg_pc0"],
               manifest=dict(good["manifest"], feature_columns=FEATS + ["kg_pc0"]))
    with pytest.raises(ValueError):
        bot.validate_model_artifact_schema(bad)


def test_v3_live_universe_from_classification(tmp_path):
    c = pd.DataFrame({"fetch_symbol": ["AAA", "BBB", "CCC", "DDD", None],
                      "in_universe": [True, True, False, True, True],
                      "status": ["active", "inactive", "active", "active", "active"]})
    c.to_parquet(tmp_path / "classification.parquet")
    pd.DataFrame({"canonical": ["AAA"], "alias": ["DDD"], "mode": ["alias"]}).to_parquet(
        tmp_path / "duplicates.parquet")
    assert bot.v3_live_universe(str(tmp_path / "classification.parquet"), verbose=False) == ["AAA"]
