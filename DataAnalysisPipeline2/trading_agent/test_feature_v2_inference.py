"""End-to-end live inference through the feature_version-2 path (no network)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from DataAnalysisPipeline2.features import core
from DataAnalysisPipeline2.trading_agent import bot, market_data, operational

DATES = pd.bdate_range("2025-01-01", periods=140).strftime("%Y-%m-%d").tolist()
TICKERS = [f"T{i:02d}" for i in range(15)]


def _prices():
    rows = []
    for i, t in enumerate(TICKERS + ["PNY", "OLD"]):
        rng = np.random.default_rng(i)
        start = 0.4 if t == "PNY" else 40.0
        px = start * np.exp(np.cumsum(rng.normal(0, 0.02, len(DATES))))
        ds = DATES[:-15] if t == "OLD" else DATES   # OLD: stale
        rows += [{"ticker": t, "Date": d, "company_close": p, "company_volume": 1e5 * (1 + rng.random()),
                  "Open": p, "High": p, "Low": p} for d, p in zip(ds, px)]
    return pd.DataFrame(rows)


@pytest.fixture
def v2_agent(tmp_path, monkeypatch):
    old = operational.AGENT_LOG_DIR
    operational.set_log_dir(str(tmp_path))
    rng = np.random.default_rng(0)
    emb = {t: rng.normal(size=4) for t in TICKERS + ["PNY", "OLD"]}
    pca = PCA(n_components=2).fit(np.array(list(emb.values())))
    cols = core.FEATURE_COLUMNS + ["kg_pc0", "kg_pc1"]
    X = pd.DataFrame(rng.normal(size=(200, len(cols))), columns=cols)
    m1 = LinearRegression().fit(X, rng.normal(size=200))
    m2 = LinearRegression().fit(X, rng.normal(size=200))

    a = bot.BDATradingAgent()
    a.trained_models, a.mix_models = {"A": m1, "B": m2}, ["A", "B"]
    a.pca, a.pca_cols, a.tabular_cols = pca, ["kg_pc0", "kg_pc1"], list(core.FEATURE_COLUMNS)
    a.company_embeddings, a.cs_z_standardize, a.feature_version = emb, True, 2

    monkeypatch.setattr(bot, "fetch_company_metadata", lambda: pd.DataFrame(
        {"ticker": TICKERS + ["PNY", "OLD"], "Sector": ["Technology"] * 8 + ["Finance"] * 9}))
    spx = pd.DataFrame({"Date": DATES, "Close": 5000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DATES))))})
    monkeypatch.setattr(market_data, "fetch_benchmark_closes", lambda *a, **k: spx)
    yield a
    operational.set_log_dir(old)


def test_v2_inference_scores_only_eligible_fresh_names(v2_agent):
    out = v2_agent.run_inference(_prices())
    assert set(out["ticker"]) == set(TICKERS)            # no penny (PNY), no stale (OLD)
    assert out["pred_rank"].between(0, 1).all()
    assert len(out) == len(TICKERS)  # (averaging 2 models' ranks can legitimately tie)
    X = v2_agent.last_feature_matrix
    assert list(X.columns) == core.FEATURE_COLUMNS + ["kg_pc0", "kg_pc1"]
    assert np.isfinite(X.to_numpy()).all() and X.abs().max().max() <= 6.0


def test_v2_matches_offline_feature_pipeline(v2_agent):
    """Live features at the decision date == features built offline (training path)."""
    prices = _prices()
    v2_agent.run_inference(prices)
    live = v2_agent.last_predictions_df.set_index("ticker")[core.FEATURE_COLUMNS].sort_index()
    offline = core.decision_rows(core.compute_features(
        prices, pd.DataFrame({"ticker": TICKERS + ["PNY", "OLD"],
                              "Sector": ["Technology"] * 8 + ["Finance"] * 9})))
    offline = offline.set_index("ticker")[core.FEATURE_COLUMNS].sort_index()
    pd.testing.assert_frame_equal(live, offline, check_dtype=False)
