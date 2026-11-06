#!/usr/bin/env python3
"""End-to-end check that a (feature_version 2) artefact loads and scores through
the LIVE bot path: BDATradingAgent.load_model() + run_inference().

Read-only: no broker calls; market-data only (benchmark closes, and Alpaca news
if the model uses news features). Decision logs go to a temp dir, never to the
tracked agent_logs/.

Also cross-checks parity: the live pred_rank on the last panel date must equal
the offline score built with the training feature path
(kg_embeddings_classifier.build_v2_scored_frame) for models without news.

Usage:
  python check_v2_artifact.py ExploitationZone/best_model_<tag>.pkl [--days 400] [--asof YYYY-MM-DD]
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
for p in (PIPELINE_DIR, SCRIPTS_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ACTIVE_ACCOUNT", "monthly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--days", type=int, default=400, help="calendar days of history fed to the bot")
    ap.add_argument("--asof", default=None, help="last panel date (default: last date in the CSV)")
    args = ap.parse_args()
    path = args.model if os.path.isabs(args.model) else os.path.join(ROOT_DIR, args.model)

    from trading_agent import config, operational, bot
    import kg_embeddings_classifier as K

    operational.set_log_dir(tempfile.mkdtemp(prefix="v2check_"))
    config.MODEL_PATH = path
    agent = bot.BDATradingAgent()
    agent.load_model()
    print(f"[check] feature_version={agent.feature_version} cs_z={agent.cs_z_standardize} "
          f"mix={agent.mix_models} n_cols={len(agent.tabular_cols) + len(agent.pca_cols)}")
    assert agent.feature_version == 2, "not a feature_version-2 artefact"

    panel = K.load_price_panel()
    asof = args.asof or panel["Date"].max()
    lo = (pd.Timestamp(asof) - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")
    panel = panel[(panel["Date"] >= lo) & (panel["Date"] <= asof)]
    print(f"[check] feeding {len(panel):,} bars, {panel['ticker'].nunique()} tickers, {lo}..{asof}")

    agent.run_inference(panel)
    out = agent.last_predictions_df
    X = agent.last_feature_matrix
    assert list(X.columns) == agent.tabular_cols + agent.pca_cols
    assert np.isfinite(X.to_numpy()).all() and X.abs().max().max() <= 6.0
    assert out["pred_rank"].between(0, 1).all()
    print(f"[check] live run_inference OK: {len(out)} names scored on {out['Date'].max()}; "
          f"top-5: {out.sort_values('pred_rank', ascending=False)['ticker'].head(5).tolist()}")

    if not any(c.startswith("news_") for c in agent.tabular_cols):
        meta = bot.fetch_company_metadata()[["ticker", "Sector"]]
        fr, _ = K.build_v2_scored_frame(panel, meta, agent.company_embeddings, agent.pca, agent.pca_cols)
        fr = fr[fr["Date"] == out["Date"].max()]
        Xo = fr[agent.tabular_cols + agent.pca_cols]
        ranks = []
        for m in agent.mix_models:
            ranks.append(pd.Series(agent.trained_models[m].predict(Xo)).rank(pct=True).values)
        fr = fr.assign(off=pd.Series(np.mean(ranks, axis=0)).rank(pct=True).values)
        j = out.merge(fr[["ticker", "off"]], on="ticker")
        diff = float((j["pred_rank"] - j["off"]).abs().max())
        print(f"[check] parity live vs offline training path: {len(j)}/{len(out)} names, "
              f"max |pred_rank diff| = {diff:.2e}")
        assert len(j) == len(out) and diff < 1e-6
    print("[check] PASS")


if __name__ == "__main__":
    main()
