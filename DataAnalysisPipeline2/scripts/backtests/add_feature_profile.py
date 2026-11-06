"""Add a training ``feature_profile`` (PSI reference) to a feature_version-2
model artefact, so live monitoring can compute feature drift.

The profile is built on exactly the matrix the model was trained on: the shared
``kg_embeddings_classifier.build_v2_scored_frame`` (per-date robust z-score of
core features + the artefact's own PCA), restricted to the manifest's training
window. Model weights are untouched; only ``metadata``/``manifest`` gain the
``feature_profile`` key (and the JSON manifest copy, if given).

    python DataAnalysisPipeline2/scripts/backtests/add_feature_profile.py \
        --model ExploitationZone/best_model.pkl \
        --manifest-json DataAnalysisPipeline2/results/model_manifest_production.json
"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPE, ".."))
for p in (PIPE, os.path.join(PIPE, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ExploitationZone/best_model.pkl")
    ap.add_argument("--manifest-json", default=None)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--null-quantile", type=float, default=0.99)
    ap.add_argument("--all-asset-classes", action="store_true",
                    help="profile every training row; default = common stocks only, i.e. the "
                         "population live scores after bot.exclude_non_common (needs Alpaca read access)")
    args = ap.parse_args()
    path = args.model if os.path.isabs(args.model) else os.path.join(ROOT_DIR, args.model)

    import kg_embeddings_classifier as K
    from trading_agent import bot, monitoring

    with open(path, "rb") as f:
        m = pickle.load(f)
    meta = m.get("metadata") or m.get("manifest") or {}
    if int(meta.get("feature_version", 0)) != 2:
        raise SystemExit("only feature_version-2 artefacts are handled here (v3 builds its own profile)")
    if any(c.startswith("news_") for c in m["tabular_cols"]):
        raise SystemExit("news-feature artefacts are not supported by this script")
    lo, hi = meta["training_date_min"], meta["training_date_max"]

    panel = K.load_price_panel()
    # 200 calendar days of warm-up before the window so rolling features are complete
    import pandas as pd
    start = (pd.Timestamp(lo) - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    panel = panel[(panel["Date"] >= start) & (panel["Date"] <= hi)]
    cmeta = bot.fetch_company_metadata()[["ticker", "Sector"]]
    fr, _ = K.build_v2_scored_frame(panel, cmeta, m["company_embeddings"], m["pca"], m["pca_cols"])
    fr = fr[(fr["Date"] >= lo) & (fr["Date"] <= hi)]
    if not args.all_asset_classes:
        kept, removed = bot.exclude_non_common(sorted(fr["ticker"].unique()))
        if not removed:
            raise SystemExit("non-common filter removed nothing (Alpaca unreachable?); "
                             "rerun with ACTIVE_ACCOUNT set or pass --all-asset-classes")
        fr = fr[fr["ticker"].isin(set(kept))]
    cols = list(m["tabular_cols"]) + list(m["pca_cols"])
    prof = monitoring.build_feature_profile(fr[cols], n_bins=args.n_bins, dates=fr["Date"].to_numpy(),
                                         null_quantile=args.null_quantile)
    prof["population"] = "all" if args.all_asset_classes else "common stocks (bot.exclude_non_common)"
    prof["window"] = [lo, hi]
    print(f"[profile] {prof['n_rows']:,} rows, {len(prof['features'])} features, window {lo}..{hi} "
          f"(manifest n_training_rows={meta.get('n_training_rows')})")

    for k in ("metadata", "manifest"):
        if isinstance(m.get(k), dict):
            m[k]["feature_profile"] = prof
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"[profile] written into {path}")

    if args.manifest_json:
        jp = args.manifest_json if os.path.isabs(args.manifest_json) else os.path.join(ROOT_DIR, args.manifest_json)
        with open(jp) as f:
            j = json.load(f)
        j["feature_profile"] = prof
        with open(jp, "w") as f:
            json.dump(j, f, indent=2)
        print(f"[profile] written into {jp}")


if __name__ == "__main__":
    main()
