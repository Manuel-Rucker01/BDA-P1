"""Final v3 artefact: refit the chosen recipe on all labelled data (WS3 step 6).

    python -m DataAnalysisPipeline2.v3.artefact --variant base --construction lo50

writes ``<data>/v3model/best_model_v3.pkl`` and ``best_model_v3.manifest.json``.
The pickle follows the bot's artefact layout (``trained_models``,
``mix_models``, ``tabular_cols``, ``cs_z_standardize``, ``manifest``) with no
KG (``pca``/``scaler`` None, empty ``pca_cols``/``company_embeddings``); the
bot routes it to the feature_version-3 path through ``manifest.feature_version``.
It is NOT copied into ExploitationZone/ -- promotion is a manual step.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import subprocess

import pandas as pd

from DataAnalysisPipeline2.trading_agent import monitoring

from . import backtest as bt
from . import dataset as ds
from . import train as tr

ARTIFACT_SCHEMA_VERSION = 3


def _git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_artifact(frame: pd.DataFrame, target: str, calendar, *, variant: str, construction: str | None,
                   extra: dict | None = None, models=tr.MODEL_NAMES, log=print) -> dict:
    """Fit the ensemble on every row of ``frame`` with a label and return the
    artefact dict (models + manifest)."""
    feats = list(ds.FEATURE_COLUMNS)
    fr = frame[frame[target].notna()]
    ms, info = tr.fit_ensemble(fr, feats, target, calendar, models=models, log=log)
    horizon = int(target.rsplit("_", 1)[1])
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_version": ds.FEATURE_VERSION,
        "model_id": f"v3_{variant}_{dt.date.today():%Y%m%d}",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "feature_columns": feats,
        "feature_code": "DataAnalysisPipeline2/v3/dataset.py compute_v3_features (features/core.py "
                        "time_series_features + add_cross_sectional_features)",
        "variant": variant,
        "target": target,
        "label_horizon": horizon,
        "label_definition": "forward H-session return on adjusted closes, last available close if the "
                            f"stock stops trading inside the horizon, winsorized to {list(ds.LABEL_CLIP)}; "
                            "target = per-date percentile rank over eligible names"
                            + (" after sector-demeaning" if "_sn_" in target else ""),
        "eligibility_rule": {**ds.ELIGIBILITY_RULE, "price": "raw (unadjusted) close",
                             "adv": "20-session mean of adjusted close x volume",
                             "bars": "count since the first bar (store starts 2016-01-01)",
                             "requires_bar_on_decision_date": True},
        "universe": "WS1 survivorship-free US common stocks (NASDAQ/NYSE/AMEX); live = active "
                    "common stocks from the WS1 classification passing eligibility on the decision date",
        "sector_method": f"legacy DuckDB metadata where available, else proxy sector = known sector "
                         f"with max market-residual return correlation over the {ds.PROXY_WINDOW} sessions "
                         f"before the decision month (>= {ds.PROXY_MIN_OBS} obs), else 'Unknown'",
        "zscore_method": "features.core.cross_sectional_zscore per date over eligible names "
                         "(winsor 1%/99% inward quantiles, float64, constant-column guard, clip +/-6)",
        "uses_kg": False,
        "decision_dates": "weekly (last session of each week) from 2017-01",
        "training_range": info["train_range"],
        "fit_range": info["fit_range"],
        "early_stopping_val_range": info["val_range"],
        "n_train_rows": info["n_train_rows"],
        "n_train_dates": info["n_train_dates"],
        "models": {"mix": list(ms), "params": tr.MODEL_PARAMS, "best_iter": info["best_iter"],
                   "ensemble": "per-date rank average of member predictions, re-ranked"},
        "construction": {"name": construction, **(bt.CONSTRUCTIONS.get(construction, {}) if construction else {})},
        "feature_profile": monitoring.build_feature_profile(fr[feats]),
    }
    manifest.update(extra or {})
    art = {"trained_models": ms, "mix_models": list(ms), "scaler": None, "pca": None,
           "tabular_cols": feats, "pca_cols": [], "company_embeddings": {},
           "cs_z_standardize": True, "manifest": manifest}
    return art


def save_artifact(art: dict, out_dir: str | None = None, name: str = "best_model_v3") -> tuple[str, str]:
    out_dir = out_dir or ds.v3_dir()
    pkl = os.path.join(out_dir, f"{name}.pkl")
    js = os.path.join(out_dir, f"{name}.manifest.json")
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(art, f)
    os.replace(pkl + ".tmp", pkl)
    json.dump(art["manifest"], open(js, "w"), indent=1, default=str)
    return pkl, js


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--construction", default=None)
    ap.add_argument("--extra-json", default=None, help="JSON file merged into the manifest")
    a = ap.parse_args()
    target = tr.VARIANTS[a.variant][0]
    extra = json.load(open(a.extra_json)) if a.extra_json else None
    art = build_artifact(tr.load_weekly(), target, ds.calendar_from_dataset(), variant=a.variant,
                         construction=a.construction, extra=extra)
    print(save_artifact(art))
