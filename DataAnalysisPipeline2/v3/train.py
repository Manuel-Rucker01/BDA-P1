"""v3 training: GBDT ensemble on per-date rank targets, walk-forward by year (WS3).

* Inputs: weekly decision dates (last session of each week, 2017+), eligible
  rows only, features z-scored per date with ``core.cross_sectional_zscore``
  over the eligible population (``dataset.zscore``).
* Model: LightGBM + XGBoost + CatBoost regressors with moderate, FIXED
  hyperparameters (``MODEL_PARAMS``), each early-stopped on an embargoed
  validation split (last ``VAL_WEEKS`` weekly dates of the training window;
  the fit part ends ``EMBARGO_SESSIONS`` sessions before the validation
  starts) and then refit on the whole training window with the early-stopped
  number of trees. Predictions are rank-averaged per date.
* Walk-forward: for test year Y, train on decision dates t with
  ``pos(t) < pos(first session of Y) - EMBARGO_SESSIONS`` (embargo 40 sessions
  >= label horizon 30 + 10), predict every weekly date of Y.
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd

from . import dataset as ds

EMBARGO_SESSIONS = 40
VAL_WEEKS = 52
TEST_YEARS = tuple(range(2019, 2027))
SEED = 42
N_THREADS = int(os.environ.get("V3_THREADS", "8"))
MODEL_NAMES = ("lgbm", "xgb", "cat")
MODEL_PARAMS = {
    "lgbm": dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=6,
                 min_child_samples=1000, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                 reg_lambda=10.0, max_bin=127),
    "xgb": dict(n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=1000,
                subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0, tree_method="hist", max_bin=128),
    "cat": dict(iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=10.0, border_count=128,
                rsm=0.7),
}
EARLY_STOPPING = 100
MIN_TREES = 50
KG_N_COMPONENTS = 16


# --------------------------------------------------------------------------- #
# Weekly training frame
# --------------------------------------------------------------------------- #
KEEP_RAW = ["ticker", "Date", "Sector", "sector_source", "company_close", "return_volatility_20d",
            "adv_usd", "fwd_ret_21", "fwd_ret_30", "target_raw_21", "target_raw_30",
            "target_sn_21", "target_sn_30"]


def build_weekly_frame(out_dir: str | None = None, start: str = ds.DECISION_START, log=print) -> pd.DataFrame:
    """Eligible rows on weekly decision dates with z-scored features (float32).
    Cached to ``<data>/v3model/weekly.parquet``."""
    cal = ds.calendar_from_dataset(out_dir)
    dates = ds.weekly_decision_dates(cal, start=start)
    frames = []
    for y in sorted({d[:4] for d in dates}):
        dd = [d for d in dates if d[:4] == y]
        df = ds.load_daily(years=[y], dates=dd, out_dir=out_dir,
                           columns=KEEP_RAW + ["eligible"] + ds.FEATURE_COLUMNS)
        df = df[df["eligible"]].drop(columns=["eligible"]).reset_index(drop=True)
        Z = ds.zscore(df)
        df = pd.concat([df[KEEP_RAW], Z], axis=1)
        frames.append(df)
        log(f"[weekly] {y}: {len(dd)} dates, {len(df):,} rows")
    out = pd.concat(frames, ignore_index=True)
    return out


def weekly_path() -> str:
    return os.path.join(ds.v3_dir(), "weekly.parquet")


def load_weekly() -> pd.DataFrame:
    return pd.read_parquet(weekly_path())


# --------------------------------------------------------------------------- #
# KG embeddings (ablation only)
# --------------------------------------------------------------------------- #
def kg_embedding_frame(model_path: str, n_components: int = KG_N_COMPONENTS) -> pd.DataFrame:
    """ticker -> kg_pc0..kg_pc{n-1} using the production artefact's
    ``company_embeddings`` and its fitted ``pca``."""
    import pickle
    import sys
    import __main__
    try:  # the legacy pickle may reference a class pickled under __main__
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "scripts"))
        from kg_embeddings_classifier import TorchMLPRegressor  # noqa: F401
        __main__.TorchMLPRegressor = TorchMLPRegressor
    except Exception:
        pass
    d = pickle.load(open(model_path, "rb"))
    emb = d["company_embeddings"]
    tick = sorted(emb)
    E = np.array([np.asarray(emb[t], dtype=float) for t in tick])
    P = d["pca"].transform(E)[:, :n_components]
    cols = [f"kg_pc{i}" for i in range(P.shape[1])]
    return pd.DataFrame(P, columns=cols).assign(ticker=tick)[["ticker"] + cols]


def add_kg_features(frame: pd.DataFrame, kg: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Restrict to the common universe (tickers with an embedding; reused-symbol
    segments ``SYM~k`` are other companies and are excluded) and add per-date
    z-scored KG PCs. Base features are re-z-scored over the common universe."""
    cols = [c for c in kg.columns if c != "ticker"]
    f = frame[frame["ticker"].isin(set(kg["ticker"]))].reset_index(drop=True)
    f = f.merge(kg, on="ticker", how="left")
    from DataAnalysisPipeline2.features import core
    f[ds.FEATURE_COLUMNS + cols] = core.cross_sectional_zscore(f, ds.FEATURE_COLUMNS + cols)
    return f, cols


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _make(name: str, n_estimators: int | None = None, early_stopping: bool = True):
    p = dict(MODEL_PARAMS[name])
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        if n_estimators:
            p["n_estimators"] = n_estimators
        return LGBMRegressor(random_state=SEED, n_jobs=N_THREADS, verbose=-1, **p)
    if name == "xgb":
        from xgboost import XGBRegressor
        if n_estimators:
            p["n_estimators"] = n_estimators
        return XGBRegressor(random_state=SEED, n_jobs=N_THREADS, objective="reg:squarederror",
                            early_stopping_rounds=EARLY_STOPPING if early_stopping else None, **p)
    if name == "cat":
        from catboost import CatBoostRegressor
        if n_estimators:
            p["iterations"] = n_estimators
        return CatBoostRegressor(random_seed=SEED, thread_count=N_THREADS, verbose=0,
                                 loss_function="RMSE", allow_writing_files=False, **p)
    raise ValueError(name)


def _fit_es(name, Xf, yf, Xv, yv):
    m = _make(name)
    if name == "lgbm":
        import lightgbm as lgb
        m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        best = m.best_iteration_ or m.n_estimators
    elif name == "xgb":
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        best = int(m.best_iteration) + 1
    else:
        m.fit(Xf, yf, eval_set=(Xv, yv), early_stopping_rounds=EARLY_STOPPING)
        best = int(m.get_best_iteration()) + 1
    return m, max(int(best), MIN_TREES)


def split_fit_val(train_dates, calendar, val_weeks=VAL_WEEKS, embargo=EMBARGO_SESSIONS):
    """(fit_dates, val_dates): validation = last ``val_weeks`` decision dates
    (at most a quarter of the window, at least 13);
    fit dates end ``embargo`` sessions before the first validation date."""
    td = sorted(set(train_dates))
    val_weeks = int(min(val_weeks, max(13, len(td) // 4)))
    val = td[-val_weeks:]
    pos = {d: i for i, d in enumerate(calendar)}
    cut = pos[val[0]] - embargo
    fit = [d for d in td if pos[d] < cut]
    return fit, val


def train_window_dates(dates, calendar, test_start: str, embargo=EMBARGO_SESSIONS) -> list:
    """Decision dates allowed for training a model tested from ``test_start``:
    pos(t) < pos(first session >= test_start) - embargo."""
    cal = np.asarray(calendar)
    k0 = int(np.searchsorted(cal, test_start))
    pos = {d: i for i, d in enumerate(cal)}
    return sorted(d for d in set(dates) if pos[d] < k0 - embargo)


def fit_ensemble(frame: pd.DataFrame, feats: list, target: str, calendar, models=MODEL_NAMES,
                 refit: bool = True, log=print) -> tuple[dict, dict]:
    """Fit the ensemble on ``frame`` (rows of the training window). Returns
    (models, info) with per-model early-stopped tree counts and date ranges."""
    f = frame[frame[target].notna()]
    fit_d, val_d = split_fit_val(f["Date"].unique(), calendar)
    fm, vm = f["Date"].isin(set(fit_d)).to_numpy(), f["Date"].isin(set(val_d)).to_numpy()
    X = f[feats].astype(np.float32).reset_index(drop=True)
    y = f[target].to_numpy(dtype=np.float32)
    out, info = {}, {"fit_range": [fit_d[0], fit_d[-1]], "val_range": [val_d[0], val_d[-1]],
                     "train_range": [str(f["Date"].min()), str(f["Date"].max())],
                     "n_train_rows": int(len(f)), "n_train_dates": int(f["Date"].nunique()),
                     "best_iter": {}, "seconds": {}}
    for name in models:
        t0 = time.time()
        m, best = _fit_es(name, X[fm], y[fm], X[vm], y[vm])
        if refit:
            m = _make(name, n_estimators=best, early_stopping=False)
            if name == "xgb":
                m.fit(X, y, verbose=False)
            else:
                m.fit(X, y)
        out[name] = m
        info["best_iter"][name] = best
        info["seconds"][name] = round(time.time() - t0, 1)
        log(f"    {name}: best_iter={best} ({time.time() - t0:.0f}s)")
    return out, info


def predict_ensemble(models: dict, frame: pd.DataFrame, feats: list) -> np.ndarray:
    """Per-date rank average of the members' predictions, re-ranked to (0, 1]."""
    X = frame[feats].astype(np.float32).reset_index(drop=True)
    dates = frame["Date"].to_numpy()
    ranks = []
    for name, m in models.items():
        p = np.asarray(m.predict(X), dtype=float)
        ranks.append(pd.Series(p).groupby(dates).rank(pct=True).to_numpy())
    avg = np.mean(ranks, axis=0)
    return pd.Series(avg).groupby(dates).rank(pct=True).to_numpy()


def walk_forward(frame: pd.DataFrame, feats: list, target: str, calendar, years=TEST_YEARS,
                 models=MODEL_NAMES, log=print) -> tuple[pd.DataFrame, list]:
    """Out-of-sample predictions for every weekly date of each test year."""
    preds, folds = [], []
    for y in years:
        start = f"{y}-01-01"
        tr_dates = train_window_dates(frame["Date"].unique(), calendar, start)
        te = frame[(frame["Date"] >= start) & (frame["Date"] <= f"{y}-12-31")]
        if te.empty or not tr_dates:
            continue
        tr = frame[frame["Date"].isin(set(tr_dates))]
        log(f"  fold {y}: train {tr_dates[0]}..{tr_dates[-1]} ({len(tr):,} rows), "
            f"test {te['Date'].min()}..{te['Date'].max()} ({len(te):,} rows)")
        ms, info = fit_ensemble(tr, feats, target, calendar, models=models, log=log)
        p = predict_ensemble(ms, te, feats)
        preds.append(pd.DataFrame({"ticker": te["ticker"].to_numpy(), "Date": te["Date"].to_numpy(),
                                   "pred": p, "fold": y}))
        folds.append({"year": y, "test_range": [te["Date"].min(), te["Date"].max()], **info})
        del ms, tr
        gc.collect()
    return pd.concat(preds, ignore_index=True), folds


VARIANTS = {
    # name: (target column, universe, KG)
    "base": ("target_raw_30", "all", False),
    "h21": ("target_raw_21", "all", False),
    "sn30": ("target_sn_30", "all", False),
    "common_nokg": ("target_raw_30", "common", False),
    "common_kg": ("target_raw_30", "common", True),
}


def variant_frame(name: str, frame: pd.DataFrame, kg_model_path: str | None = None):
    target, univ, use_kg = VARIANTS[name]
    feats = list(ds.FEATURE_COLUMNS)
    if univ == "common":
        kg = kg_embedding_frame(kg_model_path or default_kg_path())
        f, kcols = add_kg_features(frame, kg)
        if use_kg:
            feats += kcols
        return f, feats, target
    return frame, feats, target


def default_kg_path() -> str:
    from DataAnalysisPipeline2.trading_agent import config
    return config.MODEL_PATH


def run_variant(name: str, models=MODEL_NAMES, log=print) -> pd.DataFrame:
    frame = load_weekly()
    cal = ds.calendar_from_dataset()
    f, feats, target = variant_frame(name, frame)
    del frame
    log(f"[wf] variant={name} target={target} rows={len(f):,} feats={len(feats)}")
    t0 = time.time()
    preds, folds = walk_forward(f, feats, target, cal, models=models, log=log)
    out = ds.v3_dir("wf")
    preds.to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
    json.dump({"variant": name, "target": target, "features": feats, "folds": folds,
               "embargo_sessions": EMBARGO_SESSIONS, "val_weeks": VAL_WEEKS,
               "model_params": MODEL_PARAMS, "seconds": round(time.time() - t0)},
              open(os.path.join(out, f"{name}.json"), "w"), indent=1)
    log(f"[wf] {name} done in {time.time() - t0:.0f}s")
    return preds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weekly", "wf"])
    ap.add_argument("--variant", default="base")
    a = ap.parse_args()
    if a.cmd == "weekly":
        w = build_weekly_frame()
        w.to_parquet(weekly_path(), index=False)
        print(f"weekly frame: {len(w):,} rows, {w['Date'].nunique()} dates")
    else:
        run_variant(a.variant)
