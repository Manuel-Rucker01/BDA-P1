"""Quarterly retrain of the paper model ``ExploitationZone/model_v3_altall.pkl``
(signal ``sn30_altall_hd``, ``BDA_SIGNAL=altall``).

Engineering only -- NOT a new evaluation trial: the recipe is fixed
(``final_altall.train``: same features, target, parameters, 52-week embargoed
early stopping, refit on every labelled week). A retrain only adds the weeks
whose 30-session label has matured since the last fit, so N_TRIALS of the
walk-forward report does not change and nothing here touches ``v3/gate.py``.

Steps
-----
1. ``refresh`` (``--skip-refresh`` to skip): the existing data CLIs, in order
   (each one a subprocess, stops at the first failure):

       python -m DataAnalysisPipeline2.v3.prices update             # Alpaca SIP daily bars (incremental)
       python -m DataAnalysisPipeline2.v3.edgar all --force         # SEC companyfacts/submissions + CIK/sector maps
       python -m DataAnalysisPipeline2.v3.fundamentals build        # PIT fundamentals_asof
       python -m DataAnalysisPipeline2.v3.dataset --edgar           # daily v3 dataset (EDGAR sectors + fundamentals)
       python -m DataAnalysisPipeline2.v3.train weekly --dataset dataset_edgar
       python -m DataAnalysisPipeline2.v3.round6 labels             # weekly_mh (labels, z-scored core + fund)
       python -m DataAnalysisPipeline2.v3.altlive refresh           # daily Form 4 feed + FINRA files

   These rewrite SHARED research files (``dataset_edgar``, ``weekly_dataset_edgar``,
   ``weekly_mh``) in ``$BDA_DATA_V3``: never run the refresh while a research
   round is reading them. The base ``dataset`` (Round 1-3 frame) is not
   rebuilt; training takes the session calendar from ``dataset_edgar``.
   New listings only enter the price store after a universe refresh
   (``v3.universe snapshot/discover/classify``), which is not automated here.
2. ``frame`` (``--skip-frame``): ``final_altall.build_frame`` (alt_weekly_final +
   weekly_dataset_r8_final).
3. ``train``: ``final_altall.train`` into ``$BDA_DATA_V3/v3model/retrain/
   model_v3_altall.candidate.pkl`` (never straight into ExploitationZone/).
4. ``validate`` the candidate against the live model:
   (1) schema: feature list == the recipe's == the live model's, same members,
       target, feature version; the bot's ``validate_model_artifact_schema``;
       feature profile over every feature;
   (2) agreement: Spearman of candidate vs live predictions over the eligible
       names of the latest decision date >= ``MIN_RANK_CORR`` (0.8); top-100
       overlap reported;
   (3) live == offline parity on the latest decision date (the bot's
       ``altall.model_input`` path vs the training frame, thresholds of
       ``trading_agent/test_altall.py``) with the candidate's predictions;
   (4) informational: out-of-sample IC of the LIVE model on the labelled weeks
       added since its training range ended.
5. ``promote`` only if (1)-(3) pass: live pickle + manifest copied to
   ``model_v3_altall.prev.{pkl,manifest.json}`` (gitignored; older versions
   live in git history), candidate written over the live artefact with a new
   dated ``model_id`` and a ``retrain`` block in its manifest, and an entry
   appended to ``DataAnalysisPipeline2/results/retrain_log.md``.

CLI::

    python -m DataAnalysisPipeline2.v3.retrain_altall                     # full run (refresh..promote)
    python -m DataAnalysisPipeline2.v3.retrain_altall --skip-refresh      # data already fresh
    python -m DataAnalysisPipeline2.v3.retrain_altall --validate-only     # dry run: live model vs itself
    python -m DataAnalysisPipeline2.v3.retrain_altall --validate-only --candidate PATH
    python -m DataAnalysisPipeline2.v3.retrain_altall --rollback          # restore model_v3_altall.prev.*

Exit codes: 0 = promoted (or validation passed with ``--no-promote`` /
``--validate-only``), 3 = validation failed (live model untouched), 1 = error.
``DataAnalysisPipeline2/scripts/retrain_altall.sh`` wraps it for a scheduler
(retrain -> pytest -> commit -> push, rollback if the tests fail).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIVE_DIR = os.path.join(ROOT, "ExploitationZone")
NAME = "model_v3_altall"
PREV_NAME = NAME + ".prev"
CANDIDATE_NAME = NAME + ".candidate"
LOG_MD = os.path.join(ROOT, "DataAnalysisPipeline2", "results", "retrain_log.md")

MIN_RANK_CORR = 0.8
TOP_K = 100
# live == offline thresholds (same as trading_agent/test_altall.py)
PARITY = {"exact_max": 1e-4, "cell_tol": 1e-3, "cell_share_max": 1e-4, "max_abs": 0.1,
          "pred_spearman_min": 0.9999, "top_overlap_min": 97}
MIN_IC_NAMES = 30

REFRESH_STEPS = [
    ("prices", ["DataAnalysisPipeline2.v3.prices", "update"]),
    ("edgar", ["DataAnalysisPipeline2.v3.edgar", "all", "--force"]),
    ("fundamentals", ["DataAnalysisPipeline2.v3.fundamentals", "build"]),
    ("dataset_edgar", ["DataAnalysisPipeline2.v3.dataset", "--edgar"]),
    ("weekly_edgar", ["DataAnalysisPipeline2.v3.train", "weekly", "--dataset", "dataset_edgar"]),
    ("labels", ["DataAnalysisPipeline2.v3.round6", "labels"]),
    ("altlive", ["DataAnalysisPipeline2.v3.altlive", "refresh"]),
]


def _v3(name):
    import importlib
    return importlib.import_module(f"DataAnalysisPipeline2.v3.{name}")


def live_paths(live_dir: str = LIVE_DIR) -> dict:
    j = lambda n, ext: os.path.join(live_dir, f"{n}.{ext}")
    return {"pkl": j(NAME, "pkl"), "json": j(NAME, "manifest.json"),
            "prev_pkl": j(PREV_NAME, "pkl"), "prev_json": j(PREV_NAME, "manifest.json")}


def candidate_dir() -> str:
    return _v3("dataset").v3_dir("retrain")


# --------------------------------------------------------------------------- #
# Artefact helpers
# --------------------------------------------------------------------------- #
def load_artifact(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{path} is a git-lfs pointer (run `git lfs pull`)")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(art: dict, X: pd.DataFrame, dates=None) -> np.ndarray:
    """The bot's score: per-date rank average of the members, re-ranked."""
    tr = _v3("train")
    fr = X[list(art["tabular_cols"])].copy()
    fr["Date"] = np.asarray(dates) if dates is not None else "d"
    models = {m: art["trained_models"][m] for m in art["mix_models"]}
    return tr.predict_ensemble(models, fr, list(art["tabular_cols"]))


def _spearman(a, b) -> float:
    return float(pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method="spearman"))


def _top(scores, k: int) -> set:
    return set(np.argsort(-np.asarray(scores, float), kind="stable")[:k])


# --------------------------------------------------------------------------- #
# Validation (pure; unit-tested on synthetic artefacts)
# --------------------------------------------------------------------------- #
_SAME_MANIFEST_KEYS = ("feature_version", "target", "label_horizon", "variant", "signal", "artifact_schema_version")


def check_schema(old: dict, new: dict, expected_feats: list, validator=None) -> dict:
    """(1) Same feature list as the recipe and the live model, same members and
    target; the bot's artefact contract; a feature profile for every feature."""
    problems = []
    feats = list(new.get("tabular_cols", []))
    if feats != list(expected_feats):
        problems.append(f"candidate features != recipe features ({len(feats)} vs {len(expected_feats)}; "
                        f"diff {sorted(set(feats) ^ set(expected_feats))[:10]})")
    if feats != list(old.get("tabular_cols", [])):
        problems.append("candidate features (names or order) != live model features")
    if list(new.get("mix_models", [])) != list(old.get("mix_models", [])):
        problems.append(f"members {new.get('mix_models')} != live {old.get('mix_models')}")
    mo, mn = old.get("manifest", {}), new.get("manifest", {})
    for k in _SAME_MANIFEST_KEYS:
        if mo.get(k) != mn.get(k):
            problems.append(f"manifest[{k!r}] {mn.get(k)!r} != live {mo.get(k)!r}")
    if mn.get("feature_columns") is not None and list(mn["feature_columns"]) != feats:
        problems.append("manifest feature_columns != tabular_cols")
    prof = set((mn.get("feature_profile") or {}).get("features", {}))
    if prof != set(feats):
        problems.append(f"feature_profile covers {len(prof)} of {len(feats)} features")
    missing_models = [m for m in new.get("mix_models", []) if m not in new.get("trained_models", {})]
    if missing_models:
        problems.append(f"trained_models lacks {missing_models}")
    if validator is not None:
        try:
            validator(new)
        except Exception as e:
            problems.append(f"bot artefact contract: {type(e).__name__}: {e}")
    return {"ok": not problems, "problems": problems, "n_features": len(feats)}


def check_agreement(old: dict, new: dict, X: pd.DataFrame, min_corr: float = MIN_RANK_CORR,
                    top_k: int = TOP_K) -> dict:
    """(2) Spearman of candidate vs live scores on one cross-section."""
    po, pn = predict(old, X), predict(new, X)
    rho = _spearman(po, pn)
    k = min(top_k, len(X))
    ov = len(_top(po, k) & _top(pn, k))
    return {"ok": bool(np.isfinite(rho) and rho >= min_corr), "n": int(len(X)), "spearman": round(rho, 4),
            "min_spearman": min_corr, "top_k": k, "top_overlap": ov, "top_overlap_share": round(ov / max(k, 1), 3)}


def check_parity(art: dict, live: pd.DataFrame, off: pd.DataFrame, exact_cols: list, thr: dict = PARITY) -> dict:
    """(3) live model input == offline training rows (``ticker`` + features,
    both sorted by ticker), and the candidate's scores on both agree."""
    feats = list(art["tabular_cols"])
    problems = []
    res = {"n_live": int(len(live)), "n_offline": int(len(off))}
    if list(live["ticker"]) != list(off["ticker"]):
        a, b = set(live["ticker"]), set(off["ticker"])
        problems.append(f"eligible universe differs: {len(a - b)} live-only, {len(b - a)} offline-only")
        return {**res, "ok": False, "problems": problems}
    cols = [c for c in live.columns if c != "ticker"]
    if cols != feats or not set(feats) <= set(off.columns):
        problems.append("live/offline columns != model features")
        return {**res, "ok": False, "problems": problems}
    L, O = live[feats].astype(float), off[feats].astype(float)
    d = (L - O).abs()
    ex = [c for c in exact_cols if c in feats]
    res.update({
        "exact_cols_max_abs": float(d[ex].max().max()) if ex else 0.0,
        "share_cells_gt_tol": float((d > thr["cell_tol"]).to_numpy().mean()),
        "max_abs": float(np.nan_to_num(d.max().max())),
        "nan_pattern_mismatch": int((L.isna() != O.isna()).to_numpy().sum()),
    })
    pl, po = predict(art, live), predict(art, off)
    k = min(TOP_K, len(live))
    res.update({"pred_spearman": round(_spearman(pl, po), 6), "top_overlap": len(_top(pl, k) & _top(po, k)),
                "top_k": k})
    if not res["exact_cols_max_abs"] < thr["exact_max"]:
        problems.append(f"alt-data/fundamental columns differ by {res['exact_cols_max_abs']:.2g}")
    if not res["share_cells_gt_tol"] < thr["cell_share_max"]:
        problems.append(f"{res['share_cells_gt_tol']:.2%} of cells differ by > {thr['cell_tol']}")
    if not res["max_abs"] < thr["max_abs"]:
        problems.append(f"max abs difference {res['max_abs']:.3g}")
    if not res["pred_spearman"] > thr["pred_spearman_min"]:
        problems.append(f"prediction spearman {res['pred_spearman']}")
    top_min = thr["top_overlap_min"] * k / TOP_K
    if not res["top_overlap"] >= top_min:
        problems.append(f"top-{k} overlap {res['top_overlap']} < {top_min:g}")
    return {**res, "ok": not problems, "problems": problems}


def oos_ic(art: dict, frame: pd.DataFrame, after: str, target: str) -> dict:
    """(4) Per-week Spearman IC of ``art`` on labelled rows dated after ``after``
    (its training range end): weeks it has never seen."""
    f = frame[(frame["Date"].astype(str) > str(after)) & frame[target].notna()].reset_index(drop=True)
    out = {"after": str(after), "n_weeks": 0, "weekly": {}}
    if f.empty:
        return out
    p = predict(art, f, dates=f["Date"].astype(str).to_numpy())
    f = f.assign(_p=p)
    ics = {d: _spearman(g["_p"], g[target]) for d, g in f.groupby(f["Date"].astype(str)) if len(g) > MIN_IC_NAMES}
    s = pd.Series(ics, dtype=float).dropna()
    if s.empty:
        return out
    out.update({"n_weeks": int(len(s)), "first": s.index[0], "last": s.index[-1],
                "mean_ic": round(float(s.mean()), 4),
                "t_stat": round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), 2)
                if len(s) > 1 and s.std(ddof=1) > 0 else None,
                "hit_rate": round(float((s > 0).mean()), 3),
                "weekly": {k: round(float(v), 4) for k, v in s.items()}})
    return out


def decide(report: dict) -> tuple[bool, list]:
    """Promote only if schema, agreement and parity all pass (OOS IC is informational)."""
    reasons = []
    for k in ("schema", "agreement", "parity"):
        c = report.get(k)
        if c is None or c.get("skipped"):
            reasons.append(f"{k}: not run")
        elif not c.get("ok"):
            reasons.append(f"{k}: " + ("; ".join(c.get("problems", [])) or "failed"))
    return not reasons, reasons


# --------------------------------------------------------------------------- #
# Promotion / rollback / log
# --------------------------------------------------------------------------- #
def new_model_id(new_manifest: dict, old_id: str | None) -> str:
    base = f"v3_{new_manifest.get('variant', 'sn30_altall_hd')}_{str(new_manifest['created_at'])[:10].replace('-', '')}"
    mid, i = base, 2
    while mid == old_id:
        mid, i = f"{base}_{i}", i + 1
    return mid


def retrain_block(report: dict, old_manifest: dict) -> dict:
    keep = lambda c: {k: v for k, v in (c or {}).items() if k not in ("weekly", "problems")}
    return {"previous_model_id": old_manifest.get("model_id"),
            "previous_training_range": old_manifest.get("training_range"),
            "validated_at": report.get("finished_at") or dt.datetime.now().isoformat(timespec="seconds"),
            "latest_decision_date": report.get("latest_date"),
            "checks": {k: keep(report.get(k)) for k in ("schema", "agreement", "parity", "oos_ic")},
            "rule": "fixed recipe (final_altall.train); promote iff schema + agreement (spearman >= "
                    f"{MIN_RANK_CORR}) + live==offline parity pass; not a new evaluation trial",
            "tool": "DataAnalysisPipeline2/v3/retrain_altall.py"}


def promote(candidate_pkl: str, report: dict, live_dir: str = LIVE_DIR) -> str:
    """Keep the live artefact as ``.prev``, write the candidate over it with a
    new dated model_id and the ``retrain`` block. Returns the new model_id."""
    af = _v3("artefact")
    p = live_paths(live_dir)
    old = load_artifact(p["pkl"])
    shutil.copy2(p["pkl"], p["prev_pkl"])
    if os.path.exists(p["json"]):
        shutil.copy2(p["json"], p["prev_json"])
    art = load_artifact(candidate_pkl)
    art["manifest"]["model_id"] = new_model_id(art["manifest"], old["manifest"].get("model_id"))
    art["manifest"]["retrain"] = retrain_block(report, old["manifest"])
    af.save_artifact(art, live_dir, NAME)
    return art["manifest"]["model_id"]


def rollback(live_dir: str = LIVE_DIR) -> str:
    p = live_paths(live_dir)
    if not os.path.exists(p["prev_pkl"]):
        raise FileNotFoundError(f"{p['prev_pkl']} not found (older versions: git history of {p['pkl']})")
    shutil.copy2(p["prev_pkl"], p["pkl"])
    if os.path.exists(p["prev_json"]):
        shutil.copy2(p["prev_json"], p["json"])
    return load_artifact(p["pkl"])["manifest"].get("model_id")


_LOG_HEADER = """# Paper-model retrain log (`model_v3_altall`, signal `sn30_altall_hd`)

Quarterly retrain of the paper opt-in model with the FIXED recipe
(`v3/final_altall.py`; engineering, not an evaluation trial -- N_TRIALS and the
gate are unchanged). Each run validates the candidate against the live model
before replacing it: (1) identical schema / feature list, (2) Spearman of the
two models' scores on the latest decision date >= 0.8 (top-100 overlap
reported), (3) live == offline parity of the bot's model input on that date,
(4) informational out-of-sample IC of the live model on the newly labelled weeks.

How to run (locally; needs `$BDA_DATA_V3`, Alpaca keys and `SEC_USER_AGENT` in
`DataAnalysisPipeline2/trading_agent/.env`; ~1-2 h with the data refresh):

    python -m DataAnalysisPipeline2.v3.retrain_altall                  # refresh -> frame -> train -> validate -> promote
    python -m DataAnalysisPipeline2.v3.retrain_altall --validate-only  # dry run on the live model
    DataAnalysisPipeline2/scripts/retrain_altall.sh                    # + pytest, commit, push (scheduler entry point)

Scheduler entry (not installed automatically), e.g. cron on the first Saturday
of Jan/Apr/Jul/Oct at 06:00 when no research round is running:

    0 6 1-7 1,4,7,10 6 cd /path/to/BDA-P1 && DataAnalysisPipeline2/scripts/retrain_altall.sh >> /tmp/retrain_altall.log 2>&1

Rollback: `python -m DataAnalysisPipeline2.v3.retrain_altall --rollback`
(restores `ExploitationZone/model_v3_altall.prev.*`), or `git revert` the commit.

"""


def format_log_entry(report: dict) -> str:
    sc, ag, pa, ic = (report.get(k) or {} for k in ("schema", "agreement", "parity", "oos_ic"))
    st = report.get("status", "?")
    lines = [f"## {report.get('started_at', '?')[:16].replace('T', ' ')} -- {st}", ""]
    lines.append(f"- mode: {report.get('mode')}; live model `{report.get('old_model_id')}` "
                 f"(training to {report.get('old_training_end')}); candidate trained to "
                 f"{report.get('new_training_end')}" + (f"; new model_id `{report['new_model_id']}`"
                                                         if report.get("new_model_id") else ""))
    if report.get("steps"):
        lines.append("- steps: " + ", ".join(f"{k} {v}s" for k, v in report["steps"].items()))
    lines.append(f"- (1) schema: {'pass' if sc.get('ok') else 'FAIL'} ({sc.get('n_features')} features)"
                 + ("; " + "; ".join(sc.get("problems", [])) if sc.get("problems") else ""))
    if ag:
        lines.append(f"- (2) agreement on {report.get('latest_date')}: spearman {ag.get('spearman')} "
                     f"(min {ag.get('min_spearman')}), top-{ag.get('top_k')} overlap {ag.get('top_overlap')} "
                     f"over {ag.get('n')} names -> {'pass' if ag.get('ok') else 'FAIL'}")
    if pa.get("skipped"):
        lines.append("- (3) parity: skipped")
    elif pa:
        lines.append(f"- (3) live==offline on {report.get('latest_date')}: {pa.get('n_live')} names, alt/fund max "
                     f"|d| {pa.get('exact_cols_max_abs', float('nan')):.2g}, cells > 1e-3 "
                     f"{pa.get('share_cells_gt_tol', float('nan')):.2e}, max |d| {pa.get('max_abs', float('nan')):.3g}, "
                     f"pred spearman {pa.get('pred_spearman')}, top-{pa.get('top_k')} overlap {pa.get('top_overlap')}"
                     f" -> {'pass' if pa.get('ok') else 'FAIL'}"
                     + ("; " + "; ".join(pa.get("problems", [])) if pa.get("problems") else ""))
    if ic:
        lines.append(f"- (4) live-model OOS IC after {ic.get('after')}: "
                     + (f"{ic['n_weeks']} weeks {ic.get('first')}..{ic.get('last')}, mean {ic.get('mean_ic')}, "
                        f"t {ic.get('t_stat')}, hit {ic.get('hit_rate')}" if ic.get("n_weeks")
                        else "no newly labelled weeks"))
    if report.get("reasons"):
        lines.append("- not promoted: " + "; ".join(report["reasons"]))
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    return "\n".join(lines) + "\n\n"


def append_log(report: dict, path: str = LOG_MD) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(_LOG_HEADER)
        f.write(format_log_entry(report))
    return path


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def refresh(log=print, steps=REFRESH_STEPS, python: str = sys.executable) -> dict:
    took = {}
    for name, args in steps:
        t0 = time.time()
        log(f"[retrain] refresh {name}: python -m {' '.join(args)}")
        subprocess.run([python, "-m", *args], cwd=ROOT, check=True)
        took[name] = round(time.time() - t0)
    return took


def validate(old_pkl: str, cand_pkl: str, frame_path: str | None = None, parity: bool = True,
             min_corr: float = MIN_RANK_CORR, log=print) -> dict:
    fa, fu, altlive = _v3("final_altall"), _v3("fundamentals"), _v3("altlive")
    from DataAnalysisPipeline2.trading_agent import bot
    frame_path = frame_path or fa.frame_path()
    old, new = load_artifact(old_pkl), load_artifact(cand_pkl)
    feats = fa.feature_columns()
    rep = {"old_model_id": old["manifest"].get("model_id"), "new_model_id_candidate": new["manifest"].get("model_id"),
           "old_training_end": (old["manifest"].get("training_range") or [None, None])[1],
           "new_training_end": (new["manifest"].get("training_range") or [None, None])[1]}
    rep["schema"] = check_schema(old, new, feats, validator=bot.validate_model_artifact_schema)
    log(f"[retrain] (1) schema: {rep['schema']}")
    dates = pd.read_parquet(frame_path, columns=["Date"])["Date"].astype(str)
    latest = str(dates.max())
    rep["latest_date"] = latest
    if not rep["schema"]["ok"]:
        return rep
    X = pd.read_parquet(frame_path, columns=["ticker", "Date"] + feats, filters=[("Date", "==", latest)])
    X = X.sort_values("ticker").reset_index(drop=True)
    rep["agreement"] = check_agreement(old, new, X, min_corr=min_corr)
    log(f"[retrain] (2) agreement on {latest}: {rep['agreement']}")
    if parity:
        from DataAnalysisPipeline2.trading_agent import altall
        t0 = time.time()
        live, off, info = altall.offline_vs_live(latest, frame_path=frame_path)
        rep["parity"] = check_parity(new, live, off, list(altlive.MODEL_COLUMNS) + list(fu.FEATURE_COLUMNS))
        rep["parity"]["seconds"] = round(time.time() - t0)
        rep["parity"]["live_info"] = {k: v for k, v in info.items() if k != "sector_source"}
        log(f"[retrain] (3) parity: { {k: v for k, v in rep['parity'].items() if k != 'live_info'} }")
    else:
        rep["parity"] = {"skipped": True}
    target = old["manifest"].get("target", fa.TARGET)
    after = rep["old_training_end"]
    f = pd.read_parquet(frame_path, columns=["ticker", "Date", target] + feats, filters=[("Date", ">", after)])
    rep["oos_ic"] = oos_ic(old, f, after, target)
    log(f"[retrain] (4) live-model OOS IC: { {k: v for k, v in rep['oos_ic'].items() if k != 'weekly'} }")
    return rep


def run(a, log=print) -> int:
    fa, ds = _v3("final_altall"), _v3("dataset")
    p = live_paths(a.live_dir)
    report = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "steps": {},
              "mode": "validate-only" if a.validate_only else ("no-promote" if a.no_promote else "retrain")}
    rc = 1
    try:
        if not a.validate_only and not a.skip_refresh:
            report["steps"].update(refresh(log=log))
        if not a.validate_only and not a.skip_frame:
            t0 = time.time()
            fa.build_frame(log=log)
            report["steps"]["frame"] = round(time.time() - t0)
        if a.candidate:
            cand = a.candidate
        elif a.validate_only:
            cand = p["pkl"]
            log("[retrain] dry run: validating the live model against itself")
        else:
            t0 = time.time()
            cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
            cand, _ = fa.train(log=log, out_dir=candidate_dir(), name=CANDIDATE_NAME, calendar=cal)
            report["steps"]["train"] = round(time.time() - t0)
        report["candidate"] = cand
        t0 = time.time()
        report.update(validate(p["pkl"], cand, parity=not a.skip_parity, min_corr=a.min_corr, log=log))
        report["steps"]["validate"] = round(time.time() - t0)
        ok, reasons = decide(report)
        report["reasons"] = reasons
        if not ok:
            report["status"], rc = "REJECTED (live model kept)", 3
        elif a.validate_only or a.no_promote or cand == p["pkl"]:
            report["status"], rc = "validated (not promoted: " + report["mode"] + ")", 0
        else:
            report["new_model_id"] = promote(cand, report, a.live_dir)
            report["status"], rc = "PROMOTED", 0
    except Exception as e:
        report["status"], report["error"] = "ERROR", f"{type(e).__name__}: {e}"
        log(f"[retrain] ERROR {report['error']}")
        rc = 1
    report["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    out = os.path.join(candidate_dir(), f"retrain_{report['started_at'].replace(':', '')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    log(f"[retrain] {report['status']} (report {out})")
    if not a.no_log:
        append_log(report, a.log_md)
        log(f"[retrain] log entry appended to {a.log_md}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m DataAnalysisPipeline2.v3.retrain_altall",
                                 description="Quarterly retrain of ExploitationZone/model_v3_altall.pkl")
    ap.add_argument("--skip-refresh", action="store_true", help="do not run the data refresh CLIs")
    ap.add_argument("--skip-frame", action="store_true", help="reuse the existing final frame")
    ap.add_argument("--validate-only", action="store_true",
                    help="no refresh/frame/train/promotion; validate --candidate (default: the live model)")
    ap.add_argument("--candidate", default=None, help="validate this pickle instead of training one")
    ap.add_argument("--no-promote", action="store_true", help="train + validate, keep the live model")
    ap.add_argument("--skip-parity", action="store_true", help="skip check (3) (then nothing is promoted)")
    ap.add_argument("--min-corr", type=float, default=MIN_RANK_CORR)
    ap.add_argument("--live-dir", default=LIVE_DIR)
    ap.add_argument("--log-md", default=LOG_MD)
    ap.add_argument("--no-log", action="store_true", help="do not append to retrain_log.md")
    ap.add_argument("--rollback", action="store_true", help="restore model_v3_altall.prev.* and exit")
    a = ap.parse_args(argv)
    if a.rollback:
        print(f"[retrain] restored {rollback(a.live_dir)}")
        return 0
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
