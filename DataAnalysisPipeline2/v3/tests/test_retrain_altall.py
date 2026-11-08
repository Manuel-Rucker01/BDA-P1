"""Quarterly retrain of the paper model (v3/retrain_altall.py): validation logic,
promotion / rollback and the log, on synthetic artefacts (no data store)."""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from DataAnalysisPipeline2.v3 import retrain_altall as ra

FEATS = ["f1", "f2", "f3", "a1"]


def _art(coefs, created="2026-11-07T10:00:00", model_id="v3_sn30_altall_hd_20261107", feats=FEATS,
         train_end="2026-09-18"):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATS))), columns=FEATS)
    models = {}
    for name, c in zip(("lgbm", "xgb"), coefs):
        m = LinearRegression().fit(X[FEATS], X[FEATS].to_numpy() @ np.asarray(c, float))
        models[name] = m
    man = {"artifact_schema_version": 3, "feature_version": 3, "model_id": model_id, "created_at": created,
           "variant": "sn30_altall_hd", "signal": "sn30_altall_hd", "target": "target_sn_30", "label_horizon": 30,
           "feature_columns": list(feats), "training_range": ["2017-01-06", train_end],
           "feature_profile": {"features": {f: {} for f in feats}}}
    return {"trained_models": models, "mix_models": list(models), "tabular_cols": list(feats), "manifest": man}


OLD = ([1, 0.5, 0, 0.2], [0.8, 0.6, 0.1, 0.2])


def _xsec(n=500, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATS))), columns=FEATS)
    X.insert(0, "ticker", [f"T{i:04d}" for i in range(n)])
    return X


# --------------------------------------------------------------------------- #
# (1) schema
# --------------------------------------------------------------------------- #
def test_schema_pass_and_failures():
    old, new = _art(OLD), _art(OLD)
    assert ra.check_schema(old, new, FEATS)["ok"]
    swapped = _art(OLD, feats=["f2", "f1", "f3", "a1"])
    r = ra.check_schema(old, swapped, FEATS)
    assert not r["ok"] and any("recipe" in p for p in r["problems"]) and any("live" in p for p in r["problems"])
    noprof = _art(OLD)
    noprof["manifest"]["feature_profile"]["features"].pop("a1")
    assert any("feature_profile" in p for p in ra.check_schema(old, noprof, FEATS)["problems"])
    tgt = _art(OLD)
    tgt["manifest"]["target"] = "target_sn_5"
    assert not ra.check_schema(old, tgt, FEATS)["ok"]
    members = _art(OLD)
    members["mix_models"] = ["lgbm"]
    assert not ra.check_schema(old, members, FEATS)["ok"]

    def bad(_):
        raise ValueError("nope")
    r = ra.check_schema(old, new, FEATS, validator=bad)
    assert not r["ok"] and "nope" in r["problems"][0]


# --------------------------------------------------------------------------- #
# (2) agreement
# --------------------------------------------------------------------------- #
def test_agreement_identical_close_and_reversed():
    X = _xsec()
    old = _art(OLD)
    same = ra.check_agreement(old, _art(OLD), X)
    assert same["ok"] and same["spearman"] == 1.0 and same["top_overlap"] == 100 and same["n"] == 500
    close = ra.check_agreement(old, _art(([1, 0.55, 0.05, 0.2], [0.8, 0.6, 0.1, 0.25])), X)
    assert close["ok"] and 0.95 < close["spearman"] < 1.0 and 50 < close["top_overlap"] <= 100
    rev = ra.check_agreement(old, _art(([-1, -0.5, 0, -0.2], [-0.8, -0.6, -0.1, -0.2])), X)
    assert not rev["ok"] and rev["spearman"] < -0.9 and rev["top_overlap"] == 0
    other = ra.check_agreement(old, _art(([0, 0, 1, 0], [0, 0, 1, 0])), X, min_corr=0.8)
    assert not other["ok"]


def test_predict_is_per_date_rank_average():
    X = _xsec(50)
    art = _art(OLD)
    p = ra.predict(art, X)
    assert p.min() > 0 and p.max() == 1.0
    ranks = [pd.Series(art["trained_models"][m].predict(X[FEATS])).rank(pct=True) for m in art["mix_models"]]
    np.testing.assert_allclose(p, pd.concat(ranks, axis=1).mean(axis=1).rank(pct=True).to_numpy())


# --------------------------------------------------------------------------- #
# (3) live == offline parity
# --------------------------------------------------------------------------- #
def test_parity_pass_and_failures():
    art = _art(OLD)
    off = _xsec(400)
    live = off.copy()
    live["f1"] = live["f1"] + 1e-6                       # float32 storage noise
    r = ra.check_parity(art, live, off, exact_cols=["a1"])
    assert r["ok"] and r["pred_spearman"] > 0.9999 and r["top_overlap"] >= 97 and r["nan_pattern_mismatch"] == 0
    bad = live.copy()
    bad["a1"] = bad["a1"] + 0.01                         # alt-data column must be exact
    r = ra.check_parity(art, bad, off, exact_cols=["a1"])
    assert not r["ok"] and any("alt-data" in p for p in r["problems"])
    drift = live.copy()
    drift.loc[:40, "f2"] += 0.05                          # >1e-4 of the cells off by >1e-3
    assert not ra.check_parity(art, drift, off, exact_cols=["a1"])["ok"]
    r = ra.check_parity(art, live.iloc[1:], off, exact_cols=["a1"])
    assert not r["ok"] and "universe" in r["problems"][0]
    r = ra.check_parity(art, live[["ticker", "f2", "f1", "f3", "a1"]], off, exact_cols=["a1"])
    assert not r["ok"] and "columns" in r["problems"][0]


# --------------------------------------------------------------------------- #
# (4) OOS IC of the live model on new weeks
# --------------------------------------------------------------------------- #
def test_oos_ic_only_after_training_end():
    art = _art(OLD)
    frames = []
    for i, d in enumerate(pd.date_range("2026-09-04", periods=8, freq="W-FRI")):
        X = _xsec(200, seed=i)
        X["Date"] = d.strftime("%Y-%m-%d")
        X["target_sn_30"] = pd.Series(ra.predict(art, X)).rank(pct=True).to_numpy()
        if i >= 6:
            X["target_sn_30"] = np.nan                     # label not matured yet
        frames.append(X)
    f = pd.concat(frames, ignore_index=True)
    r = ra.oos_ic(art, f, "2026-09-18", "target_sn_30")
    # weeks 09-25, 10-02, 10-09 (labelled, after the training end; 10-16 / 10-23 not matured)
    assert r["n_weeks"] == 3 and r["first"] == "2026-09-25" and r["last"] == "2026-10-09"
    assert np.isclose(r["mean_ic"], 1.0) and r["hit_rate"] == 1.0
    assert ra.oos_ic(art, f, "2026-12-31", "target_sn_30")["n_weeks"] == 0


def test_decide():
    good = {"schema": {"ok": True}, "agreement": {"ok": True}, "parity": {"ok": True}, "oos_ic": {"n_weeks": 0}}
    assert ra.decide(good) == (True, [])
    ok, why = ra.decide({**good, "parity": {"skipped": True}})
    assert not ok and why == ["parity: not run"]
    ok, why = ra.decide({**good, "agreement": {"ok": False, "problems": []}})
    assert not ok and why[0].startswith("agreement")
    assert not ra.decide({"schema": {"ok": False, "problems": ["x"]}})[0]


# --------------------------------------------------------------------------- #
# Promotion / rollback / log / run
# --------------------------------------------------------------------------- #
def _write(art, path):
    with open(path, "wb") as f:
        pickle.dump(art, f)


@pytest.fixture
def live_dir(tmp_path):
    d = tmp_path / "ez"
    d.mkdir()
    old = _art(OLD)
    _write(old, d / "model_v3_altall.pkl")
    json.dump(old["manifest"], open(d / "model_v3_altall.manifest.json", "w"))
    return d


def test_promote_keeps_prev_and_rollback(live_dir, tmp_path):
    cand = tmp_path / "cand.pkl"
    _write(_art(OLD, created="2026-11-07T23:00:00", train_end="2026-10-02"), cand)   # same day as the live one
    rep = {"schema": {"ok": True, "n_features": 4}, "agreement": {"ok": True, "spearman": 0.97},
           "parity": {"ok": True}, "oos_ic": {"n_weeks": 2, "weekly": {"a": 1}}, "latest_date": "2026-11-04"}
    mid = ra.promote(str(cand), rep, str(live_dir))
    assert mid == "v3_sn30_altall_hd_20261107_2"
    p = ra.live_paths(str(live_dir))
    new = ra.load_artifact(p["pkl"])
    assert new["manifest"]["model_id"] == mid and new["manifest"]["training_range"][1] == "2026-10-02"
    blk = new["manifest"]["retrain"]
    assert blk["previous_model_id"] == "v3_sn30_altall_hd_20261107" and "weekly" not in blk["checks"]["oos_ic"]
    assert json.load(open(p["json"]))["model_id"] == mid
    assert ra.load_artifact(p["prev_pkl"])["manifest"]["model_id"] == "v3_sn30_altall_hd_20261107"
    assert json.load(open(p["prev_json"]))["model_id"] == "v3_sn30_altall_hd_20261107"
    assert ra.rollback(str(live_dir)) == "v3_sn30_altall_hd_20261107"
    assert ra.load_artifact(p["pkl"])["manifest"]["model_id"] == "v3_sn30_altall_hd_20261107"


def test_new_model_id_is_dated():
    assert ra.new_model_id({"variant": "sn30_altall_hd", "created_at": "2027-01-04T06:00:00"},
                           "v3_sn30_altall_hd_20261107") == "v3_sn30_altall_hd_20270104"


def test_lfs_pointer_is_rejected(tmp_path):
    p = tmp_path / "m.pkl"
    p.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(RuntimeError, match="lfs"):
        ra.load_artifact(str(p))


def test_log_header_once(tmp_path):
    path = tmp_path / "retrain_log.md"
    rep = {"started_at": "2026-11-08T10:00:00", "status": "PROMOTED", "mode": "retrain", "old_model_id": "a",
           "new_model_id": "b", "old_training_end": "2026-09-18", "new_training_end": "2026-12-18",
           "steps": {"train": 300}, "schema": {"ok": True, "n_features": 65},
           "agreement": {"ok": True, "spearman": 0.93, "min_spearman": 0.8, "top_k": 100, "top_overlap": 71, "n": 3000},
           "parity": {"ok": True, "n_live": 3000, "exact_cols_max_abs": 0.0, "share_cells_gt_tol": 0.0, "max_abs": 1e-6,
                      "pred_spearman": 1.0, "top_k": 100, "top_overlap": 100},
           "oos_ic": {"after": "2026-09-18", "n_weeks": 13, "first": "x", "last": "y", "mean_ic": 0.05,
                      "t_stat": 2.1, "hit_rate": 0.7}, "latest_date": "2027-01-01"}
    ra.append_log(rep, str(path))
    ra.append_log({**rep, "status": "REJECTED (live model kept)", "reasons": ["agreement: x"]}, str(path))
    txt = path.read_text()
    assert txt.count("# Paper-model retrain log") == 1 and txt.count("## 2026-11-08 10:00") == 2
    assert "spearman 0.93" in txt and "13 weeks" in txt and "not promoted: agreement: x" in txt


def _args(live_dir, tmp_path, **kw):
    base = dict(skip_refresh=True, skip_frame=True, validate_only=False, candidate=None, no_promote=False,
                skip_parity=False, min_corr=0.8, live_dir=str(live_dir), log_md=str(tmp_path / "log.md"),
                no_log=False, rollback=False)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("passes", [True, False])
def test_run_promotes_only_when_checks_pass(live_dir, tmp_path, monkeypatch, passes):
    cand = tmp_path / "cand.pkl"
    _write(_art(OLD, created="2026-12-20T06:00:00", train_end="2026-10-30"), cand)
    monkeypatch.setattr(ra, "candidate_dir", lambda: str(tmp_path))
    canned = {"schema": {"ok": True, "n_features": 4}, "agreement": {"ok": passes, "spearman": 0.9 if passes else 0.5,
              "problems": []}, "parity": {"ok": True}, "oos_ic": {"n_weeks": 0}, "latest_date": "2026-12-18"}
    monkeypatch.setattr(ra, "validate", lambda *a, **k: dict(canned))
    rc = ra.run(_args(live_dir, tmp_path, candidate=str(cand)), log=lambda *_: None)
    p = ra.live_paths(str(live_dir))
    mid = ra.load_artifact(p["pkl"])["manifest"]["model_id"]
    if passes:
        assert rc == 0 and mid == "v3_sn30_altall_hd_20261220" and os.path.exists(p["prev_pkl"])
        assert "PROMOTED" in (tmp_path / "log.md").read_text()
    else:
        assert rc == 3 and mid == "v3_sn30_altall_hd_20261107" and not os.path.exists(p["prev_pkl"])
        assert "REJECTED" in (tmp_path / "log.md").read_text()
    assert list(tmp_path.glob("retrain_*.json"))


def test_run_no_promote_and_errors(live_dir, tmp_path, monkeypatch):
    cand = tmp_path / "cand.pkl"
    _write(_art(OLD, created="2026-12-20T06:00:00"), cand)
    monkeypatch.setattr(ra, "candidate_dir", lambda: str(tmp_path))
    good = {"schema": {"ok": True}, "agreement": {"ok": True}, "parity": {"ok": True}, "oos_ic": {}}
    monkeypatch.setattr(ra, "validate", lambda *a, **k: dict(good))
    assert ra.run(_args(live_dir, tmp_path, candidate=str(cand), no_promote=True), log=lambda *_: None) == 0
    assert not os.path.exists(ra.live_paths(str(live_dir))["prev_pkl"])

    def boom(*a, **k):
        raise RuntimeError("store missing")
    monkeypatch.setattr(ra, "validate", boom)
    assert ra.run(_args(live_dir, tmp_path, candidate=str(cand)), log=lambda *_: None) == 1
    assert "error: RuntimeError: store missing" in (tmp_path / "log.md").read_text()


def test_refresh_steps_are_existing_clis():
    import importlib.util
    for name, args in ra.REFRESH_STEPS:
        assert importlib.util.find_spec(args[0]) is not None, name
    names = [n for n, _ in ra.REFRESH_STEPS]
    # order: prices before the dataset, fundamentals before the EDGAR dataset, labels after the weekly frame
    assert names.index("prices") < names.index("dataset_edgar")
    assert names.index("fundamentals") < names.index("dataset_edgar") < names.index("weekly_edgar") < names.index("labels")
