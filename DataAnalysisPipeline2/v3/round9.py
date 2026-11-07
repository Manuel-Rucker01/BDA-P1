"""Round 9 (pre-registered in ``v3/gate.py``): SEC Form 13F institutional
holdings + SEC fails-to-deliver on top of sn30_altall_hd; leak probes; the
+0.005 IC book screen; books through Round 5's ``mv_abs_all``; gate.

    python -m DataAnalysisPipeline2.v3.round9 features frame probe wf ics books gate
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import altdata_r9 as a9
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import train as tr

R9_DATASET = "dataset_r9"
NEW_COLS = a9.INST_COLUMNS + a9.FTD_COLUMNS
RAW_COLS = [f"{c}_raw" for c in NEW_COLS]
MAPPED = {"inst13f": "inst_mapped", "ftd": "ftd_mapped"}


def frame_path() -> str:
    return tr.weekly_path(R9_DATASET)


def alt9_path() -> str:
    return os.path.join(ds.v3_dir(), "alt9_weekly.parquet")


def block_hotdeck(w: pd.DataFrame, cols: list, has: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Round 8 hot-deck, generalised: rows with ``has`` False get the whole
    ``cols`` vector of a same-date donor with ``has`` True among their k nearest
    neighbours on round8.HD_MATCH (deterministic hash pick)."""
    vals = w[cols].to_numpy(dtype=float).copy()
    M = w[r8.HD_MATCH].to_numpy(dtype=float)
    dates = w["Date"].to_numpy().astype(str)
    tick = w["ticker"].to_numpy().astype(str)
    order = np.argsort(dates, kind="mergesort")
    ud, starts = np.unique(dates[order], return_index=True)
    bounds = list(starts) + [len(order)]
    for i, d in enumerate(ud):
        idx = order[bounds[i]:bounds[i + 1]]
        donors, rec = idx[has[idx]], idx[~has[idx]]
        if len(donors) == 0 or len(rec) == 0:
            continue
        Xd, Xr = np.nan_to_num(M[donors]), np.nan_to_num(M[rec])
        d2 = (Xr ** 2).sum(1)[:, None] + (Xd ** 2).sum(1)[None, :] - 2.0 * Xr @ Xd.T
        kk = min(k, len(donors))
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(d2, nn, axis=1), axis=1, kind="stable"), axis=1)
        h = ds._stable_u64([f"{tick[r]}|{d}" for r in rec])
        vals[rec] = vals[donors[nn[np.arange(len(rec)), (h % np.uint64(kk)).astype(np.int64)]]]
    return pd.DataFrame(vals, columns=cols, index=w.index)


def build_frame(log=print) -> pd.DataFrame:
    """Round-8 evaluation frame + the round-9 blocks: raw (``*_raw``, unmapped
    rows missing) and hot-decked (canonical names), z-scored per date."""
    w = pd.read_parquet(r8.frame_path())
    alt = pd.read_parquet(alt9_path(), columns=["ticker", "Date", "inst_mapped", "ftd_mapped"] + NEW_COLS)
    w = w.merge(alt, on=["ticker", "Date"], how="left")
    for blk, flag in MAPPED.items():
        w[flag] = w[flag].fillna(0.0)
    w[RAW_COLS] = w[NEW_COLS].to_numpy()
    w[a9.INST_COLUMNS] = block_hotdeck(w, a9.INST_COLUMNS, w["inst_mapped"].to_numpy() > 0)
    w[a9.FTD_COLUMNS] = block_hotdeck(w, a9.FTD_COLUMNS, w["ftd_mapped"].to_numpy() > 0)
    w[NEW_COLS + RAW_COLS] = core.cross_sectional_zscore(w, NEW_COLS + RAW_COLS)
    w.to_parquet(frame_path(), index=False)
    log(f"[r9] frame: {len(w):,} rows; 13F mapped {w['inst_mapped'].mean():.1%}, FTD mapped {w['ftd_mapped'].mean():.1%}")
    return w


def feats_for(name: str) -> list:
    return r8.feats_for(gate.R9_BASE) + [c for b in gate.R9_VARIANTS[name] for c in a9.BLOCKS[b]]


# --------------------------------------------------------------------------- #
# Leak probes (before any book)
# --------------------------------------------------------------------------- #
def leak_probe(log=print, end: str = "2022-12-31", seed: int = 0) -> dict:
    """Held-out AUC (30% of dates) of LightGBM classifiers for (i) "no EDGAR
    record" and (ii) "was delisted" (last weekly bar before R9_DELISTED_BEFORE)."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    w = pd.read_parquet(frame_path())
    alt = pd.read_parquet(alt9_path(), columns=["ticker", "Date"])
    last = alt.groupby("ticker")["Date"].max()
    w = w[w["Date"] <= end].reset_index(drop=True)
    targets = {"no_edgar": w["cik"].isna().astype(int).to_numpy(),
               "delisted": (w["ticker"].map(last) < gate.R9_DELISTED_BEFORE).astype(int).to_numpy()}
    dates = np.array(sorted(w["Date"].unique()))
    rng = np.random.default_rng(seed)
    te = w["Date"].isin(set(dates[rng.random(len(dates)) < 0.3])).to_numpy()
    base = r8.feats_for(gate.R9_BASE)
    raw13, rawf = [f"{c}_raw" for c in a9.INST_COLUMNS], [f"{c}_raw" for c in a9.FTD_COLUMNS]
    sets = {"base (core+fund+alt8_hd)": base,
            "+13F raw": base + raw13, "+13F hd": base + a9.INST_COLUMNS,
            "+FTD raw": base + rawf, "+FTD hd": base + a9.FTD_COLUMNS,
            "+both hd": base + a9.INST_COLUMNS + a9.FTD_COLUMNS,
            "13F hd only": a9.INST_COLUMNS, "FTD hd only": a9.FTD_COLUMNS,
            "13F mapped flag only": ["inst_mapped"], "FTD mapped flag only": ["ftd_mapped"]}
    out = {"period": f"2017..{end}", "rows": int(len(w)),
           "base_rate": {k: float(v.mean()) for k, v in targets.items()},
           "mapped_share": {
               f"{flag}|{k}={val}": float(w.loc[targets[k] == val, flag].mean())
               for flag in ("inst_mapped", "ftd_mapped") for k in targets for val in (0, 1)}}
    for tname, y in targets.items():
        out[tname] = {}
        for sname, cols in sets.items():
            m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1, n_jobs=8)
            m.fit(w.loc[~te, cols], y[~te])
            out[tname][sname] = float(roc_auc_score(y[te], m.predict_proba(w.loc[te, cols])[:, 1]))
            log(f"  [{tname}] {sname}: AUC {out[tname][sname]:.3f}")
    json.dump(out, open(ex.eval_dir("r9_leak.json"), "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def run_wf(log=print, names=None) -> None:
    w = pd.read_parquet(frame_path())
    cal = ds.calendar_from_dataset()
    for name in names or list(gate.R9_VARIANTS):
        t0 = time.time()
        preds, folds = tr.walk_forward(w, feats_for(name), "target_sn_30", cal, log=log)
        preds.to_parquet(os.path.join(ds.v3_dir("wf"), f"{name}.parquet"), index=False)
        json.dump({"variant": name, "features": feats_for(name), "folds": folds, "seconds": round(time.time() - t0)},
                  open(os.path.join(ds.v3_dir("wf"), f"{name}.json"), "w"), indent=1, default=str)
        log(f"[r9] {name} done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Signal ICs, the book screen, standalone feature ICs
# --------------------------------------------------------------------------- #
def _ic(p: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    return r6.per_date_ic(p, lab, "fwd_ret_30").dropna()


def signal_ics(log=print) -> dict:
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30"])
    fr = pd.read_parquet(frame_path(), columns=["ticker", "Date", "cik", "inst_mapped", "ftd_mapped"])
    top = pd.read_parquet(r6.topn_path())
    masks = {"sn30_r9_13f": fr["cik"].notna() & (fr["inst_mapped"] > 0),
             "sn30_r9_ftd": fr["cik"].notna() & (fr["ftd_mapped"] > 0),
             "sn30_r9_all": fr["cik"].notna() & (fr["inst_mapped"] > 0) & (fr["ftd_mapped"] > 0)}
    subsets = {k: fr.loc[m, ["ticker", "Date"]] for k, m in masks.items()}
    base = ex.load_preds(gate.R9_BASE)
    out = {"signals": {}, "screen": {}}
    for v in (gate.R9_BASE, "sn30_fund", "stack") + tuple(gate.R9_VARIANTS):
        p = ex.load_preds(v)
        s = _ic(p, lab)
        t1 = r6.ic_within(p, lab, top)["all"]
        out["signals"][v] = {"ic_all": float(s.mean()), "nw_t": ev.newey_west_t(s, gate.NW_LAGS),
                             "by_year": s.groupby(s.index.str[:4]).mean().round(4).to_dict(),
                             "top1000": t1["mean_ic"], "top1000_t": t1["nw_t"],
                             "holdout_proxy": float(_ic(p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)],
                                                        lab).mean())}
    for v in gate.R9_VARIANTS:
        sub = subsets[v]
        icv = float(_ic(ex.load_preds(v).merge(sub, on=["ticker", "Date"]), lab).mean())
        icb = float(_ic(base.merge(sub, on=["ticker", "Date"]), lab).mean())
        sv, sb = out["signals"][v], out["signals"][gate.R9_BASE]
        ok = gate.r9_bookable({v: sv["ic_all"], gate.R9_BASE: sb["ic_all"]}, {v: icv, gate.R9_BASE: icb})
        out["screen"][v] = {"ic": sv["ic_all"], "ic_base": sb["ic_all"], "gain": sv["ic_all"] - sb["ic_all"],
                            "ic_mapped": icv, "ic_base_mapped": icb, "gain_mapped": icv - icb,
                            "mapped_rows_share": float(masks[v].mean()), "bookable": bool(ok)}
        log(f"[r9 screen] {v}: IC {sv['ic_all']:.4f} (base {sb['ic_all']:.4f}, {sv['ic_all'] - sb['ic_all']:+.4f}); "
            f"mapped-only {icv:.4f} vs {icb:.4f} ({icv - icb:+.4f}) -> {'BOOK' if ok else 'no book'}")
    json.dump(out, open(ex.eval_dir("r9_signal_ics.json"), "w"), indent=1, default=float)
    return out


def feature_ics(log=print) -> dict:
    alt = pd.read_parquet(alt9_path())
    alt = alt[alt["fwd_ret_30"].notna()]
    top = pd.read_parquet(r6.topn_path()).assign(_top=True)
    alt = alt.merge(top, on=["ticker", "Date"], how="left")
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date"] + [f"fwd_ret_{h}" for h in r6.DECAY])
    out = {}
    for c in NEW_COLS:
        row = {}
        for uname, m in (("all", np.ones(len(alt), bool)), ("tradable", alt["eligible"].astype(bool).to_numpy()),
                         ("top1000", alt["_top"].fillna(False).to_numpy(dtype=bool))):
            sub = alt.loc[m, ["ticker", "Date", c, "fwd_ret_30"]].dropna()
            sub = sub[sub.groupby("Date")[c].transform("nunique") > 1]
            s = sub.groupby("Date").apply(lambda g: g[c].corr(g["fwd_ret_30"], method="spearman") if len(g) > 30
                                          else np.nan, include_groups=False).dropna()
            row[uname] = {"ic": float(s.mean()) if len(s) else float("nan"),
                          "nw_t": ev.newey_west_t(s, gate.NW_LAGS) if len(s) > 3 else float("nan"),
                          "coverage": float(alt.loc[m, c].notna().mean())}
        p = alt.loc[alt["eligible"].astype(bool), ["ticker", "Date", c]].rename(columns={c: "pred"}).dropna()
        row["decay"] = {h: float(r6.per_date_ic(p, lab, f"fwd_ret_{h}").mean()) for h in r6.DECAY}
        out[c] = row
        log(f"  {c}: IC all {row['all']['ic']:.4f} tradable {row['tradable']['ic']:.4f} "
            f"top1000 {row['top1000']['ic']:.4f} cov {row['tradable']['coverage']:.0%}")
    json.dump(out, open(ex.eval_dir("r9_feature_ics.json"), "w"), indent=1, default=float)
    return out


def coverage(log=print) -> dict:
    alt = pd.read_parquet(alt9_path(), columns=["ticker", "Date", "eligible", "cik", "inst_mapped", "ftd_mapped",
                                                 "inst_breadth"])
    alt["year"] = alt["Date"].str[:4]
    el = alt[alt["eligible"].astype(bool)]
    last = alt.groupby("ticker")["Date"].max()
    el = el.assign(delisted=el["ticker"].map(last) < gate.R9_DELISTED_BEFORE)
    out = {"all_rows": {"inst_mapped": float(alt["inst_mapped"].mean()), "ftd_mapped": float(alt["ftd_mapped"].mean())},
           "eligible": {"inst_mapped": float(el["inst_mapped"].mean()), "ftd_mapped": float(el["ftd_mapped"].mean()),
                        "held_by_any_13f_filer": float((el["inst_breadth"] > 0).mean())},
           "eligible_by_year": el.groupby("year")[["inst_mapped", "ftd_mapped"]].mean().round(3).to_dict(),
           "eligible_by_delisted": el.groupby("delisted")[["inst_mapped", "ftd_mapped"]].mean().round(3).to_dict(),
           "eligible_by_cik": el.groupby(el["cik"].notna())[["inst_mapped", "ftd_mapped"]].mean().round(3).to_dict()}
    json.dump(out, open(ex.eval_dir("r9_coverage.json"), "w"), indent=1, default=float)
    log(json.dumps(out, indent=1, default=float))
    return out


# --------------------------------------------------------------------------- #
# Books + gate
# --------------------------------------------------------------------------- #
def run_books(log=print) -> dict:
    scr = json.load(open(ex.eval_dir("r9_signal_ics.json")))["screen"]
    sigs = [v for v in gate.R9_VARIANTS if scr[v]["bookable"]]
    res = {"configs": {}, "trials": {}, "booked": sigs}
    if sigs:
        base = r5.load_prep()
        spy = ex.spy_returns()
        ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
        cw = r5._load_run("cw_all_100m")[0]
        for sig in sigs:
            t0 = time.time()
            out = r5.run_config("mv_abs_all", gate.R5_PRIMARY_AUM, r8.prep_with(ex.load_preds(sig), base), log=log)
            m = r5.metrics(out["ret"], out["log"], spy, ew, cw)
            m["all"]["realized_over_exante_vol"] = float(m["all"]["vol"] / out["log"]["ex_ante_vol"].mean())
            res["configs"][sig] = m
            res["trials"][f"r9:{sig}/mv_abs_all_100m"] = float(out["ret"].mean() / out["ret"].std(ddof=1))
            out["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r9_{sig}_daily.parquet"))
            a = m["all"]
            log(f"  {sig}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
                f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} ({time.time() - t0:.0f}s)")
    else:
        log("[r9] no variant passed the IC screen: no book")
    json.dump(res, open(ex.eval_dir("results_r9.json"), "w"), indent=1, default=float)
    return res


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r9.json")))
    trials = r6.all_trials()
    trials.update(json.load(open(ex.eval_dir("results_r7.json")))["trials"])
    trials.update(json.load(open(ex.eval_dir("results_r8.json")))["trials"])
    assert len(trials) == gate.N_TRIALS_R8, f"expected {gate.N_TRIALS_R8} prior trials, got {len(trials)}"
    trials.update(res["trials"])
    perf = {k: v["all"] for k, v in res["configs"].items()}
    choice = gate.choose_config_r9(perf)
    sic = json.load(open(ex.eval_dir("r9_signal_ics.json")))
    out = {"config": choice, "n_trials": gate.N_TRIALS_R9, "booked": res.get("booked", [])}
    if choice is None:
        out.update({"passed": False, "gate": [], "reason": "no variant passed the pre-registered IC screen"})
        json.dump(out, open(ex.eval_dir("gate_r9.json"), "w"), indent=1, default=float)
        log("[r9 gate] FAIL: no book (IC screen)")
        return out
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    p = ex.load_preds(choice)
    s = ev.signal_summary(ev.per_date_signal(p.merge(lab, on=["ticker", "Date"]), "pred", "fwd_ret_30"),
                          nw_lags=gate.NW_LAGS).to_dict(orient="index")
    r = pd.read_parquet(ex.eval_dir(f"r9_{choice}_daily.parquet"))["ret"]
    spy = ex.spy_returns().reindex(r.index).fillna(0.0)
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R9)
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    holdout = sic["signals"][choice]["holdout_proxy"]
    rows = gate.evaluate_gate({"mean_ic": s["all"]["mean_ic"], "nw_t": s["all"]["nw_t"]},
                              {y: s[y]["mean_ic"] for y in gate.A_YEARS if y in s}, perf[choice], ew_sh,
                              dsr["dsr"], holdout)
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R9})"
    rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
    rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. {boot['ir_spy']['hi']:.2f})",
                 "value": perf[choice]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                 "pass": bool(perf[choice]["ir_spy"] > gate.R5_MIN_IR_SPY)})
    out.update({"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot})
    json.dump(out, open(ex.eval_dir("gate_r9.json"), "w"), indent=1, default=float)
    log(f"[r9 gate] config={choice} passed={out['passed']} DSR={dsr['dsr']:.3f} IR_spy={perf[choice]['ir_spy']:.2f}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"features": lambda: a9.weekly_features(), "frame": build_frame, "coverage": coverage,
             "probe": leak_probe, "wf": run_wf, "ics": signal_ics, "fics": feature_ics, "books": run_books,
             "gate": apply_gate}
    for a in sys.argv[1:]:
        steps[a]()
