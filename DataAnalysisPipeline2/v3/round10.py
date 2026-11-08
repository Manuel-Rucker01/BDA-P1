"""Round 10 (pre-registered in ``v3/gate.py``): analyst-free earnings surprise
(SUE on EDGAR EPS, earnings-announcement return around 8-K 2.02) on top of
sn30_altall_hd; leak probes; the +0.005 IC book screen; books through Round 5's
``mv_abs_all``; gate.

    python -m DataAnalysisPipeline2.v3.round10 frame coverage overlap probe probe_tk wf ics fics books gate
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import altdata_r10 as a10
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import round9 as r9
from . import train as tr

R10_DATASET = "dataset_r10"
NEW_COLS = a10.SUE_COLUMNS + a10.EAR_COLUMNS
RAW_COLS = [f"{c}_raw" for c in NEW_COLS]
HAS = {"sue": "sue_has", "ear": "ear_has"}


def frame_path() -> str:
    return tr.weekly_path(R10_DATASET)


def alt10_path() -> str:
    return os.path.join(ds.v3_dir(), "alt10_weekly.parquet")


def build_frame(log=print) -> pd.DataFrame:
    """Round-8 evaluation frame + the round-10 blocks: raw (``*_raw``, missing
    rows stay missing) and hot-decked (canonical names), z-scored per date."""
    w = pd.read_parquet(r8.frame_path())
    alt = pd.read_parquet(alt10_path(), columns=["ticker", "Date"] + NEW_COLS)
    w = w.merge(alt, on=["ticker", "Date"], how="left")
    w[RAW_COLS] = w[NEW_COLS].to_numpy()
    cik = w["cik"].notna().to_numpy()
    for blk, cols in a10.BLOCKS.items():
        has = w[a10.HAS_COL[blk]].notna().to_numpy() & cik
        w[HAS[blk]] = has.astype(float)
        w[cols] = r9.block_hotdeck(w, cols, has)
    w[NEW_COLS + RAW_COLS] = core.cross_sectional_zscore(w, NEW_COLS + RAW_COLS)
    w.to_parquet(frame_path(), index=False)
    log(f"[r10] frame: {len(w):,} rows; SUE own value {w['sue_has'].mean():.1%}, EAR own value {w['ear_has'].mean():.1%}")
    return w


def feats_for(name: str) -> list:
    return r8.feats_for(gate.R10_BASE) + [c for b in gate.R10_VARIANTS[name] for c in a10.BLOCKS[b]]


# --------------------------------------------------------------------------- #
# Coverage and overlap with the incumbent's features (no returns involved)
# --------------------------------------------------------------------------- #
def _delisted(tickers: pd.Series) -> pd.Series:
    alt = pd.read_parquet(r9.alt9_path(), columns=["ticker", "Date"])
    last = alt.groupby("ticker")["Date"].max()
    return tickers.map(last) < gate.R9_DELISTED_BEFORE


def coverage(log=print) -> dict:
    a = pd.read_parquet(alt10_path(), columns=["ticker", "Date", "eligible", "cik", "sue_eps", "surge", "ear",
                                                "ear_abvol", "ear_src10q"])
    el = a[a["eligible"].astype(bool)].copy()
    el["delisted"] = _delisted(el["ticker"]).to_numpy()
    el["has_cik"] = el["cik"].notna()
    for c in ("sue_eps", "surge", "ear", "ear_abvol"):
        el[f"{c}_has"] = el[c].notna()
    hc = [f"{c}_has" for c in ("sue_eps", "surge", "ear", "ear_abvol")]
    ck = el[el["has_cik"]]
    ev_ = pd.read_parquet(os.path.join(a10.r10_dir(), "events.parquet"))
    out = {"eligible_rows": int(len(el)),
           "eligible": el[hc].mean().round(3).to_dict(),
           "eligible_with_cik": ck[hc].mean().round(3).to_dict(),
           "with_cik_by_delisted": ck.groupby("delisted")[hc].mean().round(3).to_dict(),
           "with_cik_by_year": ck.groupby(ck["Date"].str[:4])[hc].mean().round(3).to_dict(),
           "ear_src10q_share": float(ck.loc[ck["ear"].notna(), "ear_src10q"].mean()),
           "events": {"n": int(len(ev_)), "ciks": int(ev_["cik"].nunique()), "src10q": float(ev_["src10q"].mean())}}
    json.dump(out, open(ex.eval_dir("r10_coverage.json"), "w"), indent=1, default=float)
    log(json.dumps(out, indent=1, default=float))
    return out


def timing_check(log=print) -> dict:
    """Mean |market-adjusted return| on sessions day0-3 .. day0+3 over all
    8-K 2.02 events 2017-2026 (the reaction should peak on day0)."""
    cal, close, vol, col, mkt = a10.price_matrices(log=log)
    evs = pd.read_parquet(os.path.join(a10.r10_dir(), "events.parquet"))
    tick = pd.read_parquet(alt10_path(), columns=["ticker", "cik"]).dropna().drop_duplicates()
    e = evs.merge(tick.assign(cik=tick["cik"].astype("int64")), on="cik")
    e = e[(e["day0"] > 70) & (e["day0"] < len(cal) - 5) & (cal[np.minimum(e["day0"], len(cal) - 1)] >= "2017-01-01")]
    j = e["ticker"].map(col).to_numpy()
    out = {}
    for src in (0, 1):
        m = e["src10q"].to_numpy() == src
        res = {}
        for k in range(-3, 4):
            p = e["day0"].to_numpy()[m] + k
            r = np.log(close[p, j[m]] / close[p - 1, j[m]]) - (mkt[p] - mkt[p - 1])
            res[k] = float(np.nanmean(np.abs(r)))
        out["8-K 2.02" if src == 0 else "10-Q/10-K fallback"] = res
    json.dump(out, open(ex.eval_dir("r10_timing.json"), "w"), indent=1)
    log(json.dumps(out, indent=1))
    return out


def overlap(log=print) -> dict:
    """Mean per-date Spearman correlation between the raw new features and the
    incumbent's related features (and Round 4's SUE), eligible rows with a CIK
    and the block's own value. Feature-feature only: no returns."""
    w = pd.read_parquet(r8.frame_path(), columns=["ticker", "Date", "cik", "ni_growth_scaled", "sales_growth_yoy",
                                                   "fund_staleness_days", "return_5d", "return_20d", "return_50d",
                                                   "volume_ratio", "ep", "log_mcap"])
    a = pd.read_parquet(alt10_path(), columns=["ticker", "Date"] + NEW_COLS)
    r4 = pd.read_parquet(tr.weekly_path("dataset_r4"), columns=["ticker", "Date", "sue", "sue_chg", "fund_imputed"])
    r4 = r4[r4["fund_imputed"] == 0].rename(columns={"sue": "r4_sue", "sue_chg": "r4_sue_chg"})
    w = w[w["cik"].notna()].merge(a, on=["ticker", "Date"]).merge(r4[["ticker", "Date", "r4_sue", "r4_sue_chg"]],
                                                                   on=["ticker", "Date"], how="left")
    w = w[w["Date"].isin(sorted(w["Date"].unique())[::4])]          # every 4th week (enough for a mean)
    olds = ["ni_growth_scaled", "sales_growth_yoy", "fund_staleness_days", "return_5d", "return_20d", "return_50d",
            "volume_ratio", "ep", "log_mcap", "r4_sue", "r4_sue_chg"]
    out = {}
    for c in NEW_COLS:
        row = {}
        for o in olds:
            s = w[["Date", c, o]].dropna()
            cc = s.groupby("Date").apply(lambda g: g[c].corr(g[o], method="spearman") if len(g) > 50 else np.nan,
                                         include_groups=False)
            row[o] = round(float(cc.mean()), 3)
        out[c] = row
    df = pd.DataFrame(out).T
    json.dump(out, open(ex.eval_dir("r10_overlap.json"), "w"), indent=1)
    log(df.to_string())
    return out


# --------------------------------------------------------------------------- #
# Leak probes (before any walk-forward is evaluated)
# --------------------------------------------------------------------------- #
def leak_probe(log=print, end: str = "2022-12-31", seed: int = 0, split: str = "date") -> dict:
    """Round 9's probe design (see round9.leak_probe) for the round-10 blocks."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    w = pd.read_parquet(frame_path())
    w = w[w["Date"] <= end].reset_index(drop=True)
    targets = {"no_edgar": w["cik"].isna().astype(int).to_numpy(),
               "delisted": _delisted(w["ticker"]).astype(int).to_numpy()}
    dates = np.array(sorted(w["Date"].unique()))
    rng = np.random.default_rng(seed)
    if split == "ticker":
        tk = np.array(sorted(w["ticker"].unique()))
        te = w["ticker"].isin(set(tk[rng.random(len(tk)) < 0.3])).to_numpy()
    else:
        te = w["Date"].isin(set(dates[rng.random(len(dates)) < 0.3])).to_numpy()
    base = r8.feats_for(gate.R10_BASE)
    rs, re_ = [f"{c}_raw" for c in a10.SUE_COLUMNS], [f"{c}_raw" for c in a10.EAR_COLUMNS]
    sets = {"base (core+fund+alt8_hd)": base,
            "+SUE raw": base + rs, "+SUE hd": base + a10.SUE_COLUMNS,
            "+EAR raw": base + re_, "+EAR hd": base + a10.EAR_COLUMNS,
            "+both hd": base + a10.SUE_COLUMNS + a10.EAR_COLUMNS,
            "SUE hd only": a10.SUE_COLUMNS, "EAR hd only": a10.EAR_COLUMNS,
            "SUE has-flag only": ["sue_has"], "EAR has-flag only": ["ear_has"]}
    out = {"period": f"2017..{end}", "rows": int(len(w)), "split": split,
           "base_rate": {k: float(v.mean()) for k, v in targets.items()}}
    for tname, y in targets.items():
        out[tname] = {}
        for sname, cols in sets.items():
            m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1, n_jobs=8)
            m.fit(w.loc[~te, cols], y[~te])
            out[tname][sname] = float(roc_auc_score(y[te], m.predict_proba(w.loc[te, cols])[:, 1]))
            log(f"  [{split}|{tname}] {sname}: AUC {out[tname][sname]:.3f}")
    json.dump(out, open(ex.eval_dir("r10_leak.json" if split == "date" else f"r10_leak_{split}.json"), "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def run_wf(log=print, names=None) -> None:
    w = pd.read_parquet(frame_path())
    cal = ds.calendar_from_dataset()
    for name in names or list(gate.R10_VARIANTS):
        t0 = time.time()
        preds, folds = tr.walk_forward(w, feats_for(name), "target_sn_30", cal, log=log)
        preds.to_parquet(os.path.join(ds.v3_dir("wf"), f"{name}.parquet"), index=False)
        json.dump({"variant": name, "features": feats_for(name), "folds": folds, "seconds": round(time.time() - t0)},
                  open(os.path.join(ds.v3_dir("wf"), f"{name}.json"), "w"), indent=1, default=str)
        log(f"[r10] {name} done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Signal ICs, the book screen, standalone feature ICs
# --------------------------------------------------------------------------- #
def _ic(p: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    return r6.per_date_ic(p, lab, "fwd_ret_30").dropna()


def signal_ics(log=print) -> dict:
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30"])
    fr = pd.read_parquet(frame_path(), columns=["ticker", "Date", "sue_has", "ear_has"])
    top = pd.read_parquet(r6.topn_path())
    masks = {"sn30_r10_sue": fr["sue_has"] > 0, "sn30_r10_ear": fr["ear_has"] > 0,
             "sn30_r10_both": (fr["sue_has"] > 0) & (fr["ear_has"] > 0)}
    base = ex.load_preds(gate.R10_BASE)
    out = {"signals": {}, "screen": {}}
    for v in (gate.R10_BASE, "sn30_fund", "sn30_r9_all") + tuple(gate.R10_VARIANTS):
        p = ex.load_preds(v)
        s = _ic(p, lab)
        t1 = r6.ic_within(p, lab, top)["all"]
        out["signals"][v] = {"ic_all": float(s.mean()), "nw_t": ev.newey_west_t(s, gate.NW_LAGS),
                             "by_year": s.groupby(s.index.str[:4]).mean().round(4).to_dict(),
                             "top1000": t1["mean_ic"], "top1000_t": t1["nw_t"],
                             "holdout_proxy": float(_ic(p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)],
                                                        lab).mean())}
    for v in gate.R10_VARIANTS:
        sub = fr.loc[masks[v], ["ticker", "Date"]]
        icv = float(_ic(ex.load_preds(v).merge(sub, on=["ticker", "Date"]), lab).mean())
        icb = float(_ic(base.merge(sub, on=["ticker", "Date"]), lab).mean())
        sv, sb = out["signals"][v], out["signals"][gate.R10_BASE]
        ok = gate.r10_bookable({v: sv["ic_all"], gate.R10_BASE: sb["ic_all"]}, {v: icv, gate.R10_BASE: icb})
        out["screen"][v] = {"ic": sv["ic_all"], "ic_base": sb["ic_all"], "gain": sv["ic_all"] - sb["ic_all"],
                            "ic_mapped": icv, "ic_base_mapped": icb, "gain_mapped": icv - icb,
                            "mapped_rows_share": float(masks[v].mean()), "bookable": bool(ok)}
        log(f"[r10 screen] {v}: IC {sv['ic_all']:.4f} (base {sb['ic_all']:.4f}, {sv['ic_all'] - sb['ic_all']:+.4f}); "
            f"own-value rows {icv:.4f} vs {icb:.4f} ({icv - icb:+.4f}) -> {'BOOK' if ok else 'no book'}")
    json.dump(out, open(ex.eval_dir("r10_signal_ics.json"), "w"), indent=1, default=float)
    return out


def feature_ics(log=print) -> dict:
    alt = pd.read_parquet(alt10_path())
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
    json.dump(out, open(ex.eval_dir("r10_feature_ics.json"), "w"), indent=1, default=float)
    return out


# --------------------------------------------------------------------------- #
# Books + gate
# --------------------------------------------------------------------------- #
def run_books(log=print) -> dict:
    scr = json.load(open(ex.eval_dir("r10_signal_ics.json")))["screen"]
    sigs = [v for v in gate.R10_VARIANTS if scr[v]["bookable"]]
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
            res["trials"][f"r10:{sig}/mv_abs_all_100m"] = float(out["ret"].mean() / out["ret"].std(ddof=1))
            out["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r10_{sig}_daily.parquet"))
            a = m["all"]
            log(f"  {sig}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
                f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} ({time.time() - t0:.0f}s)")
    else:
        log("[r10] no variant passed the IC screen: no book")
    json.dump(res, open(ex.eval_dir("results_r10.json"), "w"), indent=1, default=float)
    return res


def _prior_trials() -> dict:
    trials = r6.all_trials()
    for r in ("r7", "r8", "r9"):
        trials.update(json.load(open(ex.eval_dir(f"results_{r}.json")))["trials"])
    return trials


def incumbent_dsr(log=print) -> dict:
    """DSR of the incumbent book (sn30_altall_hd / mv_abs_all) at N = N_TRIALS_R10."""
    trials = _prior_trials()
    r = pd.read_parquet(ex.eval_dir(f"r8_{gate.R10_BASE}_daily.parquet"))["ret"]
    d = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R10)
    log(f"[r10] incumbent {gate.R10_BASE} DSR at N={gate.N_TRIALS_R10}: {d['dsr']:.3f}")
    return d


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r10.json")))
    trials = _prior_trials()
    assert len(trials) == gate.N_TRIALS_R8, f"expected {gate.N_TRIALS_R8} booked prior trials, got {len(trials)}"
    trials.update(res["trials"])
    perf = {k: v["all"] for k, v in res["configs"].items()}
    choice = gate.choose_config_r10(perf)
    sic = json.load(open(ex.eval_dir("r10_signal_ics.json")))
    out = {"config": choice, "n_trials": gate.N_TRIALS_R10, "booked": res.get("booked", [])}
    if choice is None:
        out.update({"passed": False, "gate": [], "reason": "no variant passed the pre-registered IC screen",
                    "incumbent_dsr": incumbent_dsr(log)})
        json.dump(out, open(ex.eval_dir("gate_r10.json"), "w"), indent=1, default=float)
        log("[r10 gate] FAIL: no book (IC screen)")
        return out
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    p = ex.load_preds(choice)
    s = ev.signal_summary(ev.per_date_signal(p.merge(lab, on=["ticker", "Date"]), "pred", "fwd_ret_30"),
                          nw_lags=gate.NW_LAGS).to_dict(orient="index")
    r = pd.read_parquet(ex.eval_dir(f"r10_{choice}_daily.parquet"))["ret"]
    spy = ex.spy_returns().reindex(r.index).fillna(0.0)
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R10)
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    holdout = sic["signals"][choice]["holdout_proxy"]
    rows = gate.evaluate_gate({"mean_ic": s["all"]["mean_ic"], "nw_t": s["all"]["nw_t"]},
                              {y: s[y]["mean_ic"] for y in gate.A_YEARS if y in s}, perf[choice], ew_sh,
                              dsr["dsr"], holdout)
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R10})"
    rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
    rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. {boot['ir_spy']['hi']:.2f})",
                 "value": perf[choice]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                 "pass": bool(perf[choice]["ir_spy"] > gate.R5_MIN_IR_SPY)})
    out.update({"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot})
    json.dump(out, open(ex.eval_dir("gate_r10.json"), "w"), indent=1, default=float)
    log(f"[r10 gate] config={choice} passed={out['passed']} DSR={dsr['dsr']:.3f} IR_spy={perf[choice]['ir_spy']:.2f}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"features": lambda: a10.weekly_features(), "frame": build_frame, "coverage": coverage,
             "timing": timing_check, "overlap": overlap, "probe": leak_probe,
             "probe_tk": lambda: leak_probe(split="ticker"), "wf": run_wf, "ics": signal_ics, "fics": feature_ics,
             "books": run_books, "gate": apply_gate, "incumbent": incumbent_dsr}
    for a in sys.argv[1:]:
        steps[a]()
