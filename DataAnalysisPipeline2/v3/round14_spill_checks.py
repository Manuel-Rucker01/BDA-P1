"""Round 14 ``spill`` data checks. Definitions are in
``results/round14_addendum_spill.md``.

Checks:

* as-of audit
* coverage by year
* leak probes (Round 9-10 design)
* shuffled-date and shuffled-neighbour placebos
* standalone weekly ICs against the 30-session return

Results go to ``results/round14_spill_data.md`` and
``$BDA_DATA_V3/spill/checks.json``. No model is trained on returns.

    python -m DataAnalysisPipeline2.v3.round14_spill_checks
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import altdata_r14_spill as sp
from . import dataset as ds
from . import evaluation as ev
from . import gate

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "results", "round14_spill_data.md")
FEATS = sp.ALL_COLUMNS + sp.REF_COLUMNS


def fast_ic(df: pd.DataFrame, c: str, y: str, min_n: int = 31) -> pd.Series:
    """Per-date Spearman correlation of ``c`` and ``y`` (dates with at least
    ``min_n`` rows and more than one distinct ``c``)."""
    s = df[["Date", c, y]].dropna()
    g = s.groupby("Date")
    s = s[(g[c].transform("size") >= min_n) & (g[c].transform("nunique") > 1)]
    if s.empty:
        return pd.Series(dtype=float)
    g = s.groupby("Date")
    a = g[c].rank()
    b = g[y].rank()
    a = a - a.groupby(s["Date"]).transform("mean")
    b = b - b.groupby(s["Date"]).transform("mean")
    num = (a * b).groupby(s["Date"]).sum()
    den = np.sqrt((a * a).groupby(s["Date"]).sum() * (b * b).groupby(s["Date"]).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def _summ(s: pd.Series) -> dict:
    return {"ic": float(s.mean()) if len(s) else float("nan"),
            "nw_t": float(ev.newey_west_t(s, gate.NW_LAGS)) if len(s) > 3 else float("nan"), "n_dates": int(len(s))}


def load_panel() -> pd.DataFrame:
    a = pd.read_parquet(sp.out_path())
    top = pd.read_parquet(os.path.join(ds.v3_dir(), "weekly_top1000.parquet")).assign(_top=True)
    a = a.merge(top, on=["ticker", "Date"], how="left")
    a["_top"] = a["_top"].fillna(False).astype(bool)
    last = a.groupby("ticker")["Date"].max()
    a["delisted"] = a["ticker"].map(last) < gate.R9_DELISTED_BEFORE
    return a


def universes(a: pd.DataFrame) -> dict:
    return {"all": np.ones(len(a), bool), "tradable": a["eligible"].astype(bool).to_numpy(),
            "top1000": a["_top"].to_numpy()}


# --------------------------------------------------------------------------- #
def asof_audit(a: pd.DataFrame, log=print) -> dict:
    au = pd.read_parquet(os.path.join(sp.spill_dir(), "asof_audit.parquet"))
    et = pd.read_parquet(sp.edges_path("tnic"))
    eo = pd.read_parquet(sp.edges_path("own"))
    from .altdata_r9 import f13_avail
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    # every edge version: valid_from equals the registered rule
    tv = et.groupby("tnic_year")["valid_from"].first()
    tnic_rule_ok = all(tv[y] == sp.tnic_valid_from(int(y), cal) and tv[y] >= f"{int(y) + 1}-07-01" for y in tv.index)
    ov = eo.groupby("period")["valid_from"].first()
    own_rule_ok = all(ov[P] == pd.Timestamp(f13_avail([P]).iloc[0]).strftime("%Y-%m-%d") for P in ov.index)
    au = au.dropna(subset=["tnic_valid_from", "own_valid_from"], how="all")
    used_ok_t = bool((au["tnic_valid_from"].dropna() <= au.loc[au["tnic_valid_from"].notna(), "Date"]).all())
    used_ok_o = bool((au["own_valid_from"].dropna() <= au.loc[au["own_valid_from"].notna(), "Date"]).all())
    # tnic year Y used only from July 1 Y+1
    yr_ok = bool((au.dropna(subset=["tnic_year"]).apply(lambda r: r["Date"] >= f"{int(r['tnic_year']) + 1}-07-01",
                                                        axis=1)).all())
    ret_ok = bool((au["ret_end_session"] == au["Date"]).all())
    lag_t = (pd.to_datetime(au["Date"]) - pd.to_datetime(au["tnic_valid_from"])).dt.days
    lag_o = (pd.to_datetime(au["Date"]) - pd.to_datetime(au["own_valid_from"])).dt.days
    tnic_age = pd.to_datetime(au["Date"]).dt.year - au["tnic_year"]
    out = {"tnic_version_rule": bool(tnic_rule_ok), "own_version_rule": bool(own_rule_ok),
           "tnic_used_after_valid_from": used_ok_t, "own_used_after_valid_from": used_ok_o,
           "tnic_year_july_rule": yr_ok, "return_windows_end_at_date": ret_ok,
           "days_since_tnic_version": {"min": float(lag_t.min()), "median": float(lag_t.median()),
                                       "max": float(lag_t.max())},
           "days_since_13f_version": {"min": float(lag_o.min()), "median": float(lag_o.median()),
                                      "max": float(lag_o.max())},
           "tnic_year_by_date_year": au.assign(y=au["Date"].str[:4]).groupby("y")["tnic_year"]
           .agg(lambda s: f"{int(s.min())}-{int(s.max())}").to_dict(),
           "max_tnic_age_years": float(tnic_age.max()),
           "first_date_by_year_tnic": tv.to_dict(), "n_13f_versions": int(len(ov))}
    log(json.dumps(out, indent=1, default=str))
    return out


def coverage(a: pd.DataFrame, log=print) -> dict:
    el = a[a["eligible"].astype(bool)]
    hc = sp.HAS_COLUMNS
    out = {"rows_all": int(len(a)), "rows_eligible": int(len(el)),
           "all": a[hc].mean().round(3).to_dict(), "eligible": el[hc].mean().round(3).to_dict(),
           "eligible_by_year": el.groupby(el["Date"].str[:4])[hc].mean().round(3).to_dict(),
           "all_by_year": a.groupby(a["Date"].str[:4])[hc].mean().round(3).to_dict(),
           "eligible_by_delisted": el.groupby(el["delisted"].map({True: "delisted", False: "survivor"}))[hc].mean().round(3).to_dict(),
           "eligible_by_has_cik": el.groupby(el["cik"].notna().map({True: "cik", False: "no_cik"}))[hc].mean().round(3).to_dict(),
           "top1000": a.loc[a["_top"], hc].mean().round(3).to_dict(),
           "mean_n_eligible": {e: float(el.loc[el[f"spill_{e}_has"] > 0, f"spill_{e}_n"].mean()) for e in sp.EDGE_SETS}}
    et = pd.read_parquet(sp.edges_path("tnic"))
    eo = pd.read_parquet(sp.edges_path("own"))
    out["edges"] = {"tnic_rows": int(len(et)), "tnic_by_year": et.groupby("tnic_year").size().to_dict(),
                    "tnic_sources_by_year": et.groupby("tnic_year")["src_cik"].nunique().to_dict(),
                    "own_rows": int(len(eo)), "own_sources_by_period":
                        {str(pd.Timestamp(k).date()): int(v) for k, v in eo.groupby("period")["src"].nunique().items()}}
    log(json.dumps({k: v for k, v in out.items() if k != "edges"}, indent=1, default=str))
    return out


def leak_probe(a: pd.DataFrame, log=print, end: str = "2022-12-31", seed: int = 0) -> dict:
    """Held-out AUC (random 30% of dates, rows 2017..end) for "no EDGAR record"
    and "later delisted" from the raw spill features / has-flags only."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    w = a[a["Date"] <= end].reset_index(drop=True)
    targets = {"no_edgar": w["cik"].isna().astype(int).to_numpy(), "delisted": w["delisted"].astype(int).to_numpy()}
    dates = np.array(sorted(w["Date"].unique()))
    rng = np.random.default_rng(seed)
    te = w["Date"].isin(set(dates[rng.random(len(dates)) < 0.3])).to_numpy()
    sets = {"spill block raw (E1+E2)": sp.SPILL_COLUMNS, "E1 tnic raw": sp.cols_for("tnic"),
            "E2 own raw": sp.cols_for("own"), "E3 sic raw (control)": sp.cols_for("sic"),
            "E1 has-flag only": ["spill_tnic_has"], "E2 has-flag only": ["spill_own_has"],
            "E2 r/x only (no count)": sp.cols_for("own")[:-1], "own returns (reference)": sp.REF_COLUMNS}
    out = {"period": f"2017..{end}", "rows": int(len(w)), "base_rate": {k: float(v.mean()) for k, v in targets.items()}}
    for t, y in targets.items():
        out[t] = {}
        for name, cols in sets.items():
            m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1, n_jobs=6)
            m.fit(w.loc[~te, cols], y[~te])
            out[t][name] = float(roc_auc_score(y[te], m.predict_proba(w.loc[te, cols])[:, 1]))
            log(f"  [{t}] {name}: AUC {out[t][name]:.3f}")
    return out


def feature_ics(a: pd.DataFrame, log=print) -> dict:
    lab = pd.read_parquet(os.path.join(ds.v3_dir(), "weekly_mh.parquet"),
                          columns=["ticker", "Date", "fwd_ret_5", "fwd_ret_21", "fwd_ret_63"])
    el = a[a["eligible"].astype(bool)].merge(lab, on=["ticker", "Date"], how="left")
    U = universes(a)
    out = {}
    for c in FEATS:
        row = {}
        for u, m in U.items():
            s = fast_ic(a.loc[m], c, "fwd_ret_30")
            row[u] = _summ(s) | {"coverage": float(a.loc[m, c].notna().mean())}
            if u == "tradable":
                row["by_year"] = s.groupby(s.index.str[:4]).mean().round(4).to_dict()
        row["decay"] = {h: float(fast_ic(el, c, f"fwd_ret_{h}").mean()) for h in (5, 21, 63)}
        out[c] = row
        log(f"  {c}: IC all {row['all']['ic']:+.4f} tradable {row['tradable']['ic']:+.4f} "
            f"({row['tradable']['nw_t']:+.2f}) top1000 {row['top1000']['ic']:+.4f} cov {row['tradable']['coverage']:.0%}")
    return out


def partial_ics(a: pd.DataFrame, log=print) -> dict:
    """Tradable IC of each spill mean after removing (per date, rank OLS) the
    stock's own return over the same window: is there link information beyond
    own-return reversal / momentum?"""
    el = a[a["eligible"].astype(bool)].copy()
    out = {}
    for e in sp.EDGE_SETS:
        for k in sp.WINDOWS:
            c, o = f"spill_{e}_r{k}", f"ref_own_r{k}"
            s = el[["Date", c, o, "fwd_ret_30"]].dropna()
            g = s.groupby("Date")
            rc = g[c].rank(pct=True) - 0.5
            ro = g[o].rank(pct=True) - 0.5
            beta = (rc * ro).groupby(s["Date"]).transform("sum") / (ro * ro).groupby(s["Date"]).transform("sum")
            s["res"] = rc - beta * ro
            out[c] = _summ(fast_ic(s, "res", "fwd_ret_30"))
    log(json.dumps(out, indent=1))
    return out


def shuffled_date_placebo(a: pd.DataFrame, n_perm: int = 20, seed: int = 0, log=print) -> dict:
    el = a[a["eligible"].astype(bool)]
    dates = np.array(sorted(el["Date"].unique()))
    lab = el[["ticker", "Date", "fwd_ret_30"]]
    rng = np.random.default_rng(seed)
    feats = sp.ALL_COLUMNS
    res = {c: [] for c in feats}
    for i in range(n_perm):
        pm = dict(zip(dates, rng.permutation(dates)))
        f = el[["ticker", "Date"] + feats].assign(_ld=el["Date"].map(pm))
        m = f.merge(lab.rename(columns={"Date": "_ld"}), on=["ticker", "_ld"], how="inner")
        for c in feats:
            res[c].append(float(fast_ic(m, c, "fwd_ret_30").mean()))
    out = {c: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "max_abs": float(np.max(np.abs(v)))}
           for c, v in res.items()}
    log("  shuffled-date: max |mean IC| " + f"{max(abs(v['mean']) for v in out.values()):.4f}")
    return out


def shuffled_neighbour_placebo(a: pd.DataFrame, seed: int = 0, log=print) -> dict:
    p = sp.weekly_features(log=log, placebo_seed=seed, write=False)
    p = p[p["eligible"].astype(bool)]
    out = {}
    for c in sp.ALL_COLUMNS:
        if c.endswith("_n"):
            continue
        out[c] = _summ(fast_ic(p, c, "fwd_ret_30"))
    log("  shuffled-neighbour: " + ", ".join(f"{c} {v['ic']:+.4f}" for c, v in out.items() if "r21" in c or "x21" in c))
    return out


# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def write_report(res: dict) -> None:
    au, cov, lk, ics, pic, sdp, snp = (res[k] for k in ("asof", "coverage", "leak", "ics", "partial", "shuf_date",
                                                         "shuf_nb"))
    L = ["# Round 14: `spill` block data, checks and standalone ICs", "",
         "Data-builder report. No model was trained and no book was run. Definitions are in "
         "`round14_addendum_spill.md`, committed before any statistic involving forward returns. "
         "Code: `v3/altdata_r14_spill.py` (edges, features, KG export) and `v3/round14_spill_checks.py` (this report).",
         "", "## Storage", "",
         "* Features: `$BDA_DATA_V3/v3model/alt14_spill_weekly.parquet`. Rows are exactly those of "
         "`alt_weekly.parquet` (`ticker, Date, eligible, fwd_ret_30, cik`), plus 21 `spill_{tnic,own,sic}_*` "
         "columns (float32), `spill_*_has` flags and `ref_own_r{5,21,63}` (the stock's own trailing returns, "
         "for reference only).",
         "* Trial block `spill` = the 14 `tnic` and `own` columns (`altdata_r14_spill.SPILL_COLUMNS`). The "
         "`sic` columns are a control only.",
         "* Point-in-time edges: `$BDA_DATA_V3/spill/edges_tnic.parquet` (`src_cik, dst_cik, weight, tnic_year, "
         "valid_from, valid_to`) and `edges_own.parquet` (`src, dst` tickers, `weight, n_common, period, "
         "valid_from, valid_to`). `valid_to` is exclusive. `altdata_r14_spill.edges_asof(date, kind)` gives the "
         "ticker-space edges used on a decision date.",
         "* Knowledge graph: `$BDA_DATA_V3/spill/firm_links_graph_full.ttl.gz` holds every edge version, and "
         "`ExploitationZone/firm_links_graph.ttl` is a committed sample (links valid on the last decision date, "
         "with top-1000 names as sources). Both use `onto:productPeerOf` and `onto:sharesOwnersWith`, plus "
         "reified `onto:ProductPeerLink` and `onto:CommonOwnershipLink` nodes carrying `onto:linkWeight`, "
         "`onto:validFrom` and `onto:validTo`.", ""]
    e = cov["edges"]
    L += ["## Edges", "",
          f"* **E1 TNIC-3**: {e['tnic_rows']:,} top-10 edges over years {min(e['tnic_by_year'])}-"
          f"{max(e['tnic_by_year'])}. Source CIKs by year: "
          + ", ".join(f"{y}: {n:,}" for y, n in e["tnic_sources_by_year"].items()) + ".",
          f"* **E2 13F common ownership**: {e['own_rows']:,} top-10 edges over {len(e['own_sources_by_period'])} "
          f"quarters. Source tickers per quarter range from {min(e['own_sources_by_period'].values()):,} to "
          f"{max(e['own_sources_by_period'].values()):,}.",
          f"* Mean neighbour count (eligible rows with at least 1 neighbour): "
          + ", ".join(f"{k} {v:.1f}" for k, v in cov["mean_n_eligible"].items()) + ".", ""]
    L += ["## As-of audit", "",
          "| check | result |", "|---|---|"]
    for k in ("tnic_version_rule", "own_version_rule", "tnic_used_after_valid_from", "own_used_after_valid_from",
              "tnic_year_july_rule", "return_windows_end_at_date"):
        L.append(f"| {k} | {'PASS' if au[k] else 'FAIL'} |")
    L += ["", f"Days from the version's `valid_from` to the decision date: TNIC median "
          f"{au['days_since_tnic_version']['median']:.0f} (max {au['days_since_tnic_version']['max']:.0f}); 13F median "
          f"{au['days_since_13f_version']['median']:.0f} (max {au['days_since_13f_version']['max']:.0f}). TNIC year used, "
          "by decision year: " + ", ".join(f"{k}: {v}" for k, v in au["tnic_year_by_date_year"].items())
          + ". The 2023 pairs are the latest available; for decision dates after 2025-06-30 they are carried "
          "forward and are older than 12 months.", ""]
    hc = sp.HAS_COLUMNS
    L += ["## Coverage (share of rows with at least 1 neighbour)", "",
          "| year | " + " | ".join(f"{h.split('_')[1]} eligible" for h in hc) + " | "
          + " | ".join(f"{h.split('_')[1]} all" for h in hc) + " |",
          "|---|" + "---|" * (2 * len(hc))]
    for y in sorted(cov["eligible_by_year"][hc[0]]):
        L.append(f"| {y} | " + " | ".join(f"{cov['eligible_by_year'][h][y]:.1%}" for h in hc) + " | "
                 + " | ".join(f"{cov['all_by_year'][h][y]:.1%}" for h in hc) + " |")
    L.append("| total | " + " | ".join(f"{cov['eligible'][h]:.1%}" for h in hc) + " | "
             + " | ".join(f"{cov['all'][h]:.1%}" for h in hc) + " |")
    dl = cov["eligible_by_delisted"]
    ck = cov["eligible_by_has_cik"]
    L += ["", "Eligible rows, survivors vs delisted: " + "; ".join(
        f"{h.split('_')[1]} {dl[h]['survivor']:.1%} / {dl[h]['delisted']:.1%}"
        for h in hc) + ". With vs without a CIK: " + "; ".join(
        f"{h.split('_')[1]} {ck[h]['cik']:.1%} / {ck[h]['no_cik']:.1%}"
        for h in hc) + ". Top-1000: " + ", ".join(f"{h.split('_')[1]} {cov['top1000'][h]:.1%}" for h in hc) + ".", ""]
    L += ["## Leak probes (held-out AUC, random 30% of dates, rows 2017-2022, spill features only)", "",
          "| features | no EDGAR record | later delisted |", "|---|---|---|"]
    for k in lk["no_edgar"]:
        L.append(f"| {k} | {lk['no_edgar'][k]:.3f} | {lk['delisted'][k]:.3f} |")
    L += ["", f"Base rates: no EDGAR {lk['base_rate']['no_edgar']:.1%}, delisted {lk['base_rate']['delisted']:.1%}.", ""]
    L += ["## Placebos (tradable rows, weekly IC vs `fwd_ret_30`)", "",
          "| feature | real IC | shuffled-date mean (sd, max abs over 20) | shuffled-neighbour IC (t) |",
          "|---|---|---|---|"]
    for c in sp.ALL_COLUMNS:
        sn = snp.get(c)
        L.append(f"| {c} | {_fmt(ics[c]['tradable']['ic'])} | {_fmt(sdp[c]['mean'])} ({sdp[c]['sd']:.4f}, "
                 f"{sdp[c]['max_abs']:.4f}) | " + (f"{_fmt(sn['ic'])} ({sn['nw_t']:+.2f})" if sn else "") + " |")
    L += ["", "## Standalone weekly ICs (raw features, Spearman vs 30-session forward return)", "",
          "| feature | IC all | t | IC tradable | t | IC top-1000 | t | coverage tradable | decay 5d | 21d | 63d |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in FEATS:
        r = ics[c]
        L.append(f"| {c} | {_fmt(r['all']['ic'])} | {r['all']['nw_t']:+.2f} | {_fmt(r['tradable']['ic'])} | "
                 f"{r['tradable']['nw_t']:+.2f} | {_fmt(r['top1000']['ic'])} | {r['top1000']['nw_t']:+.2f} | "
                 f"{r['tradable']['coverage']:.0%} | {_fmt(r['decay'][5], 3)} | {_fmt(r['decay'][21], 3)} | "
                 f"{_fmt(r['decay'][63], 3)} |")
    L += ["", "Spill mean net of the stock's own same-window return (per-date rank regression residual, tradable):", "",
          "| feature | partial IC | t |", "|---|---|---|"]
    for c, v in pic.items():
        L.append(f"| {c} | {_fmt(v['ic'])} | {v['nw_t']:+.2f} |")
    L += ["", "Per-year tradable IC:", "",
          "| feature | " + " | ".join(sorted(ics[FEATS[0]]["by_year"])) + " |",
          "|---|" + "---|" * len(ics[FEATS[0]]["by_year"])]
    for c in sp.ALL_COLUMNS:
        if c.endswith("_n") or "r21" in c or "x21" in c or "r63" in c:
            by = ics[c]["by_year"]
            L.append(f"| {c} | " + " | ".join(f"{by.get(y, float('nan')):+.3f}" for y in sorted(ics[FEATS[0]]['by_year']))
                     + " |")
    L += ["", res.get("notes", ""), ""]
    open(REPORT, "w").write("\n".join(L))


def run_all(log=print) -> dict:
    t0 = time.time()
    a = load_panel()
    res = {"asof": asof_audit(a, log=log), "coverage": coverage(a, log=log)}
    res["leak"] = leak_probe(a, log=log)
    res["ics"] = feature_ics(a, log=log)
    res["partial"] = partial_ics(a, log=log)
    res["shuf_date"] = shuffled_date_placebo(a, log=log)
    res["shuf_nb"] = shuffled_neighbour_placebo(a, log=log)
    json.dump(res, open(os.path.join(sp.spill_dir(), "checks.json"), "w"), indent=1, default=str)
    write_report(res)
    log(f"[r14s] checks done ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    import sys
    if "report" in sys.argv:            # re-render from checks.json
        r = json.load(open(os.path.join(sp.spill_dir(), "checks.json")))
        for c in r["ics"]:
            r["ics"][c]["decay"] = {int(k): v for k, v in r["ics"][c]["decay"].items()}
        write_report(r)
    else:
        run_all()
