"""``optimized_v3``: the Round-5/7 cost-aware mean-variance book for the live bot.

OFF by default (the workflow default stays ``long_only_v3``). It reproduces, for
the single live decision date, the ``mv_abs_all`` construction of
``v3/round5.py`` / ``v3/round7.py``:

* universe  = names the production model scored (``pred_rank``) that are
  eligible in the v3 risk model on the last bar (raw price >= $3, 20-session
  ADV >= $2M, >= 120 bars) with a market-cap estimate;
* risk      = ``v3.risk.RiskEngine`` fitted on the bars the bot already fetched
  (``fetch_live_data`` with ``config.OPT_HISTORY_DAYS`` of history) + EDGAR
  fundamentals (shares outstanding -> size/mcap, B/P, E/P, GP/A, ROA, FF-12
  sector from SIC);
* alpha     = ``optimizer.grinold_alpha`` of the model scores (inverse-normal
  ranks) x IC (``config.OPT_IC``, default 0.0725 = the v2-recipe signal's
  traded-universe IC in Round 7) x specific vol;
* costs     = Corwin-Schultz half-spread from the bars' High/Low/Close +
  square-root impact on ADV, at ``config.OPT_AUM`` (default $1M = the R7 $1M
  configuration; the ~$100k paper account trades less than that, so costs and
  the 10%-of-ADV liquidity cap are conservative);
* spec      = ``gate.R5_COMMON`` + ``gate.R5_CONFIGS['mv_abs_all']``: long-only,
  fully invested, <= 5% per name, <= 10% x ADV / AUM per name, FF-12 sector
  weights within +-10 pp of the cap-weighted universe, lambda 10, names below
  0.1% dropped; the book is then scaled to ``config.OPT_TARGET_EXPOSURE``.

Every failure (no fundamentals, stale fundamentals, short history, too few
names, solver not converged, a constraint violated) raises
``OptimizerUnavailable``/an exception; the bot catches it and falls back to the
``long_only_v3`` K=30 book (``bot._optimized_v3_or_fallback``). Nothing is ever
traded on a partially built book.

Fundamentals for the cloud (GitHub Actions has no ``$BDA_DATA_V3`` store): a
small committed snapshot under ``trading_agent/data/fundamentals_snapshot/``
(trimmed ``cik_map`` / ``fundamentals_asof`` / ``sectors`` with a manifest).
Refresh it from the local EDGAR store with

    python -m DataAnalysisPipeline2.trading_agent.optimized snapshot

The snapshot rows are the same point-in-time ``asof_date`` rows as the store
(filings available on or before each date), so the live decision sees exactly
what the store would show for every date of the bar window; it only goes
stale. ``resolve_fund_dir`` prefers ``$BDA_OPT_FUND_DIR``, then the local
store, then the snapshot, and rejects any source whose newest filing is older
than ``config.OPT_FUND_MAX_AGE_DAYS``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(AGENT_DIR, "data", "fundamentals_snapshot")
FUND_FILES = ("cik_map.parquet", "fundamentals_asof.parquet", "sectors.parquet")
SNAPSHOT_YEARS = 3              # >= OPT_HISTORY_DAYS of bars + the last filing before them
SPEC_NAME = "mv_abs_all"
BAR_COLS = ["ticker", "Date", "High", "Low", "company_close", "company_volume"]


class OptimizerUnavailable(RuntimeError):
    """The optimized book cannot be built safely; the caller must fall back."""


# --------------------------------------------------------------------------- #
# Fundamentals source
# --------------------------------------------------------------------------- #
def _store_fund_dir() -> str:
    """The local EDGAR store's fundamentals dir (not created if absent)."""
    from DataAnalysisPipeline2.v3._alpaca import DEFAULT_DATA_DIR
    return os.path.join(os.environ.get("BDA_DATA_V3", DEFAULT_DATA_DIR), "fundamentals")


def fund_as_of(path: str) -> pd.Timestamp:
    """Newest filing date covered by a fundamentals dir (manifest if present)."""
    man = os.path.join(path, "manifest.json")
    if os.path.exists(man):
        with open(man) as f:
            return pd.Timestamp(json.load(f)["as_of"])
    fa = pd.read_parquet(os.path.join(path, "fundamentals_asof.parquet"), columns=["last_filed"])
    return pd.Timestamp(fa["last_filed"].max())


def resolve_fund_dir(explicit: str | None = None, max_age_days: int = 120, today=None,
                     candidates: list | None = None) -> dict:
    """First usable fundamentals source among ``explicit`` / the local store /
    the committed snapshot: all three parquet files present and newest filing
    no older than ``max_age_days``. Raises ``OptimizerUnavailable`` otherwise."""
    today = pd.Timestamp(today or dt.date.today())
    if candidates is None:
        candidates = ([("env", explicit)] if explicit else []) + [("store", _store_fund_dir()),
                                                                 ("snapshot", SNAPSHOT_DIR)]
    reasons = []
    for source, path in candidates:
        if not path or not all(os.path.exists(os.path.join(path, f)) for f in FUND_FILES):
            reasons.append(f"{source}: missing ({path})")
            continue
        try:
            as_of = fund_as_of(path)
        except Exception as e:      # unreadable files count as unavailable
            reasons.append(f"{source}: unreadable ({e})")
            continue
        age = int((today - as_of.normalize()).days)
        if age > max_age_days:
            reasons.append(f"{source}: stale (newest filing {as_of.date()}, {age} d > {max_age_days} d)")
            continue
        return {"path": path, "source": source, "as_of": str(as_of.date()), "age_days": age}
    raise OptimizerUnavailable("no usable fundamentals: " + "; ".join(reasons))


def build_snapshot(src: str | None = None, out: str = SNAPSHOT_DIR, years: int = SNAPSHOT_YEARS,
                   today=None) -> dict:
    """Trim the EDGAR store to what ``risk.live_panel`` needs for a live decision.

    * fundamentals_asof: ``risk.LIVE_FUND_COLS`` + ``risk.LIVE_FUND_OPTIONAL``
      (every column ``fundamentals.fundamental_features`` reads, float64 as in
      the store, so the altall model's fundamentals block is exact), rows with
      asof_date in the last ``years`` years plus each CIK's last row before
      that (so the as-of join is exact from the window start), CIKs in the
      current cik_map only;
    * cik_map: ticker, cik, valid_from, valid_to (current tickers);
    * sectors: cik, ff12, sector_name for those CIKs.
    """
    from DataAnalysisPipeline2.v3 import risk
    src = src or _store_fund_dir()
    today = pd.Timestamp(today or dt.date.today())
    cut = today - pd.DateOffset(years=years)
    cm = pd.read_parquet(os.path.join(src, "cik_map.parquet"))
    cm = cm[cm["valid_to"].isna() | (pd.to_datetime(cm["valid_to"]) >= cut)]
    cm = cm[["ticker", "cik", "valid_from", "valid_to"]].reset_index(drop=True)
    ciks = set(cm["cik"].astype("int64"))
    import pyarrow.parquet as pq
    fpath = os.path.join(src, "fundamentals_asof.parquet")
    have = set(pq.ParquetFile(fpath).schema_arrow.names)
    fa = pd.read_parquet(fpath, columns=risk.LIVE_FUND_COLS + [c for c in risk.LIVE_FUND_OPTIONAL if c in have])
    fa = fa[fa["cik"].isin(ciks)].sort_values(["cik", "asof_date"])
    recent = fa[fa["asof_date"] >= cut]
    before = fa[fa["asof_date"] < cut].groupby("cik").tail(1)
    fa = pd.concat([before, recent]).sort_values(["cik", "asof_date"]).reset_index(drop=True)
    fa["last_form"] = fa["last_form"].astype("category")
    secs = pd.read_parquet(os.path.join(src, "sectors.parquet"), columns=["cik", "ff12", "sector_name"])
    secs = secs[secs["cik"].isin(ciks)].reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    cm.to_parquet(os.path.join(out, "cik_map.parquet"), index=False, compression="zstd")
    fa.to_parquet(os.path.join(out, "fundamentals_asof.parquet"), index=False, compression="zstd")
    secs.to_parquet(os.path.join(out, "sectors.parquet"), index=False, compression="zstd")
    man = {"as_of": str(pd.Timestamp(fa["last_filed"].max()).date()), "built": str(today.date()),
           "source": "local EDGAR store ($BDA_DATA_V3/fundamentals)", "window_start": str(cut.date()),
           "rows": {"cik_map": len(cm), "fundamentals_asof": len(fa), "sectors": len(secs)},
           "bytes": {f: os.path.getsize(os.path.join(out, f)) for f in FUND_FILES},
           "note": "Point-in-time rows (asof_date = filing availability). Refresh monthly with "
                   "`python -m DataAnalysisPipeline2.trading_agent.optimized snapshot`; the bot rejects it "
                   "when the newest filing is older than BDA_OPT_FUND_MAX_AGE_DAYS."}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


# --------------------------------------------------------------------------- #
# Inputs for one live date
# --------------------------------------------------------------------------- #
def _wide(b: pd.DataFrame, col: str, dates, tickers) -> pd.DataFrame:
    return (b.pivot(index="Date", columns="ticker", values=col)
             .reindex(index=dates, columns=tickers).astype(float))


def duplicate_share_classes(tickers, adv, keep, fund_dir: str) -> set:
    """Secondary share classes to drop: tickers that map to the same EDGAR CIK
    as another candidate (GOOG/GOOGL, FOX/FOXA, ...). EDGAR shares outstanding
    are company-level, so every class gets the WHOLE company's market cap and
    the cap-weighted benchmark double-counts it (Alphabet was ~18% of b).
    Keeps the most liquid class (highest ADV) per CIK. Fails open (empty set)."""
    try:
        cm = pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet"), columns=["cik", "ticker", "valid_to"])
    except Exception:
        return set()
    cm = cm[cm["valid_to"].isna()].drop_duplicates("ticker")
    df = pd.DataFrame({"ticker": np.asarray(tickers).astype(str), "adv": np.asarray(adv, float)})[np.asarray(keep)]
    df = df.merge(cm[["ticker", "cik"]], on="ticker", how="inner")
    df = df.sort_values(["cik", "adv", "ticker"], ascending=[True, False, True])
    return set(df.loc[df.duplicated("cik", keep="first"), "ticker"])


def build_inputs(bars: pd.DataFrame, scores: pd.Series, fund_dir: str, *, min_price: float = 3.0,
                 min_adv_usd: float = 2e6, min_bars: int = 120) -> tuple[dict, dict]:
    """``optimize_portfolio`` keyword inputs for the last date of ``bars``.

    ``scores``: ticker -> model score (any monotone score; NaN = unscored).
    Returns (inputs, meta)."""
    from DataAnalysisPipeline2.v3 import costs as cs
    from DataAnalysisPipeline2.v3 import risk

    miss = [c for c in BAR_COLS if c not in bars.columns]
    if miss:
        raise OptimizerUnavailable(f"bars lack {miss} (needed for spreads / ADV)")
    b = bars[BAR_COLS].copy()
    b["Date"] = pd.to_datetime(b["Date"]).dt.strftime("%Y-%m-%d")
    b["ticker"] = b["ticker"].astype(str)
    b = b.drop_duplicates(["ticker", "Date"], keep="last").sort_values(["Date", "ticker"]).reset_index(drop=True)
    p = risk.live_panel(b, fund_dir=fund_dir, min_price=min_price, min_adv_usd=min_adv_usd, min_bars=min_bars)
    T = len(p.dates)
    if T < risk.BETA_WIN + 42:
        raise OptimizerUnavailable(f"only {T} sessions of history; the risk model needs >= {risk.BETA_WIN + 42}")
    eng = risk.RiskEngine(p)
    m = eng.model(str(p.dates[-1]))
    if len(m.tickers) == 0:
        raise OptimizerUnavailable("risk model has no eligible names on the decision date")
    if not (np.all(np.isfinite(m.F)) and np.all(np.isfinite(m.X)) and np.all(np.isfinite(m.D))):
        raise OptimizerUnavailable("risk model produced non-finite exposures / covariances")

    # cost inputs (as round5.prepare, on the live bars)
    idx = pd.Index(p.dates)
    H = _wide(b, "High", p.dates, p.tickers)
    L = _wide(b, "Low", p.dates, p.tickers)
    C = _wide(b, "company_close", p.dates, p.tickers)
    csest = cs.corwin_schultz(H, L, C)
    hs_all = cs.half_spread(csest, pd.DataFrame(p.price.astype(float), index=idx, columns=p.tickers),
                            pd.DataFrame(p.adv.astype(float), index=idx, columns=p.tickers)).iloc[-1]
    r = pd.DataFrame(np.clip(p.ret, -0.5, 1.0).astype(float), index=idx, columns=p.tickers)
    vol60 = r.iloc[-80:].rolling(60, min_periods=40).std().iloc[-1]

    pos = {t: i for i, t in enumerate(p.tickers)}
    col = np.array([pos[t] for t in m.tickers])
    sc = pd.Series(scores, dtype=float)
    sc.index = sc.index.astype(str)
    s = sc[~sc.index.duplicated(keep="last")].reindex(m.tickers).to_numpy(dtype=float)
    keep = np.isfinite(s) & np.isfinite(m.mcap) & (m.mcap > 0)          # round5 universe "all"
    adv_last = p.adv[-1, col].astype(float)
    dup = duplicate_share_classes(m.tickers, adv_last, keep, fund_dir)
    keep &= ~np.isin(m.tickers, list(dup))
    v = vol60.to_numpy()[col]
    sigma = np.where(np.isfinite(v), v, np.nanmedian(v))
    inputs = dict(tickers=m.tickers[keep], scores=s[keep], X=m.X[keep], F_daily=m.F, D_daily=m.D[keep],
                  mcap=m.mcap[keep], sector=m.sector[keep], factors=list(m.factors),
                  half_spread=hs_all.to_numpy()[col][keep], sigma_daily=sigma[keep],
                  adv=p.adv[-1, col][keep].astype(float))
    meta = {"decision_date": str(p.dates[-1]), "n_sessions": T, "n_bar_tickers": int(len(p.tickers)),
            "n_risk_eligible": int(len(m.tickers)), "n_scored": int(np.isfinite(s).sum()),
            "n_universe": int(keep.sum()), "sector_names": list(p.sector_names),
            "dropped_share_classes": sorted(dup)}
    return inputs, meta


def live_spec(overrides: dict | None = None) -> dict:
    from DataAnalysisPipeline2.v3 import gate
    return {**gate.R5_COMMON, **gate.R5_CONFIGS[SPEC_NAME], **(overrides or {})}


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #
def check_book(w: np.ndarray, inputs: dict, spec: dict, aum: float, diag: dict, tol: float = 2e-3) -> None:
    """Raise unless the (pre-scaling, sum ~ 1) book satisfies every constraint."""
    from DataAnalysisPipeline2.v3 import optimizer as op
    sp = {**op.DEFAULT_SPEC, **spec}
    if diag.get("status") != "solved":
        raise OptimizerUnavailable(f"solver status {diag.get('status')!r} after {diag.get('iters')} iterations")
    if not np.all(np.isfinite(w)):
        raise OptimizerUnavailable("non-finite weights")
    if w.min() < -1e-9:
        raise OptimizerUnavailable(f"negative weight {w.min():.2e} in a long-only book")
    if abs(w.sum() - 1.0) > tol:
        raise OptimizerUnavailable(f"budget not met: sum of weights {w.sum():.4f}")
    adv = np.asarray(inputs["adv"], float)
    cap = np.minimum(sp["name_cap"], np.where(np.isfinite(adv) & (adv > 0), sp["liq_frac"] * adv / aum, 0.0))
    if np.any(w > cap + tol):
        i = int(np.argmax(w - cap))
        raise OptimizerUnavailable(f"name cap violated: {inputs['tickers'][i]} {w[i]:.4f} > {cap[i]:.4f}")
    mc = np.asarray(inputs["mcap"], float)
    bw = mc / mc.sum()
    sec = np.asarray(inputs["sector"])
    for s_ in np.unique(sec):
        msk = sec == s_
        if abs(w[msk].sum() - bw[msk].sum()) > sp["sector_band"] + tol:
            raise OptimizerUnavailable(f"sector band violated for sector code {s_}: "
                                       f"{w[msk].sum():.3f} vs benchmark {bw[msk].sum():.3f}")


def optimized_book(preds: pd.DataFrame, bars: pd.DataFrame, *, aum: float, ic: float,
                   target_exposure: float, w_prev: dict | None = None, fund_dir: str | None = None,
                   max_age_days: int = 120, min_universe: int = 300, min_names: int = 20,
                   spec: dict | None = None, rank_col: str = "pred_rank", today=None,
                   candidates: list | None = None, cap_fn=None, log=print) -> tuple[dict, dict]:
    """Long-only ``mv_abs_all`` book for the live decision.

    ``preds``: ticker, ``rank_col``. ``bars``: the live daily bars (ticker, Date,
    High, Low, company_close, company_volume) with >= ~300 sessions. ``w_prev``:
    current weights (fraction of equity; names outside the universe are sold by
    the executor). Returns (weights summing to <= ``target_exposure``, diag).
    ``cap_fn`` (optional, the Round-11 earnings blackout): called as
    ``cap_fn(tickers, {**meta, 'fund_dir': ...})`` -> (ticker -> extra upper
    bound on the fully-invested book, info dict). ANY exception from it, or a
    capped book that fails the solver / constraint checks, means NO caps
    (logged in ``diag['earn_blackout']``); it never makes the book unavailable.
    Raises on any failure -- the caller falls back to ``long_only_v3``."""
    from DataAnalysisPipeline2.v3 import optimizer as op
    t0 = time.time()
    if preds is None or len(preds) == 0 or rank_col not in preds:
        raise OptimizerUnavailable("no model scores")
    if not (0.0 < float(target_exposure) <= 1.0):
        raise OptimizerUnavailable(f"target exposure {target_exposure} outside (0, 1]")
    fd = resolve_fund_dir(fund_dir, max_age_days, today=today, candidates=candidates)
    log(f"[optimized_v3] fundamentals: {fd['source']} ({fd['path']}), newest filing {fd['as_of']} "
        f"({fd['age_days']} d old)")
    scores = pd.Series(preds[rank_col].to_numpy(dtype=float), index=preds["ticker"].astype(str))
    inputs, meta = build_inputs(bars, scores, fd["path"])
    t_build = time.time() - t0
    log(f"[optimized_v3] decision {meta['decision_date']}: {meta['n_sessions']} sessions, "
        f"{meta['n_risk_eligible']} risk-eligible, {meta['n_scored']} scored -> universe {meta['n_universe']} "
        f"(risk model + costs {t_build:.1f}s)")
    if meta["n_universe"] < min_universe:
        raise OptimizerUnavailable(f"universe of {meta['n_universe']} names < minimum {min_universe}")
    sp = live_spec(spec)
    caps, bl = None, None
    if cap_fn is not None:
        try:
            caps, bl = cap_fn(inputs["tickers"], {**meta, "fund_dir": fd["path"]})
        except Exception as e:      # noqa: BLE001 -- the blackout is optional
            bl = {"applied": False, "skipped": f"{type(e).__name__}: {e}"[:500]}
            log(f"[optimized_v3] earnings blackout skipped -- {bl['skipped']}")
    t1 = time.time()
    wd, diag = op.optimize_portfolio(sp, aum=float(aum), ic=float(ic), w_prev=w_prev or {}, ub_cap=caps, **inputs)
    w = np.array([wd.get(t, 0.0) for t in inputs["tickers"]])
    if caps is not None:
        try:
            check_book(w, inputs, sp, float(aum), diag)
        except OptimizerUnavailable as e:
            bl = {**bl, "applied": False, "skipped": f"capped book rejected ({e}); solved without the blackout"}
            log(f"[optimized_v3] earnings blackout dropped -- {bl['skipped']}")
            caps = None
            wd, diag = op.optimize_portfolio(sp, aum=float(aum), ic=float(ic), w_prev=w_prev or {}, **inputs)
            w = np.array([wd.get(t, 0.0) for t in inputs["tickers"]])
    t_solve = time.time() - t1
    check_book(w, inputs, sp, float(aum), diag)
    # The cardinality step re-solves once; the re-solve can leave a few new
    # residual positions below min_weight (untradable: below the executor's
    # minimum trade). They go to cash -- only if that is a rounding-size amount.
    small = (w > 0) & (w < float(sp["min_weight"]))
    dropped = float(w[small].sum())
    if dropped > 0.02:
        raise OptimizerUnavailable(f"{int(small.sum())} positions below min_weight hold {dropped:.1%} of the book")
    w = np.where(small, 0.0, w)
    n = int((w > 0).sum())
    if n < min_names:
        raise OptimizerUnavailable(f"only {n} names in the optimized book (< {min_names})")

    # scale the fully-invested book to the target exposure, then make the caps
    # exact (the optimizer's budget polish can leave a name a hair above them;
    # the excess stays in cash)
    adv = np.asarray(inputs["adv"], float)
    cap = np.minimum(sp["name_cap"], np.where(np.isfinite(adv) & (adv > 0), sp["liq_frac"] * adv / float(aum), 0.0))
    w = np.minimum(w * float(target_exposure), cap)
    if caps is not None:            # blacked-out names never above their current weight
        ubc = np.array([caps.get(str(t), np.inf) for t in inputs["tickers"]])
        w = np.minimum(w, np.maximum(ubc, 0.0) * float(target_exposure))
        tick = np.asarray(inputs["tickers"]).astype(str)
        blk = np.isin(tick, list(caps))
        prev = np.array([max(float((w_prev or {}).get(t, 0.0) or 0.0), 0.0) for t in tick])
        bl = {**bl, "book_w_blackout": float(w[blk].sum()),
              "n_blackout_trimmed": int(((w < prev - 1e-9) & blk).sum()),
              "n_blackout_in_book": int(((w > 0) & blk).sum())}
    weights = {str(t): float(x) for t, x in zip(inputs["tickers"], w) if x > 0}
    names = meta["sector_names"]
    other = len(names) - 1 if "Other" not in names else names.index("Other")
    sec_name = np.array([names[c] if 0 <= c < len(names) else names[other] for c in inputs["sector"]])
    mc = np.asarray(inputs["mcap"], float)
    bw = mc / mc.sum()
    by_sec = pd.DataFrame({"s": sec_name, "w": w, "b": bw}).groupby("s")[["w", "b"]].sum()
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:10]
    out = {"construction": "optimized_v3", "spec": SPEC_NAME, "aum": float(aum), "ic": float(ic),
           "target_exposure": float(target_exposure), "status": diag["status"], "iters": diag["iters"],
           "n_names": len(weights), "long": float(w.sum()), "cash": float(1.0 - w.sum()),
           "max_weight": float(w.max()), "ex_ante_vol_ann": diag["ex_ante_vol_ann"] * float(target_exposure),
           "ex_ante_vol_ann_full": diag["ex_ante_vol_ann"], "ex_ante_te_ann_vs_cw": diag["ex_ante_te_ann"],
           "ex_ante_beta_vs_cw": diag["ex_ante_beta"], "alpha_ann": diag["alpha_ann"],
           "exp_cost": diag["exp_cost"], "turnover_oneway": diag["turnover_oneway"],
           "n_dropped_small": int(small.sum()), "dropped_small_weight": dropped,
           "n_prev_held": int(sum(1 for v in (w_prev or {}).values() if v)),
           "solve_seconds": round(t_solve, 2), "build_seconds": round(t_build, 2),
           "fundamentals": fd, **{k: v for k, v in meta.items() if k != "sector_names"},
           "earn_blackout": bl,
           "sector_weights": {s: round(float(r.w), 4) for s, r in by_sec.iterrows() if r.w > 0},
           "sector_bench": {s: round(float(r.b), 4) for s, r in by_sec.iterrows()},
           "top_weights": [(t, round(x, 4)) for t, x in top],
           "style_exposure": {k: round(v, 3) for k, v in diag["exposure"].items() if not k.startswith("ind:")}}
    return weights, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="optimized_v3 fundamentals snapshot")
    ap.add_argument("cmd", choices=["snapshot", "check"])
    ap.add_argument("--src", default=None, help="EDGAR fundamentals dir (default $BDA_DATA_V3/fundamentals)")
    ap.add_argument("--out", default=SNAPSHOT_DIR)
    ap.add_argument("--years", type=int, default=SNAPSHOT_YEARS)
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        print(json.dumps(build_snapshot(a.src, a.out, a.years), indent=1))
    else:
        print(json.dumps(resolve_fund_dir(a.src, max_age_days=10 ** 6), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
