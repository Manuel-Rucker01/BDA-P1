#!/usr/bin/env python3
"""Is the v2 (no-news) model better than SPY?  Longest honest OOS evidence.

Two prediction sources, same backtest engine:

1. QUASI-OOS: the walk-forward CV out-of-fold SoftVote predictions of the
   production recipe (oof_predictions_v2final_nonews.parquet; 5 test windows,
   2025-03 -> 2026-09, each predicted by a model trained only on data that ends
   >= 31 sessions before the window). "Quasi" because the recipe and its
   hyper-parameters were chosen with this same CV, and the OOF population is the
   labelled set (names whose 30-session forward price exists and is >= $0.5, a
   mild look-ahead filter on the universe). Only days INSIDE the 5 test windows
   are traded; the embargo gaps between windows are skipped for every strategy
   and for SPY alike.
2. HOLDOUT: v2_long_it3 (trained <= 2026-05-29) scored with live-path
   semantics on 2026-07-15 -> 2026-09-23 over all eligible+embedded names.

Engine: monthly rebalance (first trading date of each month inside a window,
plus the window's first date), daily mark-to-market on adjusted closes with
weights drifting, 10 bps per side on traded notional (initial buy included),
cash earns 0. Books: production long_only_book(K=30, 5/sector, 6%, 0.98),
K=50 (7/sector, 4%, 0.98), the old DIV20 top-K (select_top_k 5% gate, K=20,
4/sector, inverse-vol capped 10%), EW over the scored universe, SPY buy & hold.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPL = os.path.join(ROOT_DIR, "ExploitationZone")
for p in (PIPELINE_DIR, SCRIPTS_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ACTIVE_ACCOUNT", "monthly")
os.environ["FEATURE_VERSION"] = "2"

COST = 0.0010  # per side
BOOKS = ["PROD_K30", "K50", "DIV20_old", "EW_universe"]


def build_weights(book, g):
    from common import select_top_k
    from trading_agent.bot import inverse_volatility_weights
    from trading_agent.portfolio import long_only_book
    if book == "PROD_K30":
        w, _ = long_only_book(g, k=30, max_per_sector=5, max_weight=0.06, target_exposure=0.98)
    elif book == "K50":
        w, _ = long_only_book(g, k=50, max_per_sector=7, max_weight=0.04, target_exposure=0.98)
    elif book == "DIV20_old":
        sel = select_top_k(g, pct_threshold=5.0, top_k=20, max_sector_frac=0.2, signal_col="pred_rank")
        vols = sel["return_volatility_20d"].fillna(sel["return_volatility_20d"].median()).fillna(0.03).values
        w = inverse_volatility_weights(sel["ticker"].tolist(), vols, target_exposure=1.0, max_weight=0.10)
    else:
        n = len(g)
        w = {t: 1.0 / n for t in g["ticker"]}
    return {t: float(x) for t, x in w.items() if x > 0}


def simulate(segments, preds, info, closes, book):
    """segments: list of (rebalance_dates, end_date) per window. Returns daily
    net returns Series (window days only) and turnover list."""
    rets, turns = [], []
    for reb, end in segments:
        nav, hold = 1.0, {}  # hold: ticker -> value (fraction of nav at start)
        prev_close_day = None
        for i, d0 in enumerate(reb):
            d1 = reb[i + 1] if i + 1 < len(reb) else end
            g = preds[preds["Date"] == d0].merge(info, on=["ticker", "Date"], how="left")
            target = build_weights(book, g)
            cur_val = sum(hold.values())
            cur_w = {t: v / cur_val for t, v in hold.items()} if cur_val > 0 else {}
            turnover = sum(abs(target.get(t, 0) - cur_w.get(t, 0)) for t in set(target) | set(cur_w))
            turns.append(turnover)
            cost = COST * turnover
            names = [t for t in target if t in closes.columns]
            seg = closes.loc[d0:d1, names].ffill()
            p0 = seg.iloc[0]
            ok = p0.notna() & (p0 > 0)
            names = [t for t in names if ok[t]]
            w = np.array([target[t] for t in names])
            rel = seg[names].divide(p0[names])
            val = (rel * w).sum(axis=1) + (1 - w.sum())  # cash stays flat
            val = val * (1 - cost)
            day_ret = val.pct_change().dropna()
            first = val.iloc[0] - 1.0  # the cost hits on the rebalance day
            if prev_close_day is not None or i > 0:
                rets.append(pd.Series([first], index=[val.index[0]]).add(0))
            else:
                rets.append(pd.Series([first], index=[val.index[0]]))
            rets.append(day_ret)
            hold = {t: float(rel[t].iloc[-1] * target[t]) for t in names}
            prev_close_day = val.index[-1]
    s = pd.concat(rets)
    # rebalance days appear twice (end of one period, start of next): combine
    s = (1 + s).groupby(level=0).prod() - 1
    return s.sort_index(), turns


def spy_returns(spy, segments):
    out = []
    for reb, end in segments:
        s = spy.loc[reb[0]:end]
        out.append(s.pct_change().dropna())
    return pd.concat(out).sort_index()


def stats(r, spy_r, ew_r):
    r = r.dropna()
    n = len(r)
    cum = float((1 + r).prod() - 1)
    cagr = float((1 + cum) ** (252 / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    v = (1 + r).cumprod()
    mdd = float((v / v.cummax() - 1).min())
    out = {"days": n, "cum_%": 100 * cum, "CAGR_%": 100 * cagr, "vol_%": 100 * vol,
           "Sharpe": sharpe, "maxDD_%": 100 * mdd}
    for name, b in (("SPY", spy_r), ("EW", ew_r)):
        if b is None:
            continue
        a = (r - b.reindex(r.index).fillna(0.0))
        out[f"IR_vs_{name}"] = float(a.mean() / a.std(ddof=1) * np.sqrt(252)) if a.std() > 0 else np.nan
    bb = spy_r.reindex(r.index).fillna(0.0)
    out["beta_SPY"] = float(np.cov(r, bb)[0, 1] / np.var(bb, ddof=1))
    return out


def block_bootstrap_ci(active, block=21, reps=10000, seed=42):
    a = active.values
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        means[k] = a[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5]) * 252
    p_le0 = float((means <= 0).mean())
    return float(a.mean() * 252), float(lo), float(hi), p_le0


def quarterly(r):
    q = (1 + r).groupby(r.index.to_period("Q")).prod() - 1
    return (100 * q).round(2)


def monthly_rebalances(dates):
    dates = sorted(pd.to_datetime(dates))
    reb = [dates[0]]
    for d in dates[1:]:
        if (d.year, d.month) != (reb[-1].year, reb[-1].month):
            reb.append(d)
    return reb


def to_md(df, fmt=".2f"):
    def f(v):
        return format(v, fmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f(v) for v in row]) + " |")
    return "\n".join(lines)


def run(label, preds, segments, info, closes, spy):
    spy_r = spy_returns(spy, segments)
    res, series = {}, {}
    for b in BOOKS:
        r, turns = simulate(segments, preds, info, closes, b)
        series[b] = r
        res[b] = {"avg_turnover": float(np.mean(turns))}
    series["SPY"] = spy_r
    ew_r = series["EW_universe"]
    rows, ci_rows = {}, {}
    for name, r in series.items():
        st = stats(r, spy_r, None if name == "EW_universe" else ew_r)
        st.update(res.get(name, {}))
        rows[name] = st
        if name != "SPY":
            act = (r - spy_r.reindex(r.index).fillna(0.0))
            m, lo, hi, p = block_bootstrap_ci(act)
            ci_rows[name] = {"ann_active_vs_SPY_%": 100 * m, "CI95_lo_%": 100 * lo,
                             "CI95_hi_%": 100 * hi, "P(active<=0)": p}
    tab = pd.DataFrame(rows).T
    tab.index.name = label
    ci = pd.DataFrame(ci_rows).T
    ci.index.name = label
    q = pd.DataFrame({k: quarterly(v) for k, v in series.items()})
    q.index = q.index.astype(str)
    q.index.name = "quarter"
    return tab, ci, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=os.path.join(EXPL, "oof_predictions_v2final_nonews.parquet"))
    ap.add_argument("--holdout-model", default=os.path.join(EXPL, "best_model_v2_long_it3.pkl"))
    ap.add_argument("--out", default=os.path.join(PIPELINE_DIR, "results", "v2_vs_spy.md"))
    args = ap.parse_args()

    import kg_embeddings_classifier as K
    from features import core
    from trading_agent.bot import fetch_company_metadata
    from eval_holdout import fetch_spy, score_model_v2, load_model

    panel = K.load_price_panel()
    meta = fetch_company_metadata()[["ticker", "Sector"]]
    raw = core.compute_features(panel, meta)
    raw["Date"] = pd.to_datetime(raw["Date"])
    info = raw[["ticker", "Date", "Sector", "return_volatility_20d", "company_close"]]
    closes = panel.assign(Date=pd.to_datetime(panel["Date"])) \
        .pivot_table(index="Date", columns="ticker", values="company_close").sort_index()
    cal = closes.index
    spy = fetch_spy("2025-01-01", cal[-1])

    # 1) quasi-OOS on CV out-of-fold predictions
    oof = pd.read_parquet(args.oof, filters=[("model", "==", "SoftVote"),
                                             ("feature_set", "==", "tabular+embedding")])
    oof["Date"] = pd.to_datetime(oof["date"])
    oof["pred_rank"] = oof.groupby("Date")["pred_rank"].rank(pct=True)  # per-date rank
    preds_oof = oof[["ticker", "Date", "pred_rank"]]
    seg_oof, win_txt = [], []
    for f, g in oof.groupby("fold"):
        reb = monthly_rebalances(g["Date"].unique())
        end = g["Date"].max()
        seg_oof.append((reb, end))
        win_txt.append(f"fold {f}: {reb[0].date()} -> {end.date()} ({len(reb)} rebalances)")
    tab1, ci1, q1 = run("CV OOF (quasi-OOS)", preds_oof, seg_oof, info, closes, spy)

    # 2) holdout, live-path scoring of v2_long_it3
    M = load_model(args.holdout_model, "v2_long_it3")
    hsc = score_model_v2(M, panel, meta, "2026-07-15", "2026-09-23", None)
    hsc = hsc.rename(columns={"pred": "pred_rank"})
    reb = monthly_rebalances(hsc["Date"].unique())
    nxt = [c for c in cal if (c.year, c.month) > (reb[-1].year, reb[-1].month)]
    seg_h = [(reb, nxt[0])]
    tab2, ci2, q2 = run("Holdout (live path)", hsc, seg_h, info, closes, spy)

    with open(args.out, "w") as fh:
        fh.write("# Is v2 better than SPY?\n\n")
        fh.write("Production recipe = v2 no-news (feature_version 2, core features, KG PCA, RF+CatBoost+"
                 "XGBoost+LightGBM rank soft-vote). Books built with the production portfolio code. "
                 "Monthly rebalance, daily mark-to-market on adjusted closes, **10 bps per side** on traded "
                 "notional, cash at 0%. SPY = buy & hold over exactly the same days. IR = annualised mean "
                 "daily active return / its stdev. CI = 21-day moving-block bootstrap (10k reps) of the "
                 "annualised mean daily active return vs SPY.\n\n")
        fh.write("## 1. Quasi out-of-sample: walk-forward CV out-of-fold predictions\n\n")
        fh.write("Source `oof_predictions_v2final_nonews.parquet`, SoftVote (= deployed ensemble), re-ranked "
                 "per date. Each window is predicted by a model trained on data ending >= 31 sessions "
                 "before it. **Quasi-OOS caveats:** the recipe/hyper-parameters were chosen with this same "
                 "CV; the OOF universe is the labelled set (names whose 30-session forward price exists and "
                 ">= $0.5), a mild look-ahead filter that favours long books slightly; the first window's "
                 "model saw only ~4 months of data. Only days inside the windows are traded (embargo gaps "
                 "skipped for every series):\n\n")
        for t in win_txt:
            fh.write(f"- {t}\n")
        fh.write("\n" + to_md(tab1) + "\n\n**Active return vs SPY (annualised, block-bootstrap 95% CI)**\n\n")
        fh.write(to_md(ci1) + "\n\n**Per-quarter returns (%, window days only)**\n\n" + to_md(q1) + "\n\n")
        fh.write("## 2. Holdout 2026-07-15 -> 2026-09-23 (v2_long_it3, trained <= 2026-05-29, live path)\n\n")
        fh.write(f"Rebalances {', '.join(str(d.date()) for d in reb)}; last book held to "
                 f"{nxt[0].date()}. Universe = all eligible+embedded names (no label filter).\n\n")
        fh.write(to_md(tab2) + "\n\n" + to_md(ci2) + "\n\n" + to_md(q2) + "\n")
    print(tab1.round(3).to_string()); print(ci1.round(3).to_string()); print(q1.to_string())
    print(tab2.round(3).to_string()); print(ci2.round(3).to_string())
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
