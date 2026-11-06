"""Data-quality checks for the v3 price store -> ``<data>/prices/quality_report.md``.

Checks: rows/tickers per year, duplicate keys, non-positive / inconsistent
prices, missing raw closes, zero-volume filler bars, |daily return| > 80%
events (sampled and labelled: split on record / unadjusted split / adjustment
glitch / real move), stale tickers, and coverage of delisted names by
delisting year.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import prices as P
from ._alpaca import data_dir, md_table

SPLIT_RATIOS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75, 80, 100, 150, 200, 250])


def _near_split_ratio(r: float, tol: float = 0.03) -> bool:
    if not np.isfinite(r) or r <= 0:
        return False
    x = r if r >= 1 else 1.0 / r
    return bool(np.any(np.abs(x / SPLIT_RATIOS - 1) < tol)) or abs(x - 1.5) / 1.5 < tol


def basic_stats(raw: pd.DataFrame) -> pd.DataFrame:
    y = raw["Date"].str[:4]
    g = raw.groupby(y)
    return pd.DataFrame({
        "rows": g.size(), "tickers": g["ticker"].nunique(),
        "duplicates": g.apply(lambda d: int(d.duplicated(["ticker", "Date"]).sum()), include_groups=False),
        "non_positive_close": g["company_close"].apply(lambda s: int((s <= 0).sum())),
        "non_positive_ohl": g.apply(lambda d: int(((d[["Open", "High", "Low"]] <= 0).any(axis=1)).sum()),
                                    include_groups=False),
        "high_lt_low": g.apply(lambda d: int((d["High"] < d["Low"] - 1e-9).sum()), include_groups=False),
        "missing_close_raw": g["close_raw"].apply(lambda s: int(s.isna().sum())),
        "zero_volume_bars": g["company_volume"].apply(lambda s: int((s <= 0).sum())),
    }).rename_axis("year")


def big_moves(clean: pd.DataFrame, threshold: float = 0.8, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """|close/prev_close - 1| > threshold on consecutive traded bars of a ticker.

    Labels: ``split_on_record`` (a split/reverse split in Alpaca's corporate
    actions within 3 days), ``unadjusted_split?`` (adjusted move equals the raw
    move, adj_factor unchanged and the ratio is a typical split ratio),
    ``adjustment_glitch`` (raw close barely moved but the adjusted one did), else
    ``real_move`` (with the volume ratio vs the prior 20 sessions as support)."""
    d = clean[["ticker", "Date", "company_close", "close_raw", "company_volume"]].sort_values(["ticker", "Date"])
    g = d.groupby("ticker", sort=False)
    prev_c, prev_r = g["company_close"].shift(), g["close_raw"].shift()
    ret = d["company_close"] / prev_c - 1
    raw_ret = d["close_raw"] / prev_r - 1
    vol20 = g["company_volume"].transform(lambda s: s.shift().rolling(20, min_periods=5).mean())
    ev = d.assign(ret=ret, raw_ret=raw_ret, prev_close=prev_c, vol_ratio=d["company_volume"] / vol20)
    ev = ev[ev["ret"].abs() > threshold].copy()
    if splits is not None and len(splits) and len(ev):
        s = splits.dropna(subset=["symbol"]).copy()
        s["d"] = pd.to_datetime(s["ex_date"].fillna(s.get("process_date")))
        e = ev[["ticker", "Date"]].reset_index().merge(s[["symbol", "d"]], left_on="ticker", right_on="symbol")
        e = e[(pd.to_datetime(e["Date"]) - e["d"]).dt.days.abs() <= 3]
        ev["split_on_record"] = ev.index.isin(e["index"])
    else:
        ev["split_on_record"] = False
    adj_change = (ev["ret"] + 1) / (ev["raw_ret"] + 1) - 1
    lab = np.where(ev["split_on_record"] & (adj_change.abs() < 0.02), "unadjusted_split?",
                   np.where(ev["split_on_record"], "split_on_record_but_adjusted_move", ""))
    lab = np.where((lab == "") & (adj_change.abs() < 0.02) & ev["ret"].add(1).map(_near_split_ratio)
                   & (ev["vol_ratio"] < 3), "unadjusted_split?", lab)
    lab = np.where((lab == "") & (ev["raw_ret"].abs() < 0.2), "adjustment_glitch", lab)
    lab = np.where(lab == "", "real_move", lab)
    ev["label"] = lab
    return ev


def stale_tickers(clean: pd.DataFrame, classification: pd.DataFrame | None, store_end: str) -> pd.DataFrame:
    last = clean.groupby("ticker")["Date"].max()
    end = pd.Timestamp(store_end)
    lag = (end - pd.to_datetime(last)).dt.days
    out = pd.DataFrame({"last_date": last, "days_since_last_bar": lag})
    if classification is not None:
        act = set(classification.loc[(classification["status"] == "active") & classification["in_universe"],
                                     "fetch_symbol"])
        out["alpaca_active"] = out.index.isin(act)
    # flat runs: >= 20 consecutive traded bars with an unchanged close
    d = clean[["ticker", "Date", "company_close"]].sort_values(["ticker", "Date"])
    same = d["company_close"].eq(d.groupby("ticker")["company_close"].shift())
    run_id = (~same).cumsum()
    runs = same.groupby([d["ticker"], run_id]).sum()
    out["max_flat_run"] = runs.groupby(level=0).max().reindex(out.index).fillna(0).astype(int)
    return out


def delisted_coverage(classification: pd.DataFrame, clean: pd.DataFrame, store_end: str) -> pd.DataFrame:
    """Inactive / non-current common stocks: with data or not, by year of last bar."""
    last = clean.groupby(clean["ticker"].str.split("~").str[0])["Date"].max()
    c = classification[classification["cls"] == "common"].drop_duplicates("fetch_symbol").copy()
    c["last_bar"] = c["fetch_symbol"].map(last)
    c["has_data"] = c["last_bar"].notna()
    c["delisted"] = c["has_data"] & ((pd.Timestamp(store_end) - pd.to_datetime(c["last_bar"])).dt.days > 10)
    c["delist_year"] = c["last_bar"].str[:4].where(c["delisted"])
    return c


def run(write: bool = True) -> str:
    from . import universe as U
    meta = P.read_meta()
    raw = P.load_prices(clean=False)
    clean = P.load_prices(clean=True)
    store_end = meta.get("last_date") or raw["Date"].max()
    try:
        cls = U.load_classification()
    except FileNotFoundError:
        cls = None
    ca = U.load_corporate_actions()
    splits = ca[ca["ca_type"].isin(["forward_splits", "reverse_splits"])] if len(ca) else None

    st = basic_stats(raw)
    ev = big_moves(clean, splits=splits)
    stale = stale_tickers(clean, cls, store_end)
    size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(P.prices_dir("bars"))
                  for f in fs) / 1e6

    L = [f"# v3 price store — quality report",
         "",
         f"Store: `{P.prices_dir('bars')}` · last session **{store_end}** · "
         f"{len(raw):,} rows · {raw['ticker'].nunique():,} symbols · {size_mb:,.0f} MB on disk. "
         f"After symbol-reuse cleaning (`load_prices(clean=True)`): {len(clean):,} rows, "
         f"{clean['ticker'].nunique():,} ticker ids.",
         "", "## Rows, tickers and integrity per year", "", md_table(st), "",
         f"Non-positive adjusted prices come from broken Alpaca adjustment factors on "
         f"{raw.loc[raw['company_close'] <= 0, 'ticker'].nunique()} tickers "
         f"({', '.join(sorted(raw.loc[raw['company_close'] <= 0, 'ticker'].unique())[:10])}): adjusted closes go "
         "negative/zero while `close_raw` is fine; `load_prices(clean=True)` drops those rows. "
         "`high_lt_low` rows are the same tickers (negative prices invert High/Low).", "",
         "Zero-volume bars are filler Alpaca emits for some symbols on days without trades "
         "(halts, between two companies using the same symbol); `load_prices(clean=True)` drops them.", ""]
    lab = ev["label"].value_counts()
    L += ["## |daily return| > 80% (cleaned data)", "",
          f"{len(ev):,} events on {ev['ticker'].nunique():,} tickers "
          f"({len(ev) / max(len(clean), 1) * 1e6:.1f} per million bars). Labels:", "",
          md_table(lab.rename("events")), "",
          "Labelling: `unadjusted_split?` = adjusted and raw move are identical (adjustment factor "
          "unchanged), the ratio is a typical split ratio and volume is not elevated; "
          "`adjustment_glitch` = the raw close barely moved but the adjusted close did; "
          "`real_move` = everything else (volume ratio vs the prior 20 sessions shown — real "
          "moves usually trade at many times normal volume). `split_on_record_but_adjusted_move` = a "
          "split is on record within 3 days but the adjusted series still jumps by a different ratio "
          "than the raw one — Alpaca applied the split factor on the wrong day or with the wrong ratio; "
          "these are data errors (a few dozen bars) that downstream code should winsorise or drop.", ""]
    rng = np.random.default_rng(0)
    for name in lab.index:
        sub = ev[ev["label"] == name]
        smp = sub.iloc[rng.choice(len(sub), size=min(12, len(sub)), replace=False)]
        L += [f"### Sample: {name}", "",
              md_table(smp[["ticker", "Date", "prev_close", "company_close", "ret", "raw_ret", "vol_ratio"]]
              .round(3), index=False), ""]
    act_stale = stale[(stale["days_since_last_bar"] > 10) & stale.get("alpaca_active", False)]
    L += ["## Stale tickers", "",
          f"* Ticker ids whose last bar is > 10 days before the store end: "
          f"{int((stale['days_since_last_bar'] > 10).sum()):,} (delisted/renamed — expected).",
          f"* ...of which Alpaca still lists the symbol as an *active* common stock: {len(act_stale):,} "
          f"(examples: {', '.join(act_stale.sort_values('days_since_last_bar').index[:15])}).",
          f"* Tickers with >= 20 consecutive traded bars at an unchanged close: "
          f"{int((stale['max_flat_run'] >= 20).sum()):,} "
          f"(examples: {', '.join(stale[stale['max_flat_run'] >= 20].index[:15])}).", ""]
    if cls is not None:
        cov = delisted_coverage(cls, clean, store_end)
        inact = cov[cov["status"] == "inactive"]
        L += ["## Coverage of delisted names", "",
              f"* Alpaca *inactive* common stocks (NASDAQ/NYSE/AMEX): {len(inact):,}; with SIP bars since "
              f"2016: {int(inact['has_data'].sum()):,} ({inact['has_data'].mean():.0%}). Those without bars "
              "were mostly delisted before 2016 or are renamed securities whose history lives under the "
              "new symbol.",
              f"* All common-stock candidates that stopped trading before the store end (any source — "
              f"Alpaca inactive, 2018 NASDAQ list, discovery): {int(cov['delisted'].sum()):,}.", "",
              "Delisted common stocks with data, by year of last bar and source:", "",
              md_table(pd.crosstab(cov.loc[cov["delisted"], "delist_year"], cov.loc[cov["delisted"], "source"],
                          margins=True)), ""]
    txt = "\n".join(L)
    if write:
        with open(os.path.join(P.prices_dir(), "quality_report.md"), "w") as f:
            f.write(txt)
        ev.to_parquet(os.path.join(P.prices_dir(), "big_moves.parquet"))
    return txt


if __name__ == "__main__":
    print(run())
