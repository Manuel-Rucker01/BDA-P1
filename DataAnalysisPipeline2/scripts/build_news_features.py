#!/usr/bin/env python3
"""
Build point-in-time (PIT) news-sentiment FEATURES for model training  [v2].
=============================================================================

This is the backtested-feature counterpart to the live-only tilt in
``trading_agent/news_sentiment.py``. Unlike the live tilt, these features are
joined into the training matrix and walk-forward-validated — so they MUST be
constructed point-in-time to avoid look-ahead leakage.

Pipeline (medallion-style):
    Landing      : fetch every Alpaca news article for the universe over the
                   data span, each with its true ``created_at`` timestamp
                   (paginated). Score each headline+summary with the FROZEN
                   finance lexicon (no test-period fitting).
    Trusted      : enforce a leakage rule — an article may inform the feature
                   for date D only if it was published strictly BEFORE D
                   (created_at.date() < D), so nothing from the decision day or
                   the future leaks in.
    Exploitation : aggregate to AS-OF (ticker, Date) features over rolling
                   windows: 7-day mean sentiment, news volume, 7-vs-30-day
                   sentiment momentum, and 7-day sentiment dispersion.

Output: ExploitationZone/news_features.parquet  keyed on (ticker, Date),
ready to LEFT JOIN onto the training observation frame.

HONEST CAVEATS (documented, not hidden):
  * PIT integrity is *assumed* from Alpaca's created_at timestamps; we do not
    have a vendor guarantee that articles are never back-dated/revised.
  * Coverage is heavily skewed to mega-caps; small-caps get sparse/zero news,
    so the feature is dense for a minority of names and zero for the rest.
  * Entity resolution uses *current* ticker symbols (ticker-change look-ahead).
  * The data span is only ~10 months, limiting how much news signal can be fit.
"""

import os
import sys
import time
import argparse
import datetime as dt

import duckdb
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")
OUT_PARQUET = os.path.join(EXPLOITATION_DIR, "news_features.parquet")
RAW_PARQUET = os.path.join(EXPLOITATION_DIR, "news_articles_raw.parquet")

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from trading_agent import config
from trading_agent.news_sentiment import (
    sentiment_backend_name,
    fetch_historical_news as fetch_raw_articles,
    build_asof_news_features as build_asof_features,
)

# Rolling windows are defined canonically in trading_agent.news_sentiment
# (_NEWS_LB_SHORT / _NEWS_LB_LONG); this script just orchestrates the fetch
# over the training span and writes the parquet.


def load_grid(db_path, limit_tickers=None, grid_start=None, grid_end=None):
    con = duckdb.connect(db_path, read_only=True)
    grid = con.execute(
        'SELECT DISTINCT Symbol AS ticker, "Date" FROM master_dataset '
        'WHERE target_30d_rank IS NOT NULL'
    ).df()
    con.close()
    d = pd.to_datetime(grid["Date"])
    if grid_start:
        grid = grid[d >= pd.Timestamp(grid_start)].copy()
    if grid_end:
        grid = grid[pd.to_datetime(grid["Date"]) <= pd.Timestamp(grid_end)].copy()
    if limit_tickers:
        keep = sorted(grid["ticker"].unique())[:limit_tickers]
        grid = grid[grid["ticker"].isin(keep)].copy()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="restrict to first N tickers (for a fast smoke test)")
    ap.add_argument("--grid-start", default=None,
                    help="only build features for master_dataset dates >= this (YYYY-MM-DD)")
    ap.add_argument("--grid-end", default=None,
                    help="only build features for master_dataset dates <= this (YYYY-MM-DD)")
    ap.add_argument("--append", action="store_true",
                    help="merge into the existing parquet instead of overwriting it; "
                         "rows for (ticker, Date) keys rebuilt in this run replace old ones")
    ap.add_argument("--raw-out", default=RAW_PARQUET,
                    help="where to write the raw scored (article,ticker) rows")
    args = ap.parse_args()

    print(f"[news-features] backend = {sentiment_backend_name()}")
    grid = load_grid(DB_PATH, args.limit_tickers, args.grid_start, args.grid_end)
    tickers = sorted(grid["ticker"].unique())
    dmin = pd.to_datetime(grid["Date"]).min()
    dmax = pd.to_datetime(grid["Date"]).max()
    start = (dmin - pd.Timedelta(days=32)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    end = (dmax + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    print(f"[news-features] {len(tickers)} tickers | grid {dmin.date()}..{dmax.date()} "
          f"| fetch {start.date()}..{end.date()}")

    raw = fetch_raw_articles(tickers, start, end)
    print(f"[news-features] raw (article,ticker) rows = {len(raw)} "
          f"covering {raw['ticker'].nunique() if not raw.empty else 0} tickers")
    if not raw.empty:
        raw.to_parquet(args.raw_out, index=False)

    feats = build_asof_features(raw, grid)
    if args.append and os.path.exists(OUT_PARQUET):
        old = pd.read_parquet(OUT_PARQUET)
        old["Date"] = pd.to_datetime(old["Date"]).dt.tz_localize(None)
        # Drop every old row whose (ticker, Date) lies in the rebuilt grid, so a
        # grid key that now has *no* news does not keep a stale old value.
        g = grid.assign(Date=pd.to_datetime(grid["Date"]).dt.tz_localize(None))
        key = pd.MultiIndex.from_frame(g[["ticker", "Date"]])
        old = old[~pd.MultiIndex.from_frame(old[["ticker", "Date"]]).isin(key)]
        print(f"[news-features] --append: keeping {len(old)} existing rows "
              f"({old['Date'].min()}..{old['Date'].max()})")
        feats = (pd.concat([old, feats], ignore_index=True)
                   .sort_values(["ticker", "Date"]).reset_index(drop=True))
    feats.to_parquet(OUT_PARQUET, index=False)
    cov = feats["ticker"].nunique() if not feats.empty else 0
    print(f"[news-features] wrote {len(feats)} (ticker,Date) feature rows "
          f"for {cov}/{len(tickers)} tickers -> {OUT_PARQUET}")
    if not feats.empty:
        print(feats[["news_sent_mean_7d", "news_count_7d", "news_sent_momentum",
                     "news_sent_std_7d"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
