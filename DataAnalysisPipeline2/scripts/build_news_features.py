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
from trading_agent.news_sentiment import score_text, sentiment_backend_name

LOOKBACK_SHORT = 7    # days
LOOKBACK_LONG = 30    # days


# ── Landing: paginated historical fetch + frozen-lexicon scoring ─────────────
def fetch_raw_articles(tickers, start, end, chunk_size=40, sleep_s=0.0):
    """Fetch all (article,ticker) records over [start, end].

    alpaca-py's ``get_news`` auto-paginates internally up to ``limit``; passing
    ``limit=None`` therefore returns the FULL date range (no truncation). The
    response payload is ``{'news': [articles]}`` and each article lists every
    symbol it mentions, so we keep only intersections with the requested chunk.

    Returns a DataFrame: ticker, created_at (UTC tz-aware), sentiment.
    Scoring uses the FROZEN lexicon, so the scorer never sees test-period data.
    """
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    requested = set(tickers)
    rows = []
    tickers = list(tickers)
    n_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    for ci in range(0, len(tickers), chunk_size):
        chunk = tickers[ci:ci + chunk_size]
        try:
            req = NewsRequest(symbols=",".join(chunk), start=start, end=end,
                              limit=None, include_content=False)
            resp = client.get_news(req)
        except Exception as e:
            print(f"  [chunk {ci//chunk_size+1}/{n_chunks}] fetch error: {e}")
            continue
        articles = getattr(resp, "data", None)
        if isinstance(articles, dict):
            articles = articles.get("news", [])
        elif articles is None:
            articles = getattr(resp, "news", []) or []
        for a in articles:
            headline = getattr(a, "headline", "") or ""
            summary = getattr(a, "summary", "") or ""
            created = getattr(a, "created_at", None)
            syms = getattr(a, "symbols", []) or []
            hit = [s for s in syms if s in requested]
            if not hit or created is None:
                continue
            s = score_text(f"{headline}. {summary}".strip())
            if s is None:
                continue
            for sym in hit:
                rows.append((sym, pd.Timestamp(created), float(s)))
        print(f"  [chunk {ci//chunk_size+1}/{n_chunks}] cumulative rows={len(rows)}")
        if sleep_s:
            time.sleep(sleep_s)
    df = pd.DataFrame(rows, columns=["ticker", "created_at", "sentiment"])
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


# ── Exploitation: strict AS-OF (ticker, Date) feature construction ───────────
def build_asof_features(raw_df, grid_df):
    """For each (ticker, Date) in grid_df, aggregate news published STRICTLY
    BEFORE Date over rolling windows. No same-day or future articles leak in.

    Features: news_sent_mean_7d, news_count_7d, news_sent_mean_30d,
              news_sent_momentum (7d-30d), news_sent_std_7d.
    """
    feats = []
    if raw_df.empty:
        cols = ["ticker", "Date", "news_sent_mean_7d", "news_count_7d",
                "news_sent_mean_30d", "news_sent_momentum", "news_sent_std_7d"]
        return pd.DataFrame(columns=cols)

    # Per-ticker, per-calendar-day aggregate first (compresses the rolling work).
    raw_df = raw_df.copy()
    raw_df["news_date"] = raw_df["created_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    daily = (raw_df.groupby(["ticker", "news_date"])
             .agg(day_sent_sum=("sentiment", "sum"),
                  day_sent_sq=("sentiment", lambda x: float(np.sum(np.square(x)))),
                  day_n=("sentiment", "size"))
             .reset_index())
    by_ticker = {t: g.sort_values("news_date") for t, g in daily.groupby("ticker")}

    grid_df = grid_df.copy()
    grid_df["Date"] = pd.to_datetime(grid_df["Date"]).dt.tz_localize(None)
    for t, g in grid_df.groupby("ticker"):
        d = by_ticker.get(t)
        if d is None:
            continue
        nd = d["news_date"].values.astype("datetime64[ns]")
        s_sum = d["day_sent_sum"].values
        s_sq = d["day_sent_sq"].values
        n = d["day_n"].values
        for date in g["Date"].unique():
            date = pd.Timestamp(date)
            lo7 = (date - pd.Timedelta(days=LOOKBACK_SHORT)).to_datetime64()
            lo30 = (date - pd.Timedelta(days=LOOKBACK_LONG)).to_datetime64()
            hi = date.to_datetime64()  # strict: news_date < Date
            m7 = (nd >= lo7) & (nd < hi)
            m30 = (nd >= lo30) & (nd < hi)
            n7 = n[m7].sum()
            n30 = n[m30].sum()
            if n7 == 0 and n30 == 0:
                continue
            mean7 = s_sum[m7].sum() / n7 if n7 > 0 else 0.0
            mean30 = s_sum[m30].sum() / n30 if n30 > 0 else 0.0
            # std over the 7d window from sum and sum-of-squares
            if n7 > 1:
                var7 = max(s_sq[m7].sum() / n7 - mean7 ** 2, 0.0)
                std7 = float(np.sqrt(var7))
            else:
                std7 = 0.0
            feats.append((t, date, mean7, int(n7), mean30, mean7 - mean30, std7))
    cols = ["ticker", "Date", "news_sent_mean_7d", "news_count_7d",
            "news_sent_mean_30d", "news_sent_momentum", "news_sent_std_7d"]
    return pd.DataFrame(feats, columns=cols)


def load_grid(db_path, limit_tickers=None):
    con = duckdb.connect(db_path, read_only=True)
    grid = con.execute(
        'SELECT DISTINCT Symbol AS ticker, "Date" FROM master_dataset '
        'WHERE target_30d_rank IS NOT NULL'
    ).df()
    con.close()
    if limit_tickers:
        keep = sorted(grid["ticker"].unique())[:limit_tickers]
        grid = grid[grid["ticker"].isin(keep)].copy()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="restrict to first N tickers (for a fast smoke test)")
    args = ap.parse_args()

    print(f"[news-features] backend = {sentiment_backend_name()}")
    grid = load_grid(DB_PATH, args.limit_tickers)
    tickers = sorted(grid["ticker"].unique())
    dmin = pd.to_datetime(grid["Date"]).min()
    dmax = pd.to_datetime(grid["Date"]).max()
    start = (dmin - pd.Timedelta(days=LOOKBACK_LONG + 2)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    end = (dmax + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    print(f"[news-features] {len(tickers)} tickers | grid {dmin.date()}..{dmax.date()} "
          f"| fetch {start.date()}..{end.date()}")

    raw = fetch_raw_articles(tickers, start, end)
    print(f"[news-features] raw (article,ticker) rows = {len(raw)} "
          f"covering {raw['ticker'].nunique() if not raw.empty else 0} tickers")
    if not raw.empty:
        raw.to_parquet(RAW_PARQUET, index=False)

    feats = build_asof_features(raw, grid)
    feats.to_parquet(OUT_PARQUET, index=False)
    cov = feats["ticker"].nunique() if not feats.empty else 0
    print(f"[news-features] wrote {len(feats)} (ticker,Date) feature rows "
          f"for {cov}/{len(tickers)} tickers -> {OUT_PARQUET}")
    if not feats.empty:
        print(feats[["news_sent_mean_7d", "news_count_7d", "news_sent_momentum",
                     "news_sent_std_7d"]].describe().round(4).to_string())


if __name__ == "__main__":
    main()
