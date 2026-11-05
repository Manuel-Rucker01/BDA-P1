"""Market data provider for the live trading agent.

Why this exists
---------------
The live agent used to pull ~1,900 tickers through a single threaded yfinance
burst. That (a) got rate-limited by Yahoo and exhausted OS threads, silently
dropping most of the universe, and (b) fetched only 60 *calendar* days
(~41 sessions), so 50-day features such as ``return_50d`` were always
NaN -> 0 in production while being real in training (train/serve skew), and
26-span EMAs (MACD) were under-warmed.

Design
------
* Primary: Alpaca Market Data API, SIP feed, split+dividend-adjusted daily
  bars. Adjusted prices match the auto-adjusted yfinance history the model was
  trained on, and SIP volume is consolidated, so volume features keep parity.
  Symbols are requested in chunks with retry/backoff; unknown or delisted
  symbols are silently dropped by the API. The free data plan forbids "recent"
  SIP data, so bars are requested through the previous UTC midnight, i.e. up to
  the last completed session close -- exactly what a daily decision should use.
* Fallback: yfinance in small chunks with limited threads and retries.

Output schema matches what ``compute_live_features`` expects::

    ticker, Date ('YYYY-MM-DD'), Open, High, Low, company_close, company_volume
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from . import config

OUTPUT_COLUMNS = ["ticker", "Date", "Open", "High", "Low", "company_close", "company_volume"]


def _last_completed_close_utc() -> dt.datetime:
    """Previous UTC midnight: bars up to here are 'completed sessions' and are
    allowed on Alpaca's free SIP plan."""
    return dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------------------------------------------------------- #
# Alpaca (primary)
# --------------------------------------------------------------------------- #
def _alpaca_daily_bars(tickers, start, end, chunk_size=200, retries=3, verbose=True):
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        raise RuntimeError("Alpaca credentials not configured")
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    frames = []
    chunks = list(_chunks(list(tickers), chunk_size))
    for i, chunk in enumerate(chunks, 1):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                    start=start, end=end, feed="sip", adjustment=Adjustment.ALL,
                )
                df = client.get_stock_bars(req).df
                if df is not None and len(df):
                    frames.append(df.reset_index())
                last_err = None
                break
            except Exception as e:  # rate limit / transient network
                last_err = e
                time.sleep(2 ** attempt)
        if last_err is not None and verbose:
            print(f"[market_data] Alpaca chunk {i}/{len(chunks)} failed after {retries} tries: {last_err}")
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "ticker": raw["symbol"].astype(str),
        # Alpaca daily bars are stamped at session-start in UTC (04:00/05:00Z);
        # convert to New York time before taking the calendar date.
        "Date": pd.to_datetime(raw["timestamp"], utc=True)
                  .dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"),
        "Open": raw["open"].astype(float),
        "High": raw["high"].astype(float),
        "Low": raw["low"].astype(float),
        "company_close": raw["close"].astype(float),
        "company_volume": raw["volume"].astype(float),
    })
    return out


# --------------------------------------------------------------------------- #
# yfinance (fallback)
# --------------------------------------------------------------------------- #
def _yf_daily_bars(tickers, start, end, chunk_size=100, retries=3, verbose=True):
    import yfinance as yf

    frames = []
    for chunk in _chunks(list(tickers), chunk_size):
        batch = None
        for attempt in range(1, retries + 1):
            try:
                batch = yf.download(
                    tickers=" ".join(chunk), start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"), progress=False, group_by="ticker",
                    auto_adjust=True, threads=4,
                )
                break
            except Exception as e:
                if verbose:
                    print(f"[market_data] yfinance chunk retry {attempt}: {e}")
                time.sleep(2 ** attempt)
        if batch is None or batch.empty:
            continue
        for t in chunk:
            try:
                if isinstance(batch.columns, pd.MultiIndex):
                    if t not in batch.columns.get_level_values(0):
                        continue
                    sub = batch[t]
                else:
                    sub = batch
                sub = sub.dropna(how="all").reset_index()
                if sub.empty:
                    continue
                frames.append(pd.DataFrame({
                    "ticker": t,
                    "Date": pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d"),
                    "Open": sub["Open"], "High": sub["High"], "Low": sub["Low"],
                    "company_close": sub["Close"], "company_volume": sub["Volume"],
                }))
            except Exception:
                continue
        time.sleep(1)  # be polite between chunks
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fetch_daily_bars(tickers, lookback_days=250, provider="auto", verbose=True):
    """Daily OHLCV for ``tickers`` over the last ``lookback_days`` calendar days,
    through the last completed session close.

    Returns ``(df, failures)`` where ``failures`` lists tickers with no data.
    ``provider``: "auto" (Alpaca, then yfinance for whatever is missing),
    "alpaca", or "yfinance".
    """
    tickers = list(dict.fromkeys(t for t in tickers if t))
    end = _last_completed_close_utc()
    start = end - dt.timedelta(days=int(lookback_days))
    t0 = time.time()

    df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if provider in ("auto", "alpaca"):
        try:
            df = _alpaca_daily_bars(tickers, start, end, verbose=verbose)
            if verbose:
                print(f"[market_data] Alpaca SIP: {df['ticker'].nunique()}/{len(tickers)} tickers, "
                      f"{len(df):,} bars in {time.time() - t0:.1f}s")
        except Exception as e:
            if verbose:
                print(f"[market_data] Alpaca unavailable ({e}); falling back to yfinance.")

    have = set(df["ticker"])
    missing = [t for t in tickers if t not in have]
    if missing and provider in ("auto", "yfinance"):
        # Only worth a yfinance pass if Alpaca failed wholesale; tickers Alpaca
        # silently dropped are almost always delisted and Yahoo won't have them.
        if provider == "yfinance" or len(missing) > 0.5 * len(tickers):
            yf_df = _yf_daily_bars(missing, start, end, verbose=verbose)
            if len(yf_df):
                df = pd.concat([df, yf_df], ignore_index=True)
                if verbose:
                    print(f"[market_data] yfinance filled {yf_df['ticker'].nunique()} more tickers.")

    df = (df.drop_duplicates(["ticker", "Date"], keep="first")
            .sort_values(["ticker", "Date"]).reset_index(drop=True))
    have = set(df["ticker"])
    failures = [t for t in tickers if t not in have]
    return df[OUTPUT_COLUMNS], failures


def fetch_benchmark_closes(symbol=None, lookback_days=380, verbose=True):
    """Daily closes for the regime/beta benchmark as a DataFrame[Date, Close].

    Tries the true index (e.g. ^GSPC) via yfinance -- one request, rarely
    throttled -- and falls back to SPY from Alpaca, whose returns track the
    S&P 500 almost exactly.
    """
    symbol = symbol or getattr(config, "SP500_INDEX", "^GSPC")
    try:
        import yfinance as yf
        d = yf.download(symbol, period=f"{int(lookback_days)}d", progress=False, auto_adjust=True)
        if d is not None and not d.empty:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] for c in d.columns]
            d = d.reset_index()[["Date", "Close"]]
            d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
            return d.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        if verbose:
            print(f"[market_data] {symbol} via yfinance failed ({e}); using SPY from Alpaca.")
    bars, _ = fetch_daily_bars(["SPY"], lookback_days=int(lookback_days * 1.5),
                               provider="alpaca", verbose=False)
    if bars.empty:
        raise RuntimeError("No benchmark data from yfinance or Alpaca")
    return (bars.rename(columns={"company_close": "Close"})[["Date", "Close"]]
                .sort_values("Date").reset_index(drop=True))
