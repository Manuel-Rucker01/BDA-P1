"""Daily SIP price store, 2016 -> last completed close (WS1).

Layout under ``<data>/prices/``::

    bars/year=YYYY/part.parquet   one file per year, sorted by (ticker, Date)
    meta.json                     last stored session, symbols requested, history
    _staging/<run>/               per-chunk checkpoints of a running download
    logs/                         download/update logs

Columns: ``ticker, Date ('YYYY-MM-DD', New York session date), Open, High, Low,
company_close, company_volume, vwap, trade_count`` -- split+dividend adjusted
(``adjustment=all``) -- plus ``close_raw`` (``adjustment=raw``): the price that
actually printed that day. ``close_raw`` is what PIT price filters ($3 floor)
and PIT market cap (close x SEC shares-as-filed, which are not split-adjusted)
must use; ``adj_factor = company_close / close_raw`` converts between the two.

``ticker`` in the store is the Alpaca symbol. ``load_prices(clean=True)``
applies the symbol-reuse segmentation from ``universe.detect_symbol_reuse``
(reused symbols become distinct ids such as ``APC~1``) and drops zero-volume
filler bars.

Adjustment drift: with ``adjustment=all`` every new split *or dividend*
rescales the whole earlier history. ``update`` re-fetches the last
``overlap_sessions`` sessions for every symbol, compares them with the store,
and re-downloads the full history of any ticker whose adjusted closes moved
(``|new/old - 1| > tol``), logging splits (>20% moves) separately.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ._alpaca import AlpacaClient, data_dir, last_completed_close_utc, to_ny_date

HISTORY_START = "2016-01-01"
STORE_COLUMNS = ["ticker", "Date", "Open", "High", "Low", "company_close", "company_volume",
                 "vwap", "trade_count", "close_raw"]
_FLOAT_COLS = STORE_COLUMNS[2:]


def prices_dir(*parts: str) -> str:
    return data_dir("prices", *parts)


def _year_path(year: int | str) -> str:
    return os.path.join(prices_dir("bars", f"year={year}"), "part.parquet")


def _meta_path() -> str:
    return os.path.join(prices_dir(), "meta.json")


def read_meta() -> dict:
    p = _meta_path()
    return json.load(open(p)) if os.path.exists(p) else {}


def _write_meta(meta: dict) -> None:
    tmp = _meta_path() + ".tmp"
    json.dump(meta, open(tmp, "w"), indent=1)
    os.replace(tmp, _meta_path())


def _log(msg: str, logfile: str | None = None) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def bars_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """API bars (UTC session-start stamps) -> store schema (NY session date)."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=STORE_COLUMNS[:-1])
    return pd.DataFrame({
        "ticker": raw["symbol"].astype(str).values,
        "Date": to_ny_date(raw["timestamp"]).values,
        "Open": raw["open"].astype(float).values,
        "High": raw["high"].astype(float).values,
        "Low": raw["low"].astype(float).values,
        "company_close": raw["close"].astype(float).values,
        "company_volume": raw["volume"].astype(float).values,
        "vwap": pd.to_numeric(raw["vwap"], errors="coerce").astype(float).values,
        "trade_count": pd.to_numeric(raw["trade_count"], errors="coerce").astype(float).values,
    })


def fetch_chunk(client: AlpacaClient, symbols: list[str], start, end) -> tuple[pd.DataFrame, list[str]]:
    """Adjusted bars + the unadjusted close for ``symbols`` (two requests sets)."""
    adj_raw, invalid = client.get_bars(symbols, start, end, "1Day", adjustment="all")
    ok = [s for s in symbols if s not in set(invalid)]
    adj = bars_to_frame(adj_raw)
    if not len(adj):
        return pd.DataFrame(columns=STORE_COLUMNS), invalid
    have = sorted(adj["ticker"].unique())
    raw_raw, _ = client.get_bars([s for s in ok if s in set(have)], start, end, "1Day", adjustment="raw")
    raw = bars_to_frame(raw_raw)[["ticker", "Date", "company_close"]].rename(
        columns={"company_close": "close_raw"})
    out = adj.merge(raw, on=["ticker", "Date"], how="left")
    out = out.drop_duplicates(["ticker", "Date"], keep="last")
    return out[STORE_COLUMNS], invalid


def _run_chunks(client: AlpacaClient, run: str, symbols: list[str], start, end,
                chunk_size: int = 150, workers: int = 3, logfile: str | None = None) -> list[str]:
    """Fetch ``symbols`` in chunks with per-chunk checkpoints under
    ``_staging/<run>/``; re-running the same ``run`` resumes where it stopped."""
    sdir = prices_dir("_staging", run)
    man_path = os.path.join(sdir, "manifest.json")
    fmt = lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(x, dt.datetime) else str(x)
    if os.path.exists(man_path):
        man = json.load(open(man_path))
    else:
        syms = sorted(dict.fromkeys(symbols))
        man = {"start": fmt(start), "end": fmt(end),
               "chunks": [syms[i:i + chunk_size] for i in range(0, len(syms), chunk_size)]}
        json.dump(man, open(man_path, "w"))
    path = lambda k: os.path.join(sdir, f"c{k:05d}.parquet")
    todo = [k for k in range(len(man["chunks"])) if not os.path.exists(path(k))]
    _log(f"[prices] run={run}: {len(man['chunks'])} chunks, {len(todo)} to fetch "
         f"({man['start']} -> {man['end']})", logfile)
    t0, done, invalid_all = time.time(), 0, []

    def work(k):
        df, invalid = fetch_chunk(client, man["chunks"][k], man["start"], man["end"])
        tmp = path(k) + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path(k))
        return k, len(df), df["ticker"].nunique() if len(df) else 0, invalid

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(work, k) for k in todo]
        for f in as_completed(futs):
            k, n, nt, invalid = f.result()
            done += 1
            invalid_all += invalid
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            _log(f"[prices] chunk {k}: {n:,} bars / {nt} tickers"
                 f"{' invalid=' + ','.join(invalid) if invalid else ''} "
                 f"[{done}/{len(todo)} {el:.0f}s eta {eta:.0f}s, {client.n_requests} req]", logfile)
    if invalid_all:
        with open(os.path.join(sdir, "invalid_symbols.txt"), "a") as f:
            f.write("\n".join(invalid_all) + "\n")
    return [path(k) for k in range(len(man["chunks"]))]


# --------------------------------------------------------------------------- #
# Store I/O
# --------------------------------------------------------------------------- #
def _write_year(year: int, df: pd.DataFrame) -> None:
    df = (df.drop_duplicates(["ticker", "Date"], keep="last")
            .sort_values(["ticker", "Date"]).reset_index(drop=True))
    for c in _FLOAT_COLS:
        df[c] = df[c].astype("float64")
    p = _year_path(year)
    tmp = p + ".tmp"
    df[STORE_COLUMNS].to_parquet(tmp, index=False)
    os.replace(tmp, p)


def stored_years() -> list[int]:
    return sorted(int(os.path.basename(os.path.dirname(p)).split("=")[1])
                  for p in glob.glob(os.path.join(prices_dir("bars"), "year=*", "part.parquet")))


def _read_year(year: int, tickers=None, start=None, end=None, columns=None) -> pd.DataFrame:
    p = _year_path(year)
    if not os.path.exists(p):
        return pd.DataFrame(columns=columns or STORE_COLUMNS)
    filters = []
    if tickers is not None:
        filters.append(("ticker", "in", list(tickers)))
    if start:
        filters.append(("Date", ">=", start))
    if end:
        filters.append(("Date", "<=", end))
    return pd.read_parquet(p, columns=columns, filters=filters or None)


def consolidate(chunk_files: list[str], logfile: str | None = None) -> dict:
    """Rewrite the whole store (one file per year) from staged chunk files."""
    import pyarrow.dataset as ds
    files = [f for f in chunk_files if os.path.exists(f)]
    dset = ds.dataset(files, format="parquet")
    years = set()
    for f in files:  # cheap: Date column only
        d = pd.read_parquet(f, columns=["Date"])["Date"]
        years |= set(d.str[:4].astype(int).unique()) if len(d) else set()
    stats = {}
    for y in sorted(years):
        t = dset.to_table(filter=(ds.field("Date") >= f"{y}-01-01") & (ds.field("Date") <= f"{y}-12-31"))
        df = t.to_pandas()
        _write_year(y, df)
        stats[y] = (len(df), df["ticker"].nunique())
        _log(f"[prices] wrote year {y}: {len(df):,} rows, {df['ticker'].nunique()} tickers", logfile)
    return stats


# --------------------------------------------------------------------------- #
# Public API: download / update / load
# --------------------------------------------------------------------------- #
def universe_symbols() -> list[str]:
    """Symbols to fetch: every candidate the universe classified as common stock."""
    from .universe import load_classification
    cls = load_classification()
    return sorted(cls.loc[cls["in_universe"], "fetch_symbol"].dropna().unique())


def download(symbols: list[str] | None = None, start: str = HISTORY_START, end: dt.datetime | None = None,
             chunk_size: int = 150, workers: int = 3, client: AlpacaClient | None = None,
             run: str | None = None) -> dict:
    """Bulk (re)build of the store: full history for ``symbols`` (default: the
    universe). Resumable -- re-run with the same ``run`` after an interruption
    (the default run id is derived from ``end``'s date)."""
    client = client or AlpacaClient()
    end = end or last_completed_close_utc()
    symbols = symbols if symbols is not None else universe_symbols()
    run = run or f"full_{end:%Y%m%d}"
    logfile = os.path.join(prices_dir("logs"), f"{run}.log")
    t0 = time.time()
    files = _run_chunks(client, run, symbols, start, end, chunk_size, workers, logfile)
    for y in stored_years():  # full rebuild: drop years that may not be rewritten
        os.remove(_year_path(y))
    stats = consolidate(files, logfile)
    got = set()
    for f in files:
        got |= set(pd.read_parquet(f, columns=["ticker"])["ticker"].unique())
    last = max(pd.read_parquet(_year_path(max(stats)), columns=["Date"])["Date"]) if stats else None
    meta = {"start": start, "last_date": last, "requested": sorted(set(symbols)),
            "with_data": sorted(got), "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "history": read_meta().get("history", []) + [
                {"op": "download", "run": run, "end": str(end), "seconds": round(time.time() - t0),
                 "requests": client.n_requests, "rows": int(sum(v[0] for v in stats.values()))}]}
    _write_meta(meta)
    shutil.rmtree(prices_dir("_staging", run), ignore_errors=True)
    _log(f"[prices] download done: {sum(v[0] for v in stats.values()):,} rows, {len(got)} tickers, "
         f"{time.time() - t0:.0f}s, {client.n_requests} requests", logfile)
    return meta


def _overlap_start(last_date: str, overlap_sessions: int) -> str:
    """Date of the ``overlap_sessions``-th most recent session in the store."""
    y = int(last_date[:4])
    dates = set()
    for yy in (y, y - 1):
        p = _year_path(yy)
        if os.path.exists(p):
            dates |= set(pd.read_parquet(p, columns=["Date"])["Date"].unique())
    dates = sorted(dates)
    return dates[-overlap_sessions] if len(dates) >= overlap_sessions else dates[0]


def adjustment_changes(old: pd.DataFrame, new: pd.DataFrame, tol: float = 1e-4) -> pd.DataFrame:
    """Tickers whose adjusted closes on overlapping sessions moved (a split or
    dividend re-scaled their history). Returns ticker, max_rel_change, kind."""
    m = old[["ticker", "Date", "company_close"]].merge(
        new[["ticker", "Date", "company_close"]], on=["ticker", "Date"], suffixes=("_old", "_new"))
    m = m[(m["company_close_old"] > 0) & (m["company_close_new"] > 0)]
    m["rel"] = (m["company_close_new"] / m["company_close_old"] - 1.0).abs()
    g = m.groupby("ticker")["rel"].max()
    g = g[g > tol]
    return pd.DataFrame({"ticker": g.index, "max_rel_change": g.values,
                         "kind": np.where(g.values > 0.2, "split", "dividend/other")})


def update(symbols: list[str] | None = None, overlap_sessions: int = 10, tol: float = 1e-4,
           end: dt.datetime | None = None, chunk_size: int = 150, workers: int = 3,
           client: AlpacaClient | None = None) -> dict:
    """Incremental refresh: fetch [overlap_start, last close] for every symbol,
    full history for new symbols and for tickers whose adjusted history moved."""
    client = client or AlpacaClient()
    meta = read_meta()
    if not meta.get("last_date"):
        raise RuntimeError("empty store: run download() first")
    end = end or last_completed_close_utc()
    symbols = sorted(set(symbols if symbols is not None else universe_symbols()))
    ov_start = _overlap_start(meta["last_date"], overlap_sessions)
    run = f"update_{end:%Y%m%d}"
    logfile = os.path.join(prices_dir("logs"), f"{run}.log")
    t0 = time.time()
    known = set(meta.get("requested", []))
    new_syms = [s for s in symbols if s not in known]
    inc_syms = [s for s in symbols if s in known]
    inc_files = _run_chunks(client, run + "_inc", inc_syms, ov_start, end, chunk_size * 2, workers, logfile)
    inc = pd.concat([pd.read_parquet(f) for f in inc_files], ignore_index=True) if inc_files else \
        pd.DataFrame(columns=STORE_COLUMNS)
    years_ov = range(int(ov_start[:4]), int(meta["last_date"][:4]) + 1)
    old = pd.concat([_read_year(y, start=ov_start) for y in years_ov], ignore_index=True)
    changes = adjustment_changes(old, inc, tol)
    refetch = sorted(set(changes["ticker"]) | set(new_syms))
    _log(f"[prices] update: overlap from {ov_start}; {len(inc):,} bars for {inc['ticker'].nunique()} "
         f"tickers; {int((changes['kind'] == 'split').sum())} splits, "
         f"{int((changes['kind'] != 'split').sum())} other adjustments, {len(new_syms)} new symbols "
         f"-> full re-fetch of {len(refetch)}", logfile)
    full = pd.DataFrame(columns=STORE_COLUMNS)
    if refetch:
        files = _run_chunks(client, run + "_full", refetch, HISTORY_START, end, chunk_size, workers, logfile)
        full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Merge: drop refetched tickers everywhere and the overlap window of the rest.
    inc = inc[~inc["ticker"].isin(refetch)]
    new_rows = pd.concat([inc, full], ignore_index=True)
    rf = set(refetch)
    years = set(stored_years()) | set(new_rows["Date"].str[:4].astype(int).unique()) if len(new_rows) \
        else set(stored_years())
    for y in sorted(years):
        touches_full = len(full) and (full["Date"].str[:4] == str(y)).any()
        if y < int(ov_start[:4]) and not touches_full and not rf:
            continue
        cur = _read_year(y) if os.path.exists(_year_path(y)) else pd.DataFrame(columns=STORE_COLUMNS)
        keep = ~cur["ticker"].isin(rf) & ~((cur["Date"] >= ov_start) & cur["ticker"].isin(set(inc["ticker"])))
        add = new_rows[new_rows["Date"].str[:4] == str(y)]
        _write_year(y, pd.concat([cur[keep], add], ignore_index=True))
    last = max(pd.read_parquet(_year_path(max(stored_years())), columns=["Date"])["Date"])
    meta.update({
        "last_date": last, "requested": sorted(known | set(symbols)),
        "with_data": sorted(set(meta.get("with_data", [])) | set(new_rows["ticker"])),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds")})
    meta["history"] = meta.get("history", []) + [{
        "op": "update", "run": run, "overlap_start": ov_start, "end": str(end),
        "splits": changes.loc[changes["kind"] == "split", "ticker"].tolist()[:200],
        "n_adjusted": int(len(changes)), "n_new": len(new_syms), "seconds": round(time.time() - t0),
        "requests": client.n_requests}]
    _write_meta(meta)
    for r in (run + "_inc", run + "_full"):
        shutil.rmtree(prices_dir("_staging", r), ignore_errors=True)
    _log(f"[prices] update done -> last_date {last} ({time.time() - t0:.0f}s)", logfile)
    return meta


def _base_symbol(t: str) -> str:
    return t.split("~")[0]


def load_prices(start: str | None = None, end: str | None = None, tickers=None,
                clean: bool = True, with_raw: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the store for [start, end] (inclusive 'YYYY-MM-DD').

    ``clean=True`` (default) applies the symbol-reuse segmentation
    (``<data>/universe/segments.parquet``): reused symbols are split into
    distinct ids (``SYM`` for the latest company, ``SYM~1``, ``SYM~2``... for
    earlier ones), excluded segments are dropped, and zero-volume filler bars
    (no trades) are removed. ``tickers`` may use either plain symbols or
    segment ids. ``with_raw`` keeps ``close_raw`` and adds
    ``adj_factor = company_close / close_raw``.
    """
    years = stored_years()
    if start:
        years = [y for y in years if y >= int(start[:4])]
    if end:
        years = [y for y in years if y <= int(end[:4])]
    base = sorted({_base_symbol(t) for t in tickers}) if tickers is not None else None
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(["ticker", "Date", "company_volume"] + list(columns)
                                  + (["company_close", "close_raw"] if with_raw else [])))
    frames = [_read_year(y, base, start, end, cols) for y in years]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STORE_COLUMNS)
    if clean and len(df):
        df = df[df["company_volume"] > 0]
        from .universe import apply_segments
        df = apply_segments(df)
        if tickers is not None:
            want = set(tickers)
            df = df[df["ticker"].isin(want) | (df["ticker"].map(_base_symbol).isin(want)
                                               & ~df["ticker"].str.contains("~", regex=False))]
    if with_raw and "close_raw" in df.columns:
        df["adj_factor"] = df["company_close"] / df["close_raw"]
    elif not with_raw:
        df = df.drop(columns=[c for c in ("close_raw",) if c in df.columns])
    if columns is not None:
        keep = ["ticker", "Date"] + [c for c in columns if c in df.columns and c not in ("ticker", "Date")]
        if with_raw:
            keep += [c for c in ("close_raw", "adj_factor") if c in df.columns and c not in keep]
        df = df[keep]
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="v3 price store")
    ap.add_argument("cmd", choices=["download", "update"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=150)
    a = ap.parse_args()
    if a.cmd == "download":
        download(workers=a.workers, chunk_size=a.chunk_size)
    else:
        update(workers=a.workers, chunk_size=a.chunk_size)
