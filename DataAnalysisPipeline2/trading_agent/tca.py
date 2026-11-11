"""Transaction-cost analysis (TCA) of the paper account's real fills.

Read-only. For every filled order on the account (Alpaca ``/v2/orders``) this
computes the implementation shortfall (IS, bps, positive = cost)

    IS = side x (fill price / reference - 1),   side = +1 buy, -1 sell

against several references:

* ``is_decision``: the decision-date close -- the price the bot sized the
  order with (``bot.execute_alpaca_rebalance._price``: last ``company_close``
  of the bars the model scored, i.e. the decision-date close; the decision date
  is the date of the matching ``ops/predictions/<account>/<date>.csv.gz``).
  Includes the overnight + intraday market move until the fill, so it is mostly
  noise for one rebalance; ``is_decision_mktadj`` subtracts SPY's move over the
  same interval (beta 1);
* ``is_arrival``: the NBBO mid just before submission (SIP quotes) -- the
  realised half-spread paid by a market order;
* ``is_open`` / ``is_vwap`` / ``is_close``: the fill day's open, VWAP and close
  (SIP daily bars, unadjusted). Pending (NaN) until the session's bar is final.

Each fill is compared with what ``v3/costs.py`` (the backtest cost model)
charges for that name and that trade size at the actual account size:
half-spread (0.5 x 63-day mean Corwin-Schultz estimate, proxy if < 20
estimates) + ``IMPACT_K x sigma60 x sqrt(notional / ADV20)``, with the same
inputs the live optimizer uses (``optimized.build_inputs``: split-adjusted
High/Low/Close, ADV = 20-session mean of close x volume, sigma = 60-session std
of daily returns), all as of the decision date. ``impact_100m_bps`` is the
impact the same trade would carry scaled to the backtest's $100M book.

CAVEAT: Alpaca PAPER fills are simulated: a market order fills at the
prevailing quote with no market impact and no queueing, so the realised numbers
here are a LOWER BOUND on live costs (they measure the quoted spread at the
moment of the order, not impact or adverse selection).

    python -m DataAnalysisPipeline2.trading_agent.tca [--account monthly] [--since 2026-10-01]

writes ``ops/tca.csv`` (one row per fill) and ``ops/tca.md`` (aggregates by
liquidity bucket). Credentials come from ``config.resolve_account`` (env /
trading_agent/.env) and are never printed. The output is deterministic for a
given set of fills and bars (no generation timestamp), so re-running it on a
no-op day leaves the files unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ORDER_PREFIX = "bda-"            # bot.BOT_ORDER_PREFIX (not imported: bot.py pulls in the model stack)
DEFAULT_SINCE = "2026-10-01"
HISTORY_DAYS = 150               # calendar days of daily bars before the decision date (>= 63 + 60 sessions)
BUCKETS = [(0.0, 10e6, "<$10M"), (10e6, 50e6, "$10-50M"), (50e6, 250e6, "$50-250M"),
           (250e6, np.inf, ">$250M")]
BACKTEST_AUM = 100e6
QUOTE_WINDOWS_S = (5, 60, 900)   # look-back windows tried for the arrival quote
CSV_COLUMNS = [
    "trade_date", "decision_date", "symbol", "side", "qty", "fill_price", "notional", "order_type",
    "bot_tagged", "submitted_at", "filled_at", "fill_latency_s", "equity", "ref_decision_close",
    "arrival_bid", "arrival_ask", "arrival_mid", "arrival_spread_bps", "day_open", "day_vwap", "day_close",
    "spy_decision_close", "spy_arrival_mid", "is_decision_bps", "is_decision_mktadj_bps", "is_arrival_bps",
    "is_open_bps", "is_vwap_bps", "is_close_bps", "adv_usd", "participation", "sigma_daily", "hs_source",
    "hs_model_bps", "impact_model_bps", "model_bps", "impact_100m_bps", "liquidity_bucket"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def side_sign(side) -> int:
    return 1 if str(side).lower().startswith("buy") else -1


def shortfall_bps(side, fill, ref) -> float:
    """side x (fill / ref - 1) in bps; NaN when either price is missing."""
    try:
        fill, ref = float(fill), float(ref)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(fill) and np.isfinite(ref) and ref > 0):
        return float("nan")
    return side_sign(side) * (fill / ref - 1.0) * 1e4


def liquidity_bucket(adv) -> str:
    try:
        a = float(adv)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(a):
        return "unknown"
    for lo, hi, name in BUCKETS:
        if lo <= a < hi:
            return name
    return "unknown"


def decision_date_for(trade_date: str, decision_dates) -> str | None:
    """Latest recorded decision date strictly before ``trade_date`` (the bot
    trades the session after the decision date's close)."""
    ds = sorted(str(d) for d in decision_dates if str(d) < str(trade_date))
    return ds[-1] if ds else None


def prediction_dates(pred_dir) -> list[str]:
    p = Path(pred_dir)
    if not p.exists():
        return []
    return sorted(f.name[:10] for f in p.glob("????-??-??.csv.gz"))


def cost_inputs(bars: pd.DataFrame, decision_date: str) -> pd.DataFrame:
    """Per-ticker costs.py inputs as of ``decision_date`` from daily bars
    (ticker, Date, High, Low, company_close, company_volume; split-adjusted;
    optional close_raw = printed close for the price/ADV proxy), exactly as
    ``optimized.build_inputs`` / ``round5.prepare`` build them.

    Returns index ticker, columns hs, cs_ok, sigma, adv, price."""
    from DataAnalysisPipeline2.v3 import costs as cs
    b = bars.copy()
    b["Date"] = pd.to_datetime(b["Date"]).dt.strftime("%Y-%m-%d")
    b = b[b["Date"] <= str(decision_date)].drop_duplicates(["ticker", "Date"], keep="last")
    if b.empty:
        return pd.DataFrame(columns=["hs", "cs_ok", "sigma", "adv", "price"])
    wide = lambda c: b.pivot(index="Date", columns="ticker", values=c).sort_index().astype(float)
    H, L, C, V = wide("High"), wide("Low"), wide("company_close"), wide("company_volume")
    P = wide("close_raw") if "close_raw" in b else C
    adv = (C * V).rolling(20, min_periods=20).mean()
    est = cs.corwin_schultz(H, L, C)
    hs = cs.half_spread(est, P, adv)
    n_cs = est.notna().rolling(cs.CS_WINDOW, min_periods=1).sum()
    r = C.pct_change(fill_method=None).clip(-0.5, 1.0)
    vol60 = r.rolling(60, min_periods=40).std()
    last = lambda f: f.iloc[-1] if len(f) else pd.Series(dtype=float)
    out = pd.DataFrame({"hs": last(hs), "cs_ok": last(n_cs) >= cs.CS_MIN_OBS, "sigma": last(vol60),
                        "adv": last(adv), "price": last(P)})
    out.index.name = "ticker"
    return out


def modelled_cost_bps(hs, sigma, adv, notional, k=None) -> tuple[float, float]:
    """(half-spread bps, impact bps) of ``costs.py`` for one trade of
    ``notional`` dollars: hs and k sigma sqrt(notional / ADV)."""
    from DataAnalysisPipeline2.v3 import costs as cs
    k = cs.IMPACT_K if k is None else k
    hs = float(hs) if hs is not None and np.isfinite(hs) else cs.DEFAULT_HS
    if sigma is None or adv is None or not (np.isfinite(sigma) and np.isfinite(adv) and adv > 0):
        return hs * 1e4, float("nan")
    return hs * 1e4, float(k * sigma * np.sqrt(abs(float(notional)) / float(adv))) * 1e4


def mid_from_quotes(q: pd.DataFrame | None, t) -> tuple[float, float, float]:
    """(bid, ask, mid) of the last valid quote at or before ``t``; NaNs if none.
    ``q``: timestamp, bid_price, ask_price."""
    nan = (float("nan"),) * 3
    if q is None or not len(q):
        return nan
    t = pd.Timestamp(t)
    ts = pd.to_datetime(q["timestamp"], utc=True)
    ok = (ts <= t) & (q["bid_price"] > 0) & (q["ask_price"] >= q["bid_price"])
    if not ok.any():
        return nan
    last = q[ok].iloc[np.argmax(ts[ok].to_numpy())]
    bid, ask = float(last["bid_price"]), float(last["ask_price"])
    return bid, ask, (bid + ask) / 2.0


# --------------------------------------------------------------------------- #
# Analysis (data injected; no I/O)
# --------------------------------------------------------------------------- #
def analyse(fills: pd.DataFrame, *, bars_adj: pd.DataFrame, bars_raw: pd.DataFrame, decision_dates,
            quote_fn=None, equity_by_date: dict | None = None, spy_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-fill TCA rows (``CSV_COLUMNS``).

    ``fills``: symbol, side, filled_qty, filled_avg_price, submitted_at, filled_at, type, client_order_id.
    ``bars_adj``: split-adjusted daily bars (ticker, Date, High, Low, company_close, company_volume);
    ``bars_raw``: unadjusted daily bars (ticker, Date, Open, company_close, VWAP);
    ``quote_fn(symbol, t)`` -> (bid, ask, mid) just before ``t`` (or None to skip);
    ``spy_raw``: unadjusted SPY daily bars (Date, company_close)."""
    if fills is None or not len(fills):
        return pd.DataFrame(columns=CSV_COLUMNS)
    raw = bars_raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
    raw = raw.set_index(["ticker", "Date"])
    spy = None
    if spy_raw is not None and len(spy_raw):
        spy = spy_raw.assign(Date=pd.to_datetime(spy_raw["Date"]).dt.strftime("%Y-%m-%d")) \
            .set_index("Date")["company_close"]
    f = fills.copy()
    f["submitted_at"] = pd.to_datetime(f["submitted_at"], utc=True)
    f["filled_at"] = pd.to_datetime(f["filled_at"], utc=True)
    f["trade_date"] = f["filled_at"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    f["decision_date"] = [decision_date_for(d, decision_dates) for d in f["trade_date"]]
    ci_cache: dict = {}
    rows = []
    for r in f.sort_values(["filled_at", "symbol"]).itertuples(index=False):
        dd = r.decision_date
        if dd is None:          # no recorded decision: fall back to the previous bar before the trade day
            prior = raw.index.get_level_values("Date")
            prior = sorted(set(d for d in prior if d < r.trade_date))
            dd = prior[-1] if prior else None
        if dd not in ci_cache:
            ci_cache[dd] = cost_inputs(bars_adj, dd) if dd else pd.DataFrame()
        ci = ci_cache[dd]
        sym, side = str(r.symbol), str(r.side)
        qty, px = float(r.filled_qty), float(r.filled_avg_price)
        notional = qty * px
        get = lambda d, c: float(raw.loc[(sym, d), c]) if (sym, d) in raw.index and c in raw else float("nan")
        ref = get(dd, "company_close") if dd else float("nan")
        d_open, d_vwap, d_close = (get(r.trade_date, c) for c in ("Open", "VWAP", "company_close"))
        bid, ask, mid = quote_fn(sym, r.submitted_at) if quote_fn else (float("nan"),) * 3
        s_bid, s_ask, s_mid = quote_fn("SPY", r.submitted_at) if quote_fn else (float("nan"),) * 3
        spy_dc = float(spy.get(dd, np.nan)) if (spy is not None and dd) else float("nan")
        is_dec = shortfall_bps(side, px, ref)
        spy_move = (s_mid / spy_dc - 1.0) * 1e4 if (np.isfinite(s_mid) and np.isfinite(spy_dc) and spy_dc > 0) \
            else float("nan")
        c = ci.loc[sym] if sym in ci.index else None
        hs, sig, adv = (float(c["hs"]), float(c["sigma"]), float(c["adv"])) if c is not None \
            else (float("nan"),) * 3
        hs_bps, imp_bps = modelled_cost_bps(hs, sig, adv, notional)
        eq = (equity_by_date or {}).get(dd) or (equity_by_date or {}).get("latest")
        scale = BACKTEST_AUM / float(eq) if eq else float("nan")
        _, imp100 = modelled_cost_bps(hs, sig, adv, notional * scale) if np.isfinite(scale) else (0, float("nan"))
        rows.append({
            "trade_date": r.trade_date, "decision_date": dd, "symbol": sym, "side": side.lower(), "qty": qty,
            "fill_price": px, "notional": notional, "order_type": getattr(r, "type", None),
            "bot_tagged": str(getattr(r, "client_order_id", "") or "").startswith(ORDER_PREFIX),
            "submitted_at": r.submitted_at.isoformat(), "filled_at": r.filled_at.isoformat(),
            "fill_latency_s": (r.filled_at - r.submitted_at).total_seconds(), "equity": eq,
            "ref_decision_close": ref, "arrival_bid": bid, "arrival_ask": ask, "arrival_mid": mid,
            "arrival_spread_bps": (ask - bid) / mid * 1e4 if np.isfinite(mid) and mid > 0 else float("nan"),
            "day_open": d_open, "day_vwap": d_vwap, "day_close": d_close,
            "spy_decision_close": spy_dc, "spy_arrival_mid": s_mid,
            "is_decision_bps": is_dec,
            "is_decision_mktadj_bps": is_dec - side_sign(side) * spy_move if np.isfinite(spy_move) else float("nan"),
            "is_arrival_bps": shortfall_bps(side, px, mid), "is_open_bps": shortfall_bps(side, px, d_open),
            "is_vwap_bps": shortfall_bps(side, px, d_vwap), "is_close_bps": shortfall_bps(side, px, d_close),
            "adv_usd": adv, "participation": notional / adv if np.isfinite(adv) and adv > 0 else float("nan"),
            "sigma_daily": sig, "hs_source": ("cs" if bool(c["cs_ok"]) else "proxy") if c is not None else "default",
            "hs_model_bps": hs_bps, "impact_model_bps": imp_bps,
            "model_bps": hs_bps + (imp_bps if np.isfinite(imp_bps) else 0.0), "impact_100m_bps": imp100,
            "liquidity_bucket": liquidity_bucket(adv)})
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


AGG_METRICS = ["model_bps", "hs_model_bps", "impact_model_bps", "impact_100m_bps", "quoted_half_spread_bps",
               "is_arrival_bps", "is_vwap_bps", "is_close_bps", "is_open_bps", "is_decision_bps",
               "is_decision_mktadj_bps"]


def _wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else float("nan")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Notional-weighted mean bps by liquidity bucket (+ 'all'), with n, notional."""
    if df is None or not len(df):
        return pd.DataFrame(columns=["bucket", "n", "notional"] + AGG_METRICS)
    d = df.copy()
    d["quoted_half_spread_bps"] = d["arrival_spread_bps"] / 2.0
    order = [b[2] for b in BUCKETS] + ["unknown"]
    out = []
    for name, g in list(d.groupby("liquidity_bucket")) + [("all", d)]:
        row = {"bucket": name, "n": int(len(g)), "notional": float(g["notional"].sum())}
        for m in AGG_METRICS:
            row[m] = _wmean(g[m].astype(float), g["notional"].astype(float))
        out.append(row)
    res = pd.DataFrame(out)
    res["_o"] = res["bucket"].map(lambda b: order.index(b) if b in order else len(order))
    return res.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _f(x, nd=1):
    return "–" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"


def _usd(x) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}k"


def render_markdown(df: pd.DataFrame, agg: pd.DataFrame, account: str, diagnosis_link: str | None = None) -> str:
    L = ["# Transaction-cost analysis (paper fills)", "",
         f"Account `{account}`. Generated by `python -m DataAnalysisPipeline2.trading_agent.tca` "
         "(also refreshed by `.github/workflows/rebalance.yml` after a rebalance). Per-fill rows: "
         "[`ops/tca.csv`](tca.csv). Method: module docstring of `trading_agent/tca.py`.", "",
         "> **Caveat — paper fills are optimistic.** Alpaca's paper engine fills a market order at the "
         "prevailing quote, with no market impact, no queueing and no adverse selection. The realised "
         "columns below therefore measure (at best) the quoted half-spread at the moment of the order; "
         "they are a *lower bound* on what the same orders would cost live. The modelled column is what "
         "the backtest (`v3/costs.py`) charges for the same name and trade size at the actual account size.", ""]
    if df is None or not len(df):
        return "\n".join(L + ["No filled orders yet.", ""])
    days = sorted(df["trade_date"].unique())
    L += [f"Fills: {len(df)} on {', '.join(days)}; notional ${df['notional'].sum():,.0f} "
          f"({(df['side'] == 'buy').sum()} buys, {(df['side'] != 'buy').sum()} sells). "
          f"Decision date(s): {', '.join(sorted(set(map(str, df['decision_date'].dropna()))))}.", ""]
    pend = df["day_close"].isna().sum()
    if pend:
        L += [f"*{pend} fill(s) have no final daily bar yet (same-day VWAP/close pending; "
              "the next run fills them in).*", ""]
    L += ["## Modelled vs realised (bps of traded notional, notional-weighted; + = cost)", "",
          "| liquidity (ADV20) | n | notional | **modelled** (costs.py) | of which half-spread | of which impact "
          "| impact if $100M book | quoted half-spread | **realised vs arrival mid** | vs day VWAP | vs day close "
          "| vs open | vs decision close | vs decision close, SPY-adj. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples(index=False):
        b = f"**{r.bucket}**" if r.bucket == "all" else r.bucket
        L.append(f"| {b} | {r.n} | ${r.notional:,.0f} | **{_f(r.model_bps)}** | {_f(r.hs_model_bps)} | "
                 f"{_f(r.impact_model_bps, 2)} | {_f(r.impact_100m_bps)} | {_f(r.quoted_half_spread_bps)} | "
                 f"**{_f(r.is_arrival_bps)}** | {_f(r.is_vwap_bps)} | {_f(r.is_close_bps)} | {_f(r.is_open_bps)} | "
                 f"{_f(r.is_decision_bps)} | {_f(r.is_decision_mktadj_bps)} |")
    a = agg[agg["bucket"] == "all"].iloc[0]
    L += ["", "How to read it:",
          "* *vs arrival mid* is the cost the paper engine actually charged (fill vs NBBO mid just before "
          "submission); the like-for-like comparison with the model's half-spread.",
          "* *vs day VWAP / close / open* and *vs decision close* also contain the price move between the "
          "reference and the fill (timing noise, either sign); with one rebalance they say little about cost.",
          f"* Modelled/realised(arrival) ratio: **{_f(a.model_bps / a.is_arrival_bps if a.is_arrival_bps else np.nan, 1)}x**"
          f" (modelled {_f(a.model_bps)} bps vs realised {_f(a.is_arrival_bps)} bps; quoted half-spread "
          f"{_f(a.quoted_half_spread_bps)} bps).", ""]
    L += ["## Per fill", "",
          "| date | symbol | side | notional | ADV20 | particip. | hs src | modelled | of which impact "
          "| quoted ½-spr | vs arrival | vs VWAP | vs decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.sort_values(["trade_date", "adv_usd"]).itertuples(index=False):
        L.append(f"| {r.trade_date} | {r.symbol} | {r.side} | ${r.notional:,.0f} | "
                 f"{_usd(r.adv_usd)} | {_f(r.participation * 100, 3)}% | {r.hs_source} | {_f(r.model_bps)} | "
                 f"{_f(r.impact_model_bps, 2)} | {_f(r.arrival_spread_bps / 2)} | {_f(r.is_arrival_bps)} | "
                 f"{_f(r.is_vwap_bps)} | {_f(r.is_decision_bps)} |")
    if diagnosis_link:
        L += ["", f"Cost-model diagnosis (why `mv_abs_all` at $1M did worse than at $100M; half-spread vs "
                  f"impact at $100k): [{diagnosis_link}]({diagnosis_link}).", ""]
    return "\n".join(L) + "\n"


def write_outputs(df: pd.DataFrame, agg: pd.DataFrame, out_dir, account: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_p, md_p = out / "tca.csv", out / "tca.md"
    df.to_csv(csv_p, index=False, float_format="%.6g")
    md_p.write_text(render_markdown(df, agg, account,
                                    "../DataAnalysisPipeline2/results/cost_model_diagnosis.md"))
    return csv_p, md_p


# --------------------------------------------------------------------------- #
# Alpaca I/O (read-only)
# --------------------------------------------------------------------------- #
def fetch_fills(api, since: str = DEFAULT_SINCE) -> pd.DataFrame:
    """All filled / partially filled orders submitted after ``since`` (paginated)."""
    after = pd.Timestamp(since, tz="UTC").isoformat()
    seen, rows = set(), []
    for _ in range(50):
        page = api.get("/v2/orders", status="all", after=after, limit=500, direction="asc")
        new = [o for o in page if o["id"] not in seen]
        for o in new:
            seen.add(o["id"])
            if float(o.get("filled_qty") or 0) > 0 and o.get("filled_avg_price"):
                rows.append(o)
        if len(page) < 500 or not new:
            break
        after = page[-1]["submitted_at"]
    cols = ["id", "client_order_id", "symbol", "side", "type", "qty", "filled_qty", "filled_avg_price",
            "submitted_at", "filled_at", "status"]
    return pd.DataFrame([{c: o.get(c) for c in cols} for o in rows], columns=cols)


def _data_client(acct):
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(acct["api_key"], acct["secret_key"])


def fetch_daily(client, tickers, start, end, adjustment: str) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=sorted(set(tickers)), timeframe=TimeFrame.Day, start=start, end=end,
                           feed="sip", adjustment=Adjustment.RAW if adjustment == "raw" else Adjustment.ALL)
    df = client.get_stock_bars(req).df
    if df is None or not len(df):
        return pd.DataFrame(columns=["ticker", "Date", "Open", "High", "Low", "company_close", "company_volume",
                                     "VWAP"])
    df = df.reset_index()
    return pd.DataFrame({
        "ticker": df["symbol"].astype(str),
        "Date": pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"),
        "Open": df["open"].astype(float), "High": df["high"].astype(float), "Low": df["low"].astype(float),
        "company_close": df["close"].astype(float), "company_volume": df["volume"].astype(float),
        "VWAP": df["vwap"].astype(float) if "vwap" in df else np.nan})


def make_quote_fn(client):
    from alpaca.data.requests import StockQuotesRequest
    cache = {}

    def quote(sym, t):
        t = pd.Timestamp(t)
        key = (sym, t.floor("s"))
        if key in cache:
            return cache[key]
        res = (float("nan"),) * 3
        for w in QUOTE_WINDOWS_S if sym != "SPY" else (2, 10):
            try:
                q = client.get_stock_quotes(StockQuotesRequest(
                    symbol_or_symbols=sym, start=(t - pd.Timedelta(seconds=w)).to_pydatetime(),
                    end=t.to_pydatetime(), feed="sip")).df
            except Exception:       # noqa: BLE001 -- data gaps never fail the report
                q = None
            if q is not None and len(q):
                q = q.reset_index()
                res = mid_from_quotes(q, t)
                if np.isfinite(res[2]):
                    break
        cache[key] = res
        return res

    return quote


def equity_history(api) -> dict:
    """date (NY) -> end-of-day equity, plus 'latest'."""
    out = {}
    try:
        h = api.get("/v2/account/portfolio/history", period="6M", timeframe="1D")
        for ts, eq in zip(h.get("timestamp") or [], h.get("equity") or []):
            if eq:
                out[pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")] = float(eq)
    except Exception:               # noqa: BLE001
        pass
    try:
        out["latest"] = float(api.get("/v2/account")["equity"])
    except Exception:               # noqa: BLE001
        pass
    return out


def run(account: str = "monthly", since: str = DEFAULT_SINCE, ops_dir=None, log=print) -> dict:
    from . import config
    from . import monitoring as mon
    from .scheduled_rebalance import Alpaca
    acct = config.resolve_account(account)
    if not acct["configured"]:
        raise RuntimeError(f"Alpaca credentials for account '{account}' are not configured")
    if not acct["paper"]:
        raise RuntimeError("TCA is meant for the paper account (ALPACA_PAPER_TRADING is not true)")
    api = Alpaca(acct["api_key"], acct["secret_key"], acct["url"])
    fills = fetch_fills(api, since)
    out_dir = Path(ops_dir) if ops_dir else mon.ops_dir()
    ddates = prediction_dates(mon.predictions_dir(account, ops_dir))
    if not len(fills):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        write_outputs(df, aggregate(df), out_dir, account)
        return {"fills": 0}
    client = _data_client(acct)
    tdays = pd.to_datetime(fills["filled_at"], utc=True).dt.tz_convert("America/New_York")
    start = (tdays.min().normalize().tz_localize(None) - pd.Timedelta(days=HISTORY_DAYS)).to_pydatetime()
    # the free data plan serves SIP bars up to 15 minutes ago; a same-day bar is not final until the close
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).to_pydatetime()
    syms = sorted(set(fills["symbol"].astype(str)))
    adj = fetch_daily(client, syms, start, end, "all")
    raw = fetch_daily(client, syms + ["SPY"], start, end, "raw")
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if now_ny.hour < 20:            # today's bar is still forming (16:00 close + consolidation): drop it
        raw = raw[raw["Date"] < now_ny.strftime("%Y-%m-%d")]
    adj = adj.merge(raw[["ticker", "Date", "company_close"]].rename(columns={"company_close": "close_raw"}),
                    on=["ticker", "Date"], how="left")
    spy = raw[raw["ticker"] == "SPY"][["Date", "company_close"]]
    df = analyse(fills, bars_adj=adj, bars_raw=raw[raw["ticker"] != "SPY"], decision_dates=ddates,
                 quote_fn=make_quote_fn(client), equity_by_date=equity_history(api), spy_raw=spy)
    agg = aggregate(df)
    csv_p, md_p = write_outputs(df, agg, out_dir, account)
    a = agg[agg["bucket"] == "all"].iloc[0].to_dict()
    summary = {"fills": int(len(df)), "notional": round(float(df["notional"].sum()), 2),
               **{k: (round(float(a[k]), 2) if np.isfinite(a[k]) else None) for k in AGG_METRICS},
               "csv": str(csv_p), "md": str(md_p)}
    log("TCA_JSON: " + json.dumps(summary))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TCA of the paper account's fills -> ops/tca.{csv,md} (read-only)")
    ap.add_argument("--account", default=os.getenv("BDA_ACCOUNT", "monthly"))
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--ops-dir", default=None)
    a = ap.parse_args(argv)
    run(a.account.lower(), a.since, a.ops_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
