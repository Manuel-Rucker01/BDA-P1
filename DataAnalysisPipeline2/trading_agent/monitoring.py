"""Live-model monitoring (WS5 of docs/IMPROVEMENT_PLAN_v3.md).

What it does
------------
* ``record_predictions`` -- after every LIVE rebalance, persist the scored
  universe (ticker, pred_rank, weight, model_id) to
  ``ops/predictions/<account>/<decision_date>.csv.gz``. Small, deterministic
  (sorted, rounded, gzip mtime=0) so it can be committed by CI, which has no
  persistent disk.
* ``record_drift`` / ``feature_drift`` / ``build_feature_profile`` -- PSI of the
  live (cross-sectionally z-scored) model inputs against training quantile bins
  stored in the model manifest under ``feature_profile``.
* ``realized_metrics`` -- for one recorded decision, the realised Spearman IC,
  top-minus-bottom decile spread, and the held book vs SPY and vs the
  equal-weight universe over the holding period.
* ``update_monitoring_report`` -- recompute every period that has data and write
  ``ops/monitoring.csv`` (one row per account x period) and ``ops/monitoring.md``.
* ``evaluate_alerts`` -- pure alert rule.

Period convention (no look-ahead)
---------------------------------
``decision_date`` D is the date of the last daily bar the model saw (the live
run trades during the next session using features as of close D). A period runs
from close(D) to close(E) where E is the next recorded decision date for the
account, or the last available bar if the period is still open. The base price
is the last close on/before D; the end price is the last close on/before E and
strictly after D. No bar dated after E is read, and nothing before D except the
base close.

CLI::

    python -m DataAnalysisPipeline2.trading_agent.monitoring --account monthly
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Alert thresholds (see evaluate_alerts)
IC_WINDOW = 3
PSI_HIGH = 0.25
MAX_DRIFT_FEATURES = 3
UNDERPERF_PP = 10.0
UNDERPERF_WINDOW = 3

MIN_NAMES_FOR_IC = 10
MIN_NAMES_FOR_DECILES = 20
BASE_STALENESS_DAYS = 10  # a base close older than this before D is not used

CSV_COLUMNS = [
    "account", "decision_date", "end_date", "status", "model_id", "n_sessions",
    "n_eligible", "n_held", "ic", "spread_d10_d1", "book_return", "spy_return",
    "ew_return", "excess_vs_spy", "excess_vs_ew", "book_coverage",
]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def ops_dir(override=None) -> Path:
    """``ops/`` at the repo root; ``BDA_OPS_DIR`` or ``override`` relocate it."""
    return Path(override or os.getenv("BDA_OPS_DIR") or (REPO_ROOT / "ops"))


def predictions_dir(account: str, base=None) -> Path:
    return ops_dir(base) / "predictions" / str(account).lower()


def _date_str(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Model id
# --------------------------------------------------------------------------- #
def model_id_from(metadata: dict | None, model_path: str | None = None) -> str:
    """``model_id`` (or name/version) from the artefact manifest, else a short
    SHA256 of the model file, else ``"unknown"``."""
    md = metadata or {}
    for key in ("model_id", "model_name", "name", "version"):
        if md.get(key):
            return str(md[key])
    if model_path and os.path.exists(model_path):
        from .operational import file_sha256
        return "sha256:" + file_sha256(model_path)[:12]
    return "unknown"


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def _write_gz_csv(df: pd.DataFrame, path: Path) -> None:
    """Byte-deterministic gzip CSV (fixed mtime, no filename in the header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = df.to_csv(index=False, lineterminator="\n").encode()
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def record_predictions(predictions_df, weights, decision_date, account, model_id,
                       base_dir=None) -> Path:
    """Write ``ops/predictions/<account>/<decision_date>.csv.gz``.

    Columns: ticker, pred_rank, weight, model_id. One row per scored ticker
    (weight 0 if not held); held tickers missing from the predictions are kept
    with an empty pred_rank. Re-recording the same date overwrites the file."""
    if predictions_df is None or "ticker" not in predictions_df or "pred_rank" not in predictions_df:
        raise ValueError("predictions_df must have 'ticker' and 'pred_rank' columns")
    p = (predictions_df[["ticker", "pred_rank"]].dropna(subset=["ticker"])
         .assign(ticker=lambda d: d["ticker"].astype(str))
         .drop_duplicates("ticker", keep="last"))
    w = pd.Series({str(k): float(v) for k, v in (weights or {}).items()}, dtype=float)
    out = p.set_index("ticker").reindex(sorted(set(p["ticker"]) | set(w.index)))
    out["weight"] = w.reindex(out.index).fillna(0.0)
    out["pred_rank"] = pd.to_numeric(out["pred_rank"], errors="coerce").round(6)
    out["weight"] = out["weight"].round(6)
    out["model_id"] = str(model_id)
    out.index.name = "ticker"
    out = out.reset_index()[["ticker", "pred_rank", "weight", "model_id"]]
    path = predictions_dir(account, base_dir) / f"{_date_str(decision_date)}.csv.gz"
    _write_gz_csv(out, path)
    return path


def record_drift(drift: dict, decision_date, account, base_dir=None) -> Path:
    """Persist a ``feature_drift`` result next to the predictions file."""
    path = predictions_dir(account, base_dir) / f"{_date_str(decision_date)}.drift.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(drift, sort_keys=True, indent=1) + "\n")
    return path


def load_predictions(pred_file) -> pd.DataFrame:
    return pd.read_csv(pred_file, compression="gzip", dtype={"ticker": str})


def list_prediction_files(account, base_dir=None) -> list[tuple[str, Path]]:
    d = predictions_dir(account, base_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("????-??-??.csv.gz"))
    return [(f.name[:10], f) for f in files]


# --------------------------------------------------------------------------- #
# Feature drift (PSI)
# --------------------------------------------------------------------------- #
def _bin_fractions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Fractions in bins (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def build_feature_profile(train_X: pd.DataFrame, n_bins: int = 10, dates=None,
                          null_quantile: float = 0.95, max_null_dates: int = 120,
                          seed: int = 0) -> dict:
    """Training-distribution profile for PSI, to be stored in the model manifest
    under ``feature_profile``. Build it on the SAME transformed matrix the model
    is fed (for feature_version>=2: the cross-sectionally z-scored X).

    Per numeric feature: interior quantile edges (deduplicated) and the training
    fraction in each of the ``len(edges)+1`` bins."""
    feats = {}
    for col in train_X.columns:
        s = pd.to_numeric(train_X[col], errors="coerce").to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        qs = np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges = sorted({round(float(q), 8) for q in qs})
        fr = _bin_fractions(s, edges)
        feats[str(col)] = {"edges": edges, "fractions": [round(float(x), 8) for x in fr]}
    out = {"version": 1, "n_bins": int(n_bins), "n_rows": int(len(train_X)), "features": feats}
    if dates is not None:
        # Live drift is measured on ONE decision date, while the reference is
        # pooled over hundreds. Features that are constant within a sector or
        # driven by the day's market move (sector_return_5d, stoch_20d) show
        # a large single-date PSI even with no drift at all. Calibrate: PSI of
        # individual TRAINING dates vs the pooled profile; its upper quantile
        # becomes that feature's null level (feature_drift uses
        # max(threshold, null)).
        d = pd.Series(np.asarray(dates)).reset_index(drop=True)
        uniq = np.array(sorted(d.unique()))
        rng = np.random.default_rng(seed)
        pick = uniq if len(uniq) <= max_null_dates else rng.choice(uniq, max_null_dates, replace=False)
        X = train_X.reset_index(drop=True)
        for col, spec in feats.items():
            vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            ps = []
            for day in pick:
                v = vals[(d == day).to_numpy()]
                v = v[np.isfinite(v)]
                if v.size:
                    ps.append(psi(spec["fractions"], _bin_fractions(v, spec["edges"])))
            if ps:
                spec["psi_null_q"] = round(float(np.quantile(ps, null_quantile)), 6)
        out.update({"null_quantile": float(null_quantile), "n_null_dates": int(len(pick))})
    return out


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index: sum((a - e) * ln(a / e)), fractions floored at eps."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(live_X: pd.DataFrame | None, train_profile: dict | None,
                  threshold: float = PSI_HIGH) -> dict:
    """PSI per feature of ``live_X`` against ``train_profile``.

    ``train_profile`` is a ``build_feature_profile`` result, or a manifest dict
    that contains one under ``feature_profile``. If it is absent the result has
    ``available: False`` and a ``reason`` instead of raising."""
    prof = train_profile
    if isinstance(prof, dict) and "feature_profile" in prof:
        prof = prof["feature_profile"]
    if not prof or not isinstance(prof, dict) or not prof.get("features"):
        return {"available": False, "psi": {}, "high": [], "n_high": 0,
                "reason": "no feature_profile in the model manifest -- drift not computed"}
    if live_X is None or len(live_X) == 0:
        return {"available": False, "psi": {}, "high": [], "n_high": 0,
                "reason": "no live feature matrix"}
    out, missing = {}, []
    for name, spec in prof["features"].items():
        if name not in live_X.columns:
            missing.append(name)
            continue
        v = pd.to_numeric(live_X[name], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            missing.append(name)
            continue
        out[name] = round(psi(spec["fractions"], _bin_fractions(v, spec["edges"])), 6)
    # per-feature threshold: the global one, raised to the feature's calibrated
    # single-date null level when the profile carries one
    thr = {k: max(threshold, float(prof["features"][k].get("psi_null_q", 0.0))) for k in out}
    high = sorted([k for k, x in out.items() if x > thr[k]], key=lambda k: -out[k])
    res = {"available": True, "psi": out, "high": high, "n_high": len(high),
           "threshold": threshold, "missing": sorted(missing), "n_live_rows": int(len(live_X))}
    if any("psi_null_q" in prof["features"][k] for k in out):
        res["thresholds"] = {k: round(v, 6) for k, v in thr.items()}
    return res


# --------------------------------------------------------------------------- #
# Realised metrics
# --------------------------------------------------------------------------- #
def _spearman(x: pd.Series, y: pd.Series) -> float:
    rx, ry = x.rank(), y.rank()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _close_asof(px: pd.DataFrame, when: str, after: str | None = None) -> pd.Series:
    """Per ticker: close of the last bar with Date <= ``when`` (and > ``after``)."""
    sub = px[px["Date"] <= when]
    if after is not None:
        sub = sub[sub["Date"] > after]
    last = sub.sort_values("Date").groupby("ticker").tail(1)
    return last.set_index("ticker")[["Date", "company_close"]]


def realized_metrics(pred_file, prices: pd.DataFrame, benchmark: pd.DataFrame | None,
                     next_decision_date=None, today=None) -> dict | None:
    """Realised metrics for one recorded decision.

    ``prices``: long frame [ticker, Date, company_close] (``market_data``
    schema). ``benchmark``: [Date, Close] of SPY. The period ends at
    ``next_decision_date`` (closed) or at the last bar <= ``today`` (open).
    Returns ``None`` if there is no bar after the decision date yet."""
    pred_file = Path(pred_file)
    D = pred_file.name[:10]
    preds = load_predictions(pred_file) if pred_file.exists() else None
    if preds is None:
        return None
    px = prices[["ticker", "Date", "company_close"]].copy()
    px["Date"] = px["Date"].astype(str).str[:10]
    px = px[pd.to_numeric(px["company_close"], errors="coerce") > 0]
    cutoff = _date_str(next_decision_date) if next_decision_date is not None else \
        _date_str(today or dt.date.today())
    px = px[px["Date"] <= cutoff]  # never read beyond the period end
    later = px[px["Date"] > D]
    if later.empty:
        return None
    end_date = later["Date"].max()
    closed = next_decision_date is not None and end_date >= _date_str(next_decision_date)

    stale = _date_str(pd.Timestamp(D) - pd.Timedelta(days=BASE_STALENESS_DAYS))
    base = _close_asof(px[px["Date"] >= stale], D)
    end = _close_asof(px, end_date, after=D)
    rets = (end["company_close"] / base["company_close"].reindex(end.index) - 1.0).dropna()

    p = preds.set_index("ticker")
    ranked = p["pred_rank"].dropna()
    elig = pd.DataFrame({"pred_rank": ranked, "ret": rets.reindex(ranked.index)}).dropna()
    n = len(elig)
    ic = _spearman(elig["pred_rank"], elig["ret"]) if n >= MIN_NAMES_FOR_IC else float("nan")
    spread = float("nan")
    if n >= MIN_NAMES_FOR_DECILES:
        dec = pd.qcut(elig["pred_rank"].rank(method="first"), 10, labels=False)
        g = elig["ret"].groupby(dec).mean()
        spread = float(g.iloc[-1] - g.iloc[0])
    ew = float(elig["ret"].mean()) if n else float("nan")

    w = p["weight"].fillna(0.0)
    held = w[w != 0]
    priced = held[held.index.isin(rets.index)]
    book = float((priced * rets.reindex(priced.index)).sum()) if len(held) else float("nan")
    gross = float(held.abs().sum())
    coverage = float(priced.abs().sum() / gross) if gross > 0 else float("nan")

    spy = float("nan")
    if benchmark is not None and len(benchmark):
        b = benchmark[["Date", "Close"]].copy()
        b["Date"] = b["Date"].astype(str).str[:10]
        b0 = b[(b["Date"] <= D) & (b["Date"] >= stale)].sort_values("Date")
        b1 = b[(b["Date"] > D) & (b["Date"] <= end_date)].sort_values("Date")
        if len(b0) and len(b1):
            spy = float(b1["Close"].iloc[-1] / b0["Close"].iloc[-1] - 1.0)

    sessions = px.loc[(px["Date"] > D) & (px["Date"] <= end_date), "Date"].nunique()
    model_id = str(preds["model_id"].iloc[0]) if "model_id" in preds and len(preds) else "unknown"
    return {
        "decision_date": D, "end_date": end_date, "status": "closed" if closed else "open",
        "model_id": model_id, "n_sessions": int(sessions), "n_eligible": int(n),
        "n_held": int(len(held)), "ic": ic, "spread_d10_d1": spread, "book_return": book,
        "spy_return": spy, "ew_return": ew, "excess_vs_spy": book - spy,
        "excess_vs_ew": book - ew, "book_coverage": coverage,
    }


# --------------------------------------------------------------------------- #
# Alerts (pure)
# --------------------------------------------------------------------------- #
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def evaluate_alerts(periods: list[dict], drift: dict | None = None, *,
                    ic_window: int = IC_WINDOW, psi_threshold: float = PSI_HIGH,
                    max_drift_features: int = MAX_DRIFT_FEATURES,
                    underperf_pp: float = UNDERPERF_PP,
                    underperf_window: int = UNDERPERF_WINDOW) -> list[dict]:
    """Alert rules over CLOSED periods (chronological) and the latest drift.

    1. ``ic``: mean IC of the last ``ic_window`` closed periods < 0.
    2. ``drift``: >= ``max_drift_features`` features with PSI > ``psi_threshold``.
    3. ``underperformance``: compounded book return trails the compounded
       equal-weight universe by more than ``underperf_pp`` percentage points over
       the last ``underperf_window`` closed periods.
    Rules with too little data are not evaluated (no alert)."""
    alerts = []
    closed = [p for p in periods if p.get("status") == "closed"]
    ics = [float(p["ic"]) for p in closed if _finite(p.get("ic"))][-ic_window:]
    if len(ics) == ic_window and float(np.mean(ics)) < 0:
        alerts.append({"rule": "ic", "message":
                       f"rolling {ic_window}-period mean IC is {np.mean(ics):+.3f} (< 0): "
                       + ", ".join(f"{x:+.3f}" for x in ics)})
    if drift and drift.get("available"):
        thr = drift.get("thresholds") or {}
        high = [k for k, v in (drift.get("psi") or {}).items()
                if _finite(v) and float(v) > max(psi_threshold, float(thr.get(k, 0.0)))]
        if len(high) >= max_drift_features:
            top = sorted(high, key=lambda k: -float(drift["psi"][k]))[:8]
            alerts.append({"rule": "drift", "message":
                           f"{len(high)} features with PSI > {psi_threshold}: "
                           + ", ".join(f"{k}={float(drift['psi'][k]):.2f}" for k in top)})
    win = [p for p in closed if _finite(p.get("book_return")) and _finite(p.get("ew_return"))]
    win = win[-underperf_window:]
    if len(win) == underperf_window:
        book = float(np.prod([1 + float(p["book_return"]) for p in win]) - 1)
        ew = float(np.prod([1 + float(p["ew_return"]) for p in win]) - 1)
        gap = (book - ew) * 100
        if gap < -underperf_pp:
            alerts.append({"rule": "underperformance", "message":
                           f"book {book:+.1%} vs equal-weight universe {ew:+.1%} over the last "
                           f"{underperf_window} periods ({gap:+.1f} pp < -{underperf_pp:.0f} pp)"})
    return alerts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _default_prices(tickers, start_date):
    from . import market_data
    days = (dt.date.today() - pd.Timestamp(start_date).date()).days + 15
    bars, _ = market_data.fetch_daily_bars(sorted(tickers), lookback_days=max(days, 30), verbose=False)
    return bars


def _default_benchmark(start_date):
    from . import market_data
    days = (dt.date.today() - pd.Timestamp(start_date).date()).days + 15
    return market_data.fetch_benchmark_closes("SPY", lookback_days=max(days, 30), verbose=False)


def _latest_drift(account, base_dir=None) -> tuple[str | None, dict | None]:
    files = sorted(predictions_dir(account, base_dir).glob("????-??-??.drift.json"))
    if not files:
        return None, None
    try:
        return files[-1].name[:10], json.loads(files[-1].read_text())
    except Exception:
        return files[-1].name[:10], None


def _fmt(x, pct=False, signed=True):
    if not _finite(x):
        return "–"
    x = float(x)
    if pct:
        return f"{x * 100:+.2f}%" if signed else f"{x * 100:.2f}%"
    return f"{x:+.3f}"


def _rolling_ic(rows: list[dict], n: int) -> float:
    ics = [float(r["ic"]) for r in rows if r.get("status") == "closed" and _finite(r.get("ic"))]
    return float(np.mean(ics[-n:])) if len(ics) >= n else float("nan")


def _account_section(account: str, rows: list[dict], drift_date, drift, alerts) -> list[str]:
    L = [f"## Account `{account}`", ""]
    if not rows:
        L += ["No recorded predictions with realised data yet.", ""]
    else:
        L += ["| decision | end | status | model | n | IC | D10-D1 | book | SPY | EW | book-SPY | book-EW | IC roll3 | IC roll6 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for i, r in enumerate(rows):
            upto = rows[: i + 1]
            L.append(f"| {r['decision_date']} | {r['end_date']} | {r['status']} | `{r['model_id']}` | "
                     f"{int(r['n_eligible'])} | {_fmt(r['ic'])} | {_fmt(r['spread_d10_d1'], True)} | "
                     f"{_fmt(r['book_return'], True)} | {_fmt(r['spy_return'], True)} | "
                     f"{_fmt(r['ew_return'], True)} | {_fmt(r['excess_vs_spy'], True)} | "
                     f"{_fmt(r['excess_vs_ew'], True)} | {_fmt(_rolling_ic(upto, 3))} | "
                     f"{_fmt(_rolling_ic(upto, 6))} |")
        L.append("")
    L.append("**Verdict**")
    L.append("")
    r3, r6 = _rolling_ic(rows, 3), _rolling_ic(rows, 6)
    L.append(f"- Rolling IC (closed periods): 3-period {_fmt(r3)}, 6-period {_fmt(r6)}"
             + ("" if _finite(r3) else " (needs >= 3 closed periods)"))
    if drift is None:
        L.append("- Feature drift: no drift record yet.")
    elif not drift.get("available"):
        L.append(f"- Feature drift ({drift_date}): skipped -- {drift.get('reason', 'unavailable')}.")
    else:
        top = ", ".join(f"{k} {drift['psi'][k]:.2f}" for k in drift.get("high", [])[:5])
        L.append(f"- Feature drift ({drift_date}): {drift.get('n_high', 0)} of {len(drift.get('psi', {}))} "
                 f"features with PSI > {drift.get('threshold', PSI_HIGH)}" + (f" ({top})" if top else ""))
    if alerts:
        for a in alerts:
            L.append(f"- **ALERT [{a['rule']}]** {a['message']}")
    else:
        L.append("- OK: no alert rule triggered.")
    L.append("")
    return L


def _write_markdown(df: pd.DataFrame, base_dir, extra: dict) -> Path:
    L = ["# Live model monitoring", "",
         "Auto-generated by `DataAnalysisPipeline2/trading_agent/monitoring.py` -- do not edit.",
         "Periods run from close of the decision date (last bar the model saw) to close of the next",
         "decision (or the latest bar while open). IC = Spearman(pred_rank, realised return) over the",
         "eligible universe; D10-D1 = top-minus-bottom decile mean return; EW = equal-weight universe.",
         f"Alert rules: rolling {IC_WINDOW}-period IC < 0; >= {MAX_DRIFT_FEATURES} features with PSI > {PSI_HIGH}; "
         f"book trails EW by > {UNDERPERF_PP:.0f} pp over {UNDERPERF_WINDOW} periods.", ""]
    accounts = sorted(set(df["account"]) | set(extra)) if len(df) else sorted(extra)
    for acc in accounts:
        rows = df[df["account"] == acc].sort_values("decision_date").to_dict("records") if len(df) else []
        dd, drift = _latest_drift(acc, base_dir)
        L += _account_section(acc, rows, dd, drift, evaluate_alerts(rows, drift))
    path = ops_dir(base_dir) / "monitoring.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L).rstrip() + "\n")
    return path


def update_monitoring_report(account, *, base_dir=None, prices_fetcher=None,
                             benchmark_fetcher=None, today=None, recompute_all=False) -> dict:
    """Recompute realised metrics for ``account``'s recorded decisions and rewrite
    ``ops/monitoring.csv`` and ``ops/monitoring.md``.

    Closed periods already in the CSV are kept as-is (their window cannot
    change) unless ``recompute_all``; open/missing periods are recomputed. The
    CSV is keyed by (account, decision_date), so re-running never duplicates.
    Fetchers default to read-only market data (``market_data``) and are
    injectable for tests: ``prices_fetcher(tickers, start_date) -> DataFrame``,
    ``benchmark_fetcher(start_date) -> DataFrame[Date, Close]``."""
    account = str(account).lower()
    prices_fetcher = prices_fetcher or _default_prices
    benchmark_fetcher = benchmark_fetcher or _default_benchmark
    today = _date_str(today or dt.date.today())
    csv_path = ops_dir(base_dir) / "monitoring.csv"
    existing = pd.read_csv(csv_path, dtype={"account": str, "decision_date": str, "end_date": str}) \
        if csv_path.exists() else pd.DataFrame(columns=CSV_COLUMNS)

    files = list_prediction_files(account, base_dir)
    dates = [d for d, _ in files]
    keep = existing[existing["account"] == account] if len(existing) else existing
    done = set() if recompute_all else set(keep.loc[keep["status"] == "closed", "decision_date"])
    todo = [(i, d, f) for i, (d, f) in enumerate(files) if d not in done and d < today]

    new_rows, notes = [], []
    if todo:
        tickers = set()
        for _, _, f in todo:
            tickers |= set(load_predictions(f)["ticker"].astype(str))
        start = _date_str(pd.Timestamp(todo[0][1]) - pd.Timedelta(days=BASE_STALENESS_DAYS + 5))
        prices = prices_fetcher(sorted(tickers), start)
        try:
            bench = benchmark_fetcher(start)
        except Exception as e:  # benchmark is optional
            bench = None
            notes.append(f"benchmark unavailable: {e}")
        for i, d, f in todo:
            nxt = dates[i + 1] if i + 1 < len(dates) else None
            m = realized_metrics(f, prices, bench, next_decision_date=nxt, today=today)
            if m is None:
                notes.append(f"{d}: no market data after the decision date yet")
                continue
            new_rows.append({"account": account, **m})

    recomputed = {r["decision_date"] for r in new_rows}
    others = existing[existing["account"] != account] if len(existing) else existing
    kept = keep[~keep["decision_date"].isin(recomputed)] if len(keep) else keep
    kept = kept[kept["decision_date"].isin(dates)] if len(kept) else kept  # drop orphans
    frames = [x for x in (others, kept, pd.DataFrame(new_rows)) if len(x)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLUMNS)
    out = (out.reindex(columns=CSV_COLUMNS)
              .drop_duplicates(["account", "decision_date"], keep="last")
              .sort_values(["account", "decision_date"]).reset_index(drop=True))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False, float_format="%.6f", lineterminator="\n")
    md_path = _write_markdown(out, base_dir, {account: True})

    rows = out[out["account"] == account].to_dict("records")
    drift_date, drift = _latest_drift(account, base_dir)
    alerts = evaluate_alerts(rows, drift)
    latest = rows[-1] if rows else None
    clean = lambda v: (None if isinstance(v, float) and not math.isfinite(v) else v)  # noqa: E731
    return {
        "account": account, "n_recorded": len(files), "n_periods": len(rows),
        "n_recomputed": len(new_rows),
        "latest": {k: clean(v) for k, v in latest.items()} if latest else None,
        "rolling_ic_3": clean(_rolling_ic(rows, 3)), "rolling_ic_6": clean(_rolling_ic(rows, 6)),
        "drift": ({"date": drift_date, "available": bool(drift.get("available")),
                   "n_high": drift.get("n_high", 0), "reason": drift.get("reason")}
                  if drift else None),
        "alerts": alerts, "notes": notes,
        "csv": str(csv_path), "markdown": str(md_path),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Update ops/monitoring.{csv,md} (read-only market data).")
    ap.add_argument("--account", default=os.getenv("BDA_ACCOUNT", "monthly"))
    ap.add_argument("--recompute-all", action="store_true")
    ap.add_argument("--fail-on-alert", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("ACTIVE_ACCOUNT", args.account.lower())  # market data keys
    summary = update_monitoring_report(args.account, recompute_all=args.recompute_all)
    print("MONITOR_JSON: " + json.dumps(summary, default=str))
    for a in summary["alerts"]:
        print(f"[monitor] ALERT [{a['rule']}] {a['message']}")
    return 4 if (args.fail_on_alert and summary["alerts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
