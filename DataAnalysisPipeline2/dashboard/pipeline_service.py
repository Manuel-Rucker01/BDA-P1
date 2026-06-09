"""
Dashboard pipeline service.

A thin, web-friendly wrapper around the production ``BDATradingAgent``. It does
NOT re-implement any trading logic — it drives the exact same methods the CLI
runner (`trading_agent/run.py`) uses, and returns plain dictionaries the Flask
layer can serialise to JSON.

Two user actions are mediated here:

  1. run_pipeline(...)   -> load model, detect regime, fetch live prices, run
                            the ensemble, (optionally) tilt by news, compute the
                            target portfolio. Returns the *proposed* book and
                            caches the (weights, prices_df) so a later execute
                            uses the very numbers the user reviewed.

  2. execute(job_id)     -> submit the cached proposal to Alpaca for real, then
                            persist attribution state exactly like run.py does.

A single agent (with its 46 MB model) is loaded lazily and reused across runs.
"""

from __future__ import annotations

import os
import json
import threading
import traceback
import uuid
from datetime import datetime, timezone

import pandas as pd

# Import the production agent + config. Support both "run as package" and
# "run as a loose script" invocation styles.
try:  # package style: python -m DataAnalysisPipeline2.dashboard.app
    from ..trading_agent.bot import BDATradingAgent
    from ..trading_agent import config
    from .tradingview_service import artifact_payload, build_tradingview_package
except ImportError:  # script style: python app.py from the dashboard dir
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PIPE = os.path.abspath(os.path.join(_HERE, ".."))
    _ROOT = os.path.abspath(os.path.join(_PIPE, ".."))
    for p in (_ROOT, _PIPE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from DataAnalysisPipeline2.trading_agent.bot import BDATradingAgent
    from DataAnalysisPipeline2.trading_agent import config
    from DataAnalysisPipeline2.dashboard.tradingview_service import (
        artifact_payload,
        build_tradingview_package,
    )


_AGENT_LOCK = threading.Lock()
_AGENT: BDATradingAgent | None = None

# In-process job registry. Each run produces a job whose result holds the
# proposed weights + the prices_df used to compute them, so "Send to Alpaca"
# trades exactly what the user reviewed.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

AGENT_LOGS_DIR = os.path.join(config.AGENT_DIR, "agent_logs")


# --------------------------------------------------------------------------- #
# Agent lifecycle
# --------------------------------------------------------------------------- #
def _get_agent() -> BDATradingAgent:
    """Lazily build + load the agent once; reuse the loaded model thereafter."""
    global _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            agent = BDATradingAgent()
            agent.load_model()
            _AGENT = agent
        return _AGENT


def _apply_universe(universe: str, num_tickers: int = 50, tickers_file: str | None = None):
    """Mirror run.py's universe resolution. Returns (basket_name, deferred_full)."""
    if universe == "safe":
        config.TICKERS = config.SAFE_TICKERS
        return "Safe Sector-Diversified Basket (20 Mid-Caps)", False
    if universe == "top_mcap":
        import duckdb
        try:
            conn = duckdb.connect(config.DB_PATH, read_only=True)
            q = f"""
                SELECT Symbol, ANY_VALUE(MarketCap) as mcap
                FROM master_dataset GROUP BY Symbol
                ORDER BY mcap DESC LIMIT {int(num_tickers)}
            """
            config.TICKERS = conn.execute(q).df()["Symbol"].tolist()
            conn.close()
            return f"Dynamic Top Market-Cap Basket (Top {num_tickers})", False
        except Exception as e:
            config.TICKERS = config.HIGH_ALPHA_TICKERS
            return f"High-Alpha Basket (top_mcap query failed: {e})", False
    if universe == "full":
        config.TICKERS = []  # deferred: populated after model load
        return "Full Modelled Universe (~1,890 tickers)", True
    if universe == "custom" and tickers_file:
        try:
            with open(tickers_file) as f:
                config.TICKERS = [ln.strip().upper() for ln in f if ln.strip()]
            return f"Custom Basket ({len(config.TICKERS)} tickers)", False
        except Exception as e:
            config.TICKERS = config.HIGH_ALPHA_TICKERS
            return f"High-Alpha Basket (custom load failed: {e})", False
    config.TICKERS = config.HIGH_ALPHA_TICKERS
    return "High-Alpha Alphabetical Basket (20 Small-Caps)", False


# --------------------------------------------------------------------------- #
# Core: run the pipeline and build the proposal
# --------------------------------------------------------------------------- #
def _run_pipeline_blocking(job_id: str, universe: str, strategy: str,
                           top_k: int | None, top_pct: float | None,
                           force_regime: str | None):
    """Heavy work; runs inside a worker thread. Writes results into _JOBS."""
    def _set(**kw):
        with _JOBS_LOCK:
            _JOBS[job_id].update(**kw)

    try:
        _set(status="running", stage="Resolving universe")
        # Apply overrides (these are module-level globals on config, same as CLI).
        if top_pct is not None:
            config.TOP_PCT_THRESHOLD = float(top_pct)
        if top_k is not None:
            config.TOP_K_HOLDINGS = int(top_k)
        basket_name, deferred_full = _apply_universe(universe)

        _set(stage="Loading model")
        agent = _get_agent()

        if deferred_full:
            config.TICKERS = sorted(agent.company_embeddings.keys())
        # Keep only tickers the model can actually score.
        config.TICKERS = [t for t in config.TICKERS if t in agent.company_embeddings]

        _set(stage="Detecting market regime")
        is_bull = agent.check_market_regime(force_regime=force_regime)
        regime = "bull" if is_bull else "bear"

        _set(stage=f"Fetching live prices ({len(config.TICKERS)} tickers)")
        prices_df = agent.fetch_live_data()

        _set(stage="Running ensemble inference")
        predictions = agent.run_inference(prices_df)

        # LIVE-ONLY post-hoc tilt (OFF by default in v2; news is now a model feature)
        predictions = agent.apply_news_sentiment_tilt(predictions)

        _set(stage="Computing target portfolio")
        weights = agent.calculate_target_weights(predictions, is_bull, strategy=strategy)

        proposals = _build_proposals(predictions, weights)

        # Cache everything execute() needs. We serialise prices_df to disk so a
        # restart does not strand a pending proposal mid-flight.
        cache_path = os.path.join(AGENT_LOGS_DIR, "dashboard_pending.parquet")
        os.makedirs(AGENT_LOGS_DIR, exist_ok=True)
        try:
            prices_df.to_parquet(cache_path)
        except Exception:
            cache_path = None

        _set(
            status="done",
            stage="Proposal ready",
            result={
                "regime": regime,
                "basket_name": basket_name,
                "strategy": strategy,
                "universe": universe,
                "top_k": config.TOP_K_HOLDINGS,
                "top_pct": config.TOP_PCT_THRESHOLD,
                "n_universe": len(config.TICKERS),
                "n_holdings": len(proposals),
                "gross_exposure_pct": round(sum(abs(p["weight"]) for p in proposals) * 100, 2),
                "proposals": proposals,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            _weights=weights,
            _prices_cache=cache_path,
        )
    except Exception as e:
        _set(status="error", stage="Failed",
             error=f"{e}", traceback=traceback.format_exc())


def _build_proposals(predictions: pd.DataFrame, weights: dict) -> list[dict]:
    """Join the weight dict back onto the prediction rows for a rich table."""
    by_ticker = {row["ticker"]: row for _, row in predictions.iterrows()}
    out = []
    for ticker, w in weights.items():
        if abs(w) < 1e-9:
            continue
        row = by_ticker.get(ticker, {})
        rank = row.get("pred_rank", row.get("pred_proba"))
        out.append({
            "ticker": ticker,
            "weight": float(w),
            "weight_pct": round(float(w) * 100, 2),
            "side": "LONG" if w > 0 else "SHORT",
            "pred_rank": None if rank is None or pd.isna(rank) else round(float(rank) * 100, 2),
            "price": _f(row.get("company_close")),
            "sector": row.get("Sector", "—") if not _isna(row.get("Sector")) else "—",
            "kalman_beta": _f(row.get("kalman_beta")),
        })
    out.sort(key=lambda r: abs(r["weight"]), reverse=True)
    return out


def _f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v), 4)
    except Exception:
        return None


def _isna(v):
    try:
        return pd.isna(v)
    except Exception:
        return v is None


# --------------------------------------------------------------------------- #
# Public API used by app.py
# --------------------------------------------------------------------------- #
def start_run(universe="full", strategy="top_k", top_k=None, top_pct=None,
              force_regime=None) -> str:
    """Kick off a pipeline run in a background thread; return its job id."""
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued", "stage": "Queued",
                         "created_at": datetime.now(timezone.utc).isoformat()}
    t = threading.Thread(
        target=_run_pipeline_blocking,
        args=(job_id, universe, strategy, top_k, top_pct, force_regime),
        daemon=True,
    )
    t.start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        # Return a client-safe view (omit internal cache handles).
        return {k: v for k, v in job.items() if not k.startswith("_")}


def execute(job_id: str) -> dict:
    """Submit the cached proposal of `job_id` to Alpaca for real."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return {"ok": False, "error": "Unknown job id."}
        if job.get("status") != "done":
            return {"ok": False, "error": f"Job not ready (status={job.get('status')})."}
        weights = job.get("_weights")
        prices_cache = job.get("_prices_cache")
        if job.get("_executed"):
            return {"ok": False, "error": "This proposal was already executed."}

    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        return {"ok": False, "error": "Alpaca credentials not configured on the server."}

    prices_df = None
    if prices_cache and os.path.exists(prices_cache):
        try:
            prices_df = pd.read_parquet(prices_cache)
        except Exception:
            prices_df = None

    agent = _get_agent()
    try:
        agent.execute_alpaca_rebalance(weights, prices_df=prices_df, dry_run=False)
        _persist_attribution_state(agent, prices_df, weights)
        with _JOBS_LOCK:
            _JOBS[job_id]["_executed"] = True
            _JOBS[job_id]["executed_at"] = datetime.now(timezone.utc).isoformat()
        return {"ok": True, "message": "Orders submitted to Alpaca (paper="
                f"{config.ALPACA_PAPER_TRADING}).",
                "paper": config.ALPACA_PAPER_TRADING}
    except Exception as e:
        return {"ok": False, "error": f"{e}", "traceback": traceback.format_exc()}


def get_tradingview_package(job_id: str, exchange_prefix: str = "NASDAQ") -> dict:
    """Return TradingView-compatible artifacts for a completed proposal."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return {"ok": False, "error": "Unknown job id."}
        if job.get("status") != "done":
            return {"ok": False, "error": f"Job not ready (status={job.get('status')})."}
        result = job.get("result") or {}
    return build_tradingview_package(job_id, result, exchange_prefix=exchange_prefix)


def get_tradingview_artifact(job_id: str, artifact: str,
                             exchange_prefix: str = "NASDAQ") -> tuple[str, str, str]:
    """Return a downloadable TradingView artifact for a completed proposal."""
    package = get_tradingview_package(job_id, exchange_prefix=exchange_prefix)
    if not package.get("ok"):
        raise ValueError(package.get("error", "Could not build TradingView package."))
    return artifact_payload(package, artifact)


def _persist_attribution_state(agent, prices_df, weights):
    """Mirror run.py's post-rebalance attribution + last_weights.json write."""
    try:
        import yfinance as yf
        state_path = os.path.join(AGENT_LOGS_DIR, "last_weights.json")
        prev_state = None
        if os.path.exists(state_path):
            with open(state_path) as f:
                prev_state = json.load(f)

        realised_returns = {}
        if prices_df is not None and prev_state is not None:
            window_start = prev_state.get("period_end")
            for ticker in set(prev_state["weights"].keys()):
                rows = prices_df[(prices_df["ticker"] == ticker)
                                 & (prices_df["Date"] >= window_start)]
                if len(rows) >= 2:
                    rs = rows.sort_values("Date")
                    first = float(rs.iloc[0]["company_close"])
                    last = float(rs.iloc[-1]["company_close"])
                    realised_returns[ticker] = (last - first) / first if first > 0 else 0.0

        bench_return = 0.0
        if prev_state is not None:
            try:
                sp = yf.download(config.SP500_INDEX, period="3mo", progress=False)
                if isinstance(sp.columns, pd.MultiIndex):
                    sp.columns = [c[0] for c in sp.columns]
                sp = sp.reset_index()
                sp["Date"] = pd.to_datetime(sp["Date"]).dt.strftime("%Y-%m-%d")
                ws = prev_state.get("period_end")
                wr = sp[sp["Date"] >= ws].sort_values("Date")
                if len(wr) >= 2:
                    bench_return = (float(wr.iloc[-1]["Close"]) - float(wr.iloc[0]["Close"])) \
                        / float(wr.iloc[0]["Close"])
            except Exception:
                pass

        if prev_state is not None and realised_returns:
            agent.record_attribution(
                weights_prev=prev_state["weights"], weights_curr=weights,
                realised_returns=realised_returns, benchmark_return=bench_return,
                period_start=prev_state["period_end"],
                period_end=str(prices_df["Date"].max()),
            )

        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump({
                "decision_id": agent.last_decision_ctx.decision_id
                    if agent.last_decision_ctx else "no_ctx",
                "period_end": str(prices_df["Date"].max()) if prices_df is not None else None,
                "weights": weights,
            }, f)
    except Exception as e:
        print(f"[dashboard] attribution persist failed: {e}")


# --------------------------------------------------------------------------- #
# Live portfolio + history (read-only)
# --------------------------------------------------------------------------- #
def get_portfolio() -> dict:
    """Live Alpaca account snapshot + per-position P&L."""
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        return {"ok": False, "error": "Alpaca credentials not configured on the server."}
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                               paper=config.ALPACA_PAPER_TRADING)
        acc = client.get_account()
        positions = client.get_all_positions()

        pos_out = []
        for p in positions:
            mv = _safe(p.market_value)
            cb = _safe(p.cost_basis)
            pos_out.append({
                "ticker": p.symbol,
                "qty": _safe(p.qty),
                "side": getattr(p.side, "value", str(p.side)),
                "avg_entry_price": _safe(p.avg_entry_price),
                "current_price": _safe(p.current_price),
                "cost_basis": cb,
                "market_value": mv,
                "unrealized_pl": _safe(p.unrealized_pl),
                "unrealized_plpc": _safe(p.unrealized_plpc, mul=100),
                "change_today_pc": _safe(getattr(p, "unrealized_intraday_plpc", None), mul=100),
            })
        pos_out.sort(key=lambda r: (r["market_value"] or 0), reverse=True)

        equity = _safe(acc.portfolio_value)
        total_pl = sum((p["unrealized_pl"] or 0) for p in pos_out)
        total_cb = sum((p["cost_basis"] or 0) for p in pos_out)
        return {
            "ok": True,
            "account": {
                "equity": equity,
                "cash": _safe(acc.cash),
                "buying_power": _safe(acc.buying_power),
                "last_equity": _safe(acc.last_equity),
                "long_market_value": _safe(getattr(acc, "long_market_value", None)),
                "paper": config.ALPACA_PAPER_TRADING,
                "currency": getattr(acc, "currency", "USD"),
            },
            "summary": {
                "n_positions": len(pos_out),
                "total_unrealized_pl": round(total_pl, 2),
                "total_cost_basis": round(total_cb, 2),
                "total_unrealized_plpc": round((total_pl / total_cb * 100) if total_cb else 0, 2),
            },
            "positions": pos_out,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"ok": False, "error": f"{e}"}


def get_history() -> dict:
    """Equity curve, drawdown, per-period attribution and recent trades."""
    out = {"ok": True, "equity": [], "attribution": [], "trades": []}

    eq_path = os.path.join(AGENT_LOGS_DIR, "equity_history.csv")
    if os.path.exists(eq_path):
        try:
            df = pd.read_csv(eq_path)
            df = df.dropna(subset=["equity"]).sort_values("ts")
            out["equity"] = [
                {"ts": r["ts"], "equity": round(float(r["equity"]), 2),
                 "peak": round(float(r["peak"]), 2),
                 "drawdown_pct": round(float(r["drawdown_pct"]), 3)}
                for _, r in df.iterrows()
            ]
        except Exception as e:
            out["equity_error"] = str(e)

    at_path = os.path.join(AGENT_LOGS_DIR, "attribution.csv")
    if os.path.exists(at_path):
        try:
            df = pd.read_csv(at_path)
            out["attribution"] = [
                {"period_end": r["period_end"],
                 "realised_return_pct": round(float(r["realised_return_pct"]), 3),
                 "benchmark_return_pct": round(float(r["benchmark_return_pct"]), 3),
                 "pnl_ml_pct": round(float(r.get("pnl_ml_pct", 0)), 3),
                 "pnl_regime_pct": round(float(r.get("pnl_regime_pct", 0)), 3),
                 "pnl_kalman_pct": round(float(r.get("pnl_kalman_pct", 0)), 3)}
                for _, r in df.iterrows()
            ]
        except Exception as e:
            out["attribution_error"] = str(e)

    tr_path = os.path.join(AGENT_LOGS_DIR, "trades.csv")
    if os.path.exists(tr_path):
        try:
            df = pd.read_csv(tr_path).tail(60).iloc[::-1]
            keep = ["decision_date", "ts", "ticker", "action", "target_weight",
                    "intended_notional_usd", "ref_price", "side", "dry_run"]
            df = df[[c for c in keep if c in df.columns]]
            out["trades"] = json.loads(df.to_json(orient="records"))
        except Exception as e:
            out["trades_error"] = str(e)

    return out


def _safe(v, mul=1.0):
    try:
        if v is None:
            return None
        return round(float(v) * mul, 4)
    except Exception:
        return None


def server_status() -> dict:
    return {
        "ok": True,
        "model_loaded": _AGENT is not None,
        "alpaca_configured": bool(config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY),
        "tradingview_export_available": True,
        "paper_trading": config.ALPACA_PAPER_TRADING,
        "alpaca_endpoint": config.ALPACA_URL,
        "defaults": {
            "top_k": config.TOP_K_HOLDINGS,
            "top_pct": config.TOP_PCT_THRESHOLD,
            "weighting": getattr(config, "WEIGHTING_SCHEME", "inverse_vol"),
        },
    }
