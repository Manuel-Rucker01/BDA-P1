#!/usr/bin/env python3
"""
CLI Execution Script for the BDA Production Trading Bot.
Use this script to trigger monthly portfolio rebalancing.
"""

import os
import sys
import argparse

import pandas as pd

from .bot import BDATradingAgent, STRATEGIES
from . import config

def main():
    parser = argparse.ArgumentParser(description="BDA Production Trading Agent CLI Rebalancer")
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real-world orders directly on active Alpaca Brokerage account (defaults to Paper Trading unless env modified)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate predictions and weights print without submitting brokerage order changes (default is True unless --live set)"
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="high_confidence",
        help="Strategy to use: 'high_confidence' (Long-Only P>=0.53), 'top_k' (concentrated top-pct of full universe), "
             "'regime_filtered' (Long/Short, shorts disabled in bull markets), "
             "'long_only_v3' (broad long-only: BDA_V3_K names, strict BDA_V3_MAX_PER_SECTOR cap, "
             "inverse-vol capped at BDA_V3_MAX_WEIGHT) or 'long_short_neutral' (dollar-neutral "
             "BDA_LS_N_LONG/BDA_LS_N_SHORT at BDA_LS_GROSS gross, ETB shorts only, sector-neutral) or "
             "'optimized_v3' (experimental, off by default: Round-5/7 cost-aware mean-variance book, "
             "spec mv_abs_all; falls back to long_only_v3 when fundamentals/risk model/solver fail)"
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=None,
        help="With --strategy top_k: only consider names in the top X%% of pred_rank "
             "(overrides config.TOP_PCT_THRESHOLD; default 5.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="With --strategy top_k: cap held positions at K names "
             "(overrides config.TOP_K_HOLDINGS; default 10)"
    )
    parser.add_argument(
        "--force-regime",
        choices=["bull", "bear"],
        default=None,
        help="Force the S&P 500 regime to 'bull' or 'bear', skipping the live index calculation"
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        default=False,
        help="Use the sector-diversified mid-cap stock basket (+59%% return, lower volatility) instead of the highly explosive alphabetical basket (+304%% return)"
    )
    parser.add_argument(
        "--recommend-only",
        action="store_true",
        default=False,
        help="Print the model predictions and optimal target portfolio recommendations, then exit without executing any orders or rebalancing simulations"
    )
    parser.add_argument(
        "--universe",
        choices=["high_alpha", "safe", "top_mcap", "custom", "full"],
        default="high_alpha",
        help="Target stock universe: 'high_alpha' (default), 'safe' (sector diversified), 'top_mcap' (top market cap companies), 'custom' (user file), or 'full' (all ~1890 modelled tickers — pairs with --strategy top_k)"
    )
    parser.add_argument(
        "--num-tickers",
        type=int,
        default=50,
        help="Number of tickers to select if --universe top_mcap is active (default is 50)"
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=None,
        help="Path to a newline-separated file of custom tickers if --universe custom is active"
    )

    args = parser.parse_args()

    # Determine target stocks basket based on universe/safe flags
    universe = args.universe
    if args.safe:
        universe = "safe"

    if universe == "safe":
        config.TICKERS = config.SAFE_TICKERS
        basket_name = "Safe Sector-Diversified Basket (20 Mid-Caps)"
    elif universe == "top_mcap":
        import duckdb
        try:
            conn = duckdb.connect(config.DB_PATH, read_only=True)
            query = f"""
                SELECT Symbol, ANY_VALUE(MarketCap) as mcap 
                FROM master_dataset 
                GROUP BY Symbol 
                ORDER BY mcap DESC 
                LIMIT {args.num_tickers}
            """
            mcap_df = conn.execute(query).df()
            config.TICKERS = mcap_df["Symbol"].tolist()
            conn.close()
            basket_name = f"Dynamic Top Market-Cap Basket (Top {args.num_tickers} Large-Caps)"
        except Exception as e:
            print(f"[ERROR] Failed to query DuckDB for top_mcap universe: {e}. Falling back to default.")
            config.TICKERS = config.HIGH_ALPHA_TICKERS
            basket_name = "High-Alpha Alphabetical Basket (20 Small-Caps)"
    elif universe == "full":
        # Defer: we need agent.company_embeddings to populate this. We tag it
        # so the post-load block (below) knows to expand to the full modelled
        # universe.
        config.TICKERS = []
        basket_name = "Full Modelled Universe (defer to model embeddings)"
    elif universe == "custom":
        if not args.tickers_file:
            print("[ERROR] Custom universe selected but no --tickers-file provided. Falling back to default.")
            config.TICKERS = config.HIGH_ALPHA_TICKERS
            basket_name = "High-Alpha Alphabetical Basket (20 Small-Caps)"
        else:
            try:
                with open(args.tickers_file, "r") as f:
                    config.TICKERS = [line.strip().upper() for line in f if line.strip()]
                basket_name = f"Custom Basket from {args.tickers_file} ({len(config.TICKERS)} tickers)"
            except Exception as e:
                print(f"[ERROR] Failed to load tickers file: {e}. Falling back to default.")
                config.TICKERS = config.HIGH_ALPHA_TICKERS
                basket_name = "High-Alpha Alphabetical Basket (20 Small-Caps)"
    else:
        config.TICKERS = config.HIGH_ALPHA_TICKERS
        basket_name = "High-Alpha Alphabetical Basket (20 Small-Caps)"

    # Determine dry_run value: dry-run defaults to True unless explicit --live is passed without --dry-run
    is_dry_run = True
    if args.live:
        if args.dry_run:
            print("[Warning] Both --live and --dry-run options set. Falling back to safe Dry-Run Simulation.")
        else:
            is_dry_run = False

    print("=" * 80)
    print("BDA PRODUCTION QUANT TRADING BOT RUNNER")
    print("=" * 80)
    print(f"Operational Mode : {'[SIMULATION DRY RUN]' if is_dry_run else '[LIVE ACCOUNT ORDER EXECUTION]'}")
    print(f"Target Strategy  : {args.strategy}")
    print(f"Stock Basket     : {basket_name}")
    print(f"Alpaca endpoint  : {config.ALPACA_URL} (Paper trading: {config.ALPACA_PAPER_TRADING})")
    print("=" * 80 + "\n")

    # Apply top-K / top-pct overrides BEFORE the agent runs.
    if args.top_pct is not None:
        config.TOP_PCT_THRESHOLD = float(args.top_pct)
    if args.top_k is not None:
        config.TOP_K_HOLDINGS = int(args.top_k)
    if args.strategy == "top_k":
        print(f"[Strategy] top_k: gate=top {config.TOP_PCT_THRESHOLD}% | "
              f"hold cap={config.TOP_K_HOLDINGS} | "
              f"weighting={getattr(config, 'WEIGHTING_SCHEME', 'inverse_vol')}")
    elif args.strategy == "long_only_v3":
        print(f"[Strategy] long_only_v3: K={config.V3_K} | <= {config.V3_MAX_PER_SECTOR}/sector "
              f"(strict) | <= {config.V3_MAX_WEIGHT:.2%}/name | exposure <= {config.V3_TARGET_EXPOSURE:.2f}")
    elif args.strategy == "optimized_v3":
        print(f"[Strategy] optimized_v3 (EXPERIMENTAL, fails the promotion gate): mv_abs_all | "
              f"AUM ${config.OPT_AUM:,.0f} | IC {config.OPT_IC_ALTALL if config.SIGNAL == 'altall' else config.OPT_IC} (signal {config.SIGNAL}) | exposure {config.OPT_TARGET_EXPOSURE:.2f} | "
              f"min weight {config.OPT_MIN_WEIGHT:.2%} | {config.OPT_HISTORY_DAYS} days of history | "
              f"leverage {config.LEVERAGE:.2f}x (altall + margin account only, else 1.0x) | "
              f"fallback long_only_v3")
    elif args.strategy == "long_short_neutral":
        print(f"[Strategy] long_short_neutral: {config.LS_N_LONG} long / {config.LS_N_SHORT} short | "
              f"gross={config.LS_GROSS} | <= {config.LS_MAX_WEIGHT:.2%}/name | "
              f"<= {config.LS_MAX_PER_SECTOR}/sector/leg | sector_neutral={config.LS_SECTOR_NEUTRAL}")

    try:
        agent = BDATradingAgent()
        requested_tickers = list(config.TICKERS)

        # 1. Load serializations (BDA_SIGNAL=altall: the Round-8 artefact, or
        #    the production one if it / its live inputs are unavailable)
        agent.load_model()

        is_bull, prices_df, predictions, weights = run_with_signal_fallback(
            agent, args, universe, requested_tickers)

        print("\nTarget Portfolio Weights Allocations:")
        print("-" * 35)
        print(f"{'Ticker':<8} | {'Target Weight Allocation %'}")
        print("-" * 35)
        for t, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(w) > 0:
                side = "LONG" if w > 0 else "SHORT"
                print(f"{t:<8} | {w * 100:>23.2f}% ({side})")
        print("-" * 35)

        import json as _json
        construction = getattr(agent, "last_construction", None)
        if construction is not None or agent.signal_requested != "production":
            construction = {**(construction or {"construction": args.strategy}), **agent.signal_info()}
            agent.last_construction = construction
        if construction:
            print("CONSTRUCTION_JSON: " + _json.dumps(construction, default=str))

        # 6. Execute differential rebalance
        if args.recommend_only:
            print("\n[Recommend Only] Exit successfully without executing any portfolio rebalancing.")
            print("=" * 80)
            return
            
        exec_status = agent.execute_alpaca_rebalance(
            weights, prices_df=prices_df, dry_run=is_dry_run) or {"status": "unknown"}

        # ── Performance attribution against the previous rebalance ────────
        # We persist the most-recent decision's target weights to a small
        # state file so the next invocation can attribute realised returns
        # between rebalances. Only when orders were actually executed: a
        # dry-run or a halted run must never mutate state.
        try:
            import json
            from . import operational, market_data

            if exec_status.get("status") != "executed":
                raise _SkipState()

            state_path = os.path.join(operational.AGENT_LOG_DIR, "last_weights.json")
            prev_state = None
            if os.path.exists(state_path):
                with open(state_path) as f:
                    prev_state = json.load(f)

            # Realised returns: average daily_return over the window for each
            # ticker (using whatever price_history_df already has).
            realised_returns = {}
            if prices_df is not None and prev_state is not None:
                window_start = prev_state.get("period_end")
                for ticker in set(prev_state["weights"].keys()):
                    rows = prices_df[(prices_df["ticker"] == ticker)
                                     & (prices_df["Date"] >= window_start)]
                    if len(rows) >= 2:
                        first = float(rows.sort_values("Date").iloc[0]["company_close"])
                        last  = float(rows.sort_values("Date").iloc[-1]["company_close"])
                        realised_returns[ticker] = (last - first) / first if first > 0 else 0.0

            # Benchmark return over the same window
            bench_return = 0.0
            if prev_state is not None:
                try:
                    sp = market_data.fetch_benchmark_closes(config.SP500_INDEX, lookback_days=120)
                    window_start = prev_state.get("period_end")
                    window_rows = sp[sp["Date"] >= window_start].sort_values("Date")
                    if len(window_rows) >= 2:
                        bench_return = (float(window_rows.iloc[-1]["Close"])
                                        - float(window_rows.iloc[0]["Close"])) / \
                                        float(window_rows.iloc[0]["Close"])
                except Exception as e:
                    print(f"[Attribution] benchmark fetch failed: {e}")

            if prev_state is not None and realised_returns:
                agent.record_attribution(
                    weights_prev=prev_state["weights"],
                    weights_curr=weights,
                    realised_returns=realised_returns,
                    benchmark_return=bench_return,
                    period_start=prev_state["period_end"],
                    period_end=str(prices_df["Date"].max()),
                )

            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                json.dump({
                    "decision_id": agent.last_decision_ctx.decision_id
                        if agent.last_decision_ctx else "no_ctx",
                    "period_end": str(prices_df["Date"].max()),
                    "weights": weights,
                }, f)
        except _SkipState:
            pass
        except Exception as e:
            print(f"[Attribution] [WARN] post-rebalance attribution failed: {e}")

        # ── Monitoring (WS5): persist the scored universe of a LIVE execution ──
        # Best-effort: a monitoring failure must never fail the run.
        if exec_status.get("status") == "executed":
            exec_status = {**exec_status, **_record_for_monitoring(agent, predictions, weights, prices_df)}

        # Machine-readable outcome + exit code for schedulers:
        # 0 ok (executed / dry-run), 2 halted, 3 partial (some orders failed), 1 error.
        if construction:
            exec_status = {**exec_status, "construction": construction.get("construction"),
                           "requested_strategy": construction.get("requested")}
            if construction.get("fallback_reason"):
                exec_status["fallback_reason"] = construction["fallback_reason"][:200]
            if construction.get("signal"):
                exec_status["signal"] = construction["signal"]
            if construction.get("signal_fallback_reason"):
                exec_status["signal_fallback_reason"] = construction["signal_fallback_reason"][:200]
        st =exec_status.get("status")
        print("RESULT_JSON: " + _json.dumps({k: v for k, v in exec_status.items() if k != "failures"}))
        if st == "halted":
            print("\nRun halted by the drawdown circuit breaker — no orders placed.")
            sys.exit(2)
        if st == "error":
            print(f"\n[CRITICAL ERROR] Order execution failed: {exec_status.get('error', exec_status)}", file=sys.stderr)
            sys.exit(1)
        if st == "executed" and exec_status.get("failed", 0) > 0:
            print(f"\n[WARN] {exec_status['failed']} orders failed: {exec_status.get('failures')}")
            sys.exit(3)
        print("\nBDA Production Trading Runner run completed successfully.")

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Bot execution terminated with failure: {e}", file=sys.stderr)
        sys.exit(1)



def run_with_signal_fallback(agent, args, universe, requested_tickers, score=None):
    """``score`` (default ``_score_and_construct``) with the loaded artefact.

    With the altall signal, ANY exception -- or an optimized_v3 request that
    did not end in an optimized_v3 book on the altall scores -- switches the
    agent to the production artefact and re-runs the whole production path
    (universe, data, inference, optimized_v3 -> long_only_v3) from scratch.
    Production-signal failures propagate as before."""
    score = score or _score_and_construct
    try:
        out = score(agent, args, universe)
        c = agent.last_construction or {}
        if (agent.signal == "altall" and args.strategy == "optimized_v3"
                and c.get("construction") != "optimized_v3"):
            raise RuntimeError(f"optimized_v3 not built on the altall signal "
                               f"({c.get('fallback_reason', 'unknown reason')})")
        return out
    except Exception as e:
        if agent.signal != "altall":
            raise
        agent.use_production(f"altall failed after load: {type(e).__name__}: {e}")
        config.TICKERS = list(requested_tickers)
        return score(agent, args, universe)


def _score_and_construct(agent, args, universe):
    """Universe expansion, regime, live bars, inference and target weights for
    the artefact ``agent`` has loaded. Returns (is_bull, prices_df, predictions, weights)."""
    # Full-universe expansion: after the model is loaded we know every
    # ticker the model can score, so we can now populate the trade list.
    if agent.feature_version >= 3:
        # v3 has no KG: every active common stock is a candidate; the PIT
        # eligibility rule is applied on the decision date in run_inference.
        # A --tickers-file (custom) list is honoured as given.
        from .bot import v3_live_universe
        if universe != "custom":
            config.TICKERS = v3_live_universe()
        print(f"[Filter] feature_version 3: {len(config.TICKERS)} candidate tickers "
              f"(no KG filter).")
    elif universe == "full":
        config.TICKERS = sorted(agent.company_embeddings.keys())
        print(f"[Filter] Full-universe mode: {len(config.TICKERS)} modelled tickers loaded.")
    if agent.feature_version < 3 and getattr(config, "EXCLUDE_NON_COMMON", True):
        from .bot import exclude_non_common
        config.TICKERS, _ = exclude_non_common(config.TICKERS)

    # Validate that loaded tickers have structural embeddings in best_model.pkl
    valid_tickers = ([t for t in config.TICKERS if t in agent.company_embeddings]
                     if agent.feature_version < 3 else list(config.TICKERS))
    if len(valid_tickers) != len(config.TICKERS):
        print(f"[Filter] Removed {len(config.TICKERS) - len(valid_tickers)} tickers lacking structural model embeddings.")
        config.TICKERS = valid_tickers
        print(f"[Filter] Active universe size: {len(config.TICKERS)} tickers.")

    if args.strategy == "optimized_v3":
        agent.risk_history_days = int(config.OPT_HISTORY_DAYS)

    # 2. Check S&P 500 trend regime
    is_bull = agent.check_market_regime(force_regime=args.force_regime,
                                        strategy=args.strategy)

    # 3. Download live pricing
    prices_df = agent.fetch_live_data()

    # 4. Run ensemble inference
    predictions = agent.run_inference(prices_df)

    print("\nModel Inference (30d Cross-Sectional Rank) & Kalman Betas:")
    print("-" * 75)
    print(f"{'Ticker':<8} | {'Current Close':<15} | {'Pred Rank (30d)':<20} | {'Kalman Beta'}")
    print("-" * 75)
    for _, row in predictions.iterrows():
        rank_val = row.get("pred_rank", row.get("pred_proba", float("nan")))
        print(f"{row['ticker']:<8} | ${row['company_close']:>14.2f} | "
              f"{rank_val * 100:>18.2f}% | {row['kalman_beta']:>11.2f}")
    print("-" * 75)

    # 4b. LIVE-ONLY: tilt predictions by real-time news sentiment
    #     (Alpaca news -> RDF news-event graph -> SPARQL -> per-ticker score).
    #     Never used in backtests or embeddings -- no PIT archive to do so safely.
    predictions = agent.apply_news_sentiment_tilt(predictions)

    # 5. Compute target weights
    weights = agent.calculate_target_weights(predictions, is_bull, strategy=args.strategy)
    return is_bull, prices_df, predictions, weights

class _SkipState(Exception):
    """Internal: skip attribution/state persistence (dry-run, halted, error)."""


def _record_for_monitoring(agent, predictions, weights, prices_df) -> dict:
    """Write ops/predictions/<account>/<decision_date>.csv.gz (+ feature drift).

    Returns fields to merge into RESULT_JSON; never raises."""
    out = {}
    try:
        from . import monitoring
        meta = getattr(agent, "model_metadata", None) or {}
        model_id = monitoring.model_id_from(meta, getattr(agent, "model_path", None) or config.MODEL_PATH)
        out["model_id"] = model_id
        account = os.getenv("ACTIVE_ACCOUNT", "weekly").lower()
        # decision date = last bar the model saw (see monitoring module docstring)
        decision_date = str(prices_df["Date"].max())[:10]
        preds = agent.last_predictions_df if getattr(agent, "last_predictions_df", None) is not None \
            else predictions
        path = monitoring.record_predictions(preds, weights, decision_date, account, model_id)
        out["predictions_file"] = os.path.relpath(path, monitoring.REPO_ROOT)
        print(f"[Monitor] recorded {len(preds)} predictions -> {out['predictions_file']}")
        try:
            profile = meta.get("feature_profile") or (agent.model_data or {}).get("feature_profile")
            drift = monitoring.feature_drift(agent.last_feature_matrix, profile)
            monitoring.record_drift(drift, decision_date, account)
            print(f"[Monitor] feature drift: " + (f"{drift['n_high']} features with PSI > "
                  f"{drift['threshold']}" if drift["available"] else drift["reason"]))
        except Exception as e:
            print(f"[Monitor] [WARN] drift not recorded: {e}")
    except Exception as e:
        print(f"[Monitor] [WARN] predictions not recorded: {e}")
    return out

if __name__ == "__main__":
    main()
