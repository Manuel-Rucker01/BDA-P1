#!/usr/bin/env python3
"""
CLI Execution Script for the BDA Production Trading Bot.
Use this script to trigger weekly portfolio rebalancing.
"""

import os
import sys
import argparse

import pandas as pd

from .bot import BDATradingAgent
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
        choices=["high_confidence", "regime_filtered"],
        default="high_confidence",
        help="Strategy to use: 'high_confidence' (+304% return, Long-Only P>=0.53) or 'regime_filtered' (Long/Short, shorts disabled in bull markets)"
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
        help="Use the sector-diversified mid-cap stock basket (+59% return, lower volatility) instead of the highly explosive alphabetical basket (+304% return)"
    )
    parser.add_argument(
        "--recommend-only",
        action="store_true",
        default=False,
        help="Print the model predictions and optimal target portfolio recommendations, then exit without executing any orders or rebalancing simulations"
    )
    parser.add_argument(
        "--universe",
        choices=["high_alpha", "safe", "top_mcap", "custom"],
        default="high_alpha",
        help="Target stock universe: 'high_alpha' (default), 'safe' (sector diversified), 'top_mcap' (top market cap companies), or 'custom' (user file)"
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

    try:
        agent = BDATradingAgent()
        
        # 1. Load serializations
        agent.load_model()

        # Validate that loaded tickers have structural embeddings in best_model.pkl
        valid_tickers = [t for t in config.TICKERS if t in agent.company_embeddings]
        if len(valid_tickers) != len(config.TICKERS):
            print(f"[Filter] Removed {len(config.TICKERS) - len(valid_tickers)} tickers lacking structural model embeddings.")
            config.TICKERS = valid_tickers
            print(f"[Filter] Active universe size: {len(config.TICKERS)} tickers.")
        
        # 2. Check S&P 500 trend regime
        is_bull = agent.check_market_regime(force_regime=args.force_regime)
        
        # 3. Download live pricing
        prices_df = agent.fetch_live_data()
        
        # 4. Run ensemble inference
        predictions = agent.run_inference(prices_df)
        
        print("\nModel Inference Probabilities & Kalman Betas:")
        print("-" * 70)
        print(f"{'Ticker':<8} | {'Current Close':<15} | {'P(up) Probability':<20} | {'Kalman Beta'}")
        print("-" * 70)
        for _, row in predictions.iterrows():
            print(f"{row['ticker']:<8} | ${row['company_close']:>14.2f} | {row['pred_proba'] * 100:>18.2f}% | {row['kalman_beta']:>11.2f}")
        print("-" * 70)

        # 5. Compute target weights
        weights = agent.calculate_target_weights(predictions, is_bull, strategy=args.strategy)
        
        print("\nTarget Portfolio Weights Allocations:")
        print("-" * 35)
        print(f"{'Ticker':<8} | {'Target Weight Allocation %'}")
        print("-" * 35)
        for t, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(w) > 0:
                side = "LONG" if w > 0 else "SHORT"
                print(f"{t:<8} | {w * 100:>23.2f}% ({side})")
        print("-" * 35)

        # 6. Execute differential rebalance
        if args.recommend_only:
            print("\n[Recommend Only] Exit successfully without executing any portfolio rebalancing.")
            print("=" * 80)
            return
            
        agent.execute_alpaca_rebalance(weights, prices_df=prices_df, dry_run=is_dry_run)

        # ── Performance attribution against the previous rebalance ────────
        # We persist the most-recent decision's target weights to a small
        # state file so the next invocation can attribute realised returns
        # between rebalances.
        try:
            import json
            import yfinance as yf

            state_path = os.path.join(
                os.path.dirname(__file__), "agent_logs", "last_weights.json"
            )
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
                    sp = yf.download(config.SP500_INDEX, period="3mo", progress=False)
                    if isinstance(sp.columns, pd.MultiIndex):
                        sp.columns = [c[0] for c in sp.columns]
                    sp = sp.reset_index()
                    sp["Date"] = pd.to_datetime(sp["Date"]).dt.strftime("%Y-%m-%d")
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
        except Exception as e:
            print(f"[Attribution] [WARN] post-rebalance attribution failed: {e}")

        print("\nBDA Production Trading Runner run completed successfully.")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Bot execution terminated with failure: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
