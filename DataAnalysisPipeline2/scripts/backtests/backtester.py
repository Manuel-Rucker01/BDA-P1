import os
import numpy as np
import pandas as pd

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
PRED_PATH = os.path.join(EXPLOITATION_DIR, "test_predictions.parquet")
OUTPUT_PLOT_PATH = os.path.join(EXPLOITATION_DIR, "backtest_results.png")

class PortfolioBacktester:
    """
    Quantitative Event-Driven Backtesting Engine.
    Simulates portfolio rebalancing every 5 trading days under a 0.1% (10 bps) transaction fee constraint.
    """
    def __init__(self, df_preds, transaction_fee=0.001, initial_capital=100000.0):
        self.df = df_preds.sort_values(["Date", "ticker"]).reset_index(drop=True)
        self.transaction_fee = transaction_fee
        self.initial_capital = initial_capital
        
        # Resolve rebalancing dates (every 5 trading days)
        self.unique_dates = sorted(self.df["Date"].unique())
        self.rebalance_dates = [self.unique_dates[i] for i in range(0, len(self.unique_dates), 5)]
        print(f"Backtesting over {len(self.unique_dates)} trading days ({len(self.rebalance_dates)} rebalancing periods)")

    def run_backtest(self):
        results = {}
        
        # 1. Buy and Hold Benchmark
        results["Buy & Hold"] = self._simulate_buy_and_hold()
        
        # 2. Long-Only Decile Strategy
        results["Long-Only Decile"] = self._simulate_long_only()
        
        # 3. Long-Short Decile Spread
        results["Long-Short Decile"] = self._simulate_long_short()
        
        # 4. Probabilistic Weighted Strategy
        results["Probabilistic Weighted"] = self._simulate_weighted()
        
        return results

    def _get_period_returns(self, date_idx, selected_tickers, is_short=False):
        """
        Calculates the 5-day return for a set of selected tickers from date_idx to date_idx+5.
        """
        current_date = self.rebalance_dates[date_idx]
        
        # Determine the next rebalance date
        if date_idx + 1 < len(self.rebalance_dates):
            next_date = self.rebalance_dates[date_idx + 1]
        else:
            next_date = self.unique_dates[-1]
            
        if current_date == next_date:
            return {}

        # Fetch close prices on current date
        curr_prices = self.df[self.df["Date"] == current_date].set_index("ticker")["company_close"].to_dict()
        # Fetch close prices on next date
        next_prices = self.df[self.df["Date"] == next_date].set_index("ticker")["company_close"].to_dict()
        
        returns = {}
        for ticker in selected_tickers:
            if ticker in curr_prices and ticker in next_prices:
                p0 = curr_prices[ticker]
                p1 = next_prices[ticker]
                if p0 > 0:
                    ret = (p1 - p0) / p0
                    # If shorting, return is inverted
                    returns[ticker] = -ret if is_short else ret
        return returns

    def _simulate_buy_and_hold(self):
        """
        Buy all available stocks on the first day, and hold until the end.
        """
        portfolio_values = [self.initial_capital]
        period_returns = []
        
        # First day: Buy equal-weighted portfolio of all active stocks
        first_date = self.rebalance_dates[0]
        first_df = self.df[self.df["Date"] == first_date]
        tickers = first_df["ticker"].tolist()
        num_stocks = len(tickers)
        
        if num_stocks == 0:
            return {"values": [self.initial_capital] * len(self.rebalance_dates), "returns": [0.0] * (len(self.rebalance_dates)-1)}
            
        # Pay 0.1% entry fee
        capital = self.initial_capital * (1.0 - self.transaction_fee)
        cash_per_stock = capital / num_stocks
        
        # Track shares bought
        shares = {}
        curr_prices = first_df.set_index("ticker")["company_close"].to_dict()
        for t in tickers:
            shares[t] = cash_per_stock / curr_prices[t]
            
        # Track portfolio value at each rebalancing date
        for i in range(1, len(self.rebalance_dates)):
            date = self.rebalance_dates[i]
            prices = self.df[self.df["Date"] == date].set_index("ticker")["company_close"].to_dict()
            
            value = 0.0
            for t, sh in shares.items():
                if t in prices:
                    value += sh * prices[t]
                else:
                    # Fallback to initial buy price if delisted/missing
                    value += sh * curr_prices[t]
            
            # If it's the final period, liquidate and pay exit fee
            if i == len(self.rebalance_dates) - 1:
                value = value * (1.0 - self.transaction_fee)
                
            portfolio_values.append(value)
            ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            period_returns.append(ret)
            
        return {"values": portfolio_values, "returns": period_returns}

    def _simulate_long_only(self):
        """
        Rebalance every 5 days: Buy top decile predicted probability of going up.
        """
        portfolio_values = [self.initial_capital]
        period_returns = []
        capital = self.initial_capital
        
        for i in range(len(self.rebalance_dates) - 1):
            date = self.rebalance_dates[i]
            day_df = self.df[self.df["Date"] == date].sort_values("pred_proba", ascending=False)
            
            if len(day_df) == 0:
                portfolio_values.append(capital)
                period_returns.append(0.0)
                continue
                
            # Select top decile (at least 1 stock)
            n_select = max(1, int(len(day_df) * 0.1))
            long_tickers = day_df.head(n_select)["ticker"].tolist()
            
            # Pay 0.1% transaction fee on entry
            trade_capital = capital * (1.0 - self.transaction_fee)
            
            # Get returns for these stocks
            rets = self._get_period_returns(i, long_tickers)
            if len(rets) == 0:
                avg_ret = 0.0
            else:
                avg_ret = np.mean(list(rets.values()))
                
            # Exit position: pay 0.1% transaction fee on exit value
            capital = trade_capital * (1.0 + avg_ret) * (1.0 - self.transaction_fee)
            portfolio_values.append(capital)
            
            period_ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            period_returns.append(period_ret)
            
        return {"values": portfolio_values, "returns": period_returns}

    def _simulate_long_short(self):
        """
        Rebalance every 5 days: Long top decile, Short bottom decile.
        Cash neutral allocation: 50% capital long, 50% capital short (fully collateralized).
        """
        portfolio_values = [self.initial_capital]
        period_returns = []
        capital = self.initial_capital
        
        for i in range(len(self.rebalance_dates) - 1):
            date = self.rebalance_dates[i]
            day_df = self.df[self.df["Date"] == date].sort_values("pred_proba", ascending=False)
            
            if len(day_df) < 2:
                portfolio_values.append(capital)
                period_returns.append(0.0)
                continue
                
            n_select = max(1, int(len(day_df) * 0.1))
            long_tickers = day_df.head(n_select)["ticker"].tolist()
            short_tickers = day_df.tail(n_select)["ticker"].tolist()
            
            # Allocate 50% to long, 50% to short
            # Pay 0.1% transaction fee on entry for both sides
            long_capital = 0.5 * capital * (1.0 - self.transaction_fee)
            short_capital = 0.5 * capital * (1.0 - self.transaction_fee)
            
            # Get returns
            long_rets = self._get_period_returns(i, long_tickers, is_short=False)
            short_rets = self._get_period_returns(i, short_tickers, is_short=True)
            
            avg_long_ret = np.mean(list(long_rets.values())) if len(long_rets) > 0 else 0.0
            avg_short_ret = np.mean(list(short_rets.values())) if len(short_rets) > 0 else 0.0
            
            # Liquidate positions and pay 0.1% transaction fee on exit
            long_final = long_capital * (1.0 + avg_long_ret) * (1.0 - self.transaction_fee)
            short_final = short_capital * (1.0 + avg_short_ret) * (1.0 - self.transaction_fee)
            
            capital = long_final + short_final
            portfolio_values.append(capital)
            
            period_ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            period_returns.append(period_ret)
            
        return {"values": portfolio_values, "returns": period_returns}

    def _simulate_weighted(self):
        """
        Rebalance every 5 days: Allocate weight proportional to (probability - 0.5).
        Long if prob >= 0.55, Short if prob <= 0.45.
        """
        portfolio_values = [self.initial_capital]
        period_returns = []
        capital = self.initial_capital
        
        for i in range(len(self.rebalance_dates) - 1):
            date = self.rebalance_dates[i]
            day_df = self.df[self.df["Date"] == date].copy()
            
            if len(day_df) == 0:
                portfolio_values.append(capital)
                period_returns.append(0.0)
                continue
                
            # Calculate raw weights
            day_df["raw_weight"] = day_df["pred_proba"] - 0.5
            
            # Filter significant signals
            longs = day_df[day_df["raw_weight"] >= 0.05]
            shorts = day_df[day_df["raw_weight"] <= -0.05]
            
            if len(longs) == 0 and len(shorts) == 0:
                # No signals, keep cash
                portfolio_values.append(capital)
                period_returns.append(0.0)
                continue
                
            # Combine
            selected = pd.concat([longs, shorts])
            total_abs = selected["raw_weight"].abs().sum()
            
            if total_abs == 0:
                portfolio_values.append(capital)
                period_returns.append(0.0)
                continue
                
            # Normalize weights to sum of absolute weights = 1.0 (cash neutral / 100% exposure)
            selected["weight"] = selected["raw_weight"] / total_abs
            
            # Fetch returns for selected tickers
            # Note: _get_period_returns handles shorting manually if we pass is_short,
            # but since we have positive weights for longs and negative weights for shorts,
            # we can just fetch standard returns and multiply by the signed weight!
            long_tickers = longs["ticker"].tolist()
            short_tickers = shorts["ticker"].tolist()
            
            long_rets = self._get_period_returns(i, long_tickers, is_short=False)
            short_rets = self._get_period_returns(i, short_tickers, is_short=False)
            
            # Combine returns dictionary
            all_rets = {**long_rets, **short_rets}
            
            weighted_ret = 0.0
            actual_weight_sum = 0.0
            
            for _, row in selected.iterrows():
                ticker = row["ticker"]
                w = row["weight"]
                if ticker in all_rets:
                    weighted_ret += w * all_rets[ticker]
                    actual_weight_sum += abs(w)
                    
            # Pay transaction fees proportional to exposure (100% exposure rebalanced)
            trade_capital = capital * (1.0 - self.transaction_fee)
            
            # Liquidate positions and pay exit transaction fee
            capital = trade_capital * (1.0 + weighted_ret) * (1.0 - self.transaction_fee)
            portfolio_values.append(capital)
            
            period_ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            period_returns.append(period_ret)
            
        return {"values": portfolio_values, "returns": period_returns}


def calculate_metrics(values, returns):
    values = np.array(values)
    returns = np.array(returns)
    
    cum_return = (values[-1] - values[0]) / values[0] * 100.0
    
    # Sharpe Ratio: 252 trading days = 50.4 periods of 5 days
    ann_factor = 50.4
    mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
    std_ret = np.std(returns) if len(returns) > 0 else 1.0
    if std_ret > 0:
        sharpe = np.sqrt(ann_factor) * (mean_ret / std_ret)
    else:
        sharpe = 0.0
        
    # Max Drawdown
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks * 100.0
    max_drawdown = np.min(drawdowns)
    
    # Win Rate
    wins = np.sum(returns > 0)
    win_rate = (wins / len(returns) * 100.0) if len(returns) > 0 else 0.0
    
    # Profit Factor
    gains = np.sum(returns[returns > 0])
    losses = np.sum(returns[returns < 0])
    profit_factor = (gains / abs(losses)) if losses != 0 else float("inf")
    
    return {
        "cum_return": cum_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "final_value": values[-1]
    }


def main():
    print("=" * 80)
    print("BDA QUANTITATIVE PORTFOLIO BACKTESTING ENGINE")
    print("=" * 80)
    
    if not os.path.exists(PRED_PATH):
        print(f"[ERROR] Test predictions file not found at: {PRED_PATH}")
        print("Please run `kg_embeddings_classifier.py` first to generate predictions.")
        return
        
    df_preds = pd.read_parquet(PRED_PATH)
    backtester = PortfolioBacktester(df_preds)
    results = backtester.run_backtest()
    
    metrics = {}
    print("\n" + "=" * 80)
    print(f"{'Strategy Name':<28} | {'Final Value':<12} | {'Return (%)':<10} | {'Sharpe':<6} | {'Max DD (%)':<10} | {'Win Rate (%)':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        m = calculate_metrics(res["values"], res["returns"])
        metrics[name] = m
        print(f"{name:<28} | ${m['final_value']:<11.2f} | {m['cum_return']:>9.2f}% | {m['sharpe']:>6.3f} | {m['max_drawdown']:>9.2f}% | {m['win_rate']:>9.2f}%")
        
    print("=" * 80)
    
    # Generate Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Use a premium aesthetic dark theme for chart
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        
        # Harmonies palettes
        colors = {
            "Buy & Hold": "#7f8c8d",          # Grey
            "Long-Only Decile": "#2ecc71",    # Emerald green
            "Long-Short Decile": "#e74c3c",   # Coral red
            "Probabilistic Weighted": "#3498db" # Sleek blue
        }
        
        dates = pd.to_datetime(backtester.rebalance_dates)
        
        for name, res in results.items():
            # Normalized to 100 base index
            cum_perf = np.array(res["values"]) / backtester.initial_capital * 100.0
            plt.plot(dates, cum_perf, label=f"{name} (Return: {metrics[name]['cum_return']:.1f}%, Sharpe: {metrics[name]['sharpe']:.2f})", 
                     color=colors.get(name, "#000000"), linewidth=2)
            
        plt.title("Quantitative Portfolio Strategies Performance Comparison", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Rebalancing Date", fontsize=12)
        plt.ylabel("Portfolio Performance (Base 100)", fontsize=12)
        plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()
        
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
        print(f"\nSUCCESS: Performance chart saved to {OUTPUT_PLOT_PATH}")
    except Exception as e:
        print(f"\n[WARNING] Could not generate matplotlib plot: {e}")

if __name__ == "__main__":
    main()
