"""
Data Collector for US Exchange Rates

Collects historical exchange rates for a base currency (default USD) against
all available currencies over a configurable look-back window of business days.

The look-back window, base currency, and output path are parameterised so the
collector can be re-run for different horizons (e.g. an initial deep backfill
vs. a short periodic top-up) without editing the source. Defaults reproduce the
original behaviour (USD, 100 business days).

Usage:
    python exchange.py                      # USD, 100 business days (default)
    python exchange.py --days 250           # deeper backfill
    python exchange.py --base-currency EUR  # different base
Environment overrides (used when the matching CLI flag is absent):
    EXCHANGE_DAYS_TO_FETCH, EXCHANGE_BASE_CURRENCY, EXCHANGE_OUTPUT_PATH
"""

import os
import argparse

import pandas as pd
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "datasets"))
DEFAULT_OUTPUT_PATH = os.path.join(DATASETS_DIR, "US_exchange.csv")

# Defaults (overridable via CLI flags or environment variables)
DEFAULT_BASE_CURRENCY = "USD"
DEFAULT_DAYS_TO_FETCH = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect historical exchange rates for a base currency."
    )
    parser.add_argument(
        "--days", type=int,
        default=int(os.getenv("EXCHANGE_DAYS_TO_FETCH", DEFAULT_DAYS_TO_FETCH)),
        help=f"Look-back window in business days (default {DEFAULT_DAYS_TO_FETCH}).",
    )
    parser.add_argument(
        "--base-currency", type=str,
        default=os.getenv("EXCHANGE_BASE_CURRENCY", DEFAULT_BASE_CURRENCY),
        help=f"Base currency (default {DEFAULT_BASE_CURRENCY}).",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.getenv("EXCHANGE_OUTPUT_PATH", DEFAULT_OUTPUT_PATH),
        help="Destination CSV path.",
    )
    return parser.parse_args()


def collect_exchange_rates(base_currency: str, days_to_fetch: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = CurrencyRates()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_fetch)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    dataset_rows = []
    print(f"[INFO]: Downloading {days_to_fetch} business days of {base_currency} "
          f"rates... This might take a moment.")

    for single_date in date_range:
        try:
            rates = c.get_rates(base_currency, single_date)
            row = {"Date": single_date.strftime("%Y-%m-%d")}
            row.update(rates)
            dataset_rows.append(row)
        except Exception as e:
            print(f"[ERROR]: Error on {single_date.strftime('%Y-%m-%d')}: {e}")

    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        df.to_csv(output_path, index=False)
        print(f"\n[INFO]: Dataset saved successfully to {output_path}")
    else:
        print("\n[INFO]: No data was fetched. There might be an error.")


if __name__ == "__main__":
    args = parse_args()
    collect_exchange_rates(args.base_currency, args.days, args.output)
