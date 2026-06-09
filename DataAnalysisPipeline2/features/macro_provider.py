"""
Shared macroeconomic feature provider.

Current policy
--------------
The provider deliberately reads the static Turtle graph produced by
ExploitationZone/geopolitical_macroeconomic.py. These values are not
point-in-time releases; they are a fixed snapshot used to keep training,
backtests, and live inference on the same feature contract without network
access or retraining. A future PIT implementation should preserve the columns
below and swap the data source behind get_macro_features_for_date().
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Optional

import pandas as pd


MACRO_FEATURE_COLUMNS = [
    "country",
    "gdp_usd",
    "gdp_growth_pct",
    "inflation_pct",
    "trade_pct",
    "interest_rate_pct",
]

STATIC_MACRO_POLICY = {
    "source": "static_turtle_snapshot",
    "point_in_time": False,
    "network_required": False,
    "caveat": (
        "Static macroeconomic_graph.ttl values are reused for every as-of date "
        "to preserve feature consistency across training, backtests, and live inference."
    ),
}


def _empty_macro_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=MACRO_FEATURE_COLUMNS)
    df.attrs["macro_policy"] = STATIC_MACRO_POLICY.copy()
    return df


@lru_cache(maxsize=8)
def _load_static_macro_features_cached(macro_ttl_path: str, mtime_ns: int) -> pd.DataFrame:
    from rdflib import Graph as RdfGraph, Namespace

    graph = RdfGraph()
    graph.parse(macro_ttl_path, format="turtle")
    macro_onto = Namespace("http://bda.upc.edu/macro/ontology#")
    macro_ent = Namespace("http://bda.upc.edu/macro/resource/")

    rows = []
    for subject in set(graph.subjects()):
        if not str(subject).startswith(str(macro_ent)):
            continue

        country = str(subject).replace(str(macro_ent), "").replace("_", " ")
        gdp = graph.value(subject, macro_onto.gdpUSD)
        growth = graph.value(subject, macro_onto.gdpGrowthPercent)
        inflation = graph.value(subject, macro_onto.inflationPercent)
        trade = graph.value(subject, macro_onto.tradePercentOfGDP)
        interest = graph.value(subject, macro_onto.interestRatePercent)

        if gdp is not None or growth is not None or inflation is not None or trade is not None or interest is not None:
            rows.append({
                "country": country,
                "gdp_usd": float(gdp) if gdp is not None else None,
                "gdp_growth_pct": float(growth) if growth is not None else None,
                "inflation_pct": float(inflation) if inflation is not None else None,
                "trade_pct": float(trade) if trade is not None else None,
                "interest_rate_pct": float(interest) if interest is not None else None,
            })

    df = pd.DataFrame(rows, columns=MACRO_FEATURE_COLUMNS)
    df.attrs["macro_policy"] = STATIC_MACRO_POLICY.copy()
    df.attrs["macro_ttl_path"] = macro_ttl_path
    df.attrs["macro_ttl_mtime_ns"] = mtime_ns
    return df


def load_static_macro_features(macro_ttl_path: str) -> pd.DataFrame:
    """
    Load static country-level macro features from the Turtle graph.

    The returned DataFrame always uses the legacy feature columns expected by
    existing models. Missing files or parse failures return an empty frame so
    callers can continue to apply their established fallback fills.
    """
    if not macro_ttl_path or not os.path.exists(macro_ttl_path):
        print(f"[WARNING] Macroeconomic Turtle graph not found at {macro_ttl_path}. Using empty macro features.")
        return _empty_macro_frame()

    try:
        stat = os.stat(macro_ttl_path)
        return _load_static_macro_features_cached(os.path.abspath(macro_ttl_path), stat.st_mtime_ns).copy()
    except Exception as exc:
        print(f"[WARNING] Failed to parse RDF Macro TTL graph: {exc}. Using empty macro features.")
        return _empty_macro_frame()


def _normalise_filter(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    return {str(value) for value in values if value is not None}


def get_macro_features_for_date(
    as_of_date,
    macro_ttl_path: str,
    tickers: Optional[Iterable[str]] = None,
    countries: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return macro features available for an as-of date.

    Today this is intentionally date-invariant because the only offline source
    is the static TTL snapshot. The as_of_date and tickers arguments are kept in
    the API so a PIT provider can later be introduced without changing training,
    backtest, or live-call sites.
    """
    del as_of_date, tickers
    df = load_static_macro_features(macro_ttl_path)
    country_filter = _normalise_filter(countries)
    if country_filter is not None:
        df = df[df["country"].isin(country_filter)].copy()
        df.attrs["macro_policy"] = STATIC_MACRO_POLICY.copy()
    return df
