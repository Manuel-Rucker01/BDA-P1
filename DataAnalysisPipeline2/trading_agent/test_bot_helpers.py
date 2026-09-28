import math

import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot


def test_inverse_volatility_weights_sum_and_respect_cap():
    weights = bot.inverse_volatility_weights(
        ["LOW", "MID", "HIGH"],
        [0.05, 0.10, 0.20],
        target_exposure=1.0,
        max_weight=0.50,
    )

    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-12)
    assert max(weights.values()) <= 0.50 + 1e-12
    assert weights["LOW"] > weights["MID"] > weights["HIGH"]


def test_inverse_volatility_weights_leaves_cash_when_cap_infeasible():
    weights = bot.inverse_volatility_weights(
        ["A", "B"],
        [0.01, 0.02],
        target_exposure=1.0,
        max_weight=0.25,
    )

    assert weights == {"A": pytest.approx(0.25), "B": pytest.approx(0.25)}
    assert sum(weights.values()) == pytest.approx(0.50)


def test_select_top_k_with_sector_cap_prefers_ranked_sector_breadth():
    candidates = pd.DataFrame(
        {
            "ticker": ["T1", "T2", "T3", "T4", "F1", "F2"],
            "Sector": ["Tech", "Tech", "Tech", "Tech", "Financials", "Financials"],
            "pred_rank": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
        }
    )

    selected = bot.select_top_k_with_sector_cap(candidates, top_k=4, max_sector_frac=0.50)

    assert selected["ticker"].tolist() == ["T1", "T2", "F1", "F2"]
    assert selected["Sector"].value_counts().max() == 2


def test_load_macro_features_missing_and_empty_file(tmp_path):
    missing = bot.load_macro_features(str(tmp_path / "missing.ttl"))
    assert list(missing.columns) == [
        "country",
        "gdp_usd",
        "gdp_growth_pct",
        "inflation_pct",
        "trade_pct",
        "interest_rate_pct",
    ]
    assert missing.empty

    empty_path = tmp_path / "empty.ttl"
    empty_path.write_text("", encoding="utf-8")
    empty = bot.load_macro_features(str(empty_path))
    assert empty.empty


def test_load_macro_features_parses_partial_country_metrics(tmp_path):
    macro_path = tmp_path / "macro.ttl"
    macro_path.write_text(
        """
        @prefix ent: <http://bda.upc.edu/macro/resource/> .
        @prefix onto: <http://bda.upc.edu/macro/ontology#> .

        ent:United_States onto:gdpUSD 27000000000000.0 ;
            onto:gdpGrowthPercent 2.5 ;
            onto:inflationPercent 3.1 ;
            onto:tradePercentOfGDP 25.0 ;
            onto:interestRatePercent 5.33 .
        ent:Region_Americas onto:gdpUSD 1.0 .
        """,
        encoding="utf-8",
    )

    parsed = bot.load_macro_features(str(macro_path)).set_index("country")

    assert parsed.loc["United States", "gdp_usd"] == pytest.approx(27_000_000_000_000.0)
    assert parsed.loc["United States", "interest_rate_pct"] == pytest.approx(5.33)
    assert parsed.loc["Region Americas", "gdp_usd"] == pytest.approx(1.0)


def test_top_k_target_weights_respect_exposure_and_position_caps(monkeypatch):
    monkeypatch.setattr(bot.config, "TICKERS", ["A", "B", "C", "D"])
    monkeypatch.setattr(bot.config, "TOP_PCT_THRESHOLD", 100.0)
    monkeypatch.setattr(bot.config, "TOP_K_HOLDINGS", 3)
    monkeypatch.setattr(bot.config, "TARGET_EXPOSURE", 1.0)
    monkeypatch.setattr(bot.config, "MAX_POSITION_WEIGHT", 0.40)
    monkeypatch.setattr(bot.config, "VOL_FLOOR", 1e-3)
    monkeypatch.setattr(bot.config, "MAX_SECTOR_WEIGHT", 1.0)
    monkeypatch.setattr(bot.config, "WEIGHTING_SCHEME", "inverse_vol")

    predictions = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "pred_rank": [0.99, 0.98, 0.97, 0.96],
            "return_volatility_20d": [0.01, 0.02, 0.04, 0.08],
        }
    )

    agent = object.__new__(bot.BDATradingAgent)
    weights = agent.calculate_target_weights(predictions, is_bull=True, strategy="top_k")
    active_weights = [w for w in weights.values() if w > 0]

    assert len(active_weights) == 3
    assert sum(active_weights) == pytest.approx(1.0)
    assert max(active_weights) <= 0.40 + 1e-12


def test_regime_filtered_exposure_suppresses_shorts_in_bull(monkeypatch):
    monkeypatch.setattr(bot.config, "TICKERS", ["LONG", "SHORT"])
    monkeypatch.setattr(bot.config, "TARGET_EXPOSURE", 1.0)
    monkeypatch.setattr(bot.config, "ALPACA_CHECK_BORROWABILITY", False)
    monkeypatch.setattr(bot.config, "MAX_GROSS_EXPOSURE", 1.0)
    monkeypatch.setattr(bot.config, "MAX_NET_EXPOSURE", 1.0)
    monkeypatch.setattr(bot.config, "MAX_SHORT_EXPOSURE", 0.30)
    monkeypatch.setattr(bot.config, "MAX_LONG_EXPOSURE", 1.0)
    monkeypatch.setattr(bot.config, "MAX_POSITION_WEIGHT", 0.25)

    predictions = pd.DataFrame(
        {
            "ticker": ["LONG", "SHORT"],
            "pred_rank": [0.80, 0.20],
            "kalman_beta": [1.0, 1.0],
        }
    )

    agent = object.__new__(bot.BDATradingAgent)
    bull_weights = agent.calculate_target_weights(
        predictions, is_bull=True, strategy="regime_filtered"
    )
    bear_weights = agent.calculate_target_weights(
        predictions, is_bull=False, strategy="regime_filtered"
    )

    assert bull_weights["LONG"] == pytest.approx(0.25)
    assert bull_weights["SHORT"] == 0.0
    assert bear_weights["LONG"] == pytest.approx(0.25)
    assert bear_weights["SHORT"] == pytest.approx(-0.25)
    assert sum(abs(w) for w in bear_weights.values()) <= 1.0
    assert max(abs(w) for w in bear_weights.values()) <= 0.25 + 1e-12


def test_build_regime_filtered_weights_enforces_side_and_gross_limits():
    predictions = pd.DataFrame(
        {
            "ticker": ["L1", "L2", "S1", "S2"],
            "pred_rank": [0.90, 0.80, 0.10, 0.20],
            "kalman_beta": [1.0, 1.0, 1.0, 1.0],
        }
    )

    weights, exposure = bot.build_regime_filtered_weights(
        predictions,
        is_bull=False,
        target_exposure=1.0,
        max_gross=0.80,
        max_net=0.60,
        max_short=0.20,
        max_long=0.70,
        max_position=0.40,
    )

    assert exposure["gross"] <= 0.80 + 1e-12
    assert abs(exposure["net"]) <= 0.60 + 1e-12
    assert exposure["short"] <= 0.20 + 1e-12
    assert exposure["long"] <= 0.70 + 1e-12
    assert max(abs(w) for w in weights.values()) <= 0.40 + 1e-12
