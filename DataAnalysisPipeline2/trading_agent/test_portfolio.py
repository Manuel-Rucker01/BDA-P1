"""Tests for the v3 portfolio books (trading_agent/portfolio.py) and their
wiring into calculate_target_weights. Pure; no network."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot, config, portfolio
from DataAnalysisPipeline2.trading_agent.portfolio import (
    capped_proportional, long_only_book, long_short_book,
)


def _universe(n=200, sectors=("Tech", "Health", "Fin", "Energy", "Util"), seed=0,
              semis_top=0):
    """Ranked universe; the first ``semis_top`` names (best ranks) all share one
    sector, reproducing the all-semiconductor top of the live book."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sec = "Semis" if i < semis_top else sectors[i % len(sectors)]
        rows.append({"ticker": f"T{i:03d}", "pred_rank": 1.0 - i / n, "Sector": sec,
                     "company_close": 10.0 + i,
                     "return_volatility_20d": float(rng.uniform(0.01, 0.05)),
                     "kalman_beta": float(rng.uniform(0.5, 1.5))})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # order must not matter


def _side_sector_counts(weights, df):
    sec = df.set_index("ticker")["Sector"]
    out = {}
    for t, w in weights.items():
        key = ("L" if w > 0 else "S", sec[t])
        out[key] = out.get(key, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Water-filling
# --------------------------------------------------------------------------- #
def test_capped_proportional_waterfills_and_respects_caps():
    w = capped_proportional([10, 1, 1, 1], budget=1.0, cap=0.4)
    assert w.max() <= 0.4 + 1e-12
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.4) and w[1] == pytest.approx(0.2)


def test_capped_proportional_infeasible_budget_leaves_cash():
    w = capped_proportional([1, 2, 3], budget=1.0, cap=0.1)
    assert np.allclose(w, 0.1) and w.sum() == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# long_only_book
# --------------------------------------------------------------------------- #
def test_long_only_strict_sector_cap_is_never_relaxed():
    df = _universe(n=60, semis_top=20)        # top 20 names are all one sector
    w, diag = long_only_book(df, k=30, max_per_sector=5, max_weight=0.06, target_exposure=1.0)
    counts = _side_sector_counts(w, df)
    assert max(counts.values()) <= 5
    assert counts[("L", "Semis")] == 5          # the 5 best semis, not 20
    assert len(w) == 30


def test_long_only_holds_fewer_when_cap_binds_and_says_so():
    df = _universe(n=40, sectors=("Tech", "Health"))   # 2 sectors x 5 = 10 max
    w, diag = long_only_book(df, k=30, max_per_sector=5, max_weight=0.06, target_exposure=1.0)
    assert len(w) == 10 and diag["n_held"] == 10
    assert any("holding fewer" in n for n in diag["notes"])
    # 10 x 6% cap = 60%: the rest must stay in cash, never breach the cap.
    assert sum(w.values()) == pytest.approx(0.60)
    assert diag["cash"] == pytest.approx(0.40)


def test_long_only_walks_full_ranking_in_rank_order():
    df = _universe(n=100)
    w, _ = long_only_book(df, k=10, max_per_sector=100, max_weight=0.2, target_exposure=1.0)
    best10 = df.sort_values("pred_rank", ascending=False)["ticker"].head(10)
    assert set(w) == set(best10)


@pytest.mark.parametrize("target", [1.0, 0.9, 0.5])
def test_long_only_exposure_and_per_name_caps(target):
    df = _universe(n=300)
    w, diag = long_only_book(df, k=30, max_per_sector=7, max_weight=0.06, target_exposure=target)
    assert sum(w.values()) <= target + 1e-9
    assert sum(w.values()) == pytest.approx(target)
    assert max(w.values()) <= 0.06 + 1e-12
    assert min(w.values()) > 0
    # inverse vol: lower vol -> weight at least as large (among uncapped names)
    vol = df.set_index("ticker")["return_volatility_20d"]
    free = {t: x for t, x in w.items() if x < 0.06 - 1e-9}
    ts = sorted(free, key=lambda t: vol[t])
    assert all(free[a] >= free[b] - 1e-12 for a, b in zip(ts, ts[1:]))


def test_long_only_skips_unpriced_and_nan_ranked_names():
    df = _universe(n=20)
    best = df.sort_values("pred_rank", ascending=False)["ticker"].iloc[0]
    df.loc[df["ticker"] == best, "company_close"] = np.nan
    df.loc[df["ticker"] == "T019", "pred_rank"] = np.nan
    w, _ = long_only_book(df, k=5, max_per_sector=5, max_weight=0.3, target_exposure=1.0)
    assert best not in w and "T019" not in w


# --------------------------------------------------------------------------- #
# long_short_book
# --------------------------------------------------------------------------- #
def test_long_short_dollar_neutral_gross_and_caps():
    df = _universe(n=400, semis_top=30)
    w, diag = long_short_book(df, n_long=40, n_short=40, gross=1.6, max_per_sector=8,
                              max_weight=0.04)
    longs = sum(x for x in w.values() if x > 0)
    shorts = -sum(x for x in w.values() if x < 0)
    assert longs == pytest.approx(0.8) and shorts == pytest.approx(0.8)
    assert diag["gross"] == pytest.approx(1.6) and abs(diag["net"]) < 1e-9
    assert max(abs(x) for x in w.values()) <= 0.04 + 1e-12
    assert max(_side_sector_counts(w, df).values()) <= 8
    assert diag["beta_net"] is not None and set(diag["sector_net"])
    # longs come from the top of the ranking, shorts from the bottom
    rank = df.set_index("ticker")["pred_rank"]
    assert min(rank[t] for t, x in w.items() if x > 0) > max(rank[t] for t, x in w.items() if x < 0)


def test_long_short_only_shorts_shortable_names():
    df = _universe(n=200)
    worst = df.sort_values("pred_rank")["ticker"].tolist()
    shortable = set(worst[10:60])               # the 10 worst names are not borrowable
    w, diag = long_short_book(df, n_long=20, n_short=20, gross=1.0, max_per_sector=10,
                              max_weight=0.1, shortable=shortable)
    shorts = {t for t, x in w.items() if x < 0}
    assert shorts and shorts <= shortable
    assert not shorts & set(worst[:10])
    assert diag["rejected_not_shortable"] == 10


def test_long_short_no_shortable_names_means_no_book():
    df = _universe(n=50)
    w, diag = long_short_book(df, 10, 10, 1.0, 5, 0.1, shortable=set())
    assert w == {} and diag["gross"] == 0.0


def test_long_short_cap_shortfall_keeps_dollar_neutrality():
    df = _universe(n=100)
    w, diag = long_short_book(df, n_long=10, n_short=3, gross=1.0, max_per_sector=10,
                              max_weight=0.1)
    longs = sum(x for x in w.values() if x > 0)
    shorts = -sum(x for x in w.values() if x < 0)
    assert shorts == pytest.approx(0.3) and longs == pytest.approx(0.3)   # 3 x 10% cap
    assert any("limit each leg" in n for n in diag["notes"])


def test_long_short_sector_neutral_matches_sector_exposure():
    df = _universe(n=300, semis_top=15)          # Semis only on the long side
    w, diag = long_short_book(df, n_long=40, n_short=40, gross=1.6, max_per_sector=8,
                              max_weight=0.04, sector_neutral=True)
    assert diag["sector_neutral"]
    for sec, net in diag["sector_net"].items():
        assert abs(net) < 1e-9, (sec, net)
    sec = df.set_index("ticker")["Sector"]
    assert not any(sec[t] == "Semis" for t in w)            # one-sided sector dropped
    assert diag["dropped_one_sided"]
    assert abs(diag["net"]) < 1e-9
    assert diag["gross"] == pytest.approx(1.6)
    assert max(abs(x) for x in w.values()) <= 0.04 + 1e-12


def test_long_short_non_neutral_can_carry_sector_net():
    df = _universe(n=300, semis_top=15)
    _, diag = long_short_book(df, 40, 40, 1.6, 8, 0.04, sector_neutral=False)
    assert diag["sector_net"].get("Semis", 0.0) > 0


def test_invariants_flag_sector_cap_breach():
    w = {"A": 0.1, "B": 0.1, "C": 0.1}
    with pytest.raises(AssertionError, match="sector"):
        portfolio.assert_portfolio_invariants(
            w, max_gross=1, max_net=1, max_short=0, max_long=1, max_position=0.2,
            sectors={"A": "X", "B": "X", "C": "X"}, max_names_per_sector=2)


# --------------------------------------------------------------------------- #
# Wiring into calculate_target_weights
# --------------------------------------------------------------------------- #
@pytest.fixture
def v3_config(monkeypatch):
    df = _universe(n=300, semis_top=25)
    monkeypatch.setattr(config, "TICKERS", df["ticker"].tolist())
    for k, v in dict(V3_K=30, V3_MAX_PER_SECTOR=5, V3_MAX_WEIGHT=0.06, V3_TARGET_EXPOSURE=1.0,
                     LS_N_LONG=40, LS_N_SHORT=40, LS_GROSS=1.6, LS_MAX_WEIGHT=0.04,
                     LS_MAX_PER_SECTOR=8, LS_SECTOR_NEUTRAL=True,
                     ALPACA_API_KEY="", ALPACA_SECRET_KEY="").items():
        monkeypatch.setattr(config, k, v, raising=False)
    return df


def test_calculate_target_weights_long_only_v3(v3_config):
    agent = object.__new__(bot.BDATradingAgent)
    w = agent.calculate_target_weights(v3_config, is_bull=None, strategy="long_only_v3")
    held = {t: x for t, x in w.items() if x}
    assert len(held) == 30 and sum(held.values()) == pytest.approx(1.0)
    assert max(_side_sector_counts(held, v3_config).values()) <= 5


def test_calculate_target_weights_long_short_neutral_uses_borrow_check(v3_config, monkeypatch):
    monkeypatch.setattr(config, "ALPACA_API_KEY", "PKTEST")
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "secret")
    worst = v3_config.sort_values("pred_rank")["ticker"].tolist()
    htb = set(worst[:15])

    class Asset:
        def __init__(self, sym):
            self.shortable = True
            self.easy_to_borrow = sym not in htb

    class FakeTC:
        calls = 0

        def __init__(self, *a, **k):
            pass

        def get_asset(self, sym):
            FakeTC.calls += 1
            return Asset(sym)

    monkeypatch.setattr("alpaca.trading.client.TradingClient", FakeTC)
    agent = object.__new__(bot.BDATradingAgent)
    w = agent.calculate_target_weights(v3_config, is_bull=None, strategy="long_short_neutral")
    shorts = {t for t, x in w.items() if x < 0}
    assert shorts and not shorts & htb
    assert sum(w.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(abs(x) for x in w.values()) <= 1.6 + 1e-9
    assert FakeTC.calls <= 4 * 40


def test_existing_strategies_still_available(v3_config):
    agent = object.__new__(bot.BDATradingAgent)
    for s in ("top_k", "high_confidence"):
        agent.calculate_target_weights(v3_config, is_bull=True, strategy=s)
    with pytest.raises(ValueError):
        agent.calculate_target_weights(v3_config, is_bull=True, strategy="nope")


# --------------------------------------------------------------------------- #
# Regime HMM only where it matters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["top_k", "high_confidence", "long_only_v3",
                                      "long_short_neutral"])
def test_hmm_skipped_for_strategies_that_do_not_use_it(strategy, monkeypatch, capsys):
    from DataAnalysisPipeline2.trading_agent import market_data

    def boom(*a, **k):
        raise AssertionError("HMM data fetched for a strategy that does not use it")

    monkeypatch.setattr(market_data, "fetch_benchmark_closes", boom)
    monkeypatch.setattr(bot.GaussianHMM, "fit", boom)
    agent = object.__new__(bot.BDATradingAgent)
    assert agent.check_market_regime(strategy=strategy) is None
    assert agent.hmm_state == -1
    assert f"regime not used by {strategy}" in capsys.readouterr().out


def test_hmm_still_runs_for_regime_filtered(monkeypatch):
    from DataAnalysisPipeline2.trading_agent import market_data
    calls = []

    def fake_closes(*a, **k):
        calls.append(1)
        rng = np.random.default_rng(1)
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        return pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=300).astype(str),
                             "Close": px})

    monkeypatch.setattr(market_data, "fetch_benchmark_closes", fake_closes)
    monkeypatch.setattr(config, "REGIME_FILTER_ENABLED", True)
    agent = object.__new__(bot.BDATradingAgent)
    out = agent.check_market_regime(strategy="regime_filtered")
    assert calls and out in (True, False)
