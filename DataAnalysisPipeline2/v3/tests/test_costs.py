"""Round-5 transaction-cost model."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import costs as cs


def test_corwin_schultz_recovers_constant_spread():
    s = 0.004
    n = 30
    idx = pd.RangeIndex(n)
    close = pd.DataFrame({"A": np.full(n, 100.0)}, index=idx)
    high = close * (1 + s / 2)
    low = close * (1 - s / 2)
    est = cs.corwin_schultz(high, low, close)
    assert est["A"].iloc[1:].to_numpy() == pytest.approx(np.full(n - 1, s), rel=0.01)


def test_corwin_schultz_non_negative_and_gap_adjusted():
    rng = np.random.default_rng(0)
    px = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    close = pd.DataFrame({"A": px})
    high = close * np.exp(np.abs(rng.normal(0, 0.01, (200, 1))))
    low = close * np.exp(-np.abs(rng.normal(0, 0.01, (200, 1))))
    est = cs.corwin_schultz(high, low, close)
    assert (est.dropna() >= 0).all().all()
    hs = cs.half_spread(est, close, pd.DataFrame({"A": np.full(200, 5e7)}))
    assert hs.iloc[-1, 0] == pytest.approx(0.5 * est["A"].iloc[-63:].mean(), rel=1e-6) or hs.iloc[-1, 0] in (
        cs.HS_FLOOR, cs.HS_CAP)
    # too few estimates -> price/ADV proxy
    assert hs.iloc[5, 0] == pytest.approx(cs.proxy_half_spread(close.iloc[5, 0], 5e7))


def test_proxy_half_spread():
    assert cs.proxy_half_spread(100.0, 1e9) == pytest.approx(cs.HS_FLOOR)       # mega cap -> 1 bp
    assert cs.proxy_half_spread(100.0, 25e6) == pytest.approx(25e-4 / 5)        # 5 bp
    assert cs.proxy_half_spread(5.0, 1e9) == pytest.approx(0.005 / 5)           # half a tick on $5
    assert cs.proxy_half_spread(np.nan, np.nan) == pytest.approx(cs.DEFAULT_HS)


def test_square_root_impact_and_units():
    hs, sig, adv, aum = 3e-4, 0.02, 50e6, 100e6
    c1 = cs.trade_cost(0.01, hs, sig, adv, aum)
    c2 = cs.trade_cost(0.02, hs, sig, adv, aum)
    imp1, imp2 = c1 - hs * 0.01, c2 - hs * 0.02
    assert imp2 / imp1 == pytest.approx(2 ** 1.5)
    dollars = 0.01 * aum
    per_dollar = hs + cs.IMPACT_K * sig * np.sqrt(dollars / adv)
    assert c1 * aum == pytest.approx(per_dollar * dollars)
    # 10x the AUM -> sqrt(10)x the impact per unit of weight
    assert (cs.trade_cost(0.01, 0, sig, adv, 10 * aum) / cs.trade_cost(0.01, 0, sig, adv, aum)) == pytest.approx(
        np.sqrt(10))


def test_borrow_tiers():
    assert list(cs.borrow_rate([2e10, 5e9, 2e8])) == [cs.BORROW_GC, cs.BORROW_EASY, cs.BORROW_HARD]
