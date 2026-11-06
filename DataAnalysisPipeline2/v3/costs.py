"""Per-name transaction-cost model (round 5).

cost of trading ``t`` dollars of name i (one side), as a fraction of the traded
notional:

    half_spread_i  +  k * sigma_i * sqrt(|t| / ADV_i)

* **Half-spread**: the Corwin-Schultz (2012) high-low estimator averaged over
  the trailing ``CS_WINDOW`` sessions (two-day estimator, negative estimates set
  to 0, close-to-close overnight adjustment of the second day's range), halved;
  when fewer than ``CS_MIN_OBS`` estimates exist, the price/ADV proxy
  ``max(tick/2 / price, HS_ADV_BPS / sqrt(ADV / $1M), 1 bp)``. Either way the
  half-spread is clipped to [``HS_FLOOR``, ``HS_CAP``].
* **Market impact** (square-root law, Almgren et al. 2005; Frazzini, Israel &
  Moskowitz 2018; Toth et al. 2011): ``k * sigma_daily * sqrt(participation)``,
  participation = trade $ / 20-session ADV $. ``k = IMPACT_K = 0.2``: the
  literature puts the prefactor between ~0.1 (large, patient institutional
  flow, FIM 2018) and ~0.3-1 (aggressive, small caps); 0.2 is the
  mid-conservative choice for a one-day VWAP-style execution.
* **Borrow** for shorts: annual fee by size (general collateral ~0.25-0.5% for
  large caps, 1% for >= $1B, 5% below; hard-to-borrow names are excluded
  upstream by the round-3 safe-short filter).

For the optimizer the cost of a weight change ``dw`` (fraction of NAV, AUM
dollars) is the convex separable function

    cost(dw) = c_i |dw| + d_i |dw|^{3/2},   c_i = half_spread_i,
                                             d_i = k sigma_i sqrt(AUM / ADV_i)

which is exactly the model above (``|t| = |dw| AUM``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMPACT_K = 0.2
CS_WINDOW = 63
CS_MIN_OBS = 20
HS_FLOOR = 1e-4          # 1 bp
HS_CAP = 0.015           # 150 bp
HS_ADV_BPS = 25e-4       # 25 bp at $1M ADV, ~0.8 bp at $1B
TICK = 0.01
DEFAULT_HS = 0.005       # names with no data (e.g. delisted, sold at the frozen price)
BORROW_GC, BORROW_EASY, BORROW_HARD = 0.0025, 0.01, 0.05
GC_CAP, EASY_CAP = 10e9, 1e9


# --------------------------------------------------------------------------- #
# Spread estimators
# --------------------------------------------------------------------------- #
def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Two-day Corwin-Schultz spread estimate S_t (full spread, fraction of
    price) for dates x tickers frames. Day t+1's high/low are shifted down by
    any overnight gap above/below day t's close (CS 2012, sec. 3.2); negative
    estimates are set to 0. Row t uses days t-1 and t (known at t's close)."""
    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    pc = c.shift(1)
    # overnight adjustment of day t relative to day t-1's close
    up = (l - pc).clip(lower=0.0)
    dn = (h - pc).clip(upper=0.0)
    h_adj = h - up - dn
    l_adj = l - up - dn
    h1, l1 = h.shift(1), l.shift(1)
    beta = np.log(h1 / l1) ** 2 + np.log(h_adj / l_adj) ** 2
    hmax = np.maximum(h1, h_adj)
    lmin = np.minimum(l1, l_adj)
    gamma = np.log(hmax / lmin) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    a = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))
    return s.clip(lower=0.0)


def proxy_half_spread(price, adv):
    """max(half a tick / price, HS_ADV_BPS / sqrt(ADV/$1M), 1 bp)."""
    price = np.asarray(price, dtype=float)
    adv = np.asarray(adv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = TICK / 2.0 / price
        b = HS_ADV_BPS / np.sqrt(np.maximum(adv, 1.0) / 1e6)
    hs = np.fmax(a, b)
    hs = np.where(np.isfinite(hs), np.maximum(hs, HS_FLOOR), DEFAULT_HS)
    return np.clip(hs, HS_FLOOR, HS_CAP)


def half_spread(cs: pd.DataFrame | None, price: pd.DataFrame, adv: pd.DataFrame,
                window: int = CS_WINDOW, min_obs: int = CS_MIN_OBS) -> pd.DataFrame:
    """dates x tickers half-spread: 0.5 x trailing mean CS estimate where
    >= ``min_obs`` estimates exist, else the price/ADV proxy; clipped."""
    proxy = pd.DataFrame(proxy_half_spread(price.to_numpy(), adv.to_numpy()), index=price.index,
                         columns=price.columns)
    if cs is None:
        return proxy
    cs = cs.reindex_like(price)
    m = cs.rolling(window, min_periods=min_obs).mean() * 0.5
    return m.where(m.notna(), proxy).clip(HS_FLOOR, HS_CAP)


# --------------------------------------------------------------------------- #
# Cost functions
# --------------------------------------------------------------------------- #
def impact_coef(sigma_daily, adv, aum: float, k: float = IMPACT_K):
    """d_i such that impact cost (fraction of NAV) = d_i |dw|^{3/2}."""
    sigma_daily = np.asarray(sigma_daily, dtype=float)
    adv = np.maximum(np.asarray(adv, dtype=float), 1.0)
    d = k * sigma_daily * np.sqrt(aum / adv)
    return np.where(np.isfinite(d), d, 0.0)


def trade_cost(dw, half_spr, sigma_daily, adv, aum: float, k: float = IMPACT_K):
    """Per-name cost (fraction of NAV) of weight changes ``dw`` at ``aum``."""
    dw = np.abs(np.asarray(dw, dtype=float))
    return np.asarray(half_spr, float) * dw + impact_coef(sigma_daily, adv, aum, k) * dw ** 1.5


def borrow_rate(mcap) -> np.ndarray:
    """Annual borrow fee by market cap (general collateral / easy / hard)."""
    m = np.asarray(mcap, dtype=float)
    return np.where(m >= GC_CAP, BORROW_GC, np.where(m >= EASY_CAP, BORROW_EASY, BORROW_HARD))
