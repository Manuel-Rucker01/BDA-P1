"""Round-5 factor risk model on a synthetic factor-structured panel."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import risk

SECTORS = ["Finance", "Manufacturing", "Other"]


def _long(n=240, T=520, seed=0, shock_after=None):
    """Returns = beta_i * market + sector factor + size factor * size_i + eps_i
    (specific vol varies 1%-3% daily)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T).strftime("%Y-%m-%d")
    beta = rng.uniform(0.5, 1.5, n)
    sec = rng.integers(0, len(SECTORS), n)
    size = rng.normal(0, 1, n)
    spec = rng.uniform(0.01, 0.03, n)
    m = rng.normal(0.0004, 0.01, T)
    fs = rng.normal(0, 0.006, (T, len(SECTORS)))
    fsize = rng.normal(0, 0.004, T)
    eps = rng.normal(0, 1, (T, n)) * spec
    r = m[:, None] * beta + fs[:, sec] + fsize[:, None] * size + eps
    if shock_after is not None:
        r[shock_after:] = r[shock_after:] * 5 + 0.3       # future data that must not leak
    px = 50 * np.exp(np.cumsum(r, axis=0))
    mcap = np.exp(22 + size)[None, :] * px / px[0]
    rows = pd.DataFrame({
        "ticker": np.tile([f"T{i:03d}" for i in range(n)], T), "Date": np.repeat(dates, n),
        "company_close": px.ravel(), "close_raw": px.ravel(), "adv_usd": (mcap * 0.005).ravel(),
        "eligible": True, "Sector": np.tile(np.array(SECTORS)[sec], T), "log_mcap": np.log(mcap).ravel(),
        "fund_imputed": False, "bp": np.tile(rng.normal(0.5, 0.2, n), T), "ep": np.tile(rng.normal(0.05, 0.02, n), T),
        "gpa": np.tile(rng.normal(0.3, 0.1, n), T), "roa": np.tile(rng.normal(0.05, 0.03, n), T)})
    return rows, dict(beta=beta, spec=spec, dates=dates)


@pytest.fixture(scope="module")
def engine():
    df, truth = _long()
    p = risk.panel_from_long(df)
    return risk.RiskEngine(p), truth


def test_standardize_row():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    w = rng.lognormal(size=500)
    mask = np.ones(500, dtype=bool)
    mask[:10] = False
    x[20] = np.nan
    z = risk.standardize_row(x, w, mask)
    assert np.all(np.isnan(z[:10])) and z[20] == 0.0
    v = mask.copy()
    v[20] = False
    assert (w[mask] * z[mask]).sum() / w[mask].sum() == pytest.approx(0.0, abs=1e-9)
    assert z[v].std() == pytest.approx(1.0, abs=0.02)


def test_model_shapes_psd_and_beta_exposure(engine):
    eng, truth = engine
    m = eng.model(truth["dates"][-1])
    assert m.X.shape == (240, len(risk.STYLES) + len(SECTORS))
    assert np.all(np.linalg.eigvalsh(m.F) > 0)
    assert np.all(m.D > 0)
    # the beta style exposure ranks names like the true beta
    assert np.corrcoef(m.X[:, 0], truth["beta"])[0, 1] > 0.8
    # specific vol tracks the truth
    assert np.corrcoef(np.sqrt(m.D), truth["spec"])[0, 1] > 0.8


def test_bias_statistic_near_one(engine):
    eng, truth = engine
    p = eng.p
    t0 = 400
    m = eng.model(p.dates[t0])
    rng = np.random.default_rng(3)
    z = []
    for _ in range(30):
        w = rng.dirichlet(np.ones(len(m.tickers)))
        z.append(risk.standardized_returns(p.ret[t0 + 1:], w, m.vol(w)))
    b = risk.bias_stat(np.concatenate(z))
    assert 0.8 < b < 1.25


def test_point_in_time_no_future_leak():
    df, truth = _long(T=420)
    df2, _ = _long(T=420, shock_after=380)
    d = truth["dates"][370]
    m1 = risk.RiskEngine(risk.panel_from_long(df)).model(d)
    m2 = risk.RiskEngine(risk.panel_from_long(df2)).model(d)
    assert np.allclose(m1.X, m2.X) and np.allclose(m1.F, m2.F) and np.allclose(m1.D, m2.D)


def test_clean_mcap_replaces_gross_errors_only():
    rng = np.random.default_rng(0)
    N = 400
    lmc = rng.normal(22, 1.5, N)
    adv = np.exp(lmc - 5 + rng.normal(0, 0.5, N))
    price = np.exp(rng.normal(3, 0.5, N))
    raw = np.exp(lmc)
    raw[0] = raw[0] * 1e6           # unit error
    raw[1] = raw[1] * 3             # plausible (within 20x): kept
    raw[2] = np.nan                 # no EDGAR value: fitted
    out = risk.clean_mcap(raw[None, :], adv[None, :], price[None, :], np.ones((1, N), dtype=bool))[0]
    assert out[0] < raw[0] / 1e4
    assert out[1] == pytest.approx(raw[1], rel=1e-4)
    assert np.isfinite(out[2]) and out[3] == pytest.approx(raw[3], rel=1e-4)
