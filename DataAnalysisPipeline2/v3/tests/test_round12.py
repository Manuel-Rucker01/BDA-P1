"""Round 12 (vra) mechanics and point-in-time behaviour on toy inputs (no data store)."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round11 as r11
from DataAnalysisPipeline2.v3 import round12 as r12
from DataAnalysisPipeline2.v3.tests.test_round5 import CAL, _toy_prep


def _models(prep):
    return {d: x["model"] for d, x in prep["by_date"].items()}


def test_total_vol_matches_cov_diagonal():
    prep = _toy_prep(n=30)
    m = next(iter(_models(prep).values()))
    assert np.allclose(r12.total_vol(m), np.sqrt(np.diag(m.cov())))


def test_daily_bias_uses_previous_model_only():
    prep = _toy_prep(n=150)
    models = _models(prep)
    md = sorted(models)
    b2 = r12.daily_bias_sq(prep["dates"], prep["tickers"], prep["ret"], models)
    dates = np.asarray(prep["dates"]).astype(str)
    # no value on or before the first model's date
    assert b2[dates <= md[0]].isna().all()
    # the day after md[1] is standardised by md[1]'s model, by hand
    d = dates[np.searchsorted(dates, md[1], side="right")]
    m = models[md[1]]
    i = int(np.searchsorted(dates, d))
    r = np.clip(prep["ret"][i].astype(float), -0.5, 1.0)
    z2 = (r / r12.total_vol(m)) ** 2
    assert np.isclose(b2[d], z2.mean())
    # changing a FUTURE model never changes past values
    m2 = dict(models)
    last = md[-1]
    mm = models[last]
    m2[last] = r12.scale_model(mm, 3.0)
    b2b = r12.daily_bias_sq(prep["dates"], prep["tickers"], prep["ret"], m2)
    pre = dates <= last
    assert np.allclose(b2[pre].fillna(-1), b2b[pre].fillna(-1))
    # too few names -> NaN
    assert r12.daily_bias_sq(prep["dates"], prep["tickers"], prep["ret"], models, min_names=1000).isna().all()


def test_regime_lambda_ewma_clip_and_warmup():
    idx = [f"2020-01-{i:02d}" for i in range(1, 31)]
    b2 = pd.Series(1.0, index=idx)
    assert r12.regime_lambda(b2, idx[5]) == 1.0                  # < 20 days -> 1
    assert np.isclose(r12.regime_lambda(b2, idx[-1]), 1.0)
    assert r12.regime_lambda(b2 * 100, idx[-1]) == gate.R12_LAMBDA_CLIP[1]
    assert r12.regime_lambda(b2 * 0.01, idx[-1]) == gate.R12_LAMBDA_CLIP[0]
    x = pd.Series(np.r_[np.ones(25), np.full(5, 2.25)], index=idx)
    w = 0.5 ** (np.arange(29, -1, -1) / gate.R12_VRA_HL)
    assert np.isclose(r12.regime_lambda(x, idx[-1]), np.sqrt((w * x.to_numpy()).sum() / w.sum()))
    # future observations are ignored
    y = x.copy()
    y.iloc[26:] = 100.0
    assert r12.regime_lambda(x, idx[25]) == r12.regime_lambda(y, idx[25])


def test_scale_model_scales_cov():
    prep = _toy_prep(n=20)
    m = next(iter(_models(prep).values()))
    s = r12.scale_model(m, 1.5)
    assert np.allclose(s.cov(), 2.25 * m.cov())
    assert s.tickers is m.tickers and np.allclose(s.X, m.X)


def test_hook_none_and_unit_lambda_reproduce_runner():
    prep = _toy_prep()
    a = r11.run_book("base", prep, aum=1e8)
    b = r11.run_book("base", prep, aum=1e8, model_hook=None)
    c = r11.run_book("base", prep, aum=1e8, model_hook=r12.make_hook(pd.Series(1.0, index=sorted(prep["by_date"]))))
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy(), atol=0)
    assert np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy(), atol=1e-14)


def test_hook_scales_ex_ante_vol():
    prep = _toy_prep()
    lam = pd.Series(2.0, index=sorted(prep["by_date"]))
    a = r11.run_book("base", prep, aum=1e8)["log"]
    b = r11.run_book("base", prep, aum=1e8, model_hook=r12.make_hook(lam))["log"]
    # higher predicted risk -> the logged ex-ante vol is on the scaled model (> raw for any book)
    assert (b["ex_ante_vol"] > a["ex_ante_vol"] * 1.5).all()


def test_adoption_rule():
    inc = {"sharpe": 1.04, "max_dd": -0.277, "covid_dd": -0.30, "ir_spy": -0.10}
    ok = {"sharpe": 1.02, "max_dd": -0.255, "covid_dd": -0.25, "ir_spy": -0.14}
    assert gate.adopt_r12(ok, inc, False)["adopt"]
    for k, v in (("sharpe", 1.00), ("max_dd", -0.259), ("covid_dd", -0.30), ("ir_spy", -0.16)):
        assert not gate.adopt_r12({**ok, k: v}, inc, False)["adopt"], k
    assert gate.adopt_r12({**ok, "sharpe": 0.5}, inc, True)["adopt"]
    assert gate.N_TRIALS_R12 == gate.N_TRIALS_R11 + 1 == 104
