"""``risk.live_panel``: the live (bars + EDGAR files) risk-model panel."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import risk
from DataAnalysisPipeline2.v3 import universe as un
from DataAnalysisPipeline2.v3.tests.live_synthetic import LIVE_SECTORS, make_live_bars_and_fund


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("fund"))
    bars, meta = make_live_bars_and_fund(d, n=120, n_sessions=330)
    return bars, meta, d, risk.live_panel(bars, fund_dir=d)


def test_live_panel_shapes(live):
    bars, meta, _, p = live
    T, N = len(meta["dates"]), bars["ticker"].nunique()
    assert list(p.dates) == meta["dates"]
    assert list(p.tickers) == sorted(bars["ticker"].unique())
    for a in (p.ret, p.elig, p.mcap, p.adv, p.price, p.sector):
        assert a.shape == (T, N)
    assert set(p.fund) == {"bp", "ep", "gpa", "roa"}
    assert set(p.sector_names) <= set(LIVE_SECTORS)


def test_live_panel_eligibility_matches_universe_rule(live):
    bars, _, _, p = live
    col = {t: i for i, t in enumerate(p.tickers)}
    last = p.elig[-1]
    assert not last[col["PENNY"]]                  # below $3
    assert not p.elig[:, col["SHORTH"]].any()      # never 120 bars
    assert last[col["NOFUND"]]                     # no EDGAR record is not a reason to drop a name
    b = bars.assign(close_raw=bars["company_close"])
    det = un.eligibility_details(b, 3.0, 2e6, 120)
    want = det[det["Date"] == p.dates[-1]].set_index("ticker")["eligible"]
    got = pd.Series(last, index=p.tickers)
    assert (got.reindex(want.index) == want.astype(bool)).all()
    assert not p.elig[:119].any()                  # nobody has 120 bars before session 120


def test_live_panel_min_price_and_adv(live):
    bars, _, d, _ = live
    hi = risk.live_panel(bars, fund_dir=d, min_price=1e6)
    assert not hi.elig.any()
    liq = risk.live_panel(bars, fund_dir=d, min_adv_usd=1e15)
    assert not liq.elig.any()


def test_live_panel_mcap_sector_and_exposures(live):
    bars, meta, _, p = live
    col = {t: i for i, t in enumerate(p.tickers)}
    last = bars[bars["Date"] == p.dates[-1]].set_index("ticker")["company_close"]
    rel = [abs(p.mcap[-1, col[t]] / (last[t] * meta["shares"][t]) - 1) for t in meta["tickers"] if t != "L001"]
    assert np.median(rel) < 1e-4                   # own EDGAR shares x close
    assert np.isfinite(p.mcap[-1, col["NOFUND"]]) and p.mcap[-1, col["NOFUND"]] > 0   # ADV fit
    names = np.array(p.sector_names)
    for t in meta["tickers"][:20]:
        assert names[p.sector[-1, col[t]]] == meta["sector"][t]
    assert names[p.sector[-1, col["NOFUND"]]] == "Other"
    desc = risk.raw_descriptors(p)
    mask, X = risk.exposures_at(p, desc, len(p.dates) - 1)
    assert mask.sum() >= 100
    assert X.shape == (len(p.tickers), len(risk.factor_names(p)))
    assert np.all(np.isfinite(X[mask])) and np.all(np.isnan(X[~mask]))
    ind = X[mask][:, len(risk.STYLES):]
    assert np.allclose(ind.sum(axis=1), 1.0)       # exactly one industry per name


def test_live_panel_risk_model_is_finite_and_psd(live):
    _, _, _, p = live
    m = risk.RiskEngine(p).model(str(p.dates[-1]))
    assert len(m.tickers) >= 100
    assert np.all(np.isfinite(m.X)) and np.all(np.isfinite(m.D)) and np.all(m.D > 0)
    assert np.linalg.eigvalsh(m.F).min() > -1e-12
    w = np.full(len(m.tickers), 1.0 / len(m.tickers))
    assert 0.03 < m.vol(w, 252) < 0.5
