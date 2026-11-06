"""Round-5 optimizer: proximal operators, ADMM vs SLSQP, constraints, speed."""
import time

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from DataAnalysisPipeline2.v3 import optimizer as op


def _problem(n, k, seed, te=None, cost=True, short=False, lam=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    A = rng.normal(size=(k, k)) * 0.01
    F = A @ A.T + np.eye(k) * 1e-5
    D = rng.uniform(0.5, 3, n) * 1e-3
    G = np.vstack([np.ones(n), (rng.random(n) < 0.3).astype(float)])
    return op.Problem(alpha=rng.normal(size=n) * 2e-3, X=X, F=F, D=D, lam=lam, bench=rng.dirichlet(np.ones(n)),
                      w_prev=rng.dirichlet(np.ones(n)), lin_cost=np.full(n, 5e-4) if cost else None,
                      imp_cost=np.full(n, 2e-3) if cost else None, lb=np.full(n, -0.05 if short else 0.0),
                      ub=np.full(n, 0.2), G=G, glo=np.array([1.0, 0.0]), ghi=np.array([1.0, 0.35]), te=te,
                      short_budget=0.3 if short else None, borrow=np.full(n, 1e-3) if short else None)


def _slsqp(pb):
    S = pb.X @ pb.F @ pb.X.T + np.diag(pb.D)
    b = pb.bench
    cons = [{"type": "eq", "fun": lambda w: pb.G[0] @ w - 1.0},
            {"type": "ineq", "fun": lambda w: pb.ghi[1] - pb.G[1] @ w}]
    if pb.te is not None:
        cons.append({"type": "ineq", "fun": lambda w: pb.te ** 2 - (w - b) @ S @ (w - b)})
    if pb.short_budget is not None:
        cons.append({"type": "ineq", "fun": lambda w: pb.short_budget - np.maximum(-w, 0).sum()})
    r = minimize(lambda w: -op.objective(pb, w), np.clip(pb.w_prev, pb.lb, pb.ub), method="SLSQP",
                 bounds=list(zip(pb.lb, pb.ub)), constraints=cons, options={"maxiter": 3000, "ftol": 1e-15})
    return r.x, -r.fun


def test_prox_cost_matches_scalar_minimisation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, c, d, rho = rng.normal() * 0.05, rng.uniform(0, 0.01), rng.uniform(0, 0.1), rng.uniform(0.1, 10)
        ref = minimize_scalar(lambda t: c * abs(t) + d * abs(t) ** 1.5 + rho / 2 * (t - a) ** 2,
                              bounds=(-1, 1), method="bounded", options={"xatol": 1e-12}).x
        got = op.prox_cost(np.array([a]), np.array([c]), np.array([d]), rho)[0]
        assert abs(got - ref) < 1e-6


def test_prox_short_budget_and_fee():
    a = np.array([0.2, -0.3, -0.1, -0.05])
    z = op.prox_short(a, np.zeros(4), None, 1.0)
    assert np.allclose(z, a)
    z = op.prox_short(a, np.full(4, 0.02), 0.2, 1.0)
    assert z[0] == 0.2 and np.all(z <= 0.0 + (a > 0))
    assert -z[z < 0].sum() == pytest.approx(0.2, abs=1e-9)


def test_project_ball():
    c = np.array([1.0, 1.0])
    assert np.allclose(op.project_ball(np.array([1.1, 1.0]), c, 0.5), [1.1, 1.0])
    assert np.allclose(op.project_ball(np.array([3.0, 1.0]), c, 0.5), [1.5, 1.0])


@pytest.mark.parametrize("args", [dict(cost=False), dict(), dict(te=0.02), dict(short=True, te=0.03)])
def test_admm_matches_slsqp(args):
    pb = _problem(25, 4, 1, **args)
    res = op.solve(pb)
    _, ref = _slsqp(_problem(25, 4, 1, **args))
    assert res.status == "solved"
    assert res.obj == pytest.approx(ref, abs=2e-7)
    w = res.w
    assert w.sum() == pytest.approx(1.0, abs=5e-4)
    assert np.all(w >= pb.lb - 1e-6) and np.all(w <= pb.ub + 1e-6)
    if pb.te is not None:
        assert op.risk(pb.X, pb.F, pb.D, w - pb.bench) <= pb.te * (1 + 1e-3)
    if pb.short_budget is not None:
        assert -w[w < 0].sum() <= pb.short_budget + 1e-4


def test_te_constraint_binds_and_large_problem_is_fast():
    n, k = 2500, 21
    pb = _problem(n, k, 3, te=0.012, lam=0.01, cost=False)
    pb.ub = np.full(n, 0.05)
    t0 = time.time()
    res = op.solve(pb)
    # Generous bound: ~1-2 s on an idle M5, but CI/background training can
    # starve the CPU; this guards against pathological (minutes-long) solves only.
    assert time.time() - t0 < 300.0
    assert res.status == "solved"
    te = op.risk(pb.X, pb.F, pb.D, res.w - pb.bench)
    assert te == pytest.approx(0.012, rel=1e-2)
    assert res.w.sum() == pytest.approx(1.0, abs=1e-3)


def test_cost_reduces_turnover():
    pb0 = _problem(60, 5, 2, cost=False)
    pb1 = _problem(60, 5, 2, cost=True)
    pb1.lin_cost = pb1.lin_cost * 10
    t0 = np.abs(op.solve(pb0).w - pb0.w_prev).sum()
    t1 = np.abs(op.solve(pb1).w - pb1.w_prev).sum()
    assert t1 < t0


def test_cardinality_threshold():
    pb = _problem(80, 5, 4, cost=False)
    res = op.solve_with_cardinality(pb, min_weight=0.01)
    w = res.w
    assert np.all((w == 0) | (np.abs(w) >= 0.01 - 1e-6))
    assert w.sum() == pytest.approx(1.0, abs=5e-4)


def test_normal_scores_rank_invariant():
    x = np.array([0.1, 0.5, 0.2, 0.9, np.nan])
    z1 = op.normal_scores(x)
    z2 = op.normal_scores(np.exp(10 * x))           # monotone transform (e.g. pred vs pred_rank)
    assert np.allclose(z1, z2)
    assert z1[4] == 0.0 and abs(z1[:4].mean()) < 1e-12 and z1[:4].std() == pytest.approx(1.0)
    a = op.grinold_alpha(x[:4], np.full(4, 0.02), ic=0.05, label_h=30, horizon=21)
    assert np.allclose(a, 0.05 * 0.02 * np.sqrt(30) * z1[:4] * 21 / 30)
    assert np.all(op.grinold_alpha(x[:4], np.full(4, 0.02), ic=-0.1) == 0)


def _book_inputs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    k_style, n_sec = 3, 4
    sector = rng.integers(0, n_sec, n)
    X = np.hstack([rng.normal(size=(n, k_style)), np.eye(n_sec)[sector]])
    F = np.diag(np.r_[np.full(k_style, 2e-5), np.full(n_sec, 1e-4)])
    return dict(tickers=np.array([f"T{i}" for i in range(n)]), scores=rng.normal(size=n), X=X, F_daily=F,
                D_daily=rng.uniform(1, 5, n) * 1e-4, mcap=rng.lognormal(22, 1.5, n), sector=sector,
                factors=["a", "b", "c"] + [f"ind:{s}" for s in range(n_sec)], half_spread=np.full(n, 5e-4),
                sigma_daily=np.full(n, 0.02), adv=rng.lognormal(17, 1, n))


def test_optimize_portfolio_active_constraints():
    inp = _book_inputs()
    spec = {"kind": "active", "te": 0.04, "lam": 1.0, "active_cap": 0.03, "sector_band": 0.05, "beta_band": 0.05}
    w, diag = op.optimize_portfolio(spec, aum=1e6, ic=0.05, **inp)
    v = np.array([w.get(t, 0.0) for t in inp["tickers"]])
    b = inp["mcap"] / inp["mcap"].sum()
    assert v.min() >= 0 and v.sum() == pytest.approx(1.0, abs=2e-3)
    assert np.all(v <= b + 0.03 + 1e-4)
    assert diag["ex_ante_te_ann"] <= 0.04 * 1.01
    assert abs(diag["ex_ante_beta"] - 1.0) <= 0.05 + 1e-3
    for s in np.unique(inp["sector"]):
        m = inp["sector"] == s
        assert abs(v[m].sum() - b[m].sum()) <= 0.05 + 2e-3
    assert np.all((v == 0) | (v >= 0.001 - 1e-6))
    assert diag["active_alpha_ann"] > 0
    # zero IC -> nothing to gain from tracking error
    _, d0 = op.optimize_portfolio(spec, aum=1e6, ic=0.0, w_prev={t: x for t, x in zip(inp["tickers"], b)}, **inp)
    assert d0["ex_ante_te_ann"] < 0.005


def test_optimize_portfolio_liquidity_cap_and_130_30():
    inp = _book_inputs(seed=1)
    aum = 5e8
    spec = {"kind": "130_30", "te": 0.06, "lam": 1.0, "active_cap": 0.03, "sector_band": 0.05, "beta_band": 0.05,
            "short_budget": 0.3, "short_cap": 0.01}
    shortable = set(inp["tickers"][:60])
    w, diag = op.optimize_portfolio(spec, aum=aum, ic=0.05, shortable=shortable, **inp)
    v = np.array([w.get(t, 0.0) for t in inp["tickers"]])
    liq = 0.10 * inp["adv"] / aum
    assert np.all(np.abs(v) <= liq + 1e-6)
    assert np.all(v[60:] >= -1e-9)                         # shorts only where allowed
    assert -v[v < 0].sum() <= 0.3 + 1e-3 and v.min() >= -0.01 - 1e-6
    assert v.sum() == pytest.approx(1.0, abs=2e-3) or v.sum() < 1.0   # leftover can stay in cash
    assert diag["n_short"] > 0
