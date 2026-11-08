"""Cost-aware mean-variance portfolio optimizer (round 5).

Solves, for one rebalance,

    maximise   alpha'w - lam (w - b)' Sigma (w - b) - cost(w - w_prev) - borrow(w)
    subject to lb <= w <= ub                               (long-only / per-name caps)
               glo <= G w <= ghi                           (budget, sector, beta, style rows)
               (w - b)' Sigma (w - b) <= te^2              (optional tracking-error cap)
               sum_i max(-w_i, 0) <= short_budget          (optional, 130/30)

with a factor covariance ``Sigma = X F X' + diag(D)`` and a separable convex
trading cost ``cost(t) = sum_i c_i |t_i| + d_i |t_i|^{3/2}`` (linear half-spread
+ square-root-law impact, ``costs.py``) and a linear borrow fee on shorts.

Algorithm: ADMM in the OSQP form ``min f(x) + g(z) s.t. A x = z`` with f the
quadratic part and g the indicator / prox-friendly part, split in blocks of
rows of A:

    1. identity rows       -> box [lb, ub]            (projection: clip)
    2. G rows               -> [glo, ghi]              (clip; equality rows get 1e3 x rho)
    3. identity rows       -> trading cost around w_prev   (closed-form prox)
    4. [L'X'; sqrt(D)] rows -> ball of radius te around the benchmark   (projection)
    5. identity rows       -> borrow + short budget    (prox: threshold + bisection)

Every x-update solves ``(P + A' R A) x = rhs``. ``P`` and ``A' R A`` are both
``diagonal + U W U'`` with ``U = [X L, G']`` (N x (K + m), K + m ~ 40), so the
solve is a Woodbury identity with an (K+m) x (K+m) inverse: O(N (K+m)) per
iteration, no N x N matrix is ever formed. A 3,000-name problem solves in well
under a second; rho is adapted (OSQP rule) with a cheap re-factorisation.

All inputs are in the same per-period units (alpha and Sigma over the planning
horizon, costs as a fraction of NAV, te as a per-period standard deviation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Problem:
    alpha: np.ndarray                       # (N,) expected return over the horizon
    X: np.ndarray                           # (N, K) factor exposures
    F: np.ndarray                           # (K, K) factor covariance (per period)
    D: np.ndarray                           # (N,) specific variance (per period)
    lam: float = 1.0                        # risk aversion on (w - b)
    bench: np.ndarray | None = None         # (N,) benchmark weights (None = absolute risk)
    w_prev: np.ndarray | None = None        # (N,) current (drifted) weights
    lin_cost: np.ndarray | None = None      # (N,) c_i  (half-spread, fraction)
    imp_cost: np.ndarray | None = None      # (N,) d_i  (impact coefficient of |t|^{3/2})
    lb: np.ndarray | None = None            # (N,) default 0
    ub: np.ndarray | None = None            # (N,) default 1
    G: np.ndarray | None = None             # (m, N) linear constraint rows
    glo: np.ndarray | None = None
    ghi: np.ndarray | None = None
    te: float | None = None                 # tracking-error cap (per-period sd)
    short_budget: float | None = None       # sum of short weights <= this
    borrow: np.ndarray | None = None        # (N,) borrow fee per period on short weight

    def n(self) -> int:
        return len(self.alpha)


@dataclass
class Result:
    w: np.ndarray
    status: str
    iters: int
    obj: float
    prim_res: float
    dual_res: float
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Proximal operators / projections (pure, unit-tested)
# --------------------------------------------------------------------------- #
def prox_cost(a: np.ndarray, c: np.ndarray, d: np.ndarray, rho) -> np.ndarray:
    """argmin_t  c|t| + d|t|^{3/2} + rho/2 (t - a)^2, elementwise (c, d >= 0).

    Soft-threshold by c/rho, then |t| = s^2 with s the positive root of
    rho s^2 + 1.5 d s - rho m = 0 (m = |a| - c/rho)."""
    m = np.abs(a) - c / rho
    pos = m > 0
    s = np.zeros_like(a)
    mm = np.where(pos, m, 0.0)
    s = (-1.5 * d + np.sqrt(2.25 * d * d + 4.0 * rho * rho * mm)) / (2.0 * rho)
    return np.where(pos, np.sign(a) * s * s, 0.0)


def prox_short(a: np.ndarray, fee: np.ndarray, budget: float | None, rho) -> np.ndarray:
    """argmin_z  sum fee_i max(-z_i,0) + rho/2 ||z - a||^2  s.t. sum max(-z_i,0) <= budget."""
    def shorts(theta):
        return np.where(a < 0, np.minimum(0.0, a + (fee + theta) / rho), a)

    z = shorts(0.0)
    if budget is None or -z[z < 0].sum() <= budget + 1e-15:
        return z
    lo, hi = 0.0, float(rho * np.max(-a)) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if -np.minimum(shorts(mid), 0.0).sum() > budget:
            lo = mid
        else:
            hi = mid
    return shorts(hi)


def project_ball(v: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = v - center
    nrm = float(np.sqrt(d @ d))
    return v if nrm <= radius else center + d * (radius / nrm)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class _Ops:
    """Linear operator A (stacked blocks) and the Woodbury solver for
    (P + A' R A)."""

    def __init__(self, pb: Problem, scale: float):
        n = pb.n()
        self.n = n
        self.X = np.asarray(pb.X, dtype=float)
        F = np.asarray(pb.F, dtype=float)
        F = 0.5 * (F + F.T)
        ev, V = np.linalg.eigh(F)
        ev = np.clip(ev, 0.0, None)
        self.L = V * np.sqrt(ev)                     # F = L L'
        self.XL = self.X @ self.L                    # (N, K)
        self.D = np.clip(np.asarray(pb.D, dtype=float), 1e-12, None)
        self.sqD = np.sqrt(self.D)
        self.G = np.zeros((0, n)) if pb.G is None else np.asarray(pb.G, dtype=float)
        self.m = self.G.shape[0]
        self.K = self.XL.shape[1]
        self.has_te = pb.te is not None
        self.has_short = pb.short_budget is not None or (pb.borrow is not None and np.any(pb.lb < 0))
        self.has_cost = pb.lin_cost is not None or pb.imp_cost is not None
        self.lam2 = 2.0 * pb.lam * scale
        sizes = [n, self.m, n if self.has_cost else 0, (self.K + n) if self.has_te else 0,
                 n if self.has_short else 0]
        self.sl = []
        o = 0
        for s in sizes:
            self.sl.append(slice(o, o + s))
            o += s
        self.n_rows = o

    def A(self, x):
        out = np.empty(self.n_rows)
        out[self.sl[0]] = x
        out[self.sl[1]] = self.G @ x
        if self.has_cost:
            out[self.sl[2]] = x
        if self.has_te:
            s = self.sl[3]
            out[s.start:s.start + self.K] = self.XL.T @ x
            out[s.start + self.K:s.stop] = self.sqD * x
        if self.has_short:
            out[self.sl[4]] = x
        return out

    def At(self, y):
        x = y[self.sl[0]].copy()
        x += self.G.T @ y[self.sl[1]]
        if self.has_cost:
            x += y[self.sl[2]]
        if self.has_te:
            s = self.sl[3]
            x += self.XL @ y[s.start:s.start + self.K] + self.sqD * y[s.start + self.K:s.stop]
        if self.has_short:
            x += y[self.sl[4]]
        return x

    def factor(self, rho: np.ndarray):
        """M = diag(dm) + U diag(wu) U' ;  precompute Woodbury pieces."""
        r = lambda k: float(rho[self.sl[k]][0]) if self.sl[k].stop > self.sl[k].start else 0.0
        rte = r(3)
        dm = self.lam2 * self.D + r(0) + r(2) + r(4) + rte * self.D
        wf = self.lam2 + rte
        cols, wts = [], []
        if wf > 0 and self.K:
            cols.append(self.XL)
            wts.append(np.full(self.K, wf))
        if self.m:
            cols.append(self.G.T)
            wts.append(rho[self.sl[1]])
        self.dinv = 1.0 / dm
        if cols:
            U = np.hstack(cols)
            w = np.concatenate(wts)
            DU = U * self.dinv[:, None]
            S = np.diag(1.0 / w) + U.T @ DU
            self.U, self.DU = U, DU
            self.Sinv = np.linalg.inv(S)
        else:
            self.U = None

    def solve(self, b):
        x = self.dinv * b
        if self.U is not None:
            x -= self.DU @ (self.Sinv @ (self.U.T @ x))
        return x


def solve(pb: Problem, *, max_iter: int = 20000, eps_abs: float = 1e-7, eps_rel: float = 1e-5,
          rho0: float = 0.1, relax: float = 1.6, adapt_every: int = 25, warm: dict | None = None) -> Result:
    """ADMM solve of ``pb`` (see module docstring). ``warm`` = ``Result.info['state']``
    of a previous solve of a problem with the same shape (warm start)."""
    n = pb.n()
    lb = np.zeros(n) if pb.lb is None else np.asarray(pb.lb, dtype=float)
    ub = np.ones(n) if pb.ub is None else np.asarray(pb.ub, dtype=float)
    pb.lb, pb.ub = lb, ub
    bench = np.zeros(n) if pb.bench is None else np.asarray(pb.bench, dtype=float)
    w0 = np.zeros(n) if pb.w_prev is None else np.asarray(pb.w_prev, dtype=float)
    alpha = np.asarray(pb.alpha, dtype=float)

    # objective scaling: make the linear term O(1)
    Xb = pb.X.T @ bench
    Sb = pb.X @ (pb.F @ Xb) + pb.D * bench                # Sigma b
    q_raw = -alpha - 2.0 * pb.lam * Sb
    scale = 1.0 / max(np.abs(q_raw).max(), np.abs(alpha).max(), 1e-12)
    ops = _Ops(pb, scale)
    q = q_raw * scale
    lin = (np.zeros(n) if pb.lin_cost is None else np.asarray(pb.lin_cost, float)) * scale
    imp = (np.zeros(n) if pb.imp_cost is None else np.asarray(pb.imp_cost, float)) * scale
    fee = (np.zeros(n) if pb.borrow is None else np.asarray(pb.borrow, float)) * scale

    glo = np.zeros(0) if pb.glo is None else np.asarray(pb.glo, float)
    ghi = np.zeros(0) if pb.ghi is None else np.asarray(pb.ghi, float)
    eq = np.abs(ghi - glo) < 1e-12
    if ops.has_te:
        center = np.concatenate([ops.XL.T @ bench, ops.sqD * bench])

    rho_base = np.full(ops.n_rows, 1.0)
    rho_base[ops.sl[1]] = np.where(eq, 1e3, 1.0)
    rho_s = rho0

    def project(v, rho):
        z = np.empty_like(v)
        z[ops.sl[0]] = np.clip(v[ops.sl[0]], lb, ub)
        z[ops.sl[1]] = np.clip(v[ops.sl[1]], glo, ghi)
        if ops.has_cost:
            s = ops.sl[2]
            z[s] = w0 + prox_cost(v[s] - w0, lin, imp, rho[s])
        if ops.has_te:
            s = ops.sl[3]
            z[s] = project_ball(v[s], center, float(pb.te))
        if ops.has_short:
            s = ops.sl[4]
            z[s] = prox_short(v[s], fee, pb.short_budget, rho_s * 1.0)
        return z

    if warm is not None and len(warm.get("x", [])) == n and len(warm.get("z", [])) == ops.n_rows:
        x, z, u = warm["x"].copy(), warm["z"].copy(), warm["u"].copy()
        rho_s = warm.get("rho", rho0)
    else:
        x = np.clip(w0 if pb.w_prev is not None else bench if pb.bench is not None else np.full(n, 1.0 / n), lb, ub)
        z = project(ops.A(x), rho_base * rho_s)
        u = np.zeros(ops.n_rows)
    rho = rho_base * rho_s
    ops.factor(rho)
    status = "max_iter"
    rp = rd = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        x = ops.solve(-q + ops.At(rho * (z - u)))
        Ax = ops.A(x)
        Axr = relax * Ax + (1.0 - relax) * z
        z_prev = z
        z = project(Axr + u, rho)
        u = u + Axr - z
        if it % 5 == 0 or it == max_iter:
            rp = float(np.abs(Ax - z).max())
            rd = float(np.abs(ops.At(rho * (z - z_prev))).max())
            Px = ops.lam2 * (ops.XL @ (ops.XL.T @ x) + ops.D * x)
            e_p = eps_abs + eps_rel * max(np.abs(Ax).max(), np.abs(z).max())
            e_d = eps_abs + eps_rel * max(np.abs(Px).max(), np.abs(ops.At(rho * u)).max(), np.abs(q).max())
            if rp <= e_p and rd <= e_d:
                status = "solved"
                break
            if it % adapt_every == 0:
                ratio = np.sqrt((rp / max(np.abs(Ax).max(), np.abs(z).max(), 1e-12))
                                / max(rd / max(np.abs(ops.At(rho * u)).max(), np.abs(q).max(), 1e-12), 1e-30))
                if ratio > 5.0 or ratio < 0.2:
                    new = float(np.clip(rho_s * ratio, 1e-6, 1e6))
                    u = u * (rho_s / new)
                    rho_s = new
                    rho = rho_base * rho_s
                    ops.factor(rho)
    w = np.clip(z[ops.sl[0]], lb, ub)
    w[np.abs(w) < 1e-9] = 0.0
    res = Result(w=w, status=status, iters=it, obj=objective(pb, w), prim_res=rp, dual_res=rd,
                 info={"state": {"x": x, "z": z, "u": u, "rho": rho_s}})
    return res


def objective(pb: Problem, w: np.ndarray) -> float:
    """Value of the maximised objective (net of costs/borrow) at ``w``."""
    b = np.zeros_like(w) if pb.bench is None else pb.bench
    a = w - b
    xa = pb.X.T @ a
    var = float(xa @ pb.F @ xa + (pb.D * a * a).sum())
    t = w - (np.zeros_like(w) if pb.w_prev is None else pb.w_prev)
    cost = 0.0
    if pb.lin_cost is not None:
        cost += float((pb.lin_cost * np.abs(t)).sum())
    if pb.imp_cost is not None:
        cost += float((pb.imp_cost * np.abs(t) ** 1.5).sum())
    if pb.borrow is not None:
        cost += float((pb.borrow * np.maximum(-w, 0.0)).sum())
    return float(pb.alpha @ w - pb.lam * var - cost)


def risk(X, F, D, w) -> float:
    """Standard deviation of ``w`` under X F X' + diag(D)."""
    xw = X.T @ w
    return float(np.sqrt(max(xw @ F @ xw + (D * w * w).sum(), 0.0)))


def solve_with_cardinality(pb: Problem, min_weight: float = 0.0, **kw) -> Result:
    """Solve; then (if ``min_weight`` > 0) force every name with
    0 < |w| < min_weight (and not already held at >= min_weight) to zero and
    re-solve once, warm-started."""
    res = solve(pb, **kw)
    if min_weight <= 0:
        return res
    small = (np.abs(res.w) > 0) & (np.abs(res.w) < min_weight)
    if not small.any():
        return res
    pb.lb = np.where(small, 0.0, pb.lb)
    pb.ub = np.where(small, 0.0, pb.ub)
    res2 = solve(pb, warm=res.info["state"], **kw)
    res2.info["n_dropped_small"] = int(small.sum())
    res2.iters += res.iters
    return res2


# --------------------------------------------------------------------------- #
# Alpha and the full book (shared by the round-5 backtest and the live agent)
# --------------------------------------------------------------------------- #
def normal_scores(x) -> np.ndarray:
    """Cross-sectional rank -> inverse-normal score, re-standardised (mean 0, sd 1).
    Works for any monotone model output (v3 ``pred``, v2 ``pred_rank``)."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n < 2:
        return out
    r = stats.rankdata(x[ok])
    z = stats.norm.ppf((r - 0.5) / n)
    out[ok] = (z - z.mean()) / z.std()
    return out


def grinold_alpha(scores, spec_vol_daily, ic: float, label_h: int = 30, horizon: int = 21) -> np.ndarray:
    """alpha = IC x sigma_resid x z (Grinold & Kahn): expected residual return
    over ``horizon`` sessions; sigma_resid is the risk model's specific vol over
    the label horizon, linearly pro-rated from ``label_h`` to ``horizon``."""
    z = normal_scores(scores)
    s = np.asarray(spec_vol_daily, dtype=float) * np.sqrt(label_h)
    return max(float(ic), 0.0) * s * z * (horizon / label_h)


DEFAULT_SPEC = {"kind": "active", "te": 0.04, "lam": 1.0, "name_cap": 0.05, "active_cap": 0.03,
                "sector_band": 0.05, "beta_band": 0.05, "style_band": None, "short_budget": 0.0,
                "short_cap": 0.0, "liq_frac": 0.10, "min_weight": 0.001, "horizon": 21, "label_h": 30,
                "cost_mult": 1.0, "impact_k": 0.2}


def _cap_array(ub_cap, tickers) -> np.ndarray | None:
    """``ub_cap`` (dict ticker -> cap or (N,) array) -> (N,) array >= 0, np.inf = no cap."""
    if ub_cap is None:
        return None
    if isinstance(ub_cap, dict):
        cap = np.array([float(ub_cap.get(t, np.inf)) for t in np.asarray(tickers)])
    else:
        cap = np.asarray(ub_cap, float)
        if cap.shape != (len(tickers),):
            raise ValueError(f"ub_cap has shape {cap.shape}, expected ({len(tickers)},)")
    return np.maximum(np.where(np.isnan(cap), np.inf, cap), 0.0)


def optimize_portfolio(spec: dict, *, tickers, scores, X, F_daily, D_daily, mcap, sector, factors,
                       half_spread, sigma_daily, adv, aum: float, ic: float, w_prev: dict | None = None,
                       shortable=None, borrow_annual=None, solver_kw: dict | None = None, ub_cap=None):
    """Cost-aware mean-variance book for one rebalance.

    ``spec['kind']``:
      * ``absolute`` -- maximise alpha'w - lam w'Sigma w - cost, long-only, sum 1,
        per-name cap ``name_cap``, sector weights within +-``sector_band`` of the
        cap-weighted universe;
      * ``active``   -- benchmark = cap-weighted universe; maximise
        alpha'w - lam (w-b)'Sigma(w-b) - cost s.t. ex-ante TE <= ``te`` (annual),
        long-only, w_i <= b_i + ``active_cap``, sector active weights within
        +-``sector_band``, ex-ante beta to b in 1 +- ``beta_band``, optional
        |active style exposure| <= ``style_band`` for every style factor;
      * ``130_30``   -- as ``active`` plus shorts down to -``short_cap`` on the
        ``shortable`` names, total short <= ``short_budget`` (gross 1 + 2 x budget),
        borrow ``borrow_annual`` pro-rated to the horizon.
    Every name is also capped at ``liq_frac`` x ADV / AUM. Costs: half-spread +
    k sigma sqrt(trade/ADV) (``costs.py``) on w - w_prev, times ``cost_mult``.
    All risk/alpha terms are over ``horizon`` sessions. Names in ``w_prev`` that
    are not in ``tickers`` are NOT handled here (the caller sells them).
    ``ub_cap`` (optional): extra per-name upper bound, a dict ticker -> cap
    (missing = none) or an (N,) array (np.inf = none); negative caps count as 0.
    It is applied as an upper bound and again after the budget polish (the
    polish must not lift a capped name; the leftover stays in cash) -- for
    ``kind == 'absolute'`` exactly ``round11.optimize_ext(..., ub_cap=...)``.

    Returns (weights dict ticker -> w, diagnostics dict)."""
    from . import costs as cs
    sp = {**DEFAULT_SPEC, **spec}
    tickers = np.asarray(tickers)
    n = len(tickers)
    H = int(sp["horizon"])
    X = np.asarray(X, float)
    F = np.asarray(F_daily, float) * H
    D = np.asarray(D_daily, float) * H
    mcap = np.asarray(mcap, float)
    b = np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    b = b / b.sum()
    alpha = grinold_alpha(scores, np.sqrt(np.asarray(D_daily, float)), ic, sp["label_h"], H)
    adv = np.asarray(adv, float)
    liq = np.where(np.isfinite(adv) & (adv > 0), sp["liq_frac"] * adv / aum, 0.0)
    kind = sp["kind"]
    if kind == "absolute":
        ub = np.minimum(sp["name_cap"], liq)
    else:
        ub = np.minimum(b + sp["active_cap"], liq)
    cap = _cap_array(ub_cap, tickers)
    if cap is not None:
        ub = np.minimum(ub, cap)
    lb = np.zeros(n)
    short_budget = None
    borrow = None
    if kind == "130_30":
        sh = np.zeros(n, dtype=bool) if shortable is None else np.isin(tickers, list(shortable))
        lb = np.where(sh, -np.minimum(sp["short_cap"], liq), 0.0)
        short_budget = float(sp["short_budget"])
        ba = np.full(n, cs.BORROW_EASY) if borrow_annual is None else np.asarray(borrow_annual, float)
        borrow = ba * H / 252.0
    rows, lo, hi = [np.ones(n)], [1.0], [1.0]
    sector = np.asarray(sector)
    for s in np.unique(sector):
        m = (sector == s).astype(float)
        bs = float(m @ b)
        rows.append(m)
        lo.append(max(bs - sp["sector_band"], 0.0 if kind != "130_30" else -1.0))
        hi.append(min(bs + sp["sector_band"], 1.0 if kind != "130_30" else 2.0))
    Sb = X @ (F @ (X.T @ b)) + D * b
    beta_b = Sb / float(b @ Sb)
    if kind != "absolute" and sp.get("beta_band") is not None:
        rows.append(beta_b)
        lo.append(1.0 - sp["beta_band"])
        hi.append(1.0 + sp["beta_band"])
    n_styles = sum(1 for f in factors if not str(f).startswith("ind:"))
    if kind != "absolute" and sp.get("style_band") is not None:
        xb = X[:, :n_styles].T @ b
        for k in range(n_styles):
            rows.append(X[:, k])
            lo.append(xb[k] - sp["style_band"])
            hi.append(xb[k] + sp["style_band"])
    wp = np.zeros(n)
    if w_prev:
        idx = {t: i for i, t in enumerate(tickers)}
        for t, w in w_prev.items():
            if t in idx:
                wp[idx[t]] = w
    hs = np.asarray(half_spread, float)
    imp = cs.impact_coef(sigma_daily, adv, aum, sp["impact_k"])
    te = None if kind == "absolute" else float(sp["te"]) * np.sqrt(H / 252.0)
    pb = Problem(alpha=alpha, X=X, F=F, D=D, lam=float(sp["lam"]), bench=None if kind == "absolute" else b,
                 w_prev=wp, lin_cost=sp["cost_mult"] * np.nan_to_num(hs, nan=cs.DEFAULT_HS),
                 imp_cost=sp["cost_mult"] * imp, lb=lb, ub=np.maximum(ub, lb), G=np.vstack(rows),
                 glo=np.array(lo), ghi=np.array(hi), te=te, short_budget=short_budget, borrow=borrow)
    res = solve_with_cardinality(pb, sp["min_weight"], **(solver_kw or {}))
    w = res.w.copy()
    net = float(w.sum())
    pos = w > 0
    if abs(net - 1.0) < 1e-2 and pos.any():      # polish the ADMM residual on the budget row
        w[pos] = w[pos] * (1.0 + (1.0 - net) / w[pos].sum())
    elif net > 1.0:
        w = w / net
    if cap is not None:                          # the polish must not lift a capped name (leftover -> cash)
        w = np.minimum(w, cap + 1e-12)
    weights = {t: float(x) for t, x in zip(tickers, w) if x != 0.0}
    a = w - b
    t_ = w - wp
    diag = {"status": res.status, "iters": res.iters, "n_names": int((w != 0).sum()),
            "n_short": int((w < 0).sum()), "net": float(w.sum()), "gross": float(np.abs(w).sum()),
            "ex_ante_vol_ann": risk(X, F, D, w) * np.sqrt(252.0 / H),
            "ex_ante_te_ann": risk(X, F, D, a) * np.sqrt(252.0 / H),
            "ex_ante_beta": float(beta_b @ w), "alpha_ann": float(alpha @ w) * 252.0 / H,
            "active_alpha_ann": float(alpha @ a) * 252.0 / H,
            "exp_cost": float((pb.lin_cost * np.abs(t_)).sum() + (pb.imp_cost * np.abs(t_) ** 1.5).sum()),
            "turnover_oneway": float(np.abs(t_).sum() / 2.0), "ic": float(ic),
            "active_exposure": dict(zip([str(f) for f in factors], (X.T @ a).tolist())),
            "exposure": dict(zip([str(f) for f in factors], (X.T @ w).tolist()))}
    return weights, diag
