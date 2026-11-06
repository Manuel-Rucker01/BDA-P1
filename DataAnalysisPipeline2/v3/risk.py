"""Point-in-time fundamental factor risk model (round 5).

``Sigma = X F X' + diag(D)`` (daily units) for the eligible names on a date,
estimated from data up to and including that date's close only.

Factors
    * 8 style factors, cross-sectionally standardised over eligible names
      (winsorised at the 1st/99th percentile, cap-weighted mean 0,
      equal-weighted standard deviation 1, missing -> 0 = the cap-weighted mean):
        beta        252-session (min 120) regression slope on the cap-weighted
                    eligible market return
        size        log market cap (cleaned estimate, see ``clean_mcap``)
        value       mean of the standardised B/P and E/P (EDGAR, as filed)
        momentum    sum of log returns t-252 .. t-21 (min 200 sessions)
        reversal    sum of log returns t-20 .. t
        volatility  60-session return std, orthogonalised to beta and size
        liquidity   log(20-session ADV / mcap), orthogonalised to size
        quality     mean of the standardised GP/A and ROA
    * industry factors: EDGAR SIC -> Fama-French-12 sector dummies (proxy
      sector where EDGAR has none, as in the v3 dataset). No intercept: the
      industry factors jointly carry the market.
Factor returns
    Daily cross-sectional WLS of returns on day t (clipped to [-50%, +100%])
    on exposures at t-1, weights sqrt(mcap_{t-1}), over names eligible at t-1.
Factor covariance
    EWMA (half-life ``FACTOR_HL`` = 90 sessions) of the factor returns with a
    Newey-West (Bartlett, ``NW_LAGS`` = 2) correction of daily serial
    correlation; optional volatility-regime adjustment (VRA, USE4): F is scaled
    by lambda_F^2 = EWMA_{hl 42}(B_t^2), B_t^2 = mean_k (f_kt / sigma_k,t-1)^2.
Specific variance
    EWMA (half-life ``SPEC_HL`` = 90) of squared residuals, Bayesian-shrunk
    toward the cap-weighted mean specific vol of the name's sector (USE4
    formula, intensity ``SHRINK_Q``); names with < ``SPEC_MIN_OBS`` residuals
    get the sector mean; optional VRA with the cap-weighted specific bias
    statistic.

Inputs come as a long (ticker, Date) frame -- offline from the v3 EDGAR daily
dataset, live from ``market_data.fetch_daily_bars`` + EDGAR files (``live_panel``)
-- turned into a dense ``Panel`` of dates x tickers arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STYLES = ["beta", "size", "value", "momentum", "reversal", "volatility", "liquidity", "quality"]
FACTOR_HL = 90
NW_LAGS = 2
SPEC_HL = 90
SPEC_MIN_OBS = 42
SHRINK_Q = 0.1
VRA_HL = 42
BETA_WIN, BETA_MIN = 252, 120
MOM_LO, MOM_HI, MOM_MIN = 252, 21, 200
REV_WIN = 21
VOL_WIN, VOL_MIN = 60, 40
EST_CLIP = (-0.5, 1.0)
MCAP_OUTLIER_SD = 5.0
MCAP_OUTLIER_MIN_LOG = float(np.log(20.0))   # never flag a value within 20x of the fit


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    dates: np.ndarray            # (T,) 'YYYY-MM-DD'
    tickers: np.ndarray          # (N,)
    ret: np.ndarray              # (T, N) daily return (NaN before the first bar)
    elig: np.ndarray             # (T, N) bool
    mcap: np.ndarray             # (T, N) cleaned market cap estimate ($)
    adv: np.ndarray              # (T, N) 20-session ADV ($)
    price: np.ndarray            # (T, N) raw close
    sector: np.ndarray           # (T, N) int16 code, -1 = unknown
    sector_names: list
    fund: dict = field(default_factory=dict)   # name -> (T, N): bp, ep, gpa, roa

    def t(self, date: str) -> int:
        """Index of the last session <= date."""
        i = int(np.searchsorted(self.dates, date, side="right")) - 1
        if i < 0:
            raise KeyError(date)
        return i


def _dense(df: pd.DataFrame, col: str, ti: np.ndarray, di: np.ndarray, shape, dtype=np.float32, fill=np.nan):
    a = np.full(shape, fill, dtype=dtype)
    a[di, ti] = df[col].to_numpy(dtype=dtype)
    return a


def panel_from_long(df: pd.DataFrame, ret: pd.DataFrame | None = None, foreign=None) -> Panel:
    """``df``: ticker, Date, company_close, close_raw, adv_usd, eligible, Sector,
    log_mcap (EDGAR, may be NaN), fund_imputed (optional), bp, ep, gpa, roa.
    ``ret``: optional dates x tickers return matrix (``backtest.daily_returns_matrix``);
    computed from the closes if absent. ``foreign``: tickers whose EDGAR filer is a
    foreign private issuer (20-F/40-F: shares are ordinary shares, not ADSs, so
    close x shares is not the ADR's market cap)."""
    from . import backtest as bt
    dates = np.array(sorted(df["Date"].astype(str).unique()))
    tick = np.array(sorted(df["ticker"].astype(str).unique()))
    di = np.searchsorted(dates, df["Date"].astype(str).to_numpy())
    ti = np.searchsorted(tick, df["ticker"].astype(str).to_numpy())
    shape = (len(dates), len(tick))
    if ret is None:
        close = pd.DataFrame(_dense(df, "company_close", ti, di, shape, np.float64), index=dates, columns=tick)
        raw = pd.DataFrame(_dense(df, "close_raw", ti, di, shape, np.float64), index=dates, columns=tick)
        ret = bt.daily_returns_matrix(close, raw=raw)
    R = ret.reindex(index=dates, columns=tick).to_numpy(dtype=np.float32)
    elig = np.zeros(shape, dtype=bool)
    elig[di, ti] = df["eligible"].fillna(False).astype(bool).to_numpy()
    secs = df["Sector"].fillna("Unknown").astype(str)
    names = sorted(s for s in secs.unique() if s not in ("Unknown", "None", ""))
    code = pd.Series(np.arange(len(names)), index=names)
    sector = np.full(shape, -1, dtype=np.int16)
    sector[di, ti] = secs.map(code).fillna(-1).astype(np.int16).to_numpy()
    price = _dense(df, "close_raw", ti, di, shape)
    adv = _dense(df, "adv_usd", ti, di, shape)
    lm = df["log_mcap"].astype(float)
    if "fund_imputed" in df:
        lm = lm.where(~df["fund_imputed"].fillna(False).astype(bool))
    if foreign is not None:
        lm = lm.where(~df["ticker"].isin(set(foreign)))
    raw_mcap = np.full(shape, np.nan, dtype=np.float32)
    raw_mcap[di, ti] = np.exp(lm.to_numpy())
    fund = {c: _dense(df, c, ti, di, shape) for c in ("bp", "ep", "gpa", "roa") if c in df}
    if "fund_imputed" in df:      # descriptors from a hot-deck donor are not the name's own
        imp = np.zeros(shape, dtype=bool)
        imp[di, ti] = df["fund_imputed"].fillna(False).astype(bool).to_numpy()
        for c in fund:
            fund[c][imp] = np.nan
    mcap = clean_mcap(raw_mcap, adv, price, elig)
    return Panel(dates, tick, R, elig, mcap, adv, price, sector, names, fund)


def clean_mcap(raw_mcap: np.ndarray, adv: np.ndarray, price: np.ndarray, elig: np.ndarray,
               outlier_sd: float = MCAP_OUTLIER_SD) -> np.ndarray:
    """Per date: OLS of log EDGAR mcap on [1, log ADV, log price] over eligible
    names with an own EDGAR mcap; values more than ``outlier_sd`` robust sd from
    the fit and at least 20x off (bad share counts, unit errors) and names without an own mcap get the
    fitted value. PIT (same-date cross-section only), survivorship-neutral
    (the fill depends on the name's own trading data, not on EDGAR coverage)."""
    T, N = raw_mcap.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    lad = np.log(np.maximum(adv, 1.0))
    lpx = np.log(np.maximum(price, 1e-3))
    for t in range(T):
        ok = np.isfinite(lad[t]) & np.isfinite(lpx[t]) & (adv[t] > 0)
        if not ok.any():
            continue
        y = np.log(raw_mcap[t])
        Z = np.column_stack([np.ones(N), lad[t], lpx[t]])
        fit = ok & elig[t] & np.isfinite(y)
        if fit.sum() < 30:
            fit = ok & np.isfinite(y)
        if fit.sum() < 10:
            continue
        bcoef = np.linalg.lstsq(Z[fit], y[fit], rcond=None)[0]
        res = y - Z @ bcoef
        sd = 1.4826 * np.nanmedian(np.abs(res[fit] - np.nanmedian(res[fit])))
        thr = max(outlier_sd * sd, MCAP_OUTLIER_MIN_LOG)
        good = fit & (np.abs(res) <= thr)
        bcoef = np.linalg.lstsq(Z[good], y[good], rcond=None)[0]
        yhat = Z @ bcoef
        res = y - yhat
        keep = np.isfinite(y) & (np.abs(res) <= thr)
        est = np.where(keep, y, yhat)
        out[t] = np.where(ok, np.exp(est), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Descriptors and exposures
# --------------------------------------------------------------------------- #
def _rolling_sum(a: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing ``win``-row sums of a (NaN as 0) and counts of finite values."""
    v = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    n = np.isfinite(a).astype(np.float64)
    cv = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(v, axis=0)])
    cn = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(n, axis=0)])
    idx = np.arange(a.shape[0]) + 1
    lo = np.maximum(idx - win, 0)
    return cv[idx] - cv[lo], cn[idx] - cn[lo]


def market_return(p: Panel) -> np.ndarray:
    """Cap-weighted (mcap at t-1) return of names eligible at t-1."""
    r = np.clip(p.ret, *EST_CLIP)
    w = np.vstack([np.full((1, r.shape[1]), np.nan), np.where(p.elig, p.mcap, np.nan)[:-1]])
    ok = np.isfinite(w) & np.isfinite(r)
    num = np.where(ok, w * r, 0.0).sum(axis=1)
    den = np.where(ok, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def raw_descriptors(p: Panel) -> dict:
    """Dates x tickers raw descriptors (float32), each using data <= that date."""
    r = np.clip(p.ret.astype(np.float64), *EST_CLIP)
    m = market_return(p)[:, None]
    has = np.isfinite(r)
    mm = np.where(has, m, np.nan)
    sx, n = _rolling_sum(r, BETA_WIN)
    sm, _ = _rolling_sum(mm, BETA_WIN)
    sxm, _ = _rolling_sum(r * mm, BETA_WIN)
    smm, _ = _rolling_sum(mm * mm, BETA_WIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxm / n - sx / n * sm / n
        var = smm / n - (sm / n) ** 2
        beta = np.where(n >= BETA_MIN, cov / var, np.nan)
    lr = np.log1p(r)
    s_all, n_all = _rolling_sum(lr, MOM_LO)
    s_rec, n_rec = _rolling_sum(lr, MOM_HI)
    mom = np.where(n_all - n_rec >= MOM_MIN, s_all - s_rec, np.nan)
    s_rev, n_rev = _rolling_sum(lr, REV_WIN)
    rev = np.where(n_rev >= REV_WIN - 3, s_rev, np.nan)
    s1, nv = _rolling_sum(r, VOL_WIN)
    s2, _ = _rolling_sum(r * r, VOL_WIN)
    with np.errstate(invalid="ignore"):
        vol = np.where(nv >= VOL_MIN, np.sqrt(np.maximum(s2 / nv - (s1 / nv) ** 2, 0.0)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(p.mcap.astype(np.float64))
        liq = np.log(p.adv.astype(np.float64) / p.mcap)
    d = {"beta": beta, "size": size, "momentum": mom, "reversal": rev, "volatility": vol, "liquidity": liq}
    for c in ("bp", "ep", "gpa", "roa"):
        d[c] = p.fund[c].astype(np.float64) if c in p.fund else np.full(r.shape, np.nan)
    return {k: v.astype(np.float32) for k, v in d.items()}


def standardize_row(x: np.ndarray, w: np.ndarray, mask: np.ndarray, pct=(1.0, 99.0)) -> np.ndarray:
    """Winsorise at ``pct`` percentiles, then (x - cap-weighted mean) / equal-weighted sd over
    ``mask``; missing (in mask) -> 0; outside mask -> NaN."""
    out = np.full(x.shape, np.nan)
    v = mask & np.isfinite(x)
    if v.sum() < 5:
        out[mask] = 0.0
        return out
    lo, hi = np.percentile(x[v], pct)
    xc = np.clip(x, lo, hi)
    ww = np.where(v & np.isfinite(w), w, 0.0)
    mu = (ww * np.where(v, xc, 0)).sum() / ww.sum() if ww.sum() > 0 else xc[v].mean()
    sd = xc[v].std()
    z = (xc - mu) / sd if sd > 0 else xc * 0.0
    out[mask] = np.where(np.isfinite(z[mask]), z[mask], 0.0)
    return out


def _orth(y: np.ndarray, Z: np.ndarray, w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Residual of a weighted regression of y on [1, Z] over mask (rest unchanged)."""
    A = np.column_stack([np.ones(mask.sum()), Z[mask]])
    sw = np.sqrt(np.where(np.isfinite(w[mask]), w[mask], 0.0))
    coef = np.linalg.lstsq(A * sw[:, None], y[mask] * sw, rcond=None)[0]
    out = y.copy()
    out[mask] = y[mask] - A @ coef
    return out


def exposures_at(p: Panel, desc: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask of eligible names at t, (N, 8 + n_sectors) exposure matrix with NaN off-mask)."""
    mask = p.elig[t] & np.isfinite(p.mcap[t]) & (p.mcap[t] > 0)
    w = p.mcap[t].astype(np.float64)
    st = {}
    for k in ("beta", "size", "momentum", "reversal", "liquidity", "volatility"):
        st[k] = standardize_row(desc[k][t].astype(np.float64), w, mask)
    st["liquidity"] = standardize_row(_orth(st["liquidity"], st["size"][:, None], w, mask), w, mask)
    st["volatility"] = standardize_row(_orth(st["volatility"], np.column_stack([st["beta"], st["size"]]), w, mask),
                                       w, mask)
    v = [standardize_row(desc[c][t].astype(np.float64), w, mask) for c in ("bp", "ep")]
    st["value"] = standardize_row((v[0] + v[1]) / 2.0, w, mask)
    q = [standardize_row(desc[c][t].astype(np.float64), w, mask) for c in ("gpa", "roa")]
    st["quality"] = standardize_row((q[0] + q[1]) / 2.0, w, mask)
    S = np.column_stack([st[k] for k in STYLES])
    sec = p.sector[t]
    ind = np.zeros((len(sec), len(p.sector_names)))
    known = sec >= 0
    ind[np.flatnonzero(known), sec[known]] = 1.0
    other = p.sector_names.index("Other") if "Other" in p.sector_names else len(p.sector_names) - 1
    ind[~known, other] = 1.0
    X = np.hstack([S, ind])
    X[~mask] = np.nan
    return mask, X


def factor_names(p: Panel) -> list:
    return STYLES + [f"ind:{s}" for s in p.sector_names]


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
@dataclass
class RiskModel:
    date: str
    tickers: np.ndarray
    X: np.ndarray                # (N, K)
    F: np.ndarray                # (K, K) daily
    D: np.ndarray                # (N,) daily specific variance
    factors: list
    mcap: np.ndarray             # (N,)
    sector: np.ndarray           # (N,) codes
    sector_names: list

    def vol(self, w: np.ndarray, horizon: int = 1) -> float:
        xw = self.X.T @ w
        return float(np.sqrt(max(xw @ self.F @ xw + (self.D * w * w).sum(), 0.0) * horizon))

    def spec_vol(self) -> np.ndarray:
        return np.sqrt(self.D)

    def cov(self) -> np.ndarray:
        return self.X @ self.F @ self.X.T + np.diag(self.D)

    def subset(self, keep: np.ndarray) -> "RiskModel":
        return RiskModel(self.date, self.tickers[keep], self.X[keep], self.F, self.D[keep], self.factors,
                         self.mcap[keep], self.sector[keep], self.sector_names)


class RiskEngine:
    """Fit once over a panel (daily factor returns, residuals, EWMA states);
    ``model(date)`` then returns the PIT ``RiskModel`` for that date."""

    def __init__(self, p: Panel, vra: bool = True, start: int | None = None, log=None):
        self.p, self.vra, self.log = p, vra, log or (lambda *a: None)
        self.desc = raw_descriptors(p)
        self.names = factor_names(p)
        T, N = p.ret.shape
        K = len(self.names)
        self.K = K
        self.f = np.full((T, K), np.nan)
        self.n_reg = np.zeros(T, dtype=int)
        self.start = start if start is not None else BETA_WIN
        self._X = {}
        self.spec_var = np.full((T, N), np.nan, dtype=np.float32)   # EWMA estimate known at t
        self.spec_n = np.zeros((T, N), dtype=np.int16)
        self.bS = np.full(T, np.nan)
        self.bF = np.full(T, np.nan)
        self._fit()

    def X_at(self, t):
        if t not in self._X:
            self._X[t] = exposures_at(self.p, self.desc, t)
        return self._X[t]

    def _fit(self):
        p = self.p
        T, N = p.ret.shape
        a = 1.0 - 0.5 ** (1.0 / SPEC_HL)
        V = np.zeros(N)
        S = np.zeros(N)
        cnt = np.zeros(N, dtype=np.int32)
        r_all = np.clip(p.ret, *EST_CLIP)
        prev = None
        for t in range(self.start, T):
            if prev is None:
                prev = self.X_at(t - 1)
            mask, X = prev
            r = r_all[t].astype(np.float64)
            ok = mask & np.isfinite(r)
            u_t = np.full(N, np.nan)
            if ok.sum() > 3 * self.K:
                w = np.sqrt(p.mcap[t - 1, ok].astype(np.float64))
                Xo = X[ok]
                cols = Xo.any(axis=0) | (np.arange(self.K) < len(STYLES))
                f = np.zeros(self.K)
                f[cols] = np.linalg.lstsq(Xo[:, cols] * w[:, None], r[ok] * w, rcond=None)[0]
                self.f[t] = f
                self.n_reg[t] = ok.sum()
                u_t[ok] = r[ok] - Xo @ f
            obs = np.isfinite(u_t)
            # specific bias statistic vs the prediction known at t-1 (VRA input)
            with np.errstate(invalid="ignore", divide="ignore"):
                sv_prev = np.where(S > 0, V / S, np.nan)
            okb = obs & np.isfinite(sv_prev) & (sv_prev > 0) & (cnt >= SPEC_MIN_OBS)
            if okb.sum() > 50:
                wb = p.mcap[t - 1, okb].astype(np.float64)
                self.bS[t] = float((wb * u_t[okb] ** 2 / sv_prev[okb]).sum() / wb.sum())
            V = np.where(obs, (1 - a) * V + a * u_t ** 2, V)
            S = np.where(obs, (1 - a) * S + a, S)
            cnt = cnt + obs
            with np.errstate(invalid="ignore", divide="ignore"):
                self.spec_var[t] = np.where(S > 0, V / S, np.nan)
            self.spec_n[t] = np.minimum(cnt, 32000)
            self._X.pop(t - 1, None)
            prev = self.X_at(t)
        self._bias_series()

    def _bias_series(self):
        """Daily factor bias statistic B_t^2 = mean_k (f_kt / sigma_k,t-1)^2 with
        sigma from the EWMA (``FACTOR_HL``) known at t-1 (VRA input; the
        specific one is computed in ``_fit``)."""
        a = 1.0 - 0.5 ** (1.0 / FACTOR_HL)
        s, n_obs = 0.0, 0
        acc = np.zeros(self.K)
        for t in range(len(self.p.dates)):
            f = self.f[t]
            if not np.all(np.isfinite(f)):
                continue
            if n_obs >= 20:
                sd = np.sqrt(acc / s)
                okk = sd > 1e-5
                if okk.any():
                    self.bF[t] = np.mean((f[okk] / sd[okk]) ** 2)
            acc = (1 - a) * acc + a * f * f
            s = (1 - a) * s + a
            n_obs += 1

    def X_at_cached(self, t):
        return self.X_at(t) if t in self._X else exposures_at(self.p, self.desc, t)

    def vra_multipliers(self, t: int) -> tuple[float, float]:
        if not self.vra:
            return 1.0, 1.0
        out = []
        for b in (self.bF, self.bS):
            x = b[: t + 1]
            ok = np.isfinite(x)
            if ok.sum() < 20:
                out.append(1.0)
                continue
            idx = np.flatnonzero(ok)
            age = t - idx
            w = 0.5 ** (age / VRA_HL)
            out.append(float(np.sqrt((w * x[ok]).sum() / w.sum())))
        return out[0], out[1]

    def factor_cov(self, t: int) -> np.ndarray:
        f = self.f[: t + 1]
        ok = np.all(np.isfinite(f), axis=1)
        idx = np.flatnonzero(ok)
        f = f[ok]
        age = t - idx
        w = 0.5 ** (age / FACTOR_HL)
        w = w / w.sum()
        mu = w @ f
        fc = f - mu
        C = (fc * w[:, None]).T @ fc
        for L in range(1, NW_LAGS + 1):
            if len(fc) <= L:
                break
            wl = w[L:] / w[L:].sum()
            G = (fc[L:] * wl[:, None]).T @ fc[:-L]
            C += (1.0 - L / (NW_LAGS + 1.0)) * (G + G.T)
        C = 0.5 * (C + C.T)
        ev, V = np.linalg.eigh(C)
        return (V * np.clip(ev, 1e-12, None)) @ V.T

    def specific_var(self, t: int, mask: np.ndarray) -> np.ndarray:
        p = self.p
        sv = self.spec_var[t].astype(np.float64)
        n = self.spec_n[t]
        sig = np.sqrt(sv)
        sec = p.sector[t]
        w = p.mcap[t].astype(np.float64)
        out = np.full(len(sv), np.nan)
        good = mask & np.isfinite(sig) & (n >= SPEC_MIN_OBS)
        glob = np.nanmedian(sig[good]) if good.any() else 0.03
        for s in np.unique(sec[mask]):
            g = mask & (sec == s)
            gg = g & good
            if gg.sum() >= 5:
                mu = float((w[gg] * sig[gg]).sum() / w[gg].sum())
                delta = float(np.sqrt(np.mean((sig[gg] - mu) ** 2)))
            else:
                mu, delta = glob, 0.0
            dev = np.abs(sig[g] - mu)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(delta + SHRINK_Q * dev > 0, SHRINK_Q * dev / (delta + SHRINK_Q * dev), 1.0)
            shr = v * mu + (1 - v) * sig[g]
            ok_g = good[g]
            out[g] = np.where(ok_g, shr, mu) ** 2
        return out

    def model(self, date: str) -> RiskModel:
        p = self.p
        t = p.t(date)
        mask, X = self.X_at_cached(t)
        F = self.factor_cov(t)
        D = self.specific_var(t, mask)
        lf, ls = self.vra_multipliers(t)
        F, D = F * lf ** 2, D * ls ** 2
        keep = mask & np.isfinite(D) & np.all(np.isfinite(X), axis=1)
        return RiskModel(str(p.dates[t]), p.tickers[keep], X[keep], F, D[keep], self.names,
                         p.mcap[t, keep].astype(np.float64), p.sector[t, keep], p.sector_names)


# --------------------------------------------------------------------------- #
# Validation: bias statistics
# --------------------------------------------------------------------------- #
def bias_stat(z) -> float:
    """Standard deviation of standardised returns r / sigma_pred (1 = unbiased)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    return float(z.std(ddof=1)) if len(z) > 2 else float("nan")


def standardized_returns(ret: np.ndarray, w: np.ndarray, sigma_daily: float) -> np.ndarray:
    """Daily returns of fixed weights ``w`` over the rows of ``ret`` (NaN -> 0),
    divided by the predicted daily vol."""
    r = np.nan_to_num(ret, nan=0.0) @ w
    return r / sigma_daily if sigma_daily > 0 else np.full(len(r), np.nan)


# --------------------------------------------------------------------------- #
# Offline / live panels
# --------------------------------------------------------------------------- #
OFFLINE_COLS = ["ticker", "Date", "company_close", "close_raw", "adv_usd", "eligible", "Sector", "log_mcap",
                "fund_imputed", "bp", "ep", "gpa", "roa"]


def foreign_filers(tickers, as_of: str | None = None, fund_dir: str | None = None) -> set:
    """Tickers whose EDGAR filer ever filed a 20-F/40-F (foreign private issuer)."""
    import os
    from . import dataset as ds
    from . import fundamentals as fu
    from ._alpaca import data_dir
    fund_dir = fund_dir or data_dir("fundamentals")
    cm = pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet"))
    fa = pd.read_parquet(os.path.join(fund_dir, "fundamentals_asof.parquet"), columns=["cik", "last_form"])
    fpi = set(fa.loc[fa["last_form"].astype(str).str.match(r"^(20|40)-F"), "cik"].astype(int))
    t = pd.Series(list(tickers), dtype=object)
    keys = pd.DataFrame({"tkey": t.map(ds._base_symbol).map(fu.norm_ticker).to_numpy(),
                         "Date": pd.Timestamp(as_of or "2026-12-31")})
    cik = fu.resolve_cik(keys, cm)
    return set(t[cik.map(lambda c: pd.notna(c) and int(c) in fpi).astype(bool).to_numpy()])


def offline_panel(start: str, end: str | None = None, log=print) -> Panel:
    """Panel from the v3 EDGAR daily dataset (names eligible at least once in
    [start, end])."""
    from . import dataset as ds
    years = range(int(start[:4]), int((end or "2026")[:4]) + 1)
    frames = []
    for y in years:
        d = ds.load_daily(years=[y], columns=OFFLINE_COLS, out_dir=ds.v3_dir(ds.EDGAR_DATASET))
        d = d[(d["Date"] >= start) & ((d["Date"] <= end) if end else True)]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    keep = set(df.loc[df["eligible"].astype(bool), "ticker"])
    df = df[df["ticker"].isin(keep)].reset_index(drop=True)
    log(f"[risk] panel rows {len(df):,}, tickers {len(keep):,}")
    fpi = foreign_filers(keep)
    log(f"[risk] foreign private issuers (mcap from the ADV fit): {len(fpi)}")
    return panel_from_long(df, foreign=fpi)


def live_panel(bars: pd.DataFrame, fund_dir: str | None = None, min_price: float = 3.0,
               min_adv_usd: float = 2e6, min_bars: int = 120) -> Panel:
    """Panel for the live agent from ``market_data.fetch_daily_bars`` output
    (ticker, Date, Open, High, Low, company_close, company_volume; adjusted) +
    the EDGAR files (cik_map, fundamentals_asof, sectors). Raw closes are not
    available live, so ``close_raw`` = the adjusted close (exact for the last
    bar, which is what the decision uses)."""
    import os
    from . import fundamentals as fu
    from . import universe as un
    from ._alpaca import data_dir
    fund_dir = fund_dir or data_dir("fundamentals")
    b = bars.copy()
    b["close_raw"] = b["company_close"]
    det = un.eligibility_details(b, min_price, min_adv_usd, min_bars)
    b["adv_usd"] = det["adv_usd"].to_numpy()
    b["eligible"] = det["eligible"].to_numpy()
    cm = pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet"))
    fa = pd.read_parquet(os.path.join(fund_dir, "fundamentals_asof.parquet"))
    secs = pd.read_parquet(os.path.join(fund_dir, "sectors.parquet"))
    ff = fu.fundamental_features(b[["ticker", "Date", "company_close"]], fa, cm, secs)
    for c in ("bp", "ep", "gpa", "roa", "log_mcap"):
        b[c] = ff[c].to_numpy()
    b["Sector"] = ff["sector"].fillna("Other").to_numpy()
    fpi = foreign_filers(b["ticker"].unique(), fund_dir=fund_dir)
    return panel_from_long(b, foreign=fpi)
