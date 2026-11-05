"""Portfolio construction aligned with the signal (WS4, docs/IMPROVEMENT_PLAN_v3.md).

Pure functions, no I/O. The model's skill is in the *broad* cross-sectional
ranking (top decile - bottom decile ~ +2.3 pp/month on a fresh holdout) while
the top-10 alone did not beat the average and the concentrated, all-semis book
lost ~33% live. So these books are wide, and the per-sector cap is STRICT: a
name whose sector is full is skipped, and the cap is never relaxed to fill K
(if K names cannot be admitted the book holds fewer and says so).

  long_only_book   -- K names in rank order, strict sector cap, inverse-vol
                      weights water-filled under a per-name cap; the rest is cash.
  long_short_book  -- top-ranked longs / bottom-ranked shorts, strict per-leg
                      sector caps, inverse-vol inside each leg, dollar-neutral,
                      optional sector-neutral weighting.

Both return ``(weights, diagnostics)``; weights are signed (shorts negative) and
every book is passed through ``enforce_portfolio_constraints`` +
``assert_portfolio_invariants`` before it is returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN_SECTOR = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Exposure invariants (re-exported by bot.py for backwards compatibility)
# --------------------------------------------------------------------------- #
def portfolio_exposure_summary(weights):
    """Return long/short/gross/net exposure for a signed weight mapping."""
    vals = np.asarray(list(weights.values()), dtype=float) if weights else np.asarray([], dtype=float)
    if vals.size == 0:
        return {"long": 0.0, "short": 0.0, "gross": 0.0, "net": 0.0}
    long_exposure = float(vals[vals > 0].sum())
    short_exposure = float(-vals[vals < 0].sum())
    return {
        "long": long_exposure,
        "short": short_exposure,
        "gross": long_exposure + short_exposure,
        "net": float(vals.sum()),
    }


def enforce_portfolio_constraints(weights, *, max_gross, max_net, max_short,
                                  max_long, max_position, tol=1e-10):
    """Clip and scale signed weights to satisfy portfolio exposure guardrails.

    The transformation only reduces exposure: per-name clips first, side caps
    second, then gross/net caps. This keeps rank-based allocations intact within
    each side while ensuring the final book cannot exceed configured limits.
    """
    if not weights:
        return {}

    capped = {
        t: float(np.clip(w, -max_position, max_position))
        for t, w in weights.items()
    }

    for _ in range(4):
        summary = portfolio_exposure_summary(capped)
        long_exp = summary["long"]
        short_exp = summary["short"]

        if long_exp > max_long + tol and long_exp > 0:
            scale = max_long / long_exp
            capped = {t: (w * scale if w > 0 else w) for t, w in capped.items()}
        if short_exp > max_short + tol and short_exp > 0:
            scale = max_short / short_exp
            capped = {t: (w * scale if w < 0 else w) for t, w in capped.items()}

        summary = portfolio_exposure_summary(capped)
        if summary["gross"] > max_gross + tol and summary["gross"] > 0:
            scale = max_gross / summary["gross"]
            capped = {t: w * scale for t, w in capped.items()}

        summary = portfolio_exposure_summary(capped)
        net = summary["net"]
        if net > max_net + tol and summary["long"] > 0:
            allowed_long = max(max_net + summary["short"], 0.0)
            scale = min(1.0, allowed_long / summary["long"])
            capped = {t: (w * scale if w > 0 else w) for t, w in capped.items()}
        elif net < -max_net - tol and summary["short"] > 0:
            allowed_short = max(max_net + summary["long"], 0.0)
            scale = min(1.0, allowed_short / summary["short"])
            capped = {t: (w * scale if w < 0 else w) for t, w in capped.items()}

    return {t: (0.0 if abs(w) < tol else float(w)) for t, w in capped.items()}


def assert_portfolio_invariants(weights, *, max_gross, max_net, max_short,
                                max_long, max_position, tol=1e-8,
                                sectors=None, max_names_per_sector=None):
    """Raise if a signed weight mapping breaches configured exposure limits.

    With ``sectors`` (ticker -> sector) and ``max_names_per_sector`` it also
    checks that no side (long / short) holds more than that many names from a
    single sector.
    """
    summary = portfolio_exposure_summary(weights)
    max_abs_name = max((abs(float(w)) for w in weights.values()), default=0.0)
    checks = {
        "gross": summary["gross"] <= max_gross + tol,
        "net": abs(summary["net"]) <= max_net + tol,
        "short": summary["short"] <= max_short + tol,
        "long": summary["long"] <= max_long + tol,
        "position": max_abs_name <= max_position + tol,
    }
    worst_sector = 0
    if sectors is not None and max_names_per_sector is not None:
        counts = {}
        for t, w in weights.items():
            if abs(w) <= tol:
                continue
            key = (w > 0, _norm_sector(sectors.get(t)))
            counts[key] = counts.get(key, 0) + 1
        worst_sector = max(counts.values(), default=0)
        checks["sector"] = worst_sector <= int(max_names_per_sector)
    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise AssertionError(
            f"Portfolio invariant breach ({failed}): "
            f"long={summary['long']:.6f}, short={summary['short']:.6f}, "
            f"gross={summary['gross']:.6f}, net={summary['net']:.6f}, "
            f"max_abs_position={max_abs_name:.6f}, max_names_in_a_sector={worst_sector}"
        )
    return summary


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _norm_sector(sec):
    if sec is None or (isinstance(sec, float) and np.isnan(sec)) or str(sec).strip() == "":
        return UNKNOWN_SECTOR
    return str(sec)


def capped_proportional(raw, budget, cap):
    """Water-filling: split ``budget`` proportionally to ``raw`` with
    ``w_i <= cap_i`` (scalar or per-element caps).

    Names that would exceed their cap are frozen at the cap and the rest of the
    budget is re-spread over the free names until nothing is over. If the caps
    make the full budget infeasible (sum of caps < budget) every name ends at
    its cap and the shortfall is left unallocated (cash) -- caps are never
    breached.
    """
    raw = np.asarray(raw, dtype=float)
    n = raw.size
    w = np.zeros(n)
    if n == 0 or budget <= 0:
        return w
    caps = np.broadcast_to(np.asarray(cap, dtype=float), (n,)).astype(float)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)
    if raw.sum() <= 0:
        raw = np.ones(n)
    free = (caps > 0) & (raw > 0)
    remaining = float(budget)
    for _ in range(n + 1):
        if remaining <= 1e-15 or not free.any():
            break
        trial = raw[free] / raw[free].sum() * remaining
        idx = np.flatnonzero(free)
        over = trial > caps[idx] + 1e-15
        if not over.any():
            w[idx] = trial
            break
        hit = idx[over]
        w[hit] = caps[hit]
        free[hit] = False
        remaining = float(budget) - float(w.sum())
    return w


def _inverse_vol(df, vol_col, vol_floor):
    if vol_col in df.columns:
        v = pd.to_numeric(df[vol_col], errors="coerce").astype(float)
        fill = float(v.median()) if v.notna().any() else 1.0
        v = v.fillna(fill)
    else:
        v = pd.Series(1.0, index=df.index)
    return 1.0 / np.maximum(v.to_numpy(dtype=float), vol_floor)


def _ranked(preds, rank_col, sector_col):
    """Clean, de-duplicated, best-first frame of tradable candidates."""
    df = preds.copy()
    if rank_col not in df.columns and "pred_proba" in df.columns:
        df[rank_col] = df["pred_proba"]
    df = df[pd.to_numeric(df[rank_col], errors="coerce").notna()]
    if "company_close" in df.columns:  # can't size an order without a price
        px = pd.to_numeric(df["company_close"], errors="coerce")
        df = df[px.notna() & (px > 0)]
    df = df.drop_duplicates("ticker", keep="first")
    df["_sector"] = (df[sector_col].map(_norm_sector) if sector_col in df.columns
                     else UNKNOWN_SECTOR)
    df = df.sort_values(rank_col, ascending=False, kind="mergesort")
    return df.reset_index(drop=True)


def _admit(df, n, max_per_sector, allowed=None, exclude=()):
    """Walk ``df`` in order admitting up to ``n`` names with a STRICT per-sector
    count cap. Returns (admitted rows, stats)."""
    chosen, counts = [], {}
    skipped_sector = skipped_not_allowed = 0
    exclude = set(exclude)
    for i, row in df.iterrows():
        if len(chosen) >= n:
            break
        t = row["ticker"]
        if t in exclude:
            continue
        if allowed is not None and t not in allowed:
            skipped_not_allowed += 1
            continue
        sec = row["_sector"]
        if counts.get(sec, 0) >= max_per_sector:
            skipped_sector += 1
            continue
        chosen.append(i)
        counts[sec] = counts.get(sec, 0) + 1
    return df.loc[chosen], {"sector_counts": counts,
                            "skipped_sector_full": skipped_sector,
                            "skipped_not_allowed": skipped_not_allowed}


def _beta_net(weights, df, beta_col):
    if beta_col not in df.columns:
        return None
    betas = pd.to_numeric(df.set_index("ticker")[beta_col], errors="coerce").fillna(1.0)
    return float(sum(w * float(betas.get(t, 1.0)) for t, w in weights.items()))


def _sector_net(weights, sectors):
    out = {}
    for t, w in weights.items():
        s = sectors.get(t, UNKNOWN_SECTOR)
        out[s] = out.get(s, 0.0) + float(w)
    return {s: v for s, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def long_only_book(preds, k, max_per_sector, max_weight, target_exposure,
                   vol_col="return_volatility_20d", *, sector_col="Sector",
                   rank_col="pred_rank", beta_col="kalman_beta", vol_floor=1e-3):
    """Broad long-only book.

    Walks the FULL ranking best-first admitting names while their sector has
    fewer than ``max_per_sector`` names (strict; never relaxed) until ``k``
    names are held. Weights are inverse-volatility, water-filled under
    ``max_weight`` and summing to at most ``target_exposure``; anything the
    caps cannot place stays in cash. Holds fewer than ``k`` names when the
    sector cap/universe does not allow k (``diagnostics['notes']`` says so).

    Returns ``(weights: dict ticker->weight, diagnostics: dict)``.
    """
    notes = []
    k, max_per_sector = int(k), int(max_per_sector)
    target_exposure, max_weight = float(target_exposure), float(max_weight)
    empty_diag = {"n_held": 0, "k": k, "long": 0.0, "short": 0.0, "gross": 0.0,
                  "net": 0.0, "cash": 1.0,"sector_counts": {},
                  "sector_net": {}, "beta_net": None, "notes": notes}
    if preds is None or len(preds) == 0 or k <= 0 or target_exposure <= 0 \
            or max_per_sector <= 0 or max_weight <= 0:
        notes.append("empty book (no candidates or non-positive k/cap/exposure)")
        return {}, empty_diag

    df = _ranked(preds, rank_col, sector_col)
    held, stats = _admit(df, k, max_per_sector)
    if len(held) < k:
        notes.append(f"only {len(held)}/{k} names admissible under the strict "
                     f"{max_per_sector}-per-sector cap (universe={len(df)}); holding fewer")
    if held.empty:
        return {}, empty_diag

    w = capped_proportional(_inverse_vol(held, vol_col, vol_floor), target_exposure, max_weight)
    weights = {t: float(x) for t, x in zip(held["ticker"], w) if x > 0}
    if sum(weights.values()) < target_exposure - 1e-9:
        notes.append(f"per-name cap {max_weight:.2%} x {len(held)} names < target "
                     f"{target_exposure:.2%}; remainder held in cash")

    sectors = dict(zip(held["ticker"], held["_sector"]))
    limits = dict(max_gross=target_exposure, max_net=target_exposure, max_short=0.0,
                  max_long=target_exposure, max_position=max_weight)
    weights = {t: x for t, x in enforce_portfolio_constraints(weights, **limits).items() if x != 0}
    summary = assert_portfolio_invariants(weights, sectors=sectors,
                                          max_names_per_sector=max_per_sector, **limits)
    diag = {"n_held": len(weights), "k": k, **summary, "cash": 1.0 - summary["net"],
            "sector_counts": stats["sector_counts"],
            "skipped_sector_full": stats["skipped_sector_full"],
            "sector_net": _sector_net(weights, sectors),
            "beta_net": _beta_net(weights, held, beta_col), "notes": notes}
    return weights, diag


def long_short_book(preds, n_long, n_short, gross, max_per_sector, max_weight,
                    shortable=None, *, sector_neutral=False,
                    vol_col="return_volatility_20d", sector_col="Sector",
                    rank_col="pred_rank", beta_col="kalman_beta", vol_floor=1e-3):
    """Dollar-neutral long/short book.

    Longs: the top ``n_long`` names best-first; shorts: the bottom ``n_short``
    worst-first (only tickers in ``shortable`` when given; never a name that is
    also long). Each leg has a STRICT per-sector cap. Inside each leg names are
    inverse-vol weighted under ``max_weight``; the legs are sized to
    long = |short| = gross/2 (if a leg's caps cannot reach gross/2 the other
    leg is scaled down to match, so the book stays dollar-neutral).

    ``sector_neutral=True`` additionally matches long and short exposure inside
    every sector: only sectors present on both legs are traded, each gets the
    same budget on both legs (water-filled across sectors in proportion to the
    number of matched names, under the per-name caps), so per-sector net is ~0.
    Names in one-sided sectors are dropped and reported.

    Returns ``(weights, diagnostics)`` -- shorts negative; diagnostics hold
    gross, net, long, short, per-sector net, beta-weighted net (``kalman_beta``,
    NaN -> 1) and notes.
    """
    notes = []
    n_long, n_short, max_per_sector = int(n_long), int(n_short), int(max_per_sector)
    gross, max_weight = float(gross), float(max_weight)
    half = gross / 2.0

    def _empty(reason):
        notes.append(reason)
        return {}, {"n_long": 0, "n_short": 0, "long": 0.0, "short": 0.0, "gross": 0.0,
                    "net": 0.0, "sector_net": {}, "beta_net": None,
                    "sector_neutral": bool(sector_neutral), "notes": notes}

    if preds is None or len(preds) == 0 or n_long <= 0 or n_short <= 0 or gross <= 0 \
            or max_per_sector <= 0 or max_weight <= 0:
        return _empty("empty book (no candidates or non-positive sizes/caps)")

    df = _ranked(preds, rank_col, sector_col)
    allowed = set(shortable) if shortable is not None else None
    longs, lstats = _admit(df, n_long, max_per_sector)
    worst_first = df.iloc[::-1]
    shorts, sstats = _admit(worst_first, n_short, max_per_sector, allowed=allowed,
                            exclude=set(longs["ticker"]))
    if len(longs) < n_long:
        notes.append(f"only {len(longs)}/{n_long} longs admissible under the sector cap")
    if len(shorts) < n_short:
        notes.append(f"only {len(shorts)}/{n_short} shorts admissible "
                     f"(sector cap / {sstats['skipped_not_allowed']} not shortable)")
    if longs.empty or shorts.empty:
        return _empty("one leg is empty; a dollar-neutral book is impossible")

    dropped = []
    if not sector_neutral:
        wl = capped_proportional(_inverse_vol(longs, vol_col, vol_floor), half, max_weight)
        ws = capped_proportional(_inverse_vol(shorts, vol_col, vol_floor), half, max_weight)
        leg = min(wl.sum(), ws.sum())
        if leg < half - 1e-9:
            notes.append(f"per-name caps limit each leg to {leg:.4f} < gross/2={half:.4f}")
        wl = wl * (leg / wl.sum()) if wl.sum() > 0 else wl
        ws = ws * (leg / ws.sum()) if ws.sum() > 0 else ws
        weights = {t: float(x) for t, x in zip(longs["ticker"], wl) if x > 0}
        weights.update({t: -float(x) for t, x in zip(shorts["ticker"], ws) if x > 0})
    else:
        both = sorted(set(longs["_sector"]) & set(shorts["_sector"]))
        for leg_df in (longs, shorts):
            dropped += leg_df.loc[~leg_df["_sector"].isin(both), "ticker"].tolist()
        if not both:
            return _empty("no sector appears on both legs; sector-neutral book impossible")
        if dropped:
            notes.append(f"sector-neutral: dropped {len(dropped)} names in one-sided sectors")
        nl = longs["_sector"].value_counts()
        ns = shorts["_sector"].value_counts()
        pairs = np.array([min(nl[s], ns[s]) for s in both], dtype=float)
        sec_cap = np.array([min(nl[s], ns[s]) * max_weight for s in both], dtype=float)
        budgets = capped_proportional(pairs, half, sec_cap)
        if budgets.sum() < half - 1e-9:
            notes.append(f"per-name caps limit each leg to {budgets.sum():.4f} < gross/2={half:.4f}")
        weights = {}
        for s, b in zip(both, budgets):
            for leg_df, sign in ((longs, 1.0), (shorts, -1.0)):
                part = leg_df[leg_df["_sector"] == s]
                w = capped_proportional(_inverse_vol(part, vol_col, vol_floor), b, max_weight)
                weights.update({t: sign * float(x) for t, x in zip(part["ticker"], w) if x > 0})

    held = pd.concat([longs, shorts])
    sectors = dict(zip(held["ticker"], held["_sector"]))
    leg_cap = max(half, 0.0)
    limits = dict(max_gross=gross, max_net=1e-6, max_short=leg_cap, max_long=leg_cap,
                  max_position=max_weight)
    weights = {t: x for t, x in enforce_portfolio_constraints(weights, **limits).items() if x != 0}
    summary = assert_portfolio_invariants(weights, sectors=sectors,
                                          max_names_per_sector=max_per_sector, **limits)
    diag = {"n_long": sum(1 for x in weights.values() if x > 0),
            "n_short": sum(1 for x in weights.values() if x < 0),
            **summary,
            "sector_net": _sector_net(weights, sectors),
            "beta_net": _beta_net(weights, held, beta_col),
            "sector_neutral": bool(sector_neutral),
            "dropped_one_sided": dropped,
            "rejected_not_shortable": sstats["skipped_not_allowed"],
            "long_sector_counts": lstats["sector_counts"],
            "short_sector_counts": sstats["sector_counts"],
            "notes": notes}
    return weights, diag
