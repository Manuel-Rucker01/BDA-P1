"""PRE-REGISTERED promotion gate and selection rules for v3 (WS3).

Committed BEFORE any walk-forward result was computed; do not edit after
seeing results. ``evaluate_gate`` only reads these constants.

Recipe selection
    The primary recipe is ``base`` (30-session raw rank target, no KG, all
    v3-eligible names). ``h21`` (21-session label) or ``sn30`` (sector-neutral
    target) replaces it only if its walk-forward weekly mean IC -- measured
    against the 30-session raw forward return on the eligible universe,
    2019-2026 -- exceeds base's by >= ``SWITCH_MIN_IC_GAIN`` AND its Newey-West
    t-stat is higher; if both qualify the higher mean IC wins. KG variants are
    never adopted (the KG covers ~1.9k legacy names only, which would
    re-introduce the frozen-universe bias); they are an ablation.

Construction selection (for the chosen recipe)
    Among ``CONSTRUCTIONS``: the highest full-period (Feb 2019 - Nov 2026)
    daily Sharpe among constructions with max drawdown >= ``MAX_DD_FLOOR``;
    if none qualifies, the highest Sharpe.

Gate -- promote only if ALL hold
    (a) weekly OOS IC (chosen recipe, vs 30-session raw forward return,
        eligible universe, 2019-2026): mean > 0, Newey-West t (6 lags) >
        ``A_MIN_NW_T``, and yearly mean IC > 0 in >= ``A_MIN_POS_YEARS`` of the
        8 years;
    (b) chosen construction: Sharpe > equal-weight-universe Sharpe (EW is
        monthly rebalanced over all eligible names, gross of costs), max
        drawdown >= ``MAX_DD_FLOOR``, Deflated Sharpe probability >=
        ``B_MIN_DSR`` with N = every variant x construction backtest run
        (``N_TRIALS``) and the dispersion of their Sharpe ratios;
    (c) common recent holdout: recipe trained on weekly decision dates
        <= ``C_TRAIN_END`` only, scored on every session in
        [``C_START``, ``C_END``]; mean daily Spearman IC vs the 30-session raw
        forward return over v3-eligible names >= ``C_MIN_IC``.
"""
PRIMARY_VARIANT = "base"
CANDIDATE_VARIANTS = ("base", "h21", "sn30")
ABLATION_ONLY_VARIANTS = ("common_nokg", "common_kg")
SWITCH_MIN_IC_GAIN = 0.005
CONSTRUCTIONS = ("lo30", "lo50", "ls40", "legacy_top10")
N_TRIALS = (len(CANDIDATE_VARIANTS) + len(ABLATION_ONLY_VARIANTS)) * len(CONSTRUCTIONS)

A_MIN_NW_T = 2.0
A_MIN_POS_YEARS = 6
A_YEARS = tuple(str(y) for y in range(2019, 2027))
NW_LAGS = 6

MAX_DD_FLOOR = -0.35
B_MIN_DSR = 0.90

C_TRAIN_END = "2026-05-29"
C_START, C_END = "2026-07-15", "2026-09-23"
C_MIN_IC = 0.12
C_LABEL = "fwd_ret_30"


def choose_variant(signal: dict) -> str:
    """``signal``: variant -> {'mean_ic', 'nw_t'} (vs fwd_ret_30, all years)."""
    b = signal[PRIMARY_VARIANT]
    ok = [(v, signal[v]) for v in CANDIDATE_VARIANTS if v != PRIMARY_VARIANT and v in signal
          and signal[v]["mean_ic"] >= b["mean_ic"] + SWITCH_MIN_IC_GAIN and signal[v]["nw_t"] > b["nw_t"]]
    return max(ok, key=lambda x: x[1]["mean_ic"])[0] if ok else PRIMARY_VARIANT


def choose_construction(perf: dict) -> str:
    """``perf``: construction -> {'sharpe', 'max_dd'} (full period)."""
    ok = {k: v for k, v in perf.items() if k in CONSTRUCTIONS and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in CONSTRUCTIONS}
    return max(pool, key=lambda k: pool[k]["sharpe"])


def evaluate_gate(sig_all: dict, years_ic: dict, perf: dict, ew_sharpe: float, dsr: float,
                  holdout_ic: float) -> list[dict]:
    """Returns one row per criterion with value, threshold and PASS/FAIL."""
    pos_years = sum(1 for y in A_YEARS if years_ic.get(y, float("nan")) > 0)
    rows = [
        ("a1", "mean weekly IC > 0", sig_all["mean_ic"], "> 0", sig_all["mean_ic"] > 0),
        ("a2", "Newey-West t of IC", sig_all["nw_t"], f"> {A_MIN_NW_T}", sig_all["nw_t"] > A_MIN_NW_T),
        ("a3", "years with IC > 0", pos_years, f">= {A_MIN_POS_YEARS} of {len(A_YEARS)}",
         pos_years >= A_MIN_POS_YEARS),
        ("b1", "Sharpe(chosen) vs Sharpe(EW universe)", perf["sharpe"], f"> {ew_sharpe:.3f}",
         perf["sharpe"] > ew_sharpe),
        ("b2", "max drawdown", perf["max_dd"], f">= {MAX_DD_FLOOR}", perf["max_dd"] >= MAX_DD_FLOOR),
        ("b3", f"Deflated Sharpe prob (N={N_TRIALS})", dsr, f">= {B_MIN_DSR}", dsr >= B_MIN_DSR),
        ("c1", f"holdout IC {C_START}..{C_END}", holdout_ic, f">= {C_MIN_IC}", holdout_ic >= C_MIN_IC),
    ]
    return [{"id": i, "criterion": c, "value": v, "threshold": t, "pass": bool(p)} for i, c, v, t, p in rows]
