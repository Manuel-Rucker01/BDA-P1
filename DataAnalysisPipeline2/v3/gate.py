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


# --------------------------------------------------------------------------- #
# Round 2 (EDGAR sectors + fundamentals) -- pre-registered 2026-11-06 BEFORE any
# round-2 result was computed. Thresholds above are unchanged and reused.
# --------------------------------------------------------------------------- #
# Recipe selection: primary = base_fund (30-session raw target, core +
# fundamentals, EDGAR sectors). sn30_fund or base_sec replaces it only under
# the same rule as round 1 (mean IC >= primary + SWITCH_MIN_IC_GAIN and a
# higher NW t; if both qualify the higher mean IC wins).
PRIMARY_VARIANT_R2 = "base_fund"
CANDIDATE_VARIANTS_R2 = ("base_fund", "sn30_fund", "base_sec")
# Deflated Sharpe: every variant x construction backtested in rounds 1 and 2.
N_TRIALS_R2 = N_TRIALS + len(CANDIDATE_VARIANTS_R2) * len(CONSTRUCTIONS)
# Information only (not a gate criterion): drawdown with Feb-Apr 2020 removed.
COVID_WINDOW = ("2020-02-01", "2020-04-30")


def choose_variant_r2(signal: dict) -> str:
    b = signal[PRIMARY_VARIANT_R2]
    ok = [(v, signal[v]) for v in CANDIDATE_VARIANTS_R2 if v != PRIMARY_VARIANT_R2 and v in signal
          and signal[v]["mean_ic"] >= b["mean_ic"] + SWITCH_MIN_IC_GAIN and signal[v]["nw_t"] > b["nw_t"]]
    return max(ok, key=lambda x: x[1]["mean_ic"])[0] if ok else PRIMARY_VARIANT_R2


# --------------------------------------------------------------------------- #
# Round 3 (construction-level changes, no retraining) -- pre-registered
# 2026-11-06 BEFORE any round-3 backtest was run. Thresholds unchanged.
# --------------------------------------------------------------------------- #
# Predictions: the round-2 walk-forward OOS predictions on disk; recipe
# sn30_fund (gate a and c are therefore unchanged from round 2: a = sn30_fund
# weekly IC, c = sn30_fund holdout IC). Costs, rebalance calendar (monthly,
# first session, latest weekly prediction strictly before) and sectors (EDGAR
# FF12 + proxy) as in round 2.
#
# Common definitions (all PIT, measured on the weekly decision date):
#   mcap      = raw close x shares as filed (EDGAR as-of); names without an
#               EDGAR record use the hot-deck value (same as the model input)
#   adv       = 20-session mean dollar volume
#   vol60     = std of the last 60 daily returns
#   floor(p, a): eligible names with mcap >= per-date p-th percentile of mcap
#               over eligible names AND adv >= a
#   safe short pool: adv >= $25M, raw close >= $10, vol60 below the per-date
#               90th percentile, return_5d <= +50%
#   short mechanics (L/S configs): per-name short cap 2%; the short leg is
#               re-scaled back to its target weights at every weekly decision
#               date (costed); a short is covered (costed) the day its price is
#               >= +50% above entry and stays flat until the next monthly
#               rebalance; borrow 1%/yr if mcap >= $1B else 5%/yr.
#   vol-target overlay on a book: exposure s = min(1, 0.15 / realised vol of the
#               book's daily returns over the trailing window, annualised),
#               re-set at every weekly decision date, applied from the next
#               session, cost 10 bps x |delta s| x book gross; rest in cash;
#               trend variant additionally caps s at 0.5 while SPY's close is
#               below its 200-session moving average on the re-set date.
R3_RECIPE = "sn30_fund"
R3_CONFIGS = {
    # name: spec (interpreted by v3/round3.py)
    "flo30_k30": {"kind": "floor_long", "mcap_pct": 30, "adv_min": 10e6, "k": 30, "max_per_sector": 5, "max_weight": 0.06},
    "flo30_k50": {"kind": "floor_long", "mcap_pct": 30, "adv_min": 10e6, "k": 50, "max_per_sector": 7, "max_weight": 0.04},
    "flo50_k30": {"kind": "floor_long", "mcap_pct": 50, "adv_min": 25e6, "k": 30, "max_per_sector": 5, "max_weight": 0.06},
    "flo50_k50": {"kind": "floor_long", "mcap_pct": 50, "adv_min": 25e6, "k": 50, "max_per_sector": 7, "max_weight": 0.04},
    "vt15_20d": {"kind": "vol_target", "book": "flo30_k50", "target": 0.15, "window": 20, "trend": False},
    "vt15_60d": {"kind": "vol_target", "book": "flo30_k50", "target": 0.15, "window": 60, "trend": False},
    "vt15_20d_trend": {"kind": "vol_target", "book": "flo30_k50", "target": 0.15, "window": 20, "trend": True},
    "ls_safe": {"kind": "ls_safe", "n_long": 40, "n_short": 40, "gross": 1.6, "max_per_sector": 8, "max_weight": 0.02,
                "sector_neutral": True},
    "ls_130_30": {"kind": "ls_130_30", "long_book": "flo30_k50", "long": 1.3, "short": 0.3, "n_short": 20,
                  "max_short_per_sector": 4, "short_cap": 0.02},
}
# Comparison on the round-1 `base` predictions (legacy sectors) -- counted as trials.
R3_BASE_CONFIGS = ("flo30_k50", "vt15_20d")
N_TRIALS_R3 = N_TRIALS_R2 + len(R3_CONFIGS) + len(R3_BASE_CONFIGS)   # 32 + 9 + 2 = 43
# Selection: among R3_CONFIGS on R3_RECIPE, the highest full-period Sharpe among
# configs with max drawdown >= MAX_DD_FLOOR; if none qualifies, the highest
# Sharpe. The gate (a, b1-b3, c) is then applied to that config with
# N = N_TRIALS_R3 for the Deflated Sharpe.


def choose_config_r3(perf: dict) -> str:
    ok = {k: v for k, v in perf.items() if k in R3_CONFIGS and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in R3_CONFIGS}
    return max(pool, key=lambda k: pool[k]["sharpe"])


# --------------------------------------------------------------------------- #
# Round 4 (new SIGNAL features) -- pre-registered 2026-11-06 BEFORE any
# round-4 feature was computed or model trained. Thresholds unchanged.
# --------------------------------------------------------------------------- #
# Feature blocks (all PIT; implemented in v3/signals.py):
#  resmom  -- per ticker, rolling OLS over the last 504 bars (min 252) of the
#             daily return (clipped to [-50%, +100%]) on the market (EW mean
#             over eligible names that day) and the ticker's effective-sector
#             EW return (EDGAR FF12/proxy) WITH an intercept; residual
#             e = r - beta_m m - beta_s s (intercept kept in e).
#             resmom_12_1 = sum(e, t-251..t-21) / (sigma_e sqrt(231));
#             resmom_6_1  = sum(e, t-125..t-21) / (sigma_e sqrt(105));
#             strev_1m    = sum(e, t-20..t) / (sigma_e sqrt(21));
#             ind_mom_12_1 = cumulative sector EW log-return t-251..t-21.
#  sue     -- EDGAR, from fundamentals_asof (as known at each filing's
#             availability date = next business day after filing):
#             d_q = TTM net income(E_k) - TTM net income(E_{k-1}) for
#             consecutive period ends 75..105 days apart (= latest quarter's
#             year-over-year change); sue = d_q / std(last 8 d_q, >= 4);
#             sue_chg = sue - previous sue; days_since_report = Date - the
#             availability date of the latest new period;
#             fundamental momentum: d_gross_margin (GP/Rev TTM vs TTM a year
#             earlier), d_roa (NI/A vs NI_prev/A_1y), d_accruals ((NI-CFO)/A vs
#             a year earlier), asset_growth (A / A_1y - 1), A_1y = assets at the
#             period end 320..410 days earlier, as known at the same filing.
#             Joined through the same ticker->CIK map and imputed jointly with
#             the round-2 fundamentals (same KNN hot-deck donor); leak probe
#             re-run.
#  lvq     -- beta_mkt and idio_vol (annualised sigma_e) from the resmom
#             regression, plus lvq_composite = mean of per-date percentile ranks
#             of (-beta_mkt, -idio_vol, gpa) over eligible names.
# All features z-scored per date over eligible names (core.cross_sectional_zscore).
#
# Model variants (same walk-forward, folds, embargo, hyperparameters):
R4_VARIANTS = {
    # name: (target, extra blocks on top of core + round-2 fundamentals, EDGAR sectors)
    "sn30_resmom": ("target_sn_30", ("resmom",)),
    "sn30_sue": ("target_sn_30", ("sue",)),
    "sn30_lvq": ("target_sn_30", ("lvq",)),
    "sn30_all": ("target_sn_30", ("resmom", "sue", "lvq")),
    "base_all": ("target_raw_30", ("resmom", "sue", "lvq")),     # comparison
}
R4_CANDIDATES = ("sn30_resmom", "sn30_sue", "sn30_lvq", "sn30_all")
R4_INCUMBENT = "sn30_fund"
# Constructions (round-3 definitions): floor p50/$25M, K=50 and K=30.
R4_CONSTRUCTIONS = ("flo50_k50", "flo50_k30")
N_TRIALS_R4 = N_TRIALS_R3 + len(R4_VARIANTS) * len(R4_CONSTRUCTIONS)   # 43 + 10 = 53
# Selection: recipe = the R4 candidate with the highest walk-forward mean IC
# (vs 30-session raw return) if it beats the incumbent sn30_fund by
# >= SWITCH_MIN_IC_GAIN with a higher NW t; otherwise the incumbent (round 4
# then adds nothing and the round-3 verdict stands). Construction = the higher
# Sharpe of R4_CONSTRUCTIONS among those with max DD >= MAX_DD_FLOOR (else the
# higher Sharpe). Gate a/b/c as before with N = N_TRIALS_R4.


def choose_variant_r4(signal: dict) -> str:
    inc = signal[R4_INCUMBENT]
    ok = [(v, signal[v]) for v in R4_CANDIDATES if v in signal
          and signal[v]["mean_ic"] >= inc["mean_ic"] + SWITCH_MIN_IC_GAIN and signal[v]["nw_t"] > inc["nw_t"]]
    return max(ok, key=lambda x: x[1]["mean_ic"])[0] if ok else R4_INCUMBENT


def choose_construction_r4(perf: dict) -> str:
    ok = {k: v for k, v in perf.items() if k in R4_CONSTRUCTIONS and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in R4_CONSTRUCTIONS}
    return max(pool, key=lambda k: pool[k]["sharpe"])

# Round 4 amendment (committed while dataset_r4 was still being built, before
# any round-4 model was trained or evaluated): an LGBM-only check of the
# all-blocks recipe, as one more candidate. Trial count updated accordingly.
R4_VARIANTS["sn30_all_lgbm"] = ("target_sn_30", ("resmom", "sue", "lvq"))
R4_MODELS = {"sn30_all_lgbm": ("lgbm",)}
R4_CANDIDATES = R4_CANDIDATES + ("sn30_all_lgbm",)
N_TRIALS_R4 = N_TRIALS_R3 + len(R4_VARIANTS) * len(R4_CONSTRUCTIONS)   # 43 + 12 = 55
