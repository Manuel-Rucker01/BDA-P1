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


# --------------------------------------------------------------------------- #
# Round 5 (risk model + cost model + cost-aware mean-variance optimizer) --
# pre-registered 2026-11-06 BEFORE any round-5 backtest or bias statistic was
# computed. Thresholds above unchanged; one criterion added (r5, below).
# --------------------------------------------------------------------------- #
# Signal: the incumbent round-2 OOS predictions ``sn30_fund`` (gate a and c are
# therefore unchanged: sn30_fund weekly IC and holdout IC). No retraining.
#
# Risk model (v3/risk.py, fixed before any result): 8 styles (beta, size,
#   value, momentum 12-1, reversal 1m, volatility, liquidity, quality) + EDGAR
#   FF12 industries; daily WLS (sqrt mcap) factor returns; factor covariance
#   EWMA half-life 90 + Newey-West 2 lags; specific variance EWMA half-life 90,
#   Bayesian shrinkage to the sector cap-weighted mean (q = 0.1); VRA on
#   (half-life 42). Estimated on each decision date from data up to that date.
# Cost model (v3/costs.py): half-spread = 0.5 x 63-session mean Corwin-Schultz
#   estimate (price/ADV proxy when < 20 estimates), clipped to [1, 150] bp;
#   impact = k sigma_60d sqrt(trade/ADV20), k = 0.2; borrow 0.25%/1%/5% by mcap
#   (>= $10B / >= $1B / below). The SAME model prices the trades in the backtest.
# Alpha (v3/optimizer.grinold_alpha): alpha_i = IC_t x sigma_spec,i x z_i over
#   the horizon (z = normal score of pred over the config's universe; sigma_spec
#   from the risk model; 30-session label pro-rated to the 21-session horizon).
#   IC_t = (n x trailing mean weekly IC + 26 x 0.03) / (n + 26), floored at 0,
#   over the last 104 weekly decision dates whose 30-session label had fully
#   matured by the decision date (n of them); 0.03 is a generic prior, not
#   estimated from this walk-forward.
# Timing / accounting: monthly rebalance on the first session of each month
#   with the latest weekly prediction strictly before it (risk model, costs,
#   universe as of that prediction date), traded at the rebalance close; weekly
#   variant: the session after each weekly decision date. Daily mark-to-market,
#   delisted names frozen at their last close and sold at the next rebalance (as
#   v3/backtest.py). Per-name costs from costs.py at the book's AUM (NAV-scaled);
#   borrow accrued daily on shorts. Feb 2019 -> 2026-11-04.
# Universes (PIT, on the decision date, names with a prediction and risk-model
#   coverage): ``all`` = every v3-eligible name; ``top1000`` / ``top500`` = the
#   largest 1000 / 500 by market cap (cleaned EDGAR cap; ADV/price fit for names
#   without an own EDGAR cap, foreign private issuers and >20x outliers).
#   Benchmark of the active configs = cap-weighted universe.
# Common settings: horizon 21 sessions, cost multiplier 1 (one month of alpha
#   pays for a trade once), liquidity cap w_i <= 10% x ADV20 / AUM, post-hoc
#   cardinality threshold 0.1% (smaller positions are zeroed and the problem
#   re-solved once).
R5_RECIPE = "sn30_fund"
R5_AUMS = (100e6, 1e6)
R5_PRIMARY_AUM = 100e6
R5_COMMON = {"horizon": 21, "label_h": 30, "cost_mult": 1.0, "impact_k": 0.2, "liq_frac": 0.10,
             "min_weight": 0.001, "ic_prior": 0.03, "ic_prior_weeks": 26, "ic_window": 104}
R5_CONFIGS = {
    # (i) absolute mean-variance long-only, full universe
    "mv_abs_all": {"kind": "absolute", "universe": "all", "lam": 10.0, "name_cap": 0.05, "sector_band": 0.10,
                   "rebalance": "monthly"},
    # (ii) benchmark-relative long-only (TE cap vs the cap-weighted universe)
    "te4_all": {"kind": "active", "universe": "all", "te": 0.04, "lam": 1.0, "active_cap": 0.03,
                "sector_band": 0.05, "beta_band": 0.05, "rebalance": "monthly"},
    "te4_top1000": {"kind": "active", "universe": "top1000", "te": 0.04, "lam": 1.0, "active_cap": 0.03,
                    "sector_band": 0.05, "beta_band": 0.05, "rebalance": "monthly"},
    "te6_top1000": {"kind": "active", "universe": "top1000", "te": 0.06, "lam": 1.0, "active_cap": 0.03,
                    "sector_band": 0.05, "beta_band": 0.05, "rebalance": "monthly"},
    "te4_top500": {"kind": "active", "universe": "top500", "te": 0.04, "lam": 1.0, "active_cap": 0.03,
                   "sector_band": 0.05, "beta_band": 0.05, "rebalance": "monthly"},
    # style-neutral: |active exposure| <= 0.10 sd on every style factor
    "te4n_top1000": {"kind": "active", "universe": "top1000", "te": 0.04, "lam": 1.0, "active_cap": 0.03,
                     "sector_band": 0.05, "beta_band": 0.05, "style_band": 0.10, "rebalance": "monthly"},
    "te4_top1000_wk": {"kind": "active", "universe": "top1000", "te": 0.04, "lam": 1.0, "active_cap": 0.03,
                       "sector_band": 0.05, "beta_band": 0.05, "rebalance": "weekly"},
    # (iii) 130/30: shorts only in the round-3 safe-short pool, <= 1% each, 30% total
    "ls130_top1000": {"kind": "130_30", "universe": "top1000", "te": 0.06, "lam": 1.0, "active_cap": 0.03,
                      "sector_band": 0.05, "beta_band": 0.05, "short_budget": 0.30, "short_cap": 0.01,
                      "rebalance": "monthly"},
}
# Every config is backtested at both AUMs (the optimizer sees the AUM through
# the impact and liquidity terms) -> 16 new trials.
N_TRIALS_R5 = N_TRIALS_R4 + len(R5_CONFIGS) * len(R5_AUMS)    # 55 + 16 = 71
# Selection: among R5_CONFIGS at R5_PRIMARY_AUM, the highest full-period IR vs
# SPY among configs with max drawdown >= MAX_DD_FLOOR; if none qualifies, the
# highest IR vs SPY. Gate = a1-a3 and c1 (sn30_fund, unchanged) + b1 Sharpe >
# EW-universe Sharpe + b2 max DD >= MAX_DD_FLOOR + b3 DSR >= B_MIN_DSR with
# N = N_TRIALS_R5 + r5 (primary round-5 criterion): IR vs SPY > R5_MIN_IR_SPY,
# reported with a stationary-bootstrap 95% CI (2,000 resamples, block 21).
R5_MIN_IR_SPY = 0.0


def choose_config_r5(perf: dict) -> str:
    """``perf``: config -> {'ir_spy', 'max_dd'} at R5_PRIMARY_AUM."""
    ok = {k: v for k, v in perf.items() if k in R5_CONFIGS and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in R5_CONFIGS}
    return max(pool, key=lambda k: pool[k]["ir_spy"])


def evaluate_gate_r5(sig_all: dict, years_ic: dict, perf: dict, ew_sharpe: float, dsr: float,
                     holdout_ic: float, ir_spy: float, ir_spy_ci: tuple) -> list[dict]:
    rows = evaluate_gate(sig_all, years_ic, perf, ew_sharpe, dsr, holdout_ic)
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={N_TRIALS_R5})"
    rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {ir_spy_ci[0]:.2f} .. {ir_spy_ci[1]:.2f})",
                 "value": ir_spy, "threshold": f"> {R5_MIN_IR_SPY}", "pass": bool(ir_spy > R5_MIN_IR_SPY)})
    return rows

# Round 6 (signal side: multi-horizon, stacking, v2-recipe walk-forward) --
# pre-registered 2026-11-06 BEFORE any round-6 label, model or backtest was
# computed. Round 5 (risk model / optimizer) is being run in parallel by another
# agent; its trials are added via N_TRIALS_R5 when present (re-checked at
# evaluation time by n_trials_r6()). Thresholds unchanged.
# --------------------------------------------------------------------------- #
# 1. Multi-horizon ensemble ("mh"): sn30_fund-style models (core + EDGAR
#    fundamentals, EDGAR sectors, same hyperparameters and folds) trained on
#    sector-neutral rank targets at 5, 21 and 63 sessions (+ the existing 30).
#    Embargo per horizon: max(40, H + 10) sessions (63 -> 73). Composite at
#    weekly date t = sum_h w_h(t) * per-date rank(pred_h), with
#    w_h = 0.5 * w_ic_h + 0.5 * 1/4, w_ic_h proportional to max(0, mean IC of
#    pred_h vs its own H-session raw forward return) over the trailing 104
#    weekly OOS dates whose labels were complete by t (date <= t - H sessions);
#    equal weights while fewer than 26 such dates exist or all ICs <= 0.
# 2. Stacking ("stack"): for test year Y, a ridge (alpha = 1, intercept) of the
#    30-session sector-neutral rank target on the per-date ranks of the OOS
#    predictions of base, sn30_fund, sn30_sue, sn30_resmom, sn30_lvq on weekly
#    dates t with pos(t) < pos(start(Y)) - 40 (labels complete, prior years only);
#    equal weights for 2019 (no prior OOS year).
# 3. v2-recipe walk-forward ("v2rec_kg", "v2rec_full"): features/core.py v2
#    features with the v2 eligibility (adjusted close >= $1, >= 80 bars) and the
#    legacy-metadata sectors, robust per-date z-score, + the production
#    artefact's 16 KG PCs (kg universe = names with an embedding; full = all
#    v2-eligible, no KG); SoftVote (per-date rank average) of RandomForest
#    (100 trees, depth 14, min leaf 5), CatBoost (depth 4, lr 0.02), XGBoost and
#    LightGBM (depth 4, lr 0.02, subsample/colsample 0.8, LGBM 15 leaves), boosted
#    models early-stopped (50 rounds, max 1000) on the last 10% of training dates
#    with a 40-session embargo, then refit with the early-stopped count;
#    30-session raw rank target over the v2 population; weekly decision dates
#    and the same yearly folds (2019-2026). Deviation from production: weekly
#    instead of daily training dates (compute), early-stopped count per fold
#    instead of "median of the last 3 CV folds".
# Evaluation: IC vs 30-session raw return on each candidate's own universe;
# decay curve at 1/5/10/21/42/63 sessions; books with EDGAR sectors for caps.
R6_CANDIDATES = ("mh", "stack")
R6_INCUMBENT = "sn30_fund"
R6_HORIZONS = (5, 21, 30, 63)
R6_BOOKS = ("flo50_k50", "flo50_k30")             # round-3 definitions, for mh and stack
R6_V2_BOOK = "lo30"                                # K=30, <=5/sector, <=6%, 0.98 -- for v2rec_kg / v2rec_full
R6_V2_VARIANTS = ("v2rec_kg", "v2rec_full")
R6_NEW_TRIALS = len(R6_CANDIDATES) * len(R6_BOOKS) + len(R6_V2_VARIANTS)   # 4 + 2 = 6
# Selection: the R6 candidate with the higher walk-forward mean IC if it beats
# sn30_fund by >= SWITCH_MIN_IC_GAIN with a higher NW t, else the incumbent;
# construction = choose_construction_r4 over R6_BOOKS. The v2-recipe rows are
# the "decisive test" (information + counted trials), gate rows reported for
# v2rec_kg on its lo30 book.


def n_trials_r6() -> int:
    """All trials so far: rounds 1-4 (55), Round 5 when registered, + round 6."""
    prior = globals().get("N_TRIALS_R5", N_TRIALS_R4)
    return int(prior) + R6_NEW_TRIALS


def choose_variant_r6(signal: dict) -> str:
    inc = signal[R6_INCUMBENT]
    ok = [(v, signal[v]) for v in R6_CANDIDATES if v in signal
          and signal[v]["mean_ic"] >= inc["mean_ic"] + SWITCH_MIN_IC_GAIN and signal[v]["nw_t"] > inc["nw_t"]]
    return max(ok, key=lambda x: x[1]["mean_ic"])[0] if ok else R6_INCUMBENT

# Round 6 amendment (after merging Round 5, N_TRIALS_R5 = 71; committed before
# any round-6 evaluation was run -- horizon/v2 models were still training):
#  * information only (no trials): IC of every round-6 signal within the
#    per-date top-1000 by market cap (EDGAR mcap, hot-deck where unmapped);
#  * "lc_sn30_fund": large-cap specialist -- sn30_fund recipe trained, z-scored
#    and target-ranked only within the per-date top-1000 by mcap; IC only, no
#    book (0 trials);
#  * if choose_variant_r6 selects a round-6 candidate, it is additionally run
#    through the Round 5 optimizer config "mv_abs_all" (v3.round5) at the primary
#    AUM, with the trailing IC estimated from that signal on the universe it
#    trades: +1 trial (R6_CONDITIONAL_TRIALS), counted only if run.
R6_TOP_N = 1000
R6_LC_VARIANT = "lc_sn30_fund"
R6_CONDITIONAL_TRIALS = 1


# --------------------------------------------------------------------------- #
# Round 7 -- pre-registered 2026-11-06 (after merging main ae1e580) BEFORE any
# round-7 backtest. Thresholds and the Round 5 gate criteria unchanged.
# --------------------------------------------------------------------------- #
# Signals (existing walk-forward OOS predictions, no retraining):
#   "v2rec_tr": the v2-recipe full-universe model (round 6, v2rec_full), scored
#               ONLY on v3-eligible names (re-ranked per date among them);
#   "stack":    the round-6 PIT ridge stack (v3-eligible names).
# Constructions (Round 5 machinery, prep and risk models unchanged unless noted):
#   mv_abs_all at both R5_AUMS, with the trailing IC (R5_COMMON rule) estimated
#   from the SAME signal on the traded (v3-eligible) universe;
#   lo30 = long_only_v3 K=30 (<= 5/sector EDGAR, <= 6%, 0.98), monthly, 10 bps.
#   "mv_abs_all_eig": mv_abs_all on v2rec_tr with an eigenfactor risk adjustment
#   of every factor covariance (Menchero/USE4: M = 200 Monte Carlo draws of
#   T = 260 observations -- the effective sample of the 90-day EWMA --, bias
#   v_k = sqrt(mean true/estimated eigen-variance), scale gamma_k =
#   1.2 (v_k - 1) + 1, F_adj = U diag(gamma^2 d) U'), both AUMs.
R7_SIGNALS = ("v2rec_tr", "stack")
R7_CONFIGS = [(s, c, a) for s in R7_SIGNALS for c, a in (("mv_abs_all", 100e6), ("mv_abs_all", 1e6), ("lo30", None))] \
    + [("v2rec_tr", "mv_abs_all_eig", 100e6), ("v2rec_tr", "mv_abs_all_eig", 1e6)]
EIG_SIMS, EIG_T, EIG_SCALE = 200, 260, 1.2
N_TRIALS_R7 = N_TRIALS_R5 + R6_NEW_TRIALS + len(R7_CONFIGS)          # 71 + 6 + 8 = 85
# Selection: among the R7 configs at R5_PRIMARY_AUM plus the lo30 books, the
# highest full-period IR vs SPY among those with max DD >= MAX_DD_FLOOR; else the
# highest IR vs SPY (Round 5 rule). Gate = Round 5 criteria for that config:
# a1-a3 on the chosen signal's weekly IC vs the 30-session return over
# v3-eligible names; b1 Sharpe > EW-universe Sharpe; b2 max DD >= -35%;
# b3 DSR >= 0.90 with N = N_TRIALS_R7 (all trials' Sharpe ratios); c1 >= 0.12
# with c = mean weekly IC on the gate-c window (2026-07-15..09-23, v3-eligible)
# of the signal's 2026 walk-forward fold (trained only through 2025-10, i.e.
# stricter than the <= 2026-05-29 rule); r5 IR vs SPY > 0 with a 95% bootstrap CI.


def r7_key(sig: str, conf: str, aum) -> str:
    return f"{sig}/{conf}" + (f"_{int(aum / 1e6)}m" if aum else "")


def choose_config_r7(perf: dict) -> str:
    keys = [r7_key(s, c, a) for s, c, a in R7_CONFIGS if a in (None, R5_PRIMARY_AUM)]
    ok = {k: perf[k] for k in keys if k in perf and perf[k]["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: perf[k] for k in keys if k in perf}
    return max(pool, key=lambda k: pool[k]["ir_spy"])


# --------------------------------------------------------------------------- #
# Round 8 (new free information: SEC insider transactions, FINRA short
# interest) -- pre-registered 2026-11-06 after merging main 8d287c3, BEFORE any
# round-8 feature IC, model or backtest was computed. Thresholds and the R7
# gate criteria (incl. IR vs SPY > 0) unchanged.
# --------------------------------------------------------------------------- #
# Blocks (v3/altdata.py), all evaluated on weekly decision dates t:
#  insider -- SEC "Insider Transactions Data Sets" (Forms 3/4/5, 2018Q1 on),
#    non-derivative transactions with TRANS_CODE P (open-market purchase) or S
#    (open-market sale), keyed by ISSUERCIK, available from the business day
#    AFTER the filing date (never the transaction date). Windows by
#    availability date over (t - W calendar days, t]:
#      ins_buy_n_W, ins_sell_n_W (W = 30, 90, 180): number of distinct
#        (insider, filing) with a P / S line;
#      ins_buy_val_90/180, ins_sell_val_90/180: log1p(dollar value / 1e3);
#      ins_net_ratio_90/180: (buy$ - sell$) / (buy$ + sell$), 0 if none;
#      ins_n_buyers_90: distinct insiders buying; ins_od_buy_90: any officer or
#        director purchase; ins_cluster_30: >= 3 distinct insiders buying.
#    Joined through the same ticker->CIK map (no CIK -> all zeros = "no
#    activity"; leak probe re-run).
#  si -- FINRA bi-monthly short-interest files (exchange-listed + OTC), keyed by
#    symbol on the settlement date's listing, available settlement date + 8
#    business days; latest available record:
#      si_ratio = short interest / shares outstanding (shares = EDGAR mcap /
#        raw close, the model's mcap incl. hot-deck), si_dtc = FINRA days to
#        cover, si_chg = (SI - previous SI) / previous SI, si_age = days since
#        availability.
# Model variants (sn30_fund recipe: core + EDGAR fundamentals, sector-neutral
# 30-session target; same folds/embargo/hyperparameters):
#   sn30_ins, sn30_si, sn30_altall (both blocks);
#   stack_r8: the round-6 PIT ridge stack with the best of the three (highest
#   walk-forward mean IC) added as a sixth input.
# Book: mv_abs_all at R5_PRIMARY_AUM for each of the 4 signals (IC estimated
# from the same signal on the traded universe) -> 4 new trials.
R8_VARIANTS = {"sn30_ins": ("insider",), "sn30_si": ("si",), "sn30_altall": ("insider", "si")}
R8_STACK = "stack_r8"
R8_SIGNALS = tuple(R8_VARIANTS) + (R8_STACK,)
N_TRIALS_R8 = N_TRIALS_R7 + len(R8_SIGNALS)                          # 85 + 4 = 89
# Selection: highest IR vs SPY with max DD >= -35% among the 4 mv_abs_all books
# (else highest IR vs SPY). Gate = R7 criteria for that book (a1-a3 on the
# signal's weekly IC over v3-eligible names; b1; b2; b3 with N = N_TRIALS_R8;
# c1 on the 2026 walk-forward fold over the gate-c window; r5 IR vs SPY > 0).
# Information (no trials): standalone feature ICs (all names with a label,
# v3-eligible, top-1000), IC decay, walk-forward IC increments.


def choose_config_r8(perf: dict) -> str:
    ok = {k: v for k, v in perf.items() if k in R8_SIGNALS and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in R8_SIGNALS}
    return max(pool, key=lambda k: pool[k]["ir_spy"])

# Round 8 correction (registered after the first round-8 pass revealed a
# survivorship leak in the insider block, BEFORE re-running): names without a
# CIK mapping (mostly future delisters) had exactly zero insider activity, so
# the block identified "no EDGAR record" with AUC 0.849 (core+fund 0.700 ->
# +insider 0.901). The first-pass insider configs (sn30_ins, sn30_altall,
# stack_r8) are therefore INVALID as evidence; they stay counted as trials but
# are not eligible for selection. Correction: for eligible rows without a CIK,
# the whole insider vector is copied from a same-date donor with a CIK among its
# 5 nearest neighbours on (log dollar volume, 20d vol, 20d and 50d return) --
# the round-2 fundamentals hot-deck -- and the three configs are re-run
# ("_hd" suffix): +3 trials.
R8_INVALID = ("sn30_ins", "sn30_altall", "stack_r8")
R8_CORRECTED = ("sn30_ins_hd", "sn30_altall_hd", "stack_r8_hd")
R8_VARIANTS.update({"sn30_ins_hd": ("insider",), "sn30_altall_hd": ("insider", "si")})
N_TRIALS_R8 = N_TRIALS_R7 + len(R8_SIGNALS) + len(R8_CORRECTED)      # 85 + 4 + 3 = 92
R8_ELIGIBLE = ("sn30_si",) + R8_CORRECTED


def choose_config_r8(perf: dict) -> str:   # noqa: F811 -- the correction redefines the eligible set
    ok = {k: v for k, v in perf.items() if k in R8_ELIGIBLE and v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or {k: v for k, v in perf.items() if k in R8_ELIGIBLE}
    return max(pool, key=lambda k: pool[k]["ir_spy"])


# --------------------------------------------------------------------------- #
# Round 9 (new free information: SEC Form 13F institutional holdings, SEC
# fails-to-deliver) -- pre-registered 2026-11-07 on top of main a39b0c6, BEFORE
# any round-9 feature IC, model, leak probe or backtest was computed (only the
# raw files were downloaded and the FTD posting lag measured). Thresholds and
# the R8 gate criteria (incl. IR vs SPY > 0) unchanged.
# --------------------------------------------------------------------------- #
# Blocks (v3/altdata_r9.py), weekly decision dates t:
#  inst13f -- SEC "Form 13F Data Sets". Original 13F-HR holdings/combination
#    reports only (no amendments, no notices), one per (filer, quarter), filed
#    ON OR BEFORE the 45-day deadline; quarter P is usable from the business
#    day after its deadline (late filers never enter). Positions: SH, no
#    PUT/CALL. CUSIP -> ticker id via the FTD files' (CUSIP, symbol) pairs
#    published by P's availability date, settlement within the prior 365 days
#    and on/after the ticker id's first bar (point in time; no current ticker
#    list). On t the latest available quarter P (and P-1):
#      inst_breadth = # filers holding / # filers in P;
#      inst_dbreadth = Chen-Hong-Stein change in breadth over filers present in
#        both P and P-1 (/ # such filers);
#      inst_io = institutional shares / shares outstanding at P's quarter end
#        (clipped at 2), inst_dio = inst_io(P) - inst_io(P-1);
#      inst_new_n, inst_exit_n = log1p(# continuing filers opening / closing a
#        position).
#  ftd -- SEC CNS fails-to-deliver half-month files, a file available the
#    business day after max(period end + 21 days, its observed posting date
#    when that is < 90 days after the period); on t the latest 6 available
#    files: ftd_ratio = mean daily fails over the latest file / shares out,
#    ftd_ratio_3m = same over the 6 files, ftd_persist_3m = share of settlement
#    days with a fail record. Days without a record = zero fails.
# Missing / unmapped rows -- the Round 8 lesson: a row whose ticker has no
# CUSIP link at P (inst13f) or no FTD record in the trailing 365 days (ftd)
# gets the whole block vector of a same-date mapped donor among its 5 nearest
# neighbours on (log dollar volume, 20d vol, 20d and 50d return) (Round 8
# hot-deck), identically for survivors and delisted names. Change features
# without a P-1 link -> missing (z = 0).
# Variants (sn30_altall_hd recipe = the current best signal: core + EDGAR
# fundamentals + insider_hd + SI, sector-neutral 30-session target, same
# folds/embargo/hyperparameters, round-8 evaluation frame):
#   sn30_r9_13f (+inst13f), sn30_r9_ftd (+ftd), sn30_r9_all (+both).
# Leak checks (before any book): held-out AUC of LightGBM probes for
# (i) "no EDGAR record" (Round 8's probe) and (ii) "was delisted" (the ticker
# id's last bar is before 2026-10-01) on rows <= 2022-12-31, core+fund+alt8_hd
# vs + each new block (raw and hot-decked); and the walk-forward IC on rows
# mapped by CIK AND by the block's own link.
# Book screen (Round 8's incremental-IC bar, SWITCH_MIN_IC_GAIN): a variant is
# run through mv_abs_all at R5_PRIMARY_AUM only if its walk-forward mean
# weekly IC (v3-eligible, vs the 30-session return) >= sn30_altall_hd's
# + 0.005 AND its gain on the mapped-only rows is >= half that bar (+0.0025);
# otherwise no book (the signal stays information). No new stack.
# Trials: every variant's walk-forward counts, booked or not -> 3 new trials,
# N = 95 (DSR with n_trials = 95; dispersion from the trials that have a
# Sharpe ratio).
# Selection: highest IR vs SPY with max DD >= -35% among the round-9 books
# (else highest IR vs SPY); no book -> nothing selected, verdict FAIL. Gate =
# R8 criteria for that book: a1-a3 on the signal's weekly IC over v3-eligible
# names; b1 Sharpe > EW-universe Sharpe; b2 max DD >= -35%; b3 DSR >= 0.90 with
# N = 95; c1 >= 0.12 on the gate-c window from the 2026 walk-forward fold; r5
# IR vs SPY > 0 (bootstrap 95% CI).
R9_BASE = "sn30_altall_hd"
R9_VARIANTS = {"sn30_r9_13f": ("inst13f",), "sn30_r9_ftd": ("ftd",), "sn30_r9_all": ("inst13f", "ftd")}
R9_MIN_IC_GAIN = SWITCH_MIN_IC_GAIN                                   # 0.005
R9_MIN_MAPPED_GAIN = SWITCH_MIN_IC_GAIN / 2                           # 0.0025
R9_DELISTED_BEFORE = "2026-10-01"
N_TRIALS_R9 = N_TRIALS_R8 + len(R9_VARIANTS)                          # 92 + 3 = 95


def r9_bookable(ic: dict, ic_mapped: dict) -> list:
    """Variants passing the incremental-IC screen (both dicts: name -> mean IC,
    incl. R9_BASE)."""
    return [v for v in R9_VARIANTS if v in ic
            and ic[v] >= ic[R9_BASE] + R9_MIN_IC_GAIN
            and ic_mapped[v] >= ic_mapped[R9_BASE] + R9_MIN_MAPPED_GAIN]


# Round 9 correction (registered after the pre-registered leak probe and BEFORE
# any round-9 IC, screen or book was computed; one first-pass walk-forward,
# sn30_r9_13f, had finished and was set aside unevaluated): the 13F block
# separated "no EDGAR record" with AUC 0.90 (base 0.75; same with a ticker-held-
# out split, so not memorisation). Cause: inst_io / inst_dio divide by shares
# outstanding = EDGAR mcap / close, which for rows without a CIK is itself a
# hot-deck value from a (weekly-changing) donor -- inst_dio of such rows is
# donor noise and fingerprints "no EDGAR record" (13F without io/dio: 0.67).
# Correction: the shares-dependent columns (R9_SHARES_COLS) are treated as
# unobserved on rows without a CIK and hot-decked from CIK-mapped, block-mapped
# donors; shares-free columns keep the block's own link as "mapped". Probes
# re-run on the corrected frame before the walk-forwards. No trial was
# evaluated, so N stays 95.
R9_SHARES_COLS = ("inst_io", "inst_dio", "ftd_ratio", "ftd_ratio_3m")


def choose_config_r9(perf: dict):
    pool_all = {k: v for k, v in perf.items() if k in R9_VARIANTS}
    if not pool_all:
        return None
    ok = {k: v for k, v in pool_all.items() if v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or pool_all
    return max(pool, key=lambda k: pool[k]["ir_spy"])


# --------------------------------------------------------------------------- #
# Round 10 pre-registration (committed before any round-10 feature IC, leak
# probe, walk-forward or book). Analyst-free earnings surprise as a substitute
# for analyst estimate revisions (no point-in-time history for those yet).
# Code: v3/altdata_r10.py, v3/round10.py. Data: the EDGAR bulk companyfacts /
# submissions zips already in the store (no new download).
# Overlap with sn30_altall_hd (documented in the report before evaluation):
# the incumbent has TTM fundamentals incl. ni_growth_scaled (TTM NI yoy change
# / assets), sales_growth_yoy and fund_staleness_days, and price features up
# to 50 sessions, but no quarterly surprise and nothing anchored on the
# announcement date. Round 4's NI-based SUE (TTM differences; never part of the
# incumbent) added +0.0005 IC to sn30_fund -- the prior for the SUE block is
# small; the EAR block (event-anchored return / volume) is new to the model.
# Blocks (definitions in altdata_r10's docstring):
#  sue -- seasonal-random-walk SUE on quarterly diluted EPS (NI / diluted
#    shares if no EPS concept; Q4 = FY - Q1..Q3); std of the previous <= 8
#    recorded differences (min 4); sue_eps, sue_eps_chg, sue_streak, surge
#    (revenue analogue), sue_days. Available the NYSE business day after the
#    10-Q/10-K filed date.
#  ear -- announcement = original 8-K with item 2.02 (EDGAR acceptance time),
#    else a 10-Q/10-K with no 8-K 2.02 in the 75 days before (ear_src10q = 1);
#    day0 = acceptance date if a session and < 16:00 New York, else the next
#    session; ear = stock minus EW-eligible-market log return over
#    [day0-1, day0+1]; ear_abvol = log(mean volume [day0-1, day0+1] / mean
#    volume [day0-60, day0-11]); ear_prev (previous event), ear_days. Usable
#    from session day0+2, held until the next event.
# Missing rows (Rounds 8-9 lesson): a row without a CIK or without the block's
# own value (sue_eps / ear) gets the whole block vector of a same-date donor
# with a CIK and the value among its 5 nearest neighbours on round8.HD_MATCH
# (round9.block_hotdeck), identically for survivors and delisted names.
# Leak checks before any walk-forward is evaluated: Round 9's probes (held-out
# AUC for "no EDGAR record" and "was delisted", rows <= 2022, date and ticker
# splits), base vs + each block raw / hot-decked. A hot-decked block that
# raises either AUC by more than R10_MAX_PROBE_GAIN (Round 9's accepted
# residual, +0.06) is investigated and corrected -- correction registered in a
# commit -- before any IC is computed.
# Variants (sn30_altall_hd recipe, same folds / embargo / hyperparameters,
# Round 8 evaluation frame): sn30_r10_sue (+sue), sn30_r10_ear (+ear),
# sn30_r10_both (+both).
# Book screen = Round 9's: walk-forward mean weekly IC >= sn30_altall_hd +
# 0.005 AND gain on rows with a CIK and the block's own value >= +0.0025; only
# then mv_abs_all at R5_PRIMARY_AUM. No new stack.
# Trials: 3 walk-forwards -> N = 98. Selection / gate = Round 9's (highest IR
# vs SPY with max DD >= -35% among round-10 books; no book -> FAIL; a1-a3, b1,
# b2, b3 DSR >= 0.90 at N = 98, c1 >= 0.12 on the 2026-fold gate-c window, r5
# IR vs SPY > 0).
R10_BASE = R9_BASE
R10_VARIANTS = {"sn30_r10_sue": ("sue",), "sn30_r10_ear": ("ear",), "sn30_r10_both": ("sue", "ear")}
R10_MIN_IC_GAIN = SWITCH_MIN_IC_GAIN                                  # 0.005
R10_MIN_MAPPED_GAIN = SWITCH_MIN_IC_GAIN / 2                          # 0.0025
R10_MAX_PROBE_GAIN = 0.06
N_TRIALS_R10 = N_TRIALS_R9 + len(R10_VARIANTS)                        # 95 + 3 = 98


def r10_bookable(ic: dict, ic_mapped: dict) -> list:
    """Variants passing the incremental-IC screen (both dicts: name -> mean IC,
    incl. R10_BASE)."""
    return [v for v in R10_VARIANTS if v in ic
            and ic[v] >= ic[R10_BASE] + R10_MIN_IC_GAIN
            and ic_mapped[v] >= ic_mapped[R10_BASE] + R10_MIN_MAPPED_GAIN]


def choose_config_r10(perf: dict):
    pool_all = {k: v for k, v in perf.items() if k in R10_VARIANTS}
    if not pool_all:
        return None
    ok = {k: v for k, v in pool_all.items() if v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or pool_all
    return max(pool, key=lambda k: pool[k]["ir_spy"])


# --------------------------------------------------------------------------- #
# Round 11 pre-registration (committed before any round-11 book, walk-forward,
# residual target or blackout set was computed). Five trials aimed at the
# incumbent's failing criterion, IR vs SPY (its book is low-beta: ex-ante, risk
# model, ~0.4 vs the cap-weighted universe in the live dry run). Code:
# v3/round11.py; data: FRED DTB3 (3-month T-bill, secondary market, % p.a.,
# daily; fredgraph.csv?id=DTB3, downloaded 2026-11-08 to
# $BDA_DATA_V3/rates/DTB3.csv) + the existing stores.
# Common: signal sn30_altall_hd walk-forward OOS predictions (trial 3: its own
# retrained ones), the Round 5 prep (risk models, costs), the mv_abs_all spec,
# $100M, costs.py on every trade, cash at 0%, Feb 2019 -> 2026-11-04, the
# trailing-IC rule of R5_COMMON (IC of the book's own signal vs the raw
# 30-session return).
# "Ex-ante beta vs SPY" (trials 1, 2): risk-model beta to an SPY proxy = the
# cap-weighted top-500 (cleaned mcap) of the names the book may hold on the
# decision date, beta_i = (Sigma b500)_i / b500' Sigma b500 (S&P 500 overlap of
# that set: Round 5 coverage section).
#  1 beta1_spy -- the literal spec (incumbent sleeve scaled down, SPY fills,
#    weights sum <= 1) cannot reach beta 1 unless the book is all SPY (a mix of
#    a beta < 1 sleeve and a beta-1 asset has beta < 1 for any sleeve weight
#    > 0), so it is registered in its only non-degenerate form: SPY enters the
#    mv_abs_all problem as an extra asset (alpha 0; factor exposure X'b500,
#    specific variance sum b500^2 D; covariance with stock i =
#    (X F X'b500)_i + D_i b500_i), long-only, weights sum to 1, stock caps as
#    mv_abs_all, SPY in [0, 1], ex-ante beta vs SPY == R11_BETA_TARGET
#    (equality row), sector bands on look-through weights (SPY contributes
#    b500's sector shares). SPY trades cost R11_SPY_HALF_SPREAD +
#    k sigma60 sqrt(trade / ADV20) (SPY's own dollar volume); SPY earns its
#    adjusted daily return.
#  2 beta1_lev -- the incumbent mv_abs_all sleeve w (the optimizer sees
#    AUM x R11_LEV_CAP for impact / liquidity; w_prev = drifted sleeve weights
#    / their sum), held at leverage L = min(R11_LEV_CAP,
#    R11_BETA_TARGET / beta_exante(w)) (L < 1 -> cash), target exposure L w,
#    costed on the full exposure change; financing on the borrowed amount
#    max(sum exposure - NAV, 0), accrued daily at (DTB3 of the previous
#    session, forward-filled) + R11_FIN_SPREAD.
#  3 resid_target -- sn30_altall_hd features, folds, embargo, hyperparameters
#    (train.walk_forward on the Round 8 frame) with target = per-date
#    percentile rank (eligible rows) of the residual fwd_ret_30 - X_t f_t,
#    X_t = risk-model exposures on the weekly date t (PIT) to R11_RESID_STYLES
#    + the FF12 industries, f_t = sqrt(mcap)-WLS cross-sectional regression of
#    the 30-session forward return (clipped to risk.EST_CLIP for the fit only)
#    on X_t over eligible rows (value / reversal / liquidity / quality stay in
#    the target: characteristics the model may forecast). Rows without
#    exposures: target missing (not trained on, still predicted). Book:
#    mv_abs_all.
#  4 biweekly -- incumbent signal, mv_abs_all at every second weekly decision
#    date (from the first monthly decision date on), traded at the next
#    session; horizon = R11_BIWEEKLY_HORIZON sessions (alpha, risk and the
#    once-per-rebalance cost on the same per-rebalance scale as monthly's
#    21 = its interval; alpha pro-rated from the 30-session label by H/30 as in
#    optimizer.grinold_alpha).
#  5 earn_blackout -- incumbent mv_abs_all, except names with an expected
#    earnings announcement in [R - R11_BLACKOUT_OVERDUE_DAYS calendar days,
#    session R + R11_BLACKOUT_SESSIONS] (R = rebalance session) may not be
#    bought or increased (upper bound = current drifted weight; may be held or
#    reduced). Expected date = last retained 8-K 2.02 event +
#    R11_EXPECTED_GAP_DAYS calendar days, from sec_earn/events.parquet
#    (src10q == 0) with acceptance (America/New_York date) <= the decision
#    date; an event is retained only if >= R11_EVENT_MIN_GAP_DAYS after the
#    CIK's previous retained one (drops intra-quarter extra 2.02 filings;
#    past-only). Ticker -> CIK from the decision date's alt10_weekly row; no
#    CIK / no event -> never blacked out.
# Trials: 5 books -> N = 103. Engineering check (not a trial): the generalised
# runner with no overlay must reproduce Round 8's sn30_altall_hd book.
# Selection: highest IR vs SPY with max DD >= -35% among the five (else the
# highest IR vs SPY). Gate = Round 10's: a1-a3 (the book's signal, weekly IC
# over v3-eligible names), b1 Sharpe > EW, b2 max DD >= -35%, b3 DSR >= 0.90 at
# N = 103 (dispersion from every trial with a Sharpe ratio), c1 >= 0.12
# holdout IC proxy (2026 fold, gate-c window), r5 IR vs SPY > 0 (stationary
# bootstrap 95% CI).
R11_BASE = R10_BASE
R11_TRIALS = ("beta1_spy", "beta1_lev", "resid_target", "biweekly", "earn_blackout")
R11_AUM = R5_PRIMARY_AUM
R11_BETA_TARGET = 1.0
R11_SPY_PROXY_TOP = 500
R11_SPY_HALF_SPREAD = 0.5e-4
R11_LEV_CAP = 1.5
R11_FIN_SPREAD = 0.01
R11_RATE_SERIES = "DTB3"
R11_RESID_STYLES = ("beta", "size", "momentum", "volatility")
R11_BIWEEKLY_HORIZON = 10
R11_BLACKOUT_SESSIONS = 5
R11_BLACKOUT_OVERDUE_DAYS = 10
R11_EXPECTED_GAP_DAYS = 91
R11_EVENT_MIN_GAP_DAYS = 60
N_TRIALS_R11 = N_TRIALS_R10 + len(R11_TRIALS)                         # 98 + 5 = 103


def choose_config_r11(perf: dict):
    pool_all = {k: v for k, v in perf.items() if k in R11_TRIALS}
    if not pool_all:
        return None
    ok = {k: v for k, v in pool_all.items() if v["max_dd"] >= MAX_DD_FLOOR}
    pool = ok or pool_all
    return max(pool, key=lambda k: pool[k]["ir_spy"])
