"""
Configuration settings for the Production BDA Trading Bot.
This file handles absolute path resolutions and holds Alpaca API credentials and model parameters.
"""

import os
from dotenv import load_dotenv

# Always load the .env that ships alongside this config file, so credentials
# resolve correctly regardless of the cwd the script is invoked from
# (running from repo root via `-m DataAnalysisPipeline2.trading_agent.run`
# would otherwise look in the wrong place and silently fall back to dry-run).
_AGENT_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_AGENT_ENV)
# Also fall back to a cwd-rooted .env if one exists (developer convenience)
load_dotenv()

# --- Directory & Path Resolution ---
# Path of this config.py
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# DataAnalysisPipeline2 directory
PIPELINE_DIR = os.path.abspath(os.path.join(AGENT_DIR, ".."))
# Project Root
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))

# Exploitation Zone path
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
# MODEL_PATH is the production artefact; the agent records the artefact it
# actually loaded in ``BDATradingAgent.model_path`` (see SIGNAL below).
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")

# --- Alpaca API Credentials (multi-account) ---
# Two paper accounts you can switch between in the dashboard: one rebalanced
# weekly, one monthly. Paste each account's keys into trading_agent/.env. The
# ACTIVE account is chosen at runtime via env ACTIVE_ACCOUNT or the dashboard.
ACCOUNTS = {
    "weekly": {
        "label": "Weekly Rebalance",
        "cadence": "weekly",
        "api_key": os.getenv("ALPACA_API_KEY_WEEKLY", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY_WEEKLY", ""),
    },
    "monthly": {
        "label": "Monthly Rebalance",
        "cadence": "monthly",
        "api_key": os.getenv("ALPACA_API_KEY_MONTHLY", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY_MONTHLY", ""),
    },
}

# Legacy single-account keys (the original account) — used as a fallback so
# nothing breaks while the per-account keys are still blank.
_LEGACY_KEY = os.getenv("ALPACA_API_KEY", "")
_LEGACY_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

# Default to paper trading for safety
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "True").lower() in ("true", "1", "yes")
ALPACA_URL = ("https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING
              else "https://api.alpaca.markets")


def resolve_account(name=None):
    """Resolve an account's Alpaca credentials + metadata.

    Falls back to the legacy single-account keys when the named account's keys
    are blank or the name is unknown. ``name=None`` uses the ACTIVE_ACCOUNT env
    (default 'weekly'). Returns a dict the dashboard and agent can consume.
    """
    name = (name or os.getenv("ACTIVE_ACCOUNT", "weekly") or "weekly").lower()
    acct = ACCOUNTS.get(name, {})
    key = acct.get("api_key") or _LEGACY_KEY
    secret = acct.get("secret_key") or _LEGACY_SECRET
    return {
        "name": name,
        "label": acct.get("label", name.title()),
        "cadence": acct.get("cadence", name),
        "api_key": key,
        "secret_key": secret,
        "paper": ALPACA_PAPER_TRADING,
        "url": ALPACA_URL,
        "configured": bool(key and secret),
    }


def account_log_dir(name=None):
    """Per-account agent state directory (equity history, halt file, trades,
    decisions, last_weights). Each account gets its own so a new account's
    drawdown breaker never inherits another account's peak equity. The bare
    agent_logs/ root is kept as the archive of the single-account era."""
    name = (name or os.getenv("ACTIVE_ACCOUNT", "weekly") or "weekly").lower()
    return os.path.join(AGENT_DIR, "agent_logs", "accounts", name)


# Module-level ALPACA_API_KEY / ALPACA_SECRET_KEY resolve to the ACTIVE account
# so existing code (bot.py, backtests, the CLI runner) keeps working unchanged.
# The dashboard can re-point these at runtime via pipeline_service.set_active_account().
_active = resolve_account()
ALPACA_API_KEY = _active["api_key"]
ALPACA_SECRET_KEY = _active["secret_key"]

# --- High-Alpha Tickers Basket (Alphabetical Small-Caps: +304% Alpha Goldmine) ---
HIGH_ALPHA_TICKERS = [
    'AAL', 'AAME', 'AAOI', 'AAON', 'AAPL', 'AAXJ', 'ABCB', 'ABEO', 'ABUS', 'ACAD',
    'ACET', 'ACGL', 'ACGLO', 'ACHC', 'ACHV', 'ACIU', 'ACIW', 'ACLS', 'ACMR', 'ACNB'
]

# --- Safe Sector-Diversified Basket (Resilient Mid-Caps: +59% Conservative Growth) ---
SAFE_TICKERS = [
    'CDNS', 'AMAT', 'ACLS', 'ACMR',  # Technology & Semiconductors
    'BIIB', 'ALNY', 'BMRN', 'ACAD',  # Healthcare & Biotechnology
    'AVAV', 'BLDR', 'AAON', 'ASTE',  # Industrials, Aerospace & Infrastructure
    'CBOE', 'BPOP', 'ABCB',          # Financials & Banking
    'CAKE', 'CARG', 'BLMN',          # Consumer Discretionary (Food & Marketplace)
    'CELH', 'CENT'                   # Consumer Defensive & Essentials
]

# Default to the highly explosive High-Alpha basket. Can be dynamically overridden via CLI.
TICKERS = HIGH_ALPHA_TICKERS

# --- Model & Trading Parameters ---
CONFIDENCE_THRESHOLD = 0.53      # Probability threshold for High-Confidence Longs
def _env_num(name, default, cast=float):
    """Strategy knobs can be overridden via env (BDA_<NAME>) so a scheduler or a
    cloud job can configure them without code changes."""
    v = os.getenv(f"BDA_{name}")
    try:
        return cast(v) if v not in (None, "") else default
    except ValueError:
        return default


TARGET_EXPOSURE = _env_num("TARGET_EXPOSURE", 1.0)  # Target total portfolio exposure
# Minimum USD per order: skips churn trades on tiny weight drifts.
MIN_ORDER_VALUE = _env_num("MIN_ORDER_VALUE", 5.0)

# --- Live market data ---
# Calendar days of daily bars fetched for live inference. Must cover the
# longest feature lookback (return_50d, 26-span MACD EMA) with warm-up; the old
# 60-day window (~41 sessions) left return_50d permanently 0 in production.
LIVE_HISTORY_DAYS = 250
# feature_version-3 models: >= 120 bars for the eligibility rule, the 120-session
# proxy-sector window ending before the decision month, and EMA/RSI warm-up
# (~290 sessions) so live features equal the offline dataset's.
LIVE_HISTORY_DAYS_V3 = 420
# Refuse to rank the universe if fewer than this fraction of tickers returned
# data (a data outage would otherwise yield an arbitrary top-K book).
MIN_UNIVERSE_COVERAGE = 0.80

# --- Drawdown circuit breaker ---
# Peak-to-trough drawdown (%) at which the bot halts and refuses to trade until
# an operator manually clears the halt file. Set to 15% (was an implicit 5%
# default): this concentrated book routinely swings ~10%, so a 5% stop re-trips
# on every run. Raise/lower to match your real risk tolerance.
DRAWDOWN_LIMIT_PCT = _env_num("DRAWDOWN_LIMIT_PCT", 15.0)

# --- Portfolio exposure guardrails (used by regime_filtered long/short mode) ---
# Gross exposure is sum(abs(weights)); net exposure is abs(sum(weights)).
# Side caps are applied before gross/net scaling so short exposure cannot grow
# implicitly when long signals are scarce.
MAX_GROSS_EXPOSURE = 1.0
MAX_NET_EXPOSURE = 1.0
MAX_SHORT_EXPOSURE = 0.30
MAX_LONG_EXPOSURE = 1.0

# --- Top-K Concentrated Portfolio Mode ---
# When `--universe full` is active, the bot scores every modelled ticker
# (~1,890 names), keeps the top TOP_PCT_THRESHOLD percent of the cross-
# section, then trims to at most TOP_K_HOLDINGS names to actually hold.
# This is the "model on the full universe, trade only the very top" pattern
# that quant funds use — it preserves the cross-sectional Z preprocessing
# the model was trained with AND produces a concentrated, low-overhead
# portfolio.
TOP_PCT_THRESHOLD = _env_num("TOP_PCT_THRESHOLD", 5.0)   # only consider top X% of ranked universe
TOP_K_HOLDINGS = _env_num("TOP_K_HOLDINGS", 10, int)     # cap actually-held positions
EQUAL_WEIGHT_TOP_K = True        # (legacy) equal-weight when WEIGHTING_SCHEME='equal'

# --- Sector concentration cap (DISABLED by default; kept as a tested knob) ---
# A per-sector COUNT cap on the top-K book, derived from this fraction:
#   max_names_per_sector = max(1, round(TOP_K_HOLDINGS * MAX_SECTOR_WEIGHT)).
# Intent was to raise effective breadth (Grinold: IR ~ IC*sqrt(breadth)). We
# A/B-tested it (K=20 + cap 0.30) against the K=10 book on the clean OOS windows
# and it was WORSE on return, Sharpe and IR — because this model's skill is
# sector-conditional (strong in Tech/Mega-Cap, negative in Healthcare/Financials,
# per the subsets backtest), so forcing sector diversification adds low/negative-IC
# bets and dilutes the concentrated edge. Grinold's law assumes uniform per-bet
# skill, which does not hold here. Left at 1.0 (no cap); set <1.0 to re-enable.
MAX_SECTOR_WEIGHT = _env_num("MAX_SECTOR_WEIGHT", 1.0)

# --- Intra-basket position weighting ---
# How capital is split across the held top-K names. The alpha layer chooses
# WHICH names to hold; this chooses HOW MUCH of each.
#   "inverse_vol" : w_i proportional to 1 / realized_vol_i (risk-parity-lite).
#                   High-volatility names receive less capital, so no single
#                   name dominates portfolio drawdown — this directly targets
#                   the idiosyncratic single-name risk that a concentrated
#                   top-K book is exposed to (and that the HMM/Kalman layers,
#                   which manage systematic risk, do not address).
#   "equal"       : equal weight across the held names (legacy behaviour).
#   "pred_rank"   : weight proportional to predicted rank.
WEIGHTING_SCHEME = "inverse_vol"
# Cap on any single position so one ultra-low-volatility name cannot dominate
# the inverse-vol book, and so regime_filtered cannot concentrate gross
# exposure in one name. With K=10 the equal-weight baseline is 0.10; 0.25 lets
# the risk tilt act without letting a single name become the whole portfolio.
MAX_POSITION_WEIGHT = _env_num("MAX_POSITION_WEIGHT", 0.25)
# Floor on realized volatility used in the 1/vol calculation (guards against
# division blow-ups on a name with a near-zero trailing vol).
VOL_FLOOR = 1e-3

# --- v3 portfolio construction (WS4; see trading_agent/portfolio.py) ---
# The model's skill is in the broad cross-sectional ranking, not the top-10, so
# these books are wide and the per-sector cap is STRICT (never relaxed).
# long_only_v3: K names best-first, <= V3_MAX_PER_SECTOR per sector, inverse-vol
# weights capped at V3_MAX_WEIGHT, total <= V3_TARGET_EXPOSURE (rest = cash).
# Common stocks only: drop ETFs/funds/preferreds/... from the v2 KG universe
# before ranking (bot.exclude_non_common). BDA_EXCLUDE_NON_COMMON=0 disables.
EXCLUDE_NON_COMMON = bool(_env_num("EXCLUDE_NON_COMMON", 1, int))
V3_K = _env_num("V3_K", 30, int)
V3_MAX_PER_SECTOR = _env_num("V3_MAX_PER_SECTOR", 5, int)
V3_MAX_WEIGHT = _env_num("V3_MAX_WEIGHT", 0.06)
V3_TARGET_EXPOSURE = _env_num("V3_TARGET_EXPOSURE", TARGET_EXPOSURE)
# long_short_neutral: long top LS_N_LONG / short bottom LS_N_SHORT (shortable +
# easy-to-borrow only), dollar-neutral at LS_GROSS gross (long = |short| =
# LS_GROSS/2), per-name cap LS_MAX_WEIGHT, strict per-leg sector cap, and
# sector-neutral weighting unless LS_SECTOR_NEUTRAL=0.
LS_N_LONG = _env_num("LS_N_LONG", 40, int)
LS_N_SHORT = _env_num("LS_N_SHORT", 40, int)
LS_GROSS = _env_num("LS_GROSS", 1.6)
LS_MAX_WEIGHT = _env_num("LS_MAX_WEIGHT", 0.04)
LS_MAX_PER_SECTOR = _env_num("LS_MAX_PER_SECTOR", 8, int)
LS_SECTOR_NEUTRAL = bool(_env_num("LS_SECTOR_NEUTRAL", 1, int))

# --- Execution ---
# Skip trades smaller than this fraction of equity (and < MIN_ORDER_VALUE):
# avoids churning on small weight drifts. Full liquidations are never skipped.
MIN_TRADE_PCT = _env_num("MIN_TRADE_PCT", 0.0025)

# --- optimized_v3 (OFF by default; see trading_agent/optimized.py) ---
# Round-5/7 cost-aware mean-variance book (spec mv_abs_all: long-only, <= 5%/name,
# <= 10% ADV/AUM, FF-12 sectors within +-10 pp of the cap-weighted universe).
# Better risk than K=30 in the 2019-2026 walk-forward (Sharpe 0.82-0.98 vs 0.52-0.64,
# max DD -30..-32% vs -35..-38%) but it FAILS the pre-registered gate (IR vs SPY < 0,
# DSR < 0.90): an explicit, user-approved paper experiment only. Any failure
# falls back to long_only_v3.
# OPT_IC: IC used to scale alpha; 0.0725 = the v2-recipe signal's mean weekly IC
#   on the traded (v3-eligible) universe in Round 7 (results_r7 / report). The
#   R7 backtest's trailing-shrunk estimate at the last decision was ~0.049.
# OPT_AUM: AUM for the impact / liquidity terms; $1M = the R7 $1M configuration
#   (conservative for the ~$100k paper account).
# OPT_MIN_WEIGHT: names below this are dropped (R5 spec 0.001); default = the
#   executor's MIN_TRADE_PCT so every planned position is actually tradable.
OPT_IC = _env_num("OPT_IC", 0.0725)
OPT_AUM = _env_num("OPT_AUM", 1e6)
OPT_TARGET_EXPOSURE = _env_num("OPT_TARGET_EXPOSURE", V3_TARGET_EXPOSURE)
OPT_MIN_WEIGHT = _env_num("OPT_MIN_WEIGHT", max(0.001, MIN_TRADE_PCT))
OPT_HISTORY_DAYS = _env_num("OPT_HISTORY_DAYS", 800, int)    # calendar days of bars for the risk model
OPT_FUND_DIR = os.getenv("BDA_OPT_FUND_DIR", "")             # override; else local store, else snapshot
OPT_FUND_MAX_AGE_DAYS = _env_num("OPT_FUND_MAX_AGE_DAYS", 120, int)
OPT_MIN_UNIVERSE = _env_num("OPT_MIN_UNIVERSE", 300, int)
OPT_MIN_NAMES = _env_num("OPT_MIN_NAMES", 20, int)
# --- Signal selection: BDA_SIGNAL = altall (default since 2026-11-10, user decision) | production ---
# altall = the best Round-8 book's signal sn30_altall_hd (v3 core + EDGAR
# fundamentals + SEC insider (hot-decked) + FINRA short interest), artefact
# ExploitationZone/model_v3_altall.pkl (git LFS; trained with the exact Round-8
# recipe on every labelled week, v3/final_altall.py). PAPER opt-in only: it
# FAILS the pre-registered gate (DSR 0.863 at N = 92, IR vs SPY -0.13; Sharpe 1.02,
# max DD -28.5% through mv_abs_all in the 2019-2026 walk-forward). Live inputs:
# fundamentals snapshot + data/altdata_snapshot (trading_agent/altall.py).
# ANY failure (artefact missing / invalid, stale or missing snapshot, feature
# error, optimized_v3 not built) falls back to the production model and the
# usual optimized_v3 -> long_only_v3 path, logged in CONSTRUCTION_JSON.
SIGNAL = (os.getenv("BDA_SIGNAL", "altall") or "altall").strip().lower()
ALTALL_MODEL_PATH = os.getenv("BDA_MODEL_PATH_ALTALL", "") or os.path.join(EXPLOITATION_DIR, "model_v3_altall.pkl")
# OPT_IC for the altall signal: Round-8 trailing IC of sn30_altall_hd at the last
# walk-forward decision (2026-11-04) under the pre-registered R5 rule (last 104
# matured weekly ICs on v3-eligible names, shrunk to 0.03 with 26 pseudo-weeks).
# Its full-sample mean weekly IC is 0.0734 (OPT_IC=0.0725 is the v2 signal's mean).
OPT_IC_ALTALL = _env_num("OPT_IC_ALTALL", 0.0534)
ALT_SNAPSHOT_DIR = os.getenv("BDA_ALT_SNAPSHOT_DIR", "") or os.path.join(AGENT_DIR, "data", "altdata_snapshot")
ALT_INSIDER_MAX_AGE_DAYS = _env_num("ALT_INSIDER_MAX_AGE_DAYS", 14, int)
# In-runner top-ups: FINRA files (public CDN) when one is missing; the EDGAR Form 4
# feed only if SEC_USER_AGENT is set, time-boxed to ALT_REFRESH_BUDGET_S seconds.
ALT_REFRESH = bool(_env_num("ALT_REFRESH", 1, int))
ALT_REFRESH_BUDGET_S = _env_num("ALT_REFRESH_BUDGET_S", 900.0)
# --- Round-11 earnings blackout (optimized_v3 + the altall signal; trading_agent/earnings.py) ---
# Names whose expected earnings date (last retained 8-K 2.02 + 91 d) is in
# [rebalance - 10 d, 5th session after] cannot be bought / increased (cap = current
# weight). Round 11 selected it (Sharpe 1.04, max DD -27.7%) but it FAILS the gate
# (IR vs SPY -0.10, DSR 0.862): PAPER only. BDA_EARN_BLACKOUT=1/0 overrides; unset =
# ON iff BDA_SIGNAL=altall. Applied only while the altall signal is in use. Events
# older than EARN_MAX_AGE_DAYS before the decision date -> blackout skipped (logged).
EARN_BLACKOUT = bool(_env_num("EARN_BLACKOUT", 1 if SIGNAL == "altall" else 0, int))
EARN_MAX_AGE_DAYS = _env_num("EARN_MAX_AGE_DAYS", 14, int)

# Exposure-reducing orders (sells / covers) are sent first and the bot waits up
# to FILL_WAIT_TIMEOUT_S for them to fill (polling every FILL_POLL_INTERVAL_S)
# before sending buys / new shorts. Unfilled orders are reported.
FILL_WAIT_TIMEOUT_S = _env_num("FILL_WAIT_TIMEOUT_S", 60.0)
FILL_POLL_INTERVAL_S = _env_num("FILL_POLL_INTERVAL_S", 2.0)

# --- Broad-Market Regime Switching ---
# Enable the S&P 500 trend filter (shut off short exposure in bull regimes)
REGIME_FILTER_ENABLED = True
SP500_INDEX = "^GSPC"

# --- Gaussian HMM & Kalman Filter Configuration ---
HMM_TRAINING_DAYS = 250         # Number of historical days to train the HMM
KALMAN_Q = 1e-4                 # Process noise covariance (Beta drift)
KALMAN_R = 1e-1                 # Measurement noise covariance (trust in daily prints)

# --- Quantitative Trading Upgrades ---
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
ALPACA_CHECK_BORROWABILITY = True # Enable Alpaca Shortable / Easy-to-Borrow (ETB) check

# --- Live News-Sentiment Tilt (LIVE-ONLY — never backtested) ---
# A real-time news-sentiment signal applied as a small tilt on the model's
# predicted rank at LIVE decision time only. It is deliberately kept out of the
# backtests and the structural KG/embeddings: a leak-free backtest would need a
# point-in-time historical news archive, which current-news APIs do not provide.
# Pipeline: Alpaca News -> sentiment -> ephemeral RDF news-event graph ->
# SPARQL aggregation -> per-ticker score. Disable by setting USE_NEWS_SENTIMENT=0.
# Default OFF in the v2 (news-in-training) adoption: news is now a MODEL feature
# (PIT news_* columns trained into the model + computed live in run_inference),
# so the post-hoc tilt would double-count the same signal. Set to 1 only if
# running the older tilt-on-a-news-free model.
USE_NEWS_SENTIMENT = os.getenv("USE_NEWS_SENTIMENT", "0") == "1"
NEWS_TILT_LAMBDA = float(os.getenv("NEWS_TILT_LAMBDA", "0.10"))  # tilt strength
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "7"))

