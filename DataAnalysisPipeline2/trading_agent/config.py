"""
Configuration settings for the Production BDA Trading Bot.
This file handles absolute path resolutions and holds Alpaca API credentials and model parameters.
"""

import os
from dotenv import load_dotenv

# Load credentials from .env file if it exists
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
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")

# --- Alpaca API Credentials ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
# Default to paper trading for safety
ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "True").lower() in ("true", "1", "yes")

if ALPACA_PAPER_TRADING:
    ALPACA_URL = "https://paper-api.alpaca.markets"
else:
    ALPACA_URL = "https://api.alpaca.markets"

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
TARGET_EXPOSURE = 1.0           # Target total portfolio exposure
MIN_ORDER_VALUE = 5.0           # Minimum USD order limit to avoid tiny fraction order rejections

# --- Broad-Market Regime Switching ---
# Enable the S&P 500 trend filter (shut off short exposure in bull regimes)
REGIME_FILTER_ENABLED = True
SP500_INDEX = "^GSPC"

# --- Gaussian HMM & Kalman Filter Configuration ---
HMM_TRAINING_DAYS = 250         # Number of historical days to train the HMM
KALMAN_Q = 1e-4                 # Process noise covariance (Beta drift)
KALMAN_R = 1e-1                 # Measurement noise covariance (trust in daily prints)

