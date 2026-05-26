"""
Operational hardening for the BDA trading agent.

This module groups the production-grade plumbing that the academic core of
the bot didn't need but a live system absolutely does:

  - TradeLogger              : structured per-order log with signal decomposition
  - DrawdownCircuitBreaker   : hard peak-to-trough stop with a manual-reset file
  - DecisionContext / hashing: reproducible (model_hash, feature_snapshot,
                               regime_state) tag stamped on every order so any
                               trade can be replayed offline
  - PerformanceAttribution   : decompose realised PnL between rebalances into
                               HMM regime, ML rank, Kalman beta buckets

Everything writes to plain CSV/parquet under ./agent_logs/ so the artefacts
survive process restarts and can be inspected with any tool.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ── Filesystem layout ──────────────────────────────────────────────────────────

AGENT_LOG_DIR = os.environ.get(
    "BDA_AGENT_LOG_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "agent_logs")),
)
SNAPSHOT_DIR  = os.path.join(AGENT_LOG_DIR, "decisions")
TRADES_PATH   = os.path.join(AGENT_LOG_DIR, "trades.csv")
DECISIONS_PATH = os.path.join(AGENT_LOG_DIR, "decisions.csv")
ATTRIBUTION_PATH = os.path.join(AGENT_LOG_DIR, "attribution.csv")
EQUITY_HISTORY_PATH = os.path.join(AGENT_LOG_DIR, "equity_history.csv")
HALT_FILE     = os.path.join(AGENT_LOG_DIR, ".agent_halted")

# Bootstrap log dirs at import — every other call assumes they exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ── Reproducibility hash ──────────────────────────────────────────────────────

def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file's contents.  Used to fingerprint the loaded model so the
    decision id changes when the model changes."""
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def dataframe_hash(df: pd.DataFrame) -> str:
    """Content hash of a feature dataframe.  Sort columns first so column order
    doesn't matter, then hash the underlying bytes."""
    if df is None or len(df) == 0:
        return "empty"
    cols = sorted(df.columns.tolist())
    h = hashlib.sha256()
    h.update(("|".join(cols)).encode())
    h.update(pd.util.hash_pandas_object(df[cols], index=False).values.tobytes())
    return h.hexdigest()[:16]


@dataclass
class DecisionContext:
    """All ingredients needed to reproduce a single rebalance decision."""

    decision_ts: str                  # ISO timestamp (UTC) when the decision was taken
    decision_date: str                # the *market* date the decision applies to
    model_hash: str                   # short hash of best_model.pkl
    feature_hash: str                 # short hash of the feature snapshot
    universe: str                     # short hash of the sorted ticker list
    hmm_state: int                    # 0 = bull, 1 = bear
    hmm_prob_bull: float              # probability mass on the bull state
    regime_label: str                 # human-readable

    @property
    def decision_id(self) -> str:
        """SHA256 of (model_hash, feature_hash, universe, regime_state, date).
        Short-prefixed for readability."""
        material = "|".join([
            self.model_hash, self.feature_hash, self.universe,
            str(self.hmm_state), self.decision_date,
        ])
        return hashlib.sha256(material.encode()).hexdigest()[:16]


def build_decision_context(
    *,
    model_path: str,
    feature_df: pd.DataFrame,
    universe: list[str],
    hmm_state: int,
    hmm_probs: np.ndarray,
    decision_date: str | None = None,
) -> DecisionContext:
    """Assemble a DecisionContext from the live inputs the agent already has."""
    universe_str = ",".join(sorted(universe))
    universe_hash = hashlib.sha256(universe_str.encode()).hexdigest()[:16]
    return DecisionContext(
        decision_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        decision_date=(decision_date
                       or datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        model_hash=file_sha256(model_path),
        feature_hash=dataframe_hash(feature_df),
        universe=universe_hash,
        hmm_state=int(hmm_state),
        hmm_prob_bull=float(hmm_probs[0]) if len(hmm_probs) > 0 else float("nan"),
        regime_label="bull" if hmm_state == 0 else "bear",
    )


def save_feature_snapshot(decision_id: str, feature_df: pd.DataFrame) -> str:
    """Persist the exact feature matrix used for the decision under
    ./agent_logs/decisions/<decision_id>.parquet so the trade can be replayed."""
    path = os.path.join(SNAPSHOT_DIR, f"{decision_id}.parquet")
    try:
        feature_df.to_parquet(path, index=False)
    except Exception as e:
        # Fall back to CSV if parquet writer isn't available
        path = os.path.join(SNAPSHOT_DIR, f"{decision_id}.csv")
        feature_df.to_csv(path, index=False)
    return path


def write_decision_row(ctx: DecisionContext, extras: dict[str, Any] | None = None) -> None:
    """Append one row to decisions.csv describing this rebalance event."""
    row = asdict(ctx)
    row["decision_id"] = ctx.decision_id
    if extras:
        row.update(extras)
    _append_csv(DECISIONS_PATH, row)


# ── Per-trade logger ──────────────────────────────────────────────────────────

@dataclass
class TradeLogRow:
    """One log entry per submitted order.  Stays human-readable so you can
    eyeball the CSV in a year and understand why a trade fired."""

    decision_id: str
    decision_date: str
    ts: str
    ticker: str
    action: str                       # BUY / SELL / COVER / LIQUIDATE
    target_weight: float              # post-trade target %
    delta_weight: float               # change vs current %
    intended_qty: int
    intended_notional_usd: float
    ref_price: float                  # last close used for sizing
    raw_score: float                  # pre-friction model rank score
    ml_signal: float                  # post-PCA ensemble score
    hmm_state: int
    hmm_prob_bull: float
    kalman_beta: float
    side: str                         # "long" / "short" / "flat"
    dry_run: bool
    notes: str = ""


class TradeLogger:
    """Structured trade log.  One row per *order* (so a rebalance with N
    deltas writes N rows, sharing the same decision_id)."""

    def __init__(self, path: str = TRADES_PATH):
        self.path = path

    def log(self, row: TradeLogRow) -> None:
        _append_csv(self.path, asdict(row))

    def log_many(self, rows: list[TradeLogRow]) -> None:
        for r in rows:
            self.log(r)


# ── Drawdown circuit breaker ──────────────────────────────────────────────────

class DrawdownCircuitBreaker:
    """Track running equity peak and refuse to trade if equity falls more than
    `threshold` below it.

    State is persisted in equity_history.csv.  Tripping creates a file at
    HALT_FILE — the operator has to remove it manually to resume.  This is
    deliberately strict: bots that "auto-reset" after a drawdown tend to
    chase losses.
    """

    def __init__(self, threshold_pct: float = 5.0, history_path: str = EQUITY_HISTORY_PATH):
        self.threshold = threshold_pct / 100.0
        self.history_path = history_path

    # --- Halt file management -------------------------------------------------

    def is_halted(self) -> bool:
        return os.path.exists(HALT_FILE)

    def reset(self) -> None:
        """Manually clear the halt file.  Only meant to be called by an
        operator, never by the bot itself."""
        if os.path.exists(HALT_FILE):
            os.remove(HALT_FILE)

    def _trip(self, current_equity: float, peak_equity: float, drawdown_pct: float) -> None:
        body = {
            "tripped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "peak_equity": peak_equity,
            "current_equity": current_equity,
            "drawdown_pct": drawdown_pct,
            "threshold_pct": self.threshold * 100,
            "instructions": (
                "Delete this file manually after you have investigated the "
                "drawdown.  The bot will refuse to submit orders while this "
                "file exists."
            ),
        }
        with open(HALT_FILE, "w") as f:
            json.dump(body, f, indent=2)

    # --- Equity tracking ------------------------------------------------------

    def update_and_check(self, current_equity: float, ts: str | None = None) -> dict:
        """Append the latest equity, compute drawdown vs running peak, trip
        the halt file if the threshold is breached.

        Returns a small status dict so the caller can decide whether to
        continue with the rebalance."""
        ts = ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
        history = self._load_history()
        peak = max([float(e["equity"]) for e in history] + [current_equity])
        drawdown = (peak - current_equity) / peak if peak > 0 else 0.0
        triggered = drawdown > self.threshold

        # Persist this observation
        _append_csv(self.history_path, {
            "ts": ts, "equity": current_equity,
            "peak": peak, "drawdown_pct": drawdown * 100,
        })

        if triggered and not self.is_halted():
            self._trip(current_equity, peak, drawdown * 100)

        return {
            "halted": self.is_halted(),
            "newly_triggered": triggered and self.is_halted(),
            "peak_equity": peak,
            "current_equity": current_equity,
            "drawdown_pct": drawdown * 100,
            "threshold_pct": self.threshold * 100,
        }

    def _load_history(self) -> list[dict]:
        if not os.path.exists(self.history_path):
            return []
        try:
            return pd.read_csv(self.history_path).to_dict(orient="records")
        except Exception:
            return []


# ── Performance attribution ───────────────────────────────────────────────────

@dataclass
class AttributionSnapshot:
    """One row of three-bucket PnL attribution between rebalances."""

    decision_id: str
    period_start: str
    period_end: str
    realised_return_pct: float
    benchmark_return_pct: float        # S&P 500 over the same window
    gross_exposure_pct: float          # |long| + |short|
    hmm_state: int
    pnl_regime_pct: float              # HMM exposure delta × benchmark
    pnl_ml_pct: float                  # cross-sectional tilt over the universe
    pnl_kalman_pct: float              # beta-sized residual
    pnl_residual_pct: float            # whatever the three above don't explain


class PerformanceAttribution:
    """Decompose realised PnL between two rebalance events into three buckets:

        PnL_regime  ≈ (gross_exposure_t1 − gross_exposure_t0) × benchmark_return
        PnL_ml      ≈ Σ_i (w_i − w_universe_mean) × (r_i − r_universe_mean)
        PnL_kalman  ≈ Σ_i (β_i × benchmark_return × w_i)
        PnL_resid   = realised − (regime + ml + kalman)

    The three buckets are NOT mutually exclusive in the strict accounting
    sense (any decomposition under correlated drivers is approximate), but
    they answer the practical question "which signal explained today's PnL"
    well enough for diagnostics.  The residual catches everything we can't
    attribute, so a large residual is a sign that the bot's PnL is being
    driven by something we don't measure (idiosyncratic news, fills, etc.).
    """

    def attribute(
        self, *,
        decision_id: str,
        period_start: str,
        period_end: str,
        weights_prev: dict[str, float],
        weights_curr: dict[str, float],
        realised_returns: dict[str, float],
        benchmark_return: float,
        kalman_betas: dict[str, float],
        hmm_state: int,
    ) -> AttributionSnapshot:
        # Universe = anything that appears in either weight vector
        tickers = sorted(set(weights_prev) | set(weights_curr) | set(realised_returns))

        w_prev = np.array([weights_prev.get(t, 0.0) for t in tickers])
        w_curr = np.array([weights_curr.get(t, 0.0) for t in tickers])
        r      = np.array([realised_returns.get(t, 0.0) for t in tickers])
        beta   = np.array([kalman_betas.get(t, 1.0) for t in tickers])

        # Realised return = sum_i w_prev_i * r_i   (held the previous weights over the window)
        realised = float((w_prev * r).sum())

        gross_prev = float(np.abs(w_prev).sum())
        gross_curr = float(np.abs(w_curr).sum())

        pnl_regime = float((gross_curr - gross_prev) * benchmark_return)
        ml_demeaned_w = w_prev - w_prev.mean()
        ml_demeaned_r = r - r.mean()
        pnl_ml = float((ml_demeaned_w * ml_demeaned_r).sum())
        pnl_kalman = float((beta * benchmark_return * w_prev).sum())
        pnl_residual = realised - (pnl_regime + pnl_ml + pnl_kalman)

        snap = AttributionSnapshot(
            decision_id=decision_id,
            period_start=period_start,
            period_end=period_end,
            realised_return_pct=realised * 100,
            benchmark_return_pct=benchmark_return * 100,
            gross_exposure_pct=gross_prev * 100,
            hmm_state=int(hmm_state),
            pnl_regime_pct=pnl_regime * 100,
            pnl_ml_pct=pnl_ml * 100,
            pnl_kalman_pct=pnl_kalman * 100,
            pnl_residual_pct=pnl_residual * 100,
        )
        _append_csv(ATTRIBUTION_PATH, asdict(snap))
        return snap


# ── CSV append helper ─────────────────────────────────────────────────────────

def _append_csv(path: str, row: dict[str, Any]) -> None:
    """Append a single row to a CSV, writing the header on first write."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
