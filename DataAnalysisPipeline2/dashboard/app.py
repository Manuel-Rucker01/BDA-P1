#!/usr/bin/env python3
"""
BDA Quant Dashboard — Flask backend.

A small web UI on top of the production trading agent so non-technical users
(friends!) can:

  * click "Run Pipeline" to score the universe and see the *proposed* book,
  * click "Send to Alpaca" to submit that exact proposal (paper account),
  * inspect live holdings (qty / avg cost / current value / unrealised P&L),
  * browse history: equity curve, drawdown, per-period attribution, trades.

All trading logic is delegated to pipeline_service, which itself reuses the
unmodified BDATradingAgent. This file is just HTTP plumbing.

Run:
    # from repo root
    python -m DataAnalysisPipeline2.dashboard.app
    # or from this directory
    python app.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import json

from flask import Flask, jsonify, request, send_from_directory

try:
    from . import pipeline_service as svc
except ImportError:
    import pipeline_service as svc  # script-mode

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, static_folder=None)


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/status")
def api_status():
    status = svc.server_status()
    # Merge in saved profile / environment availability
    try:
        profile = load_profile()
    except Exception:
        profile = {}
    env_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY")
    # config.py loads trading_agent/.env, where the secret is ALPACA_SECRET_KEY;
    # accept that name too (the trading code uses it) plus the legacy variants.
    env_secret = (os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
                  or os.getenv("APCA_API_SECRET"))
    configured = bool(
        # svc.server_status() already checks config.ALPACA_* which loads the .env,
        # so trust it first instead of only the raw process environment.
        status.get("alpaca_configured")
        or (profile.get("use_env") and env_key and env_secret)
        or (profile.get("alpaca_key") and profile.get("alpaca_secret"))
        or (env_key and env_secret))
    status["alpaca_configured"] = configured
    if "paper_trading" in profile:
        status["paper_trading"] = profile.get("paper_trading")
    status["profile"] = {"name": profile.get("name"), "use_env": profile.get("use_env", False)}
    return jsonify(status)


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(silent=True) or {}
    job_id = svc.start_run(
        universe=body.get("universe", "full"),
        strategy=body.get("strategy", "top_k"),
        top_k=body.get("top_k"),
        top_pct=body.get("top_pct"),
        force_regime=body.get("force_regime"),
    )
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/run/status/<job_id>")
def api_run_status(job_id):
    job = svc.get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."}), 404
    return jsonify({"ok": True, "job": job})


@app.route("/api/execute", methods=["POST"])
def api_execute():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    if not job_id:
        return jsonify({"ok": False, "error": "Missing job_id."}), 400
    return jsonify(svc.execute(job_id))


@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(svc.get_portfolio())


@app.route("/api/history")
def api_history():
    return jsonify(svc.get_history())


# -------------------- simple profile storage -------------------- #
PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def load_profile():
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_profile(p):
    try:
        with open(PROFILE_FILE, "w") as f:
            json.dump(p, f)
    except Exception:
        pass


@app.route("/api/profile", methods=["GET", "POST"])
def api_profile():
    if request.method == "GET":
        profile = load_profile()
        env_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY")
        env_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET")
        profile["env_available"] = bool(env_key and env_secret)
        # Mask stored secrets for safety
        if not profile.get("use_env", False):
            if "alpaca_key" in profile:
                profile["alpaca_key"] = "*****"
            if "alpaca_secret" in profile:
                profile["alpaca_secret"] = "*****"
        return jsonify({"ok": True, "profile": profile})
    else:
        body = request.get_json(silent=True) or {}
        use_env = bool(body.get("use_env"))
        profile = {"name": body.get("name")}
        profile["use_env"] = use_env
        profile["paper_trading"] = bool(body.get("paper_trading"))
        if not use_env:
            profile["alpaca_key"] = body.get("alpaca_key")
            profile["alpaca_secret"] = body.get("alpaca_secret")
        save_profile(profile)
        return jsonify({"ok": True, "profile": {"name": profile.get("name"), "use_env": profile.get("use_env")}})


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    print("=" * 70)
    print(" BDA Quant Dashboard")
    print(f" Open  ->  http://127.0.0.1:{port}")
    print("=" * 70)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
