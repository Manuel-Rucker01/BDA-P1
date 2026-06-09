"""Pickle compatibility helpers for legacy backtest scripts.

The current model artifact may contain ``TorchMLPRegressor`` instances that
were serialized when ``kg_embeddings_classifier.py`` ran as ``__main__``.
Registering the class on ``sys.modules['__main__']`` keeps old standalone
backtest scripts able to unpickle the artifact without importing the full live
trading bot.
"""

from __future__ import annotations

import os
import sys


def register_pickle_compat() -> None:
    """Register training-time classes needed by ``best_model.pkl``."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..", "scripts")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "kg_embeddings_classifier.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            break

    try:
        from kg_embeddings_classifier import TorchMLPRegressor

        sys.modules["__main__"].TorchMLPRegressor = TorchMLPRegressor
    except Exception as exc:
        print(f"[shim] could not pre-register TorchMLPRegressor: {exc}")
