"""Final artefact for the best Round-8 book's signal, ``sn30_altall_hd`` (paper
opt-in; it FAILS the pre-registered gate -- DSR 0.863 at N = 92, IR vs SPY < 0).

No new trial: the exact Round-8 recipe (``round8.feats_for('sn30_altall_hd')``
= v3 core + EDGAR fundamentals + hot-decked insider block + FINRA short
interest, target ``target_sn_30``, ``train.fit_ensemble`` with the fixed
LightGBM/XGBoost/CatBoost parameters, 52-week embargoed early stopping, refit on
the whole window) fitted once on every labelled weekly row.

Data completion (same code, more complete inputs): the Round-8 evaluation frame
had insider features MISSING after 2026-03-31 (last bulk quarter) and SI up to
~2 months stale on its last weeks (header-format miss, fixed afterwards). The
final frame recomputes the alternative-data block with the bulk quarters + the
daily EDGAR Form 4 feed (``form4.load_events``, exact bulk parity) and the
refreshed FINRA files, so the training rows of 2026Q2-Q3 carry the same
information the live bot sees. Everything else (``weekly_mh`` rows, z-scoring,
hot-deck, labels) is the Round-8 frame code (``altdata.weekly_features``,
``round8.build_frame``) writing to separate files; the Round-8 evaluation
files are not touched.

    python -m DataAnalysisPipeline2.v3.final_altall frame    # alt_weekly_final + weekly_dataset_r8_final
    python -m DataAnalysisPipeline2.v3.final_altall train    # ExploitationZone/model_v3_altall.pkl (+ manifest)
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import altdata as ad
from . import artefact as af
from . import dataset as ds
from . import form4
from . import round8 as r8
from . import train as tr

SIGNAL = "sn30_altall_hd"
TARGET = "target_sn_30"
CONSTRUCTION = "mv_abs_all"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "ExploitationZone")
ARTEFACT_NAME = "model_v3_altall"
# Round-8 trailing IC (pre-registered R5 rule: last 104 matured weekly ICs on
# v3-eligible names, shrunk to a 0.03 prior with 26 pseudo-weeks, floored at 0)
# of the sn30_altall_hd walk-forward at its last decision date 2026-11-04.
# Full-sample mean weekly IC: 0.0734 (403 weeks, NW t 5.10).
TRAILING_IC = 0.0534
MEAN_IC = 0.0734


def alt_path() -> str:
    return os.path.join(ds.v3_dir(), "alt_weekly_final.parquet")


def frame_path() -> str:
    return tr.weekly_path("dataset_r8_final")


def feature_columns() -> list:
    return r8.feats_for(SIGNAL)


def build_frame(log=print) -> dict:
    t0 = time.time()
    events = form4.load_events()
    si = ad.load_si(refresh_cache=True)
    info = {"insider_last_filing": str(pd.Timestamp(events["filing_date"].max()).date()),
            "insider_lines": int(len(events)),
            "si_last_settlement": str(pd.Timestamp(si["settlement_date"].max()).date())}
    log(f"[final] events {len(events):,} (last filing {info['insider_last_filing']}), "
        f"SI last settlement {info['si_last_settlement']}")
    ad.weekly_features(log=log, events=events, si=si, out_path=alt_path())
    w = r8.build_frame(log=log, alt_path=alt_path(), out_path=frame_path())
    info.update({"rows": int(len(w)), "dates": [str(w["Date"].min()), str(w["Date"].max())],
                 "seconds": round(time.time() - t0)})
    json.dump(info, open(frame_path() + ".json", "w"), indent=1)
    log(f"[final] frame done: {info}")
    return info


def train(log=print, out_dir: str = OUT_DIR, name: str = ARTEFACT_NAME, calendar=None) -> tuple[str, str]:
    """Fit the fixed recipe on every labelled row of the final frame. ``calendar``
    (session dates, for the embargo) defaults to the base v3 dataset's; the
    quarterly retrain (``v3/retrain_altall.py``) passes the refreshed EDGAR
    dataset's calendar, which holds the same sessions plus the new ones."""
    feats = feature_columns()
    w = pd.read_parquet(frame_path(), columns=["ticker", "Date", TARGET] + feats)
    w = w[w[TARGET].notna()].reset_index(drop=True)
    cal = ds.calendar_from_dataset() if calendar is None else calendar
    frame_info = json.load(open(frame_path() + ".json")) if os.path.exists(frame_path() + ".json") else {}
    log(f"[final] training {SIGNAL}: {len(w):,} labelled rows, {w['Date'].nunique()} dates "
        f"{w['Date'].min()}..{w['Date'].max()}, {len(feats)} features")
    extra = {
        "signal": SIGNAL,
        "feature_blocks": ["core", "fundamentals", "insider_hd", "si"],
        "feature_block_columns": {"core": list(ds.FEATURE_COLUMNS), "fundamentals": tr.fund_columns(),
                                  "insider_hd": list(r8.HD_COLS), "si": list(ad.SI_COLUMNS)},
        "recipe": "round8.feats_for('sn30_altall_hd') + train.fit_ensemble (fixed params, embargo "
                  f"{tr.EMBARGO_SESSIONS}, {tr.VAL_WEEKS}-week early-stopping split, refit on all rows)",
        "training_frame": "weekly_mh rows + altdata.weekly_features(bulk + daily Form 4 events, refreshed "
                          "FINRA) + round8.build_frame (insider hot-deck k=5, per-date z-score)",
        "alt_data_rules": {"insider": "SEC Forms 4/5 open-market P/S, available the business day after "
                                      "filing; windows 30/90/180 calendar days; unmapped names get a same-date "
                                      "mapped donor among the 5 nearest on " + ", ".join(r8.HD_MATCH),
                           "short_interest": "latest FINRA exchange-listed file with settlement + 8 business "
                                             "days <= decision date; symbol on the settlement-date listing"},
        "data_as_of": frame_info,
        "fundamentals_rule": "dataset.FundamentalsBlock (as-of join on filing availability, hot-deck of "
                             "names without any EDGAR record); sectors = EDGAR SIC FF-12, proxy elsewhere",
        "construction_live": CONSTRUCTION,
        "opt_ic": {"trailing_ic": TRAILING_IC, "mean_weekly_ic": MEAN_IC,
                   "rule": "R5_COMMON trailing IC at 2026-11-04 (104 matured weeks, prior 0.03 x 26 weeks)"},
        "evaluation": {"round": 8, "n_trials": 92, "gate": "FAIL", "sharpe": 1.02, "max_dd": -0.285,
                       "dsr": 0.863, "ir_spy": -0.13, "ir_ew": 0.25,
                       "report": "DataAnalysisPipeline2/results/v3_walkforward_report.md (Round 8)"},
        "status": "paper opt-in only (fails the pre-registered gate); production stays best_model.pkl",
    }
    art = af.build_artifact(w, TARGET, cal, variant=SIGNAL, construction=None, extra=extra, log=log,
                            feats=feats, profile_dates=True)
    art["manifest"]["model_id"] = f"v3_{SIGNAL}_{art['manifest']['created_at'][:10].replace('-', '')}"
    pkl, js = af.save_artifact(art, out_dir, name)
    log(f"[final] saved {pkl} ({os.path.getsize(pkl) / 1e6:.1f} MB), {js}")
    return pkl, js


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        {"frame": build_frame, "train": train}[a]()
