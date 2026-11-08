"""Live computation of the round-8 alternative-data features (not wired into
the bot; see ``python -m DataAnalysisPipeline2.v3.altlive refresh``).

``live_altdata_features(tickers, decision_date, core_frame)`` returns, for the
decision date, exactly the columns the ``sn30_altall_hd`` model consumes --
``ins_*_hd`` (nearest-neighbour-filled insider block) and ``si_*`` -- z-scored
per date over the rows given, with the same point-in-time rules as training:

* insider: Form 4/5 open-market P/S lines (bulk quarters + the daily EDGAR feed,
  ``form4.load_events``) available the business day after filing; windows of
  30/90/180 days; names without a CIK get a same-date donor's vector among the
  5 nearest on (log dollar volume, 20d vol, 20d / 50d return);
* short interest: the latest FINRA file whose settlement date + 8 business days
  <= the decision date, symbol on the settlement-date listing, reused-symbol guard.

``core_frame`` = the decision-date cross-section the model scores (one row per
ticker): ticker, Date, eligible, close_raw (or company_close), log_mcap (raw),
first_date (first bar of the ticker id) and the z-scored HD match columns -- in
live use the v3 feature path (``dataset.compute_v3_features`` + ``zscore``).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import altdata as ad
from . import form4
from ._alpaca import data_dir

HD_MATCH = ["log_dollar_volume_20d", "ret_vol_20d", "return_20d", "return_50d"]
HD_COLS = [f"{c}_hd" for c in ad.INSIDER_COLUMNS]
MODEL_COLUMNS = HD_COLS + ad.SI_COLUMNS


def _cik(rows: pd.DataFrame, cik_map: pd.DataFrame) -> pd.Series:
    from . import fundamentals as fu
    keys = pd.DataFrame({"tkey": rows["ticker"].astype(str).str.split("~").str[0].map(fu.norm_ticker),
                         "Date": pd.to_datetime(rows["Date"]).astype("datetime64[ns]")})
    return pd.Series(fu.resolve_cik(keys, cik_map).to_numpy(), index=rows.index)


def compute(core_frame: pd.DataFrame, events: pd.DataFrame, si: pd.DataFrame, cik_map: pd.DataFrame,
            insider_end=None) -> pd.DataFrame:
    """Raw + model columns for the rows of ``core_frame`` (one or several dates)."""
    from .round8 import insider_hotdeck
    rows = core_frame.copy()
    rows["cik"] = _cik(rows, cik_map)
    ins = ad.insider_features(events, sorted(rows["Date"].unique())).rename(columns={"issuer_cik": "cik"})
    ins["cik"] = ins["cik"].astype("Int64")
    rows["cik"] = rows["cik"].astype("Int64")
    rows = rows.merge(ins, on=["cik", "Date"], how="left")
    rows[ad.INSIDER_COLUMNS] = rows[ad.INSIDER_COLUMNS].fillna(0.0)
    if insider_end is not None:
        rows.loc[pd.to_datetime(rows["Date"]) > pd.Timestamp(insider_end), ad.INSIDER_COLUMNS] = np.nan
    px = rows["close_raw"].fillna(rows["company_close"]) if "company_close" in rows else rows["close_raw"]
    rows["shares"] = np.exp(rows["log_mcap"]) / px
    rows[ad.SI_COLUMNS] = ad.si_features(si, rows[["ticker", "Date", "first_date", "shares"]]).to_numpy()
    rows[HD_COLS] = insider_hotdeck(rows).to_numpy()
    rows[ad.INSIDER_COLUMNS + ad.SI_COLUMNS + HD_COLS] = core.cross_sectional_zscore(
        rows, ad.INSIDER_COLUMNS + ad.SI_COLUMNS + HD_COLS).to_numpy()
    return rows


def live_altdata_features(tickers, decision_date: str, core_frame: pd.DataFrame,
                          events: pd.DataFrame | None = None, si: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model inputs (``MODEL_COLUMNS``, z-scored over ``core_frame``'s rows of the
    decision date) for ``tickers``. ``events``/``si`` default to the local caches
    (bulk + daily Form 4 feed; FINRA files)."""
    cf = core_frame[core_frame["Date"] == decision_date]
    events = form4.load_events() if events is None else events
    si = ad.load_si(refresh_cache=True) if si is None else si
    cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"))
    out = compute(cf, events, si, cm)
    out = out[out["ticker"].isin(set(tickers))]
    return out[["ticker", "Date"] + MODEL_COLUMNS].reset_index(drop=True)


def refresh(log=print) -> dict:
    """CLI: bring both caches up to date (Form 4 daily feed, FINRA files)."""
    info = {"form4": form4.refresh(log=log)}
    last = sorted(os.listdir(data_dir("finra_si", "raw")))[-1][4:10]
    got = ad.download_si(start=f"{last[:4]}-{last[4:6]}", log=log)
    ad.load_si(refresh_cache=True)
    info["si_files"] = len(got)
    try:                       # Round-11 earnings blackout events (8-K 2.02); optional, never blocks the rest
        from . import earnlive
        info["earnings_8k"] = earnlive.refresh(log=log)
    except Exception as e:     # noqa: BLE001
        log(f"[altlive] 8-K 2.02 refresh failed ({type(e).__name__}: {e}); the earnings snapshot keeps its date")
        info["earnings_8k"] = {"error": f"{type(e).__name__}: {e}"}
    return info


# --------------------------------------------------------------------------- #
# Validation helpers (used by the tests)
# --------------------------------------------------------------------------- #
def parity(days, out_json: bool = True) -> dict:
    """Daily-feed rows vs bulk rows for filing dates in ``days`` (bulk restricted
    to the same issuer filter). Key = (accession, code, shares, price)."""
    import json
    bulk = ad.load_insider_events()
    ds_ = [pd.Timestamp(d) for d in days]
    b = bulk[bulk["filing_date"].isin(ds_)]
    b = b[b["issuer_cik"].isin(form4.universe_issuer_ciks())]
    dly = form4.normalize(pd.concat([pd.read_parquet(os.path.join(form4.cache_dir(), f"{pd.Timestamp(d):%Y-%m-%d}.parquet"))
                                     for d in days], ignore_index=True)).drop_duplicates(
        ["accession", "code", "shares", "price"])

    def key(df):
        return (df["accession"].astype(str) + "|" + df["code"].astype(str) + "|"
                + df["shares"].round(4).astype(str) + "|" + df["price"].round(4).astype(str))

    kb, kd = key(b), key(dly)
    sb, sd = set(kb), set(kd)
    m = b.assign(_k=kb.to_numpy()).drop_duplicates("_k").merge(
        dly.assign(_k=kd.to_numpy()).drop_duplicates("_k"), on="_k", suffixes=("_b", "_d"))
    mism = ((m["issuer_cik_b"].astype("Int64") != m["issuer_cik_d"].astype("Int64"))
            | ((m["value_b"] - m["value_d"]).abs() > 0.01) | (m["avail_date_b"] != m["avail_date_d"]))
    owner_mism = m["owner_cik_b"].astype("Int64") != m["owner_cik_d"].astype("Int64")
    od_mism = m["is_od_b"] != m["is_od_d"]
    only_b = b[~kb.isin(sd)]
    rep = {"days": [str(d) for d in days], "bulk_rows": int(len(b)), "daily_rows": int(len(dly)),
           "bulk_accessions": int(b["accession"].nunique()), "daily_accessions": int(dly["accession"].nunique()),
           "share_bulk_found_in_daily": float(kb.isin(sd).mean()) if len(b) else float("nan"),
           "share_daily_found_in_bulk": float(kd.isin(sb).mean()) if len(dly) else float("nan"),
           "matched": int(len(m)), "field_mismatch_share": float(mism.mean()) if len(m) else float("nan"),
           "owner_mismatch_share": float(owner_mism.mean()) if len(m) else float("nan"),
           "officer_director_flag_mismatch_share": float(od_mism.mean()) if len(m) else float("nan"),
           "note": "field_mismatch = issuer CIK, dollar value or availability date; owner/OD differences come from "
                   "multi-owner filings whose first REPORTINGOWNER row in the bulk TSV is not the XML's first owner",
           "examples_bulk_only": only_b[["accession", "code", "shares", "price"]].head(5).astype(str).to_dict("records")}
    if out_json:
        json.dump(rep, open(os.path.join(data_dir("sec_insider"), "parity.json"), "w"), indent=1)
    return rep


def offline_vs_live(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(live, offline) model columns for every row of the round-8 frame on ``date``."""
    import glob
    from . import dataset as ds
    from . import round6 as r6
    from . import round8 as r8
    off = pd.read_parquet(r8.frame_path(), columns=["ticker", "Date"] + MODEL_COLUMNS,
                          filters=[("Date", "==", date)])
    cf = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date"] + HD_MATCH, filters=[("Date", "==", date)])
    raw = ds.load_daily(years=[int(date[:4])], dates=[date], out_dir=ds.v3_dir(ds.EDGAR_DATASET),
                        columns=["ticker", "Date", "close_raw", "company_close", "log_mcap", "eligible"])
    cf = cf.merge(raw, on=["ticker", "Date"], how="left")
    first = pd.concat([pd.read_parquet(f, columns=["ticker", "Date"]).groupby("ticker")["Date"].min()
                       for f in sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))])
    cf["first_date"] = cf["ticker"].map(first)
    live = live_altdata_features(cf["ticker"], date, cf, events=ad.load_insider_events(), si=ad.load_si())
    key = lambda x: x.sort_values("ticker").reset_index(drop=True)[["ticker"] + MODEL_COLUMNS]
    return key(live).astype({c: "float64" for c in MODEL_COLUMNS}), key(off).astype({c: "float64" for c in MODEL_COLUMNS})


if __name__ == "__main__":
    import sys
    if "refresh" in sys.argv:
        print(refresh())
