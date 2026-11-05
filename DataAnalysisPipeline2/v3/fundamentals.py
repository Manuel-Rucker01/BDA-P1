"""Point-in-time fundamentals from SEC companyfacts (WS2).

Point-in-time rule
------------------
Every XBRL fact carries ``filed`` (EDGAR filing date).  A fact becomes usable on
the **next NYSE business day after ``filed``** (``avail_date``).  "As known on
date D" = for every reporting period, the value from the latest filing with
``avail_date <= D``; a later restatement/amendment overrides a period only from
its own ``avail_date`` onwards (history before that is never rewritten).

Concepts and fallbacks (us-gaap unless noted; the first concept that reports a
given period wins, per period -- this stitches e.g. the 2018 ASC-606 switch from
``SalesRevenueNet`` to ``RevenueFromContractWithCustomer...``):

  revenue        Revenues > RevenueFromContractWithCustomerExcludingAssessedTax >
                 RevenueFromContractWithCustomerIncludingAssessedTax > SalesRevenueNet >
                 RevenuesNetOfInterestExpense (banks)
  net_income     NetIncomeLoss > ProfitLoss (incl. NCI) > NetIncomeLossAvailableToCommonStockholdersBasic
  gross_profit   GrossProfit; else revenue_TTM - cost_TTM at the same period end, cost =
                 CostOfRevenue > CostOfGoodsAndServicesSold > CostOfGoodsSold
  cfo            NetCashProvidedByUsedInOperatingActivities >
                 NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
  assets         Assets
  liabilities    Liabilities; else LiabilitiesAndStockholdersEquity - total equity (incl. NCI)
                 at the same balance-sheet date
  equity         StockholdersEquity > StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
  lt_debt        LongTermDebtNoncurrent > LongTermDebt > LongTermDebtAndCapitalLeaseObligations
  cash           CashAndCashEquivalentsAtCarryingValue >
                 CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents > Cash
  shares         dei:EntityCommonStockSharesOutstanding > CommonStockSharesOutstanding >
                 WeightedAverageNumberOfDilutedSharesOutstanding > WeightedAverageNumberOfSharesOutstandingBasic
  wdil (issuance) WeightedAverageNumberOfDilutedSharesOutstanding > ...Basic, latest period vs the
                 same-length period one year earlier *as reported in the latest filing* (split-restated)

Stock items: latest balance-sheet date known; a lower-priority concept is only
used when the preferred one is >120 days staler.  Only USD (monetary) / shares
units and forms 10-K, 10-Q, 10-KT, 10-QT, 20-F, 40-F (+ amendments) are used.

Flow items -> quarters -> TTM
  * quarter = a 75..105-day fact; else YTD differencing (same start, ends ~1 quarter
    apart: Q2 = 6M - Q1, Q3 = 9M - 6M, Q4 = FY - 9M); else Q4 = FY - (Q1+Q2+Q3)
  * TTM at period end E = a 340..390-day fact ending at E if any, else the sum of 4
    chained quarters ending at E; uses the latest E (of the 4 latest ends) where it is formable
  * ``*_ttm_prev`` = TTM at the period end ~1 year before (350..380 days), as known at D

Output ``fundamentals_asof.parquet``: event table, one row per (cik, asof_date)
(asof_date = an ``avail_date`` on which any input of that CIK changed).

CLI (from DataAnalysisPipeline2/):  python -m v3.fundamentals build [--limit N]
"""
from __future__ import annotations

import argparse
import bisect
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    import json as _json

    _loads = _json.loads

from .edgar import edgar_dir, fundamentals_dir, norm_ticker

log = logging.getLogger("v3.fundamentals")

# --------------------------------------------------------------------------- concepts
G, D = "us-gaap", "dei"
FLOW_ITEMS: dict[str, list[tuple[str, str]]] = {
    "revenue": [(G, "Revenues"), (G, "RevenueFromContractWithCustomerExcludingAssessedTax"),
                (G, "RevenueFromContractWithCustomerIncludingAssessedTax"), (G, "SalesRevenueNet"),
                (G, "RevenuesNetOfInterestExpense")],
    "net_income": [(G, "NetIncomeLoss"), (G, "ProfitLoss"),
                   (G, "NetIncomeLossAvailableToCommonStockholdersBasic")],
    "gross_profit": [(G, "GrossProfit")],
    "cost_of_revenue": [(G, "CostOfRevenue"), (G, "CostOfGoodsAndServicesSold"), (G, "CostOfGoodsSold")],
    "cfo": [(G, "NetCashProvidedByUsedInOperatingActivities"),
            (G, "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "wdil": [(G, "WeightedAverageNumberOfDilutedSharesOutstanding"),
             (G, "WeightedAverageNumberOfSharesOutstandingBasic")],
}
STOCK_ITEMS: dict[str, list[tuple[str, str]]] = {
    "assets": [(G, "Assets")],
    "liabilities": [(G, "Liabilities")],
    "liab_and_equity": [(G, "LiabilitiesAndStockholdersEquity")],
    "equity": [(G, "StockholdersEquity"),
               (G, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "equity_total": [(G, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
                     (G, "StockholdersEquity")],
    "lt_debt": [(G, "LongTermDebtNoncurrent"), (G, "LongTermDebt"),
                (G, "LongTermDebtAndCapitalLeaseObligations")],
    "cash": [(G, "CashAndCashEquivalentsAtCarryingValue"),
             (G, "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"), (G, "Cash")],
    "shares": [(D, "EntityCommonStockSharesOutstanding"), (G, "CommonStockSharesOutstanding"),
               (G, "WeightedAverageNumberOfDilutedSharesOutstanding"),
               (G, "WeightedAverageNumberOfSharesOutstandingBasic")],
}
SHARE_ITEMS = {"shares", "wdil"}
ALLOWED_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A",
                 "20-F", "20-F/A", "40-F", "40-F/A"}

# concept -> list of (kind, item, rank)
_CONCEPT_INDEX: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
for _kind, _items in (("flow", FLOW_ITEMS), ("stock", STOCK_ITEMS)):
    for _item, _concepts in _items.items():
        for _rank, _c in enumerate(_concepts):
            _CONCEPT_INDEX.setdefault(_c, []).append((_kind, _item, _rank))

Q_MIN, Q_MAX = 75, 105        # quarter length (days, end - start)
FY_MIN, FY_MAX = 340, 390     # fiscal year length (52/53-week years included)
STALE_FALLBACK_DAYS = 120     # lower-priority stock concept only if preferred is staler than this


# --------------------------------------------------------------------------- calendar
@lru_cache(maxsize=1)
def nyse_holidays() -> np.ndarray:
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                        USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                        USThanksgivingDay, nearest_workday)

    class _NYSE(AbstractHolidayCalendar):
        rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr,
                 USPresidentsDay, GoodFriday, USMemorialDay,
                 Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
                 Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay,
                 USThanksgivingDay, Holiday("Christmas", month=12, day=25, observance=nearest_workday)]

    return _NYSE().holidays("1990-01-01", "2040-12-31").values.astype("datetime64[D]")


def next_business_day(d) -> pd.Timestamp:
    """First NYSE business day strictly after calendar date ``d``."""
    x = np.datetime64(pd.Timestamp(d).date(), "D")
    return pd.Timestamp(np.busday_offset(x, 1, roll="backward", holidays=nyse_holidays()))


@lru_cache(maxsize=200_000)
def _ord(s: str) -> int:
    return date.fromisoformat(s).toordinal()


@lru_cache(maxsize=50_000)
def _avail_ord(filed: str) -> int:
    return next_business_day(filed).toordinal()


# --------------------------------------------------------------------------- state
class _FlowState:
    """Resolved duration facts of one flow item as known so far (dates as ordinals)."""

    __slots__ = ("rank", "by_end", "by_start", "ends")

    def __init__(self):
        self.rank: dict[tuple[int, int], int] = {}
        self.by_end: dict[int, dict[int, float]] = {}
        self.by_start: dict[int, dict[int, float]] = {}
        self.ends: list[int] = []

    def update(self, rank: int, s: int, e: int, v: float):
        cur = self.rank.get((s, e))
        if cur is not None and rank > cur:
            return  # a preferred concept already reports this period
        self.rank[(s, e)] = rank
        if e not in self.by_end:
            self.by_end[e] = {}
            bisect.insort(self.ends, e)
        self.by_end[e][s] = v
        self.by_start.setdefault(s, {})[e] = v

    def _prev_ends(self, e: int) -> list[int]:
        lo = bisect.bisect_left(self.ends, e - Q_MAX)
        hi = bisect.bisect_right(self.ends, e - Q_MIN)
        return sorted(self.ends[lo:hi], key=lambda x: abs(e - 91 - x))

    def quarter(self, e: int, use_fy: bool = True) -> float | None:
        starts = self.by_end.get(e)
        if not starts:
            return None
        for s, v in starts.items():                      # 1) a reported quarter
            if Q_MIN <= e - s <= Q_MAX:
                return v
        for s, v in starts.items():                      # 2) YTD differencing (same start)
            if e - s > Q_MAX:
                for e2, v2 in self.by_start.get(s, {}).items():
                    if Q_MIN <= e - e2 <= Q_MAX:
                        return v - v2
        if use_fy:                                       # 3) Q4 = FY - (Q1+Q2+Q3)
            for s, v in starts.items():
                if FY_MIN <= e - s <= FY_MAX:
                    tot, cur = 0.0, e
                    for _ in range(3):
                        nxt = next((p for p in self._prev_ends(cur)
                                    if self.quarter(p, use_fy=False) is not None), None)
                        if nxt is None:
                            break
                        tot += self.quarter(nxt, use_fy=False)
                        cur = nxt
                    else:
                        if Q_MIN <= cur - s <= Q_MAX:    # Q1 starts at the FY start
                            return v - tot
        return None

    def ttm(self, e: int) -> float | None:
        starts = self.by_end.get(e)
        if not starts:
            return None
        fy = [(abs(e - s - 365), v) for s, v in starts.items() if FY_MIN <= e - s <= FY_MAX]
        if fy:
            return min(fy)[1]
        q = self.quarter(e)
        if q is None:
            return None
        tot, cur = q, e
        for _ in range(3):
            for p in self._prev_ends(cur):
                qp = self.quarter(p)
                if qp is not None:
                    tot += qp
                    cur = p
                    break
            else:
                return None
        return tot

    def latest_ttm(self, n_try: int = 4) -> tuple[float | None, int | None]:
        for e in reversed(self.ends[-n_try:]):
            t = self.ttm(e)
            if t is not None:
                return t, e
        return None, None

    def ttm_year_before(self, e: int) -> float | None:
        lo = bisect.bisect_left(self.ends, e - 380)
        hi = bisect.bisect_right(self.ends, e - 350)
        for p in sorted(self.ends[lo:hi], key=lambda x: abs(e - 365 - x)):
            t = self.ttm(p)
            if t is not None:
                return t
        return None

    def yoy_same_length(self) -> tuple[float | None, int | None]:
        """Latest period value / same-length period one year earlier - 1 (for share counts)."""
        for e in reversed(self.ends[-2:]):
            for s, v in sorted(self.by_end[e].items(), key=lambda kv: e - kv[0]):  # shortest first
                dur = e - s
                lo = bisect.bisect_left(self.ends, e - 385)
                hi = bisect.bisect_right(self.ends, e - 345)
                for p in self.ends[lo:hi]:
                    for s2, v2 in self.by_end[p].items():
                        if abs((p - s2) - dur) <= 20 and v2 and v2 > 0 and v > 0:
                            return v / v2 - 1.0, e
        return None, None


class _StockState:
    __slots__ = ("vals", "latest")

    def __init__(self, n_ranks: int):
        self.vals: list[dict[int, float]] = [dict() for _ in range(n_ranks)]
        self.latest: list[int | None] = [None] * n_ranks

    def update(self, rank: int, e: int, v: float):
        self.vals[rank][e] = v
        if self.latest[rank] is None or e > self.latest[rank]:
            self.latest[rank] = e

    def resolve(self) -> tuple[float | None, int | None]:
        known = [le for le in self.latest if le is not None]
        if not known:
            return None, None
        m = max(known)
        for r, le in enumerate(self.latest):
            if le is not None and le >= m - STALE_FALLBACK_DAYS:
                return self.vals[r][le], le
        return None, None  # pragma: no cover

    def value_at(self, e: int) -> float | None:
        for d in self.vals:
            if e in d:
                return d[e]
        return None


# --------------------------------------------------------------------------- per-company build
def _extract_facts(cf: dict) -> list[tuple]:
    """Relevant facts as tuples (avail, filed, accn, form, kind, item, rank, start|None, end, val)."""
    out = []
    facts = cf.get("facts", {}) or {}
    for (tax, concept), targets in _CONCEPT_INDEX.items():
        node = facts.get(tax, {}).get(concept)
        if not node:
            continue
        units = node.get("units", {}) or {}
        for kind, item, rank in targets:
            unit = "shares" if item in SHARE_ITEMS else "USD"
            for f in units.get(unit, ()) or ():
                form = f.get("form")
                if form not in ALLOWED_FORMS or f.get("val") is None or not f.get("filed"):
                    continue
                st = f.get("start")
                e = _ord(f["end"])
                s = _ord(st) if st else None
                if kind == "flow" and s is None:
                    continue
                v = float(f["val"])
                if item in SHARE_ITEMS and v <= 0:
                    continue
                out.append((_avail_ord(f["filed"]), _ord(f["filed"]), f.get("accn") or "", form,
                            kind, item, rank, s, e, v))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return out


_FLOW_OUT = ["revenue", "net_income", "gross_profit", "cfo"]
_STOCK_OUT = ["assets", "liabilities", "equity", "lt_debt", "cash", "shares"]
ASOF_COLUMNS = (["cik", "asof_date", "last_filed", "last_form", "last_accn", "days_since_filing"]
                + [c for it in _FLOW_OUT for c in (f"{it}_ttm", f"{it}_ttm_prev", f"{it}_end")]
                + [c for it in _STOCK_OUT for c in (it, f"{it}_end")]
                + ["wdil_yoy", "wdil_end", "gp_source"])


def build_company_asof(cf: dict, cik: int | None = None) -> pd.DataFrame:
    """Event table for one companyfacts JSON document (see module docstring)."""
    cik = int(cik if cik is not None else cf.get("cik"))
    facts = _extract_facts(cf)
    flows = {k: _FlowState() for k in FLOW_ITEMS}
    stocks = {k: _StockState(len(v)) for k, v in STOCK_ITEMS.items()}
    rows, prev_sig = [], None
    i, n = 0, len(facts)
    while i < n:
        avail = facts[i][0]
        last = facts[i]
        while i < n and facts[i][0] == avail:
            t = facts[i]
            _, filed, accn, form, kind, item, rank, s, e, v = t
            if kind == "flow":
                flows[item].update(rank, s, e, v)
            else:
                stocks[item].update(rank, e, v)
            if (filed, accn) >= (last[1], last[2]):
                last = t
            i += 1
        row = {"cik": cik, "asof_date": avail, "last_filed": last[1], "last_form": last[3],
               "last_accn": last[2]}
        for it in ("revenue", "net_income", "cfo"):
            v, e = flows[it].latest_ttm()
            row[f"{it}_ttm"], row[f"{it}_end"] = v, e
            row[f"{it}_ttm_prev"] = flows[it].ttm_year_before(e) if e is not None else None
        # gross profit: reported GrossProfit, else revenue - cost at the revenue period end
        gv, ge = flows["gross_profit"].latest_ttm()
        rev_e = row["revenue_end"]
        src = "GrossProfit" if gv is not None else None
        if rev_e is not None and (ge is None or ge < rev_e - 100):
            c = flows["cost_of_revenue"].ttm(rev_e)
            if c is not None and row["revenue_ttm"] is not None:
                gv, ge, src = row["revenue_ttm"] - c, rev_e, "revenue-cost"
        row["gross_profit_ttm"], row["gross_profit_end"], row["gp_source"] = gv, ge, src
        row["gross_profit_ttm_prev"] = None
        if src == "GrossProfit":
            row["gross_profit_ttm_prev"] = flows["gross_profit"].ttm_year_before(ge)
        elif src == "revenue-cost":
            cp = flows["cost_of_revenue"].ttm_year_before(rev_e)
            if cp is not None and row["revenue_ttm_prev"] is not None:
                row["gross_profit_ttm_prev"] = row["revenue_ttm_prev"] - cp
        for it in ("assets", "liabilities", "equity", "lt_debt", "cash", "shares"):
            row[it], row[f"{it}_end"] = stocks[it].resolve()
        ae = row["assets_end"]
        if ae is not None and (row["liabilities"] is None or row["liabilities_end"] < ae):
            lse, eqt = stocks["liab_and_equity"].value_at(ae), stocks["equity_total"].value_at(ae)
            if lse is not None and eqt is not None:
                row["liabilities"], row["liabilities_end"] = lse - eqt, ae
        row["wdil_yoy"], row["wdil_end"] = flows["wdil"].yoy_same_length()
        sig = tuple(row.get(c) for c in ASOF_COLUMNS[6:])
        if sig == prev_sig:          # nothing derived changed -> no new event row
            continue
        prev_sig = sig
        rows.append(row)
    df = pd.DataFrame(rows, columns=ASOF_COLUMNS)
    if df.empty:
        return df
    for c in ["asof_date", "last_filed"] + [c for c in ASOF_COLUMNS if c.endswith("_end")]:
        df[c] = pd.to_datetime([date.fromordinal(int(x)) if pd.notna(x) else None for x in df[c]])
    df["days_since_filing"] = (df["asof_date"] - df["last_filed"]).dt.days
    num = [c for c in ASOF_COLUMNS if c.endswith(("_ttm", "_ttm_prev")) or c in _STOCK_OUT or c == "wdil_yoy"]
    df[num] = df[num].astype("float64")
    return df


# --------------------------------------------------------------------------- bulk build
def _build_chunk(args) -> pd.DataFrame:
    zpath, names = args
    parts = []
    with zipfile.ZipFile(zpath) as z:
        for n in names:
            try:
                cf = _loads(z.read(n))
                cik = int(cf.get("cik") or n[3:13])
                df = build_company_asof(cf, cik)
                if not df.empty:
                    parts.append(df)
            except Exception as e:  # keep going on a malformed company
                log.warning("skip %s: %s", n, e)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ASOF_COLUMNS)


def build_fundamentals_asof(zpath: Path | None = None, out: Path | None = None, *,
                            workers: int = 8, limit: int | None = None) -> Path:
    zpath = Path(zpath or edgar_dir() / "companyfacts.zip")
    out = Path(out or fundamentals_dir() / "fundamentals_asof.parquet")
    if not zpath.exists():
        raise FileNotFoundError(f"{zpath} missing - run `python -m v3.edgar download` first")
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.startswith("CIK") and n.endswith(".json"))
    if limit:
        names = names[:limit]
    chunks = [(str(zpath), names[i:i + 200]) for i in range(0, len(names), 200)]
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for k, df in enumerate(ex.map(_build_chunk, chunks), 1):
            if not df.empty:
                parts.append(df)
            if k % 10 == 0:
                log.info("companyfacts: %d/%d chunks", k, len(chunks))
    res = pd.concat(parts, ignore_index=True).sort_values(["cik", "asof_date"]).reset_index(drop=True)
    res.to_parquet(out, index=False)
    log.info("fundamentals_asof: %d rows, %d CIKs -> %s", len(res), res.cik.nunique(), out)
    return out


# --------------------------------------------------------------------------- features
FEATURE_COLUMNS = ["ep", "bp", "sp", "cfp", "gpa", "roa", "roe", "accruals", "sales_growth_yoy",
                   "ni_growth_scaled", "leverage", "net_issuance_12m", "log_mcap",
                   "fund_staleness_days"]


def _ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[ns]")


def resolve_cik(keys: pd.DataFrame, cik_map: pd.DataFrame) -> pd.Series:
    """CIK for each (tkey, Date) row of ``keys`` (index preserved; <NA> if unmapped).

    Uses cik_map rows whose [valid_from, valid_to] window contains Date (missing
    bounds = open).  When several CIKs qualify (reused/ambiguous ticker) a closed
    window (a delisted holder) wins over an open one, then the latest valid_from.
    """
    cm = cik_map.copy()
    cm["tkey"] = cm["ticker"].map(norm_ticker)
    for c in ("valid_from", "valid_to"):
        cm[c] = _ns(cm[c]) if c in cm else pd.Series(pd.NaT, index=cm.index, dtype="datetime64[ns]")
    cm = cm[["tkey", "cik", "valid_from", "valid_to"]].dropna(subset=["tkey", "cik"]).drop_duplicates()
    k = keys[["tkey", "Date"]].copy()
    k["_row"] = np.arange(len(k))
    m = k.merge(cm, on="tkey", how="inner")
    ok = (m["valid_from"].isna() | (m["Date"] >= m["valid_from"])) & \
         (m["valid_to"].isna() | (m["Date"] <= m["valid_to"]))
    m = m[ok]
    m = m.assign(_open=m["valid_to"].isna(), _vf=m["valid_from"].fillna(pd.Timestamp("1900-01-01")))
    m = m.sort_values(["_row", "_open", "_vf"], ascending=[True, True, False]).drop_duplicates("_row")
    out = pd.Series(pd.array([pd.NA] * len(k), dtype="Int64"), index=keys.index)
    out.iloc[m["_row"].to_numpy()] = m["cik"].astype("int64").to_numpy()
    return out


def fundamental_features(prices: pd.DataFrame, fund_asof: pd.DataFrame, cik_map: pd.DataFrame,
                         sectors: pd.DataFrame | None = None, *, close_col: str = "company_close",
                         split_factor_col: str | None = None) -> pd.DataFrame:
    """PIT fundamental features on a daily (ticker, Date) panel.

    ``prices`` needs ``ticker, Date, <close_col>``.  Fundamentals are attached with a
    backward as-of join on ``asof_date <= Date`` per CIK (never a future filing).
    Market cap = close x shares-as-known-at-Date.  If ``close`` is split-ADJUSTED,
    pass ``split_factor_col`` (raw_close = close x factor) -- otherwise pre-split
    market caps are off by the split ratio (filing share counts are as-reported).
    Returns a frame aligned to ``prices.index``: ticker, Date, cik, mcap, the
    columns in FEATURE_COLUMNS, ff12, sector.  Nothing is winsorized; NaN = unknown.
    """
    p = prices[["ticker", "Date", close_col] + ([split_factor_col] if split_factor_col else [])].copy()
    p["Date"] = _ns(p["Date"])
    p["tkey"] = p["ticker"].map(norm_ticker)
    p["cik"] = resolve_cik(p, cik_map)
    p["_row"] = np.arange(len(p))

    f = fund_asof.copy()
    f["asof_date"] = _ns(f["asof_date"])
    f["last_filed"] = _ns(f["last_filed"])
    f["cik"] = f["cik"].astype("int64")
    f = f.sort_values("asof_date")

    has = p[p["cik"].notna()].copy()
    has["cik"] = has["cik"].astype("int64")
    has = has.sort_values("Date")
    cols = ["cik", "asof_date", "last_filed", "revenue_ttm", "revenue_ttm_prev", "net_income_ttm",
            "net_income_ttm_prev", "gross_profit_ttm", "cfo_ttm", "assets", "liabilities", "equity",
            "shares", "wdil_yoy"]
    j = pd.merge_asof(has, f[cols], left_on="Date", right_on="asof_date", by="cik",
                      direction="backward", allow_exact_matches=True)
    # shares as known one year earlier (fallback for net issuance)
    j["_d12"] = j["Date"] - pd.Timedelta(days=365)
    j = j.sort_values("_d12")
    f12 = f[["cik", "asof_date", "shares"]].rename(columns={"asof_date": "_a12", "shares": "shares_12m"})
    j = pd.merge_asof(j, f12, left_on="_d12", right_on="_a12", by="cik", direction="backward")

    close = j[close_col].astype(float)
    if split_factor_col:
        close = close * j[split_factor_col].astype(float)
    mcap = close * j["shares"]
    mcap = mcap.where(mcap > 0)
    A = j["assets"].where(j["assets"] > 0)
    eq = j["equity"]
    out = pd.DataFrame({"_row": j["_row"].to_numpy(), "mcap": mcap.to_numpy()})
    out["ep"] = (j["net_income_ttm"] / mcap).to_numpy()
    out["bp"] = (eq / mcap).to_numpy()
    out["sp"] = (j["revenue_ttm"] / mcap).to_numpy()
    out["cfp"] = (j["cfo_ttm"] / mcap).to_numpy()
    out["gpa"] = (j["gross_profit_ttm"] / A).to_numpy()
    out["roa"] = (j["net_income_ttm"] / A).to_numpy()
    out["roe"] = (j["net_income_ttm"] / eq.where(eq > 0)).to_numpy()
    out["accruals"] = ((j["net_income_ttm"] - j["cfo_ttm"]) / A).to_numpy()
    rp = j["revenue_ttm_prev"]
    out["sales_growth_yoy"] = (j["revenue_ttm"] / rp.where(rp > 0) - 1).to_numpy()
    out["ni_growth_scaled"] = ((j["net_income_ttm"] - j["net_income_ttm_prev"]) / A).to_numpy()
    out["leverage"] = (j["liabilities"] / A).to_numpy()
    fallback_iss = j["shares"] / j["shares_12m"].where(j["shares_12m"] > 0) - 1
    out["net_issuance_12m"] = j["wdil_yoy"].fillna(fallback_iss).to_numpy()
    out["log_mcap"] = np.log(mcap).to_numpy()
    out["fund_staleness_days"] = (j["Date"] - j["last_filed"]).dt.days.to_numpy()

    res = p[["ticker", "Date", "cik", "_row"]].merge(out, on="_row", how="left")
    if sectors is not None:
        sec = sectors[["cik", "ff12", "sector_name"]].drop_duplicates("cik").rename(columns={"sector_name": "sector"})
        sec["cik"] = sec["cik"].astype("Int64")
        res = res.merge(sec, on="cik", how="left")
    else:
        res["ff12"], res["sector"] = pd.NA, None
    res = res.sort_values("_row")
    res.index = prices.index
    return res[["ticker", "Date", "cik", "mcap"] + FEATURE_COLUMNS + ["ff12", "sector"]]


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N companies (smoke test)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_fundamentals_asof(workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
