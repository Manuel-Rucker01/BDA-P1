"""Survivorship-free, point-in-time US common-stock universe (WS1).

Pipeline (``build_universe``)
-----------------------------
1. **Candidates** from three sources, because Alpaca's *inactive* asset list is
   far from complete (TWTR, ATVI, SIVB, FRC, BBBY are absent from it although
   their SIP bars exist):

   * ``/v2/assets`` active + inactive (named; snapshot ``assets_<date>.parquet``);
   * ``datasets/nasdaq_companies.csv`` -- a ~2018 NASDAQ listing (named);
   * **symbol discovery**: weekly SIP bars for *every* 1-4 letter symbol
     2016 -> today (plus every other candidate symbol). SIP only carries
     exchange-listed securities, so anything that returns bars was listed on a
     US exchange at some point since 2016. Corporate actions (mergers,
     worthless removals) are snapshotted as supporting evidence.

2. **Classification** (``classify_name``): exchange filter (NASDAQ/NYSE/AMEX),
   then name keywords AND symbol patterns -> one of ``CLASSES`` with a reason.
   Symbols with no name (discovered only, or blank asset names) are classified
   by ``classify_unnamed``: a weekly-return correlation test against every
   *named* ETF -- a dead ETF almost always has a live twin (same index/theme)
   with |corr| >= 0.95 (or >= 0.8 with low volatility), a company does not. The rule is calibrated on named
   stocks/ETFs and the calibration is printed in the report.

3. **Symbol reuse** (``detect_symbol_reuse``): Alpaca bars are keyed by symbol,
   so a reused symbol returns one series spanning two companies, sometimes
   glued by zero-volume filler bars and with the *new* company's split factors
   applied to the old one (AMTD 2016 shows $10,893). Histories are split into
   segments at long trading gaps / price breaks; each segment gets its own
   ticker id and class (see the rule in the function docstring).

4. ``eligible``: PIT rule per (ticker, Date) using only data up to that date.

Outputs under ``<data>/universe/``: ``assets_<date>.parquet``,
``corporate_actions_<date>.parquet``, ``discovery_weekly.parquet``,
``classification.parquet``, ``segments.parquet``, ``eligible_counts.parquet``
and ``universe_report.md``.
"""
from __future__ import annotations

import datetime as dt
import glob
import itertools
import json
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ._alpaca import AlpacaClient, data_dir, last_completed_close_utc, to_ny_date

LISTED_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
HISTORY_START = "2016-01-01"
CLASSES = ("common", "etf", "fund", "trust", "warrant", "unit", "right",
           "preferred", "note", "other", "non_listed_exchange")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NASDAQ_2018_CSV = os.path.join(REPO_ROOT, "datasets", "nasdaq_companies.csv")
VALID_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")


def _today() -> str:
    return dt.date.today().isoformat()


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# =========================================================================== #
# 1. Snapshots
# =========================================================================== #
def snapshot_assets(client: AlpacaClient | None = None, date: str | None = None) -> pd.DataFrame:
    """All us_equity assets (active + inactive) with every field Alpaca returns."""
    client = client or AlpacaClient()
    frames = [pd.DataFrame(client.get_assets(s)) for s in ("active", "inactive")]
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = date or _today()
    df.to_parquet(os.path.join(data_dir("universe"), f"assets_{date or _today()}.parquet"), index=False)
    return df


CA_TYPES = ("cash_merger", "stock_merger", "stock_and_cash_merger", "name_change",
            "worthless_removal", "redemption", "forward_split", "reverse_split",
            "unit_split", "spin_off")


def snapshot_corporate_actions(client: AlpacaClient | None = None, start: str = HISTORY_START,
                               end: str | None = None, date: str | None = None) -> pd.DataFrame:
    """Non-dividend corporate actions (Alpaca coverage is sparse before ~2019)."""
    client = client or AlpacaClient()
    end = end or last_completed_close_utc().date().isoformat()
    rows = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        ys, ye = max(start, f"{y}-01-01"), min(end, f"{y}-12-31")
        for typ, recs in client.get_corporate_actions(CA_TYPES, ys, ye).items():
            for r in recs:
                rows.append({"ca_type": typ, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                                for k, v in r.items()}})
    df = pd.DataFrame(rows).drop_duplicates("id")
    df.to_parquet(os.path.join(data_dir("universe"), f"corporate_actions_{date or _today()}.parquet"),
                  index=False)
    return df


def load_assets(path: str | None = None) -> pd.DataFrame:
    path = path or _latest(os.path.join(data_dir("universe"), "assets_*.parquet"))
    if not path:
        raise FileNotFoundError("no assets snapshot; run snapshot_assets()")
    return pd.read_parquet(path)


def load_corporate_actions(path: str | None = None) -> pd.DataFrame:
    path = path or _latest(os.path.join(data_dir("universe"), "corporate_actions_*.parquet"))
    return pd.read_parquet(path) if path else pd.DataFrame(columns=["ca_type"])


# =========================================================================== #
# 2. Classification of named securities
# =========================================================================== #
def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)


# Order matters: the first matching rule wins.
_ETF_NAME = _rx(
    r"\betf\b|\betns?\b|exchange[- ]traded|\bindex (fund|shares)\b|\bishares\b|proshares|direxion|"
    r"\bspdr\b|wisdomtree|vaneck|\bglobal x\b|graniteshares|yieldmax|roundhill|leverage shares|"
    r"\btidal\b|defiance|\bpacer\b|\bvictoryshares\b|\bvanguard\b.*\b(index|shares)\b|"
    r"\binvesco\b.*\b(shares|portfolio|qqq|trust, series)\b|\bfirst trust\b.*\b(fund|etf|portfolio|index)\b|"
    r"\b(daily|2x|3x|-1x|-2x|-3x)\b.*\b(bull|bear|long|short|inverse)\b|\bultra(pro)?\s?(short)?\b.*\bproshares|"
    r"\btrust,? series 1\b|\bselect sector\b|\bself-indexed\b|\bbulletshares\b|\bmicrosectors\b"
)
# An ETF-sponsor *company* (WisdomTree, Inc.) is not an ETF.
_ETF_STRONG = _rx(r"\betf|\betn|fund|trust|shares|index|portfolio|daily|\b\dx\b|ultra|bull|bear")
_ISSUER_SUFFIX = _rx(r"\b(inc|corp|corporation|ltd|limited|plc|holdings|co)\.?,?$")
_WARRANT_NAME = _rx(r"\bwarrants?\b|\bwts?\b(?!\w)|\bwt exp\b")
_RIGHT_NAME = _rx(r"\brights?\b(?! offering)|contingent value|subscription right")
_UNIT_NAME = _rx(r"\bunits?\b")
_LP_UNIT = _rx(r"common units|limited partner|partnership units|\blp units|representing limited|unitholder|"
               r"\bclass [a-z] units$")
_PREF_NAME = _rx(
    r"\bpreferred\b|\bpreference\b|\bpfd\b|non-?cumulative|\bcumulative\b|perpetual|"
    r"fixed[- ]to[- ]floating|fixed[- ]rate reset|depositary shares?,? each representing a 1/|"
    r"\bseries [a-z]{1,2}\b.*%|%.*\bseries [a-z]{1,2}\b|\btrust preferred\b"
)
_NOTE_NAME = _rx(
    r"\bnotes?\b|\bdebentures?\b|\bsubordinated\b(?! voting)|\bbaby bonds?\b|\bdue\b[^%]*\b(19|20)\d\d\b|\d{4} notes|"
    r"%\s*(senior|junior|fixed|notes|bonds)|\bmortgage bonds?\b|\bbonds?\b.*%|%.*\bbonds?\b"
)
_FUND_SPONSORS = (
    r"blackrock|nuveen|eaton vance|gabelli|pimco|calamos|cohen & steers|western asset|invesco|aberdeen|"
    r"abrdn|franklin|john hancock|royce|tekla|virtus|saba|neuberger|doubleline|flaherty|\bmfs\b|putnam|templeton|"
    r"allspring|wells fargo|kayne anderson|tortoise|clearbridge|guggenheim|duff & phelps|pioneer|"
    r"brookfield|highland|first trust|nexpoint|bancroft|ellsworth|adams|general american|liberty all-star|"
    r"source capital|central securities|tri-continental|zweig|cornerstone|oxford lane|eagle point|"
    r"pgim|voya|federated|dws|morgan stanley|lazard|apollo|blackstone|ares dynamic|barings|angel oak|"
    r"bny mellon|dreyfus|delaware|macquarie|mainstay|miller/howard|principal|thornburg|"
    r"ishares|vanguard|schwab|sprott physical|central fund|riverNorth|alliancebe?rn?stein|columbia seligman"
)
_FUND_NAME = _rx(
    r"\bfunds?\b|closed[- ]end|\bincome trust\b|\bcredit opportunities\b|\bpremier income\b|"
    r"\bcef\b|\bmunicipal\b|\bmuni\b|\binterval\b"
)
# Sponsor + fund-like word, unless the name reads like the sponsor's own operating company.
_FUND_SPONSOR_NAME = _rx(
    rf"({_FUND_SPONSORS}).*\b(trust|income|strategies|opportunit\w*|portfolio|total return|premium|"
    r"yield|dividend|credit|bond|tax[- ]?(advantaged|free|managed)|limited duration|floating rate)\b"
)
_OPERATING_CO = _rx(r"management|partners|group|holdings|corporation|financial|asset management|"
                    r"bancorp|\bcorp\b|renewable|infrastructure corp|realty|reit\b")
_TRUST_NAME = _rx(
    r"units? of beneficial interest|royalty trust|\bgrantor\b|"
    r"\btrust\b.*\b(bitcoin|ethereum|ether|gold|silver|platinum|palladium|oil|commodit\w*|currency|"
    r"crypto|digital asset|solana|xrp|litecoin)\b|"
    r"\b(bitcoin|ethereum|ether|gold|silver|platinum|palladium|crypto|digital asset|solana|xrp)\b.*\btrust\b|"
    r"\b(energy|oil|gas|minerals?|royalty|basin|petroleum|realty royalty) trust\b|"
    r"\b(liquidating|litigation|liquidation) trust\b|\bunit investment trust\b|\bdepositary receipts? trust\b"
)
_WHEN_ISSUED = _rx(r"when[- ]issued|\bw/i\b|\bwhen issued\b|\bescrow\b|\bcontingent\b|\bstub\b")
_COMMON_HINT = _rx(r"common stock|common shares|ordinary shares?|\bclass [a-c]\b|capital stock|"
                   r"american depositary|\bads\b|\badr\b|shares of beneficial interest|common units")
_SPAC_NAME = _rx(r"\bacquisition (corp|corporation|company|co|limited|ltd|inc|holdings)\b|"
                 r"\bblank check\b|\bspac\b|\bmerger (corp|corporation|sub)\b|"
                 r"\b(acquisition|merger) corp\b")


def classify_symbol(symbol: str, exchange: str | None = None) -> tuple[str | None, str]:
    """Symbol-pattern rules. Returns (class or None if inconclusive, reason)."""
    s = (symbol or "").upper().replace("_DELISTED", "")
    if not s:
        return "other", "sym:empty"
    if re.search(r"\d", s) or not re.fullmatch(r"[A-Z.\-/ ]+", s):
        return "other", "sym:cusip_or_nonstandard"
    if re.search(r"\.WS(\.[A-Z])?$|\.W$|/WS|\.WT$", s):
        return "warrant", "sym:.WS"
    if re.search(r"\.U$|\.UN$|/U$", s):
        return "unit", "sym:.U"
    if re.search(r"\.RT$|\.R$|\.RTW?I?$", s):
        return "right", "sym:.RT"
    if re.search(r"\.PR[A-Z]?$|-P[A-Z]?$|\.P[A-Z]$|/PR", s):
        return "preferred", "sym:.PR/-P"
    if re.search(r"\.(WI|WD|CL|CV|EC|PT|ID|NT)$", s):
        return "other", "sym:when_issued_or_special"
    if (exchange in (None, "NASDAQ")) and re.fullmatch(r"[A-Z]{5}", s):
        last = s[-1]
        if last == "W":
            return "warrant", "sym:nasdaq_5th_W"
        if last == "U":
            return "unit", "sym:nasdaq_5th_U"
        if last == "R":
            return "right", "sym:nasdaq_5th_R"
        if last == "X":
            return "fund", "sym:5th_X_mutual_fund"
        if last in "PONM":
            return None, f"sym:nasdaq_5th_{last}_preferred_if_not_common"
        if last in "GHIL":
            return None, f"sym:nasdaq_5th_{last}_note_if_not_common"
        if last == "Z":
            return None, "sym:nasdaq_5th_Z_misc"
    return None, "sym:plain"


def classify_name(symbol: str, name: str | None, exchange: str | None) -> tuple[str, str]:
    """Classify one security. Returns (class, reason).

    Rules, first match wins:
      1. exchange not NASDAQ/NYSE/AMEX            -> non_listed_exchange
      2. symbol pattern decisive (.WS/.U/.RT/.PR, NASDAQ 5th letter W/U/R/X,
         CUSIP-like)                               -> warrant/unit/right/preferred/fund/other
      3. name keywords: ETF/ETN, warrant, right, unit (not LP common units),
         preferred, note/debenture, closed-end fund, trust (royalty/grantor/
         commodity/crypto/liquidating), when-issued/escrow
      4. NASDAQ 5th letter P/O/N/M (preferred) or G/H/I/L (notes) unless the
         name says common/class/ordinary (e.g. GOOGL)
      5. otherwise common (ADRs and MLP common units stay common)
    """
    if exchange is not None and exchange not in LISTED_EXCHANGES:
        return "non_listed_exchange", f"exchange:{exchange}"
    sym_cls, sym_reason = classify_symbol(symbol, exchange)
    if sym_cls is not None:
        return sym_cls, sym_reason
    n = name.strip() if isinstance(name, str) else ""
    if not n:
        return "common", f"no_name;{sym_reason}"
    if _ETF_NAME.search(n) and (_ETF_STRONG.search(n) or not _ISSUER_SUFFIX.search(n)):
        return "etf", "name:etf/etn"
    if (_PREF_NAME.search(n) and not re.match(r"preferred\b", n, re.I)
            and not (re.search(r"common stock|ordinary shares", n, re.I) and "%" not in n)):
        return "preferred", "name:preferred"
    if _WARRANT_NAME.search(n):
        return "warrant", "name:warrant"
    if _UNIT_NAME.search(n) and not _LP_UNIT.search(n):
        return "unit", "name:unit"
    if _RIGHT_NAME.search(n) and not re.search(r"depositary|depository", n, re.I):
        return "right", "name:right"
    if _NOTE_NAME.search(n):
        return "note", "name:note/debenture"
    if _TRUST_NAME.search(n):
        return "trust", "name:trust"
    if _FUND_NAME.search(n) or (_FUND_SPONSOR_NAME.search(n) and not _OPERATING_CO.search(n)):
        return "fund", "name:fund/closed-end"
    if _WHEN_ISSUED.search(n):
        return "other", "name:when_issued/escrow/contingent"
    if sym_reason.startswith("sym:nasdaq_5th_") and not _COMMON_HINT.search(n):
        letter = sym_reason.split("_")[2]
        return ("preferred" if letter in "PONM" else "note" if letter in "GHIL" else "other"), sym_reason
    return "common", ("name:common_hint" if _COMMON_HINT.search(n) else "name:default_common")


def is_spac_like(name: str | None) -> bool:
    return isinstance(name, str) and bool(_SPAC_NAME.search(name))


def classify_assets(assets: pd.DataFrame) -> pd.DataFrame:
    """Vectorised ``classify_name`` over an assets snapshot."""
    out = assets[["id", "symbol", "name", "exchange", "status"]].copy()
    res = [classify_name(s, n, e) for s, n, e in zip(out["symbol"], out["name"], out["exchange"])]
    out["cls"] = [r[0] for r in res]
    out["reason"] = [r[1] for r in res]
    out["spac_like"] = out["name"].map(is_spac_like)
    out["fetch_symbol"] = out["symbol"].str.replace("_DELISTED", "", regex=False)
    out["source"] = "alpaca_assets"
    return out


def load_nasdaq_2018(path: str = NASDAQ_2018_CSV) -> pd.DataFrame:
    """~2018 NASDAQ company list shipped in ``datasets/`` (named, pre-2020 coverage)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol", "name"])
    df = pd.read_csv(path, usecols=["Symbol", "Name"]).rename(columns={"Symbol": "symbol", "Name": "name"})
    df["symbol"] = (df["symbol"].astype(str).str.strip()
                    .str.replace("^", ".PR", regex=False).str.replace("~", "", regex=False))
    return df.drop_duplicates("symbol")


# =========================================================================== #
# 3. Symbol discovery (weekly SIP bars for every 1-4 letter symbol)
# =========================================================================== #
def all_short_symbols(max_len: int = 4) -> list[str]:
    out = []
    for n in range(1, max_len + 1):
        out.extend("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=n))
    return out


def _weekly_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if not len(raw):
        return pd.DataFrame(columns=["symbol", "Date", "close", "volume"])
    return pd.DataFrame({
        "symbol": raw["symbol"].astype(str),
        "Date": to_ny_date(raw["timestamp"]),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })


def discover_symbols(client: AlpacaClient | None = None, extra_symbols=(), batch_size: int = 4000,
                     workers: int = 3, start: str = HISTORY_START, end: dt.datetime | None = None,
                     log=print) -> pd.DataFrame:
    """Weekly (split+dividend adjusted) SIP bars for every 1-4 letter symbol plus
    ``extra_symbols``. Resumable: each batch is checkpointed to
    ``universe/_discovery/b<k>.parquet`` and skipped on re-run."""
    client = client or AlpacaClient()
    ddir = data_dir("universe", "_discovery")
    man_path = os.path.join(ddir, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
    else:
        syms = all_short_symbols()
        extra = sorted({s for s in extra_symbols if isinstance(s, str) and VALID_SYMBOL.match(s)} - set(syms))
        syms = syms + extra
        end = end or last_completed_close_utc()
        man = {"start": start, "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "batches": [syms[i:i + batch_size] for i in range(0, len(syms), batch_size)]}
        json.dump(man, open(man_path, "w"))
    todo = [k for k in range(len(man["batches"]))
            if not os.path.exists(os.path.join(ddir, f"b{k:05d}.parquet"))]
    log(f"[discover] {len(man['batches'])} batches, {len(todo)} to do")
    t0 = time.time()

    def run(k):
        raw, invalid = client.get_bars(man["batches"][k], man["start"], man["end"], timeframe="1Week",
                                       adjustment="all")
        df = _weekly_frame(raw)
        tmp = os.path.join(ddir, f"b{k:05d}.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, os.path.join(ddir, f"b{k:05d}.parquet"))
        return k, df["symbol"].nunique(), invalid

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run, k) for k in todo]):
            k, n, invalid = fut.result()
            done += 1
            log(f"[discover] batch {k} -> {n} symbols{' invalid=' + str(invalid) if invalid else ''} "
                f"({done}/{len(todo)}, {time.time() - t0:.0f}s, {client.n_requests} requests)")
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ddir, "b*.parquet")))]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "Date"])
    out.to_parquet(os.path.join(data_dir("universe"), "discovery_weekly.parquet"), index=False)
    log(f"[discover] {out['symbol'].nunique()} symbols with SIP bars since {man['start']}")
    return out


CA_SYMBOL_COLS = ("symbol", "acquiree_symbol", "acquirer_symbol", "old_symbol", "new_symbol",
                  "source_symbol", "alternate_symbol")


def candidate_extra_symbols(assets: pd.DataFrame, ca: pd.DataFrame, nasdaq18: pd.DataFrame) -> list[str]:
    """Every non-1-4-letter symbol worth probing: listed-exchange assets, the
    2018 NASDAQ list and all symbols named in corporate actions."""
    syms = set(assets.loc[assets["exchange"].isin(LISTED_EXCHANGES), "symbol"]
               .str.replace("_DELISTED", "", regex=False))
    syms |= set(nasdaq18["symbol"])
    for c in CA_SYMBOL_COLS:
        if c in ca.columns:
            syms |= set(ca[c].dropna().astype(str))
    return sorted(s for s in syms if VALID_SYMBOL.match(s))


def load_discovery() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir("universe"), "discovery_weekly.parquet"))


# =========================================================================== #
# 4. Unnamed symbols: ETF-likeness by weekly-return correlation
# =========================================================================== #
ETF_CORR_THRESHOLD = 0.95      # |corr| with a named ETF -> ETF-like ...
ETF_CORR_LOW_VOL = 0.80        # ... or this |corr| together with
ETF_MAX_ANN_VOL = 0.20         # annualised weekly-return vol below 20%
MIN_OVERLAP_WEEKS = 26


def etf_like(corr, ann_vol, hi: float = ETF_CORR_THRESHOLD, lo: float = ETF_CORR_LOW_VOL,
             vmax: float = ETF_MAX_ANN_VOL):
    """The ETF-likeness rule (vectorised); NaN corr -> False (untested)."""
    corr, ann_vol = np.asarray(corr, float), np.asarray(ann_vol, float)
    with np.errstate(invalid="ignore"):
        return (corr >= hi) | ((corr >= lo) & (ann_vol < vmax))


def weekly_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    """Date x symbol weekly log returns; |r| > 1 (split glitches) -> NaN."""
    w = weekly[weekly["close"] > 0].pivot_table(index="Date", columns="symbol", values="close", aggfunc="last")
    r = np.log(w).diff()
    return r.where(r.abs() <= 1.0)


def _centered(r: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Zero-filled, own-mean-centred values (numerical stability) and the mask."""
    x = r.to_numpy(dtype="float64")
    m = np.isfinite(x)
    mu = np.nansum(np.where(m, x, 0.0), 0) / np.maximum(m.sum(0), 1)
    return np.where(m, x - mu, 0.0), m.astype("float64")


def _pairwise_corr(xt, mt, xr, mr):
    """Exact pairwise-complete Pearson correlation between all columns of two
    zero-filled matrices with masks (sums restricted to the common weeks)."""
    n = mt.T @ mr
    sx, sy = xt.T @ mr, mt.T @ xr
    sxx, syy = (xt ** 2).T @ mr, mt.T @ (xr ** 2)
    sxy = xt.T @ xr
    num = n * sxy - sx * sy
    den = np.sqrt(np.clip(n * sxx - sx ** 2, 0, None) * np.clip(n * syy - sy ** 2, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, num / den, np.nan)
    return np.clip(c, -1, 1), n


def max_abs_corr(target: pd.DataFrame, refs: pd.DataFrame, min_overlap: int = MIN_OVERLAP_WEEKS,
                 exclude_self: bool = False, block: int = 2000) -> pd.DataFrame:
    """For each column of ``target``: max |corr| with any column of ``refs`` over
    >= ``min_overlap`` common weeks (pairwise, approximated with per-series
    pairwise-complete Pearson), plus the best-matching reference symbol."""
    refs = refs.reindex(target.index)
    xr, mr = _centered(refs)
    out_c, out_r, out_n = [], [], []
    cols = list(target.columns)
    for i in range(0, len(cols), block):
        xt, mt = _centered(target[cols[i:i + block]])
        c, n = _pairwise_corr(xt, mt, xr, mr)
        c = np.where(n >= min_overlap, np.abs(c), np.nan)
        if exclude_self:
            idx = {s: k for k, s in enumerate(refs.columns)}
            for k, s in enumerate(cols[i:i + block]):
                if s in idx:
                    c[k, idx[s]] = np.nan
        allnan = np.all(np.isnan(c), 1)
        j = np.nanargmax(np.where(np.isnan(c), -1, c), 1)
        out_c.extend(np.where(allnan, np.nan, c[np.arange(len(j)), j]))
        out_r.extend(np.where(allnan, None, np.asarray(refs.columns)[j]))
        out_n.extend((mt.sum(0)).astype(int))
    return pd.DataFrame({"symbol": cols, "etf_corr": out_c, "etf_match": out_r, "n_weeks": out_n})


def etf_reference_set(ret: pd.DataFrame, named: pd.DataFrame, tracker_corr: float = 0.97) -> tuple[list, list]:
    """Named ETFs (any exchange) usable as references, minus single-stock
    trackers (|corr| >= ``tracker_corr`` with a named common stock), and the named
    listed common stocks used for calibration."""
    etfs = sorted(set(named.loc[named["name_cls"] == "etf", "fetch_symbol"]) & set(ret.columns))
    commons = sorted(set(named.loc[(named["cls"] == "common") & named["name_verified"], "fetch_symbol"])
                     & set(ret.columns))
    trk = max_abs_corr(ret[etfs], ret[commons])
    trackers = set(trk.loc[trk["etf_corr"] >= tracker_corr, "symbol"])
    return [e for e in etfs if e not in trackers], commons


def calibrate_etf_test(ret: pd.DataFrame, refs: list, commons: list, etfs: list,
                       rules=((0.90, 0, 0), (0.93, 0, 0), (0.95, 0, 0), (0.97, 0, 0),
                              (0.95, 0.80, 0.20), (0.95, 0.70, 0.20), (0.93, 0.80, 0.25))) -> pd.DataFrame:
    """Share of named commons / named ETFs flagged ETF-like by each candidate
    rule (hi corr, low corr, max annual vol); self excluded from references.
    ``*_tested`` = among series with >= 26 overlapping weeks."""
    vol = ret.std() * np.sqrt(52)
    sc = max_abs_corr(ret[commons], ret[refs], exclude_self=True)
    se = max_abs_corr(ret[etfs], ret[refs], exclude_self=True)
    rows = []
    for hi, lo, vmax in rules:
        fc = etf_like(sc["etf_corr"], vol.reindex(sc["symbol"]).values, hi, lo, vmax)
        fe = etf_like(se["etf_corr"], vol.reindex(se["symbol"]).values, hi, lo, vmax)
        te = se["etf_corr"].notna().to_numpy()
        rows.append({"rule": f"corr>={hi}" + (f" or (corr>={lo} and vol<{vmax:.0%})" if lo else ""),
                     "used": (hi, lo, vmax) == (ETF_CORR_THRESHOLD, ETF_CORR_LOW_VOL, ETF_MAX_ANN_VOL),
                     "commons_flagged_pct": round(100 * fc.mean(), 2),
                     "etfs_flagged_pct": round(100 * fe.mean(), 2),
                     "etfs_tested_flagged_pct": round(100 * fe[te].mean(), 2),
                     "n_commons": len(sc), "n_etfs": len(se), "n_etfs_tested": int(te.sum())})
    return pd.DataFrame(rows)


def corporate_action_evidence(ca: pd.DataFrame, named_cls: dict) -> pd.DataFrame:
    """Per symbol: was it an acquiree of a cash merger / of a stock merger into
    a fund or ETF, or removed as worthless (bankruptcy)?"""
    rows = []
    for _, r in ca.iterrows():
        t = r.get("ca_type")
        if t in ("cash_mergers", "stock_mergers", "stock_and_cash_mergers") and isinstance(r.get("acquiree_symbol"), str):
            acq_cls = named_cls.get(r.get("acquirer_symbol"))
            ev = "merged_into_fund" if acq_cls in ("etf", "fund", "trust") else "acquired_company"
            rows.append((r["acquiree_symbol"], ev, r.get("effective_date") or r.get("process_date")))
        elif t == "worthless_removals" and isinstance(r.get("symbol"), str):
            rows.append((r["symbol"], "worthless_removal", r.get("process_date")))
    df = pd.DataFrame(rows, columns=["symbol", "ca_evidence", "ca_date"])
    pri = {"merged_into_fund": 0, "acquired_company": 1, "worthless_removal": 2}
    return (df.assign(p=df["ca_evidence"].map(pri)).sort_values(["symbol", "p"])
              .drop_duplicates("symbol").drop(columns="p"))


# =========================================================================== #
# 5. Classification table
# =========================================================================== #
def build_classification(assets: pd.DataFrame | None = None, ca: pd.DataFrame | None = None,
                         weekly: pd.DataFrame | None = None, nasdaq18: pd.DataFrame | None = None,
                         write: bool = True, log=print):
    """One row per candidate (asset row, 2018-list row or discovered symbol).

    Columns: id, symbol, fetch_symbol, name, exchange, status, source, cls,
    reason, name_verified, spac_like, etf_corr, etf_match, ca_evidence,
    first_week, last_week, n_weeks, in_universe.
    Returns (classification, calibration_table).
    """
    assets = assets if assets is not None else load_assets()
    ca = ca if ca is not None else load_corporate_actions()
    weekly = weekly if weekly is not None else load_discovery()
    nasdaq18 = nasdaq18 if nasdaq18 is not None else load_nasdaq_2018()

    a = classify_assets(assets)
    a["name_cls"] = [classify_name(s, n, None)[0] if isinstance(n, str) and n.strip() else None
                     for s, n in zip(a["symbol"], a["name"])]
    a["name_verified"] = a["name"].fillna("").str.strip() != ""

    known = set(a["fetch_symbol"])
    n18 = nasdaq18[~nasdaq18["symbol"].isin(known)].copy()
    res = [classify_name(s, n, "NASDAQ") for s, n in zip(n18["symbol"], n18["name"])]
    n18 = pd.DataFrame({"id": None, "symbol": n18["symbol"], "name": n18["name"], "exchange": "NASDAQ",
                        "status": "unknown", "cls": [r[0] for r in res], "reason": [r[1] for r in res],
                        "spac_like": n18["name"].map(is_spac_like), "fetch_symbol": n18["symbol"],
                        "source": "nasdaq_2018", "name_cls": [r[0] for r in res], "name_verified": True})
    known |= set(n18["fetch_symbol"])

    wk = weekly.groupby("symbol").agg(first_week=("Date", "min"), last_week=("Date", "max"),
                                      n_weeks=("Date", "size")).reset_index()
    disc = wk[~wk["symbol"].isin(known)]
    d = pd.DataFrame({"id": None, "symbol": disc["symbol"], "name": None, "exchange": None,
                      "status": "unknown", "cls": None, "reason": None, "spac_like": False,
                      "fetch_symbol": disc["symbol"], "source": "discovery", "name_cls": None,
                      "name_verified": False})
    allc = pd.concat([a, n18, d], ignore_index=True)

    # Unnamed rows on listed exchanges or of unknown exchange -> corr test.
    unnamed = (~allc["name_verified"]) & (allc["exchange"].isna() | allc["exchange"].isin(LISTED_EXCHANGES))
    ret = weekly_returns(weekly)
    refs, commons = etf_reference_set(ret, allc[allc["name_verified"]])
    named_etfs = sorted(set(allc.loc[allc["name_cls"] == "etf", "fetch_symbol"]) & set(ret.columns))
    calib = calibrate_etf_test(ret, refs, commons, named_etfs)
    log(f"[universe] {len(refs)} reference ETFs; calibration:\n{calib.to_string(index=False)}")
    targets = sorted(set(allc.loc[unnamed, "fetch_symbol"]) & set(ret.columns))
    sc = max_abs_corr(ret[targets], ret[refs]).set_index("symbol") if targets else pd.DataFrame(
        columns=["etf_corr", "etf_match", "n_weeks"])

    named_cls = dict(zip(allc.loc[allc["name_verified"], "fetch_symbol"], allc.loc[allc["name_verified"], "cls"]))
    ev = corporate_action_evidence(ca, named_cls).set_index("symbol")
    allc["ca_evidence"] = allc["fetch_symbol"].map(ev["ca_evidence"])
    allc["etf_corr"] = allc["fetch_symbol"].map(sc["etf_corr"]) if len(sc) else np.nan
    allc["etf_match"] = allc["fetch_symbol"].map(sc["etf_match"]) if len(sc) else None
    allc["ann_vol"] = allc["fetch_symbol"].map(ret.std() * np.sqrt(52))

    for i in allc.index[unnamed]:
        sym, ex = allc.at[i, "fetch_symbol"], allc.at[i, "exchange"]
        scls, sreason = classify_symbol(sym, ex)
        corr, evid = allc.at[i, "etf_corr"], allc.at[i, "ca_evidence"]
        if scls is not None:
            c, r = scls, f"unnamed;{sreason}"
        elif sreason.startswith("sym:nasdaq_5th_"):
            c, r = "other", f"unnamed;{sreason}"  # 5-letter, no name: cannot tell
        elif evid == "merged_into_fund":
            c, r = "fund", "unnamed;ca:merged_into_fund"
        elif (pd.notna(corr) and corr >= ETF_CORR_THRESHOLD) or (
                etf_like(corr, allc.at[i, "ann_vol"]) and evid not in ("acquired_company", "worthless_removal")):
            c, r = "etf", (f"unnamed;etf_like corr={corr:.3f} vs {allc.at[i, 'etf_match']} "
                           f"vol={allc.at[i, 'ann_vol']:.0%}")
        elif pd.isna(corr):
            c, r = "common", "unnamed;short_history_untested" + (f";ca:{evid}" if isinstance(evid, str) else "")
        else:
            c, r = "common", f"unnamed;not_etf_like corr={corr:.3f} vol={allc.at[i, 'ann_vol']:.0%}" + (f";ca:{evid}" if isinstance(evid, str) else "")
        allc.at[i, "cls"], allc.at[i, "reason"] = c, r

    allc = allc.merge(wk.rename(columns={"symbol": "fetch_symbol"}), on="fetch_symbol", how="left")
    allc["has_bars"] = allc["n_weeks"].fillna(0) > 0
    # Delisted to the OTC market: Alpaca now reports exchange=OTC, but SIP bars
    # (exchange-listed securities only) exist for the listed period.
    otc = (allc["exchange"] == "OTC") & (allc["name_cls"] == "common") & allc["has_bars"]
    allc.loc[otc, "cls"] = "common"
    allc.loc[otc, "reason"] = "exchange_now_OTC;sip_bars_while_listed"
    allc["in_universe"] = (allc["cls"] == "common") & allc["has_bars"]
    cols = ["id", "symbol", "fetch_symbol", "name", "exchange", "status", "source", "cls", "reason",
            "name_verified", "spac_like", "etf_corr", "etf_match", "ann_vol", "ca_evidence", "first_week",
            "last_week", "n_weeks", "has_bars", "in_universe"]
    allc = allc[cols]
    if write:
        allc.to_parquet(os.path.join(data_dir("universe"), "classification.parquet"), index=False)
        calib.to_parquet(os.path.join(data_dir("universe"), "etf_test_calibration.parquet"), index=False)
    return allc, calib


def load_classification() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(data_dir("universe"), "classification.parquet"))


# =========================================================================== #
# 6. Symbol reuse
# =========================================================================== #
REUSE_GAP_DAYS = 90          # calendar days without a traded bar -> new listing
REUSE_GAP_DAYS_MULTI = 30    # ... when Alpaca knows >1 security under the symbol
REUSE_GAP_DAYS_MULTI_JUMP = 7  # ... or > 7 days if the price level also changed > 2x
GAP_JUMP_LOG = np.log(2)
PRICE_BREAK_LOG = np.log(3)  # one-day |log return| treated as a company switch (multi only)
CURRENT_DAYS = 10            # segment ending within N days of the store end is "current"


def detect_symbol_reuse(prices: pd.DataFrame, classification: pd.DataFrame | None = None,
                        write: bool = True) -> pd.DataFrame:
    """Split each symbol's history into listings ("segments").

    Rule (documented in the report):
      * only traded bars count (volume > 0): Alpaca glues reused symbols with
        zero-volume filler bars;
      * a new segment starts after a gap > 90 calendar days between traded
        bars; when the symbol is *multi* -- Alpaca lists more than one security
        under it (active+inactive duplicates or ``SYM_DELISTED``) -- after a gap
        > 30 days, or > 7 days if the price level also changed by > 2x;
      * a multi symbol with a single segment is split at its largest one-day
        move if |log return| > ln 3 (a different company's price level);
      * segment ids: the latest segment keeps the symbol, earlier ones become
        ``SYM~1``, ``SYM~2``... (oldest first);
      * segment class: the current segment (ending within 10 days of the store
        end) takes the class of the symbol's *active* security; earlier or
        non-current segments take the class shared by its other securities --
        if those disagree the segment is ``ambiguous`` and excluded;
      * kept = class common and >= 5 traded bars.
    ``prices`` needs ticker, Date, company_close, company_volume.
    """
    classification = classification if classification is not None else load_classification()
    cl = classification[classification["fetch_symbol"].notna()]
    rows_by_sym = {s: g for s, g in cl.groupby("fetch_symbol")}
    multi_syms = {s for s, g in rows_by_sym.items()
                  if (g["source"] == "alpaca_assets").sum() > 1 or g["symbol"].str.endswith("_DELISTED").any()}
    px = prices.loc[prices["company_volume"] > 0, ["ticker", "Date", "company_close"]] \
        .sort_values(["ticker", "Date"])
    store_end = pd.Timestamp(px["Date"].max())
    out = []
    for sym, g in px.groupby("ticker", sort=False):
        d = pd.to_datetime(g["Date"].to_numpy())
        c = g["company_close"].to_numpy(dtype=float)
        multi = sym in multi_syms
        gap = np.r_[0, np.diff(d.values).astype("timedelta64[D]").astype(int)]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = np.abs(np.r_[0, np.diff(np.log(np.where(c > 0, c, np.nan)))])
        if multi:
            cut = (gap > REUSE_GAP_DAYS_MULTI) | ((gap > REUSE_GAP_DAYS_MULTI_JUMP) & (jump > GAP_JUMP_LOG))
        else:
            cut = gap > REUSE_GAP_DAYS
        why = np.where(cut, "gap", "").astype(object)
        if multi and not cut.any() and len(c) > 1:
            lr = np.abs(np.diff(np.log(np.where(c > 0, c, np.nan))))
            if np.nanmax(np.r_[lr, 0]) > PRICE_BREAK_LOG:
                k = int(np.nanargmax(lr)) + 1
                cut[k], why[k] = True, "price_break"
        seg_no = np.cumsum(cut)
        n_seg = seg_no.max() + 1
        g_rows = rows_by_sym.get(sym)
        for s in range(n_seg):
            m = seg_no == s
            start, end = d[m][0], d[m][-1]
            current = (store_end - end).days <= CURRENT_DAYS
            cls, cls_reason = _segment_class(g_rows, current, s == n_seg - 1)
            ticker = sym if s == n_seg - 1 else f"{sym}~{s + 1}"
            out.append({"symbol": sym, "seg": s + 1, "n_seg": n_seg, "ticker": ticker,
                        "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
                        "n_bars": int(m.sum()), "gap_before_days": int(gap[m][0]) if s else 0,
                        "split_reason": (why[m][0] or None) if s else None, "multi_security": multi,
                        "current": current, "cls": cls, "cls_reason": cls_reason})
    seg = pd.DataFrame(out)
    seg["keep"] = (seg["cls"] == "common") & (seg["n_bars"] >= 5)
    if write:
        seg.to_parquet(os.path.join(data_dir("universe"), "segments.parquet"), index=False)
        _SEG_CACHE.clear()
    return seg


def _segment_class(rows: pd.DataFrame | None, current: bool, is_last: bool) -> tuple[str, str]:
    if rows is None or not len(rows):
        return "unknown", "no_classification_row"
    if len(rows) == 1:
        return rows["cls"].iloc[0], "single_security"
    active = rows[rows["status"] == "active"]
    others = rows[rows["status"] != "active"]
    if current and is_last and len(active):
        cls = active["cls"].unique()
        return (cls[0], "current->active_security") if len(cls) == 1 else ("ambiguous", "active_disagree")
    pool = others if len(others) else rows
    cls = pool["cls"].unique()
    if len(cls) == 1:
        return cls[0], "past->inactive_securities"
    return "ambiguous", "past_securities_disagree:" + "/".join(sorted(map(str, cls)))


DUP_MIN_SHARED = 20     # identical traded bars needed to call two symbols the same security
DUP_ALIAS_FRAC = 0.9    # shared / bars of the shorter series above which it is a pure alias


def detect_duplicates(prices: pd.DataFrame, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symbols carrying the *same* security's bars.

    Alpaca re-keys a renamed company's history to its new symbol, but the old
    symbol -- when nobody reused it -- still returns the same bars (SQ/XYZ,
    SIVB/SIVBQ). Within one request the API returns it once; across requests
    both come back. Two symbols sharing >= 20 traded bars with identical
    (Date, close, volume, trade_count) are the same security on those dates.
    The canonical one has more bars (tie: later last date, then name); the
    other is an ``alias`` (dropped entirely) when the shared bars cover > 90%
    of it, otherwise only the shared rows are dropped (``rows``).
    Returns (pairs, rows_to_drop[ticker, Date])."""
    key = ["Date", "company_close", "company_volume", "trade_count"]
    d = prices.loc[prices["company_volume"] > 0, ["ticker"] + key]
    sh = d[d.groupby(key)["ticker"].transform("size") > 1]
    pr = sh.merge(sh, on=key)
    pr = pr[pr["ticker_x"] < pr["ticker_y"]]
    cnt = pr.groupby(["ticker_x", "ticker_y"]).size().rename("shared").reset_index()
    cnt = cnt[cnt["shared"] >= DUP_MIN_SHARED]
    n = d.groupby("ticker").size()
    last = d.groupby("ticker")["Date"].max()
    rows = []
    for _, r in cnt.sort_values("shared", ascending=False).iterrows():
        x, y = r["ticker_x"], r["ticker_y"]
        kx, ky = (n[x], last[x], x), (n[y], last[y], y)
        keep, drop = (x, y) if kx > ky else (y, x)
        frac = r["shared"] / min(n[x], n[y])
        rows.append({"canonical": keep, "alias": drop, "shared": int(r["shared"]), "n_canonical": int(n[keep]),
                     "n_alias": int(n[drop]), "frac": float(frac),
                     "mode": "alias" if r["shared"] / n[drop] > DUP_ALIAS_FRAC else "rows"})
    pairs = pd.DataFrame(rows, columns=["canonical", "alias", "shared", "n_canonical", "n_alias", "frac", "mode"])
    rp = pairs[pairs["mode"] == "rows"]
    if len(rp):
        m = pr.merge(rp[["canonical", "alias"]], left_on=["ticker_x", "ticker_y"], right_on=["canonical", "alias"],
                     how="inner")
        m2 = pr.merge(rp[["canonical", "alias"]], left_on=["ticker_y", "ticker_x"], right_on=["canonical", "alias"],
                      how="inner")
        drop_rows = pd.concat([m[["alias", "Date"]], m2[["alias", "Date"]]]).rename(columns={"alias": "ticker"})
    else:
        drop_rows = pd.DataFrame(columns=["ticker", "Date"])
    drop_rows = drop_rows.drop_duplicates()
    if write:
        pairs.to_parquet(os.path.join(data_dir("universe"), "duplicates.parquet"), index=False)
        drop_rows.to_parquet(os.path.join(data_dir("universe"), "duplicate_rows.parquet"), index=False)
        _SEG_CACHE.clear()
    return pairs, drop_rows


def apply_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop alias symbols and duplicated rows (see ``detect_duplicates``)."""
    ud = data_dir("universe")
    p = os.path.join(ud, "duplicates.parquet")
    if not os.path.exists(p) or not len(df):
        return df
    pairs = pd.read_parquet(p)
    df = df[~df["ticker"].isin(set(pairs.loc[pairs["mode"] == "alias", "alias"]))]
    rp = os.path.join(ud, "duplicate_rows.parquet")
    if os.path.exists(rp):
        dr = pd.read_parquet(rp)
        if len(dr):
            m = df[["ticker", "Date"]].merge(dr.assign(_d=1), on=["ticker", "Date"], how="left")["_d"]
            df = df[m.isna().to_numpy()]
    return df


_SEG_CACHE: dict = {}


def load_segments() -> pd.DataFrame | None:
    p = os.path.join(data_dir("universe"), "segments.parquet")
    if not os.path.exists(p):
        return None
    mt = os.path.getmtime(p)
    if _SEG_CACHE.get("mtime") != mt:
        _SEG_CACHE.update(mtime=mt, df=pd.read_parquet(p))
    return _SEG_CACHE["df"]


def apply_segments(df: pd.DataFrame, segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map store rows (ticker = symbol) to segment ids and drop excluded
    segments / bars between segments. Symbols without segment info pass."""
    seg = segments if segments is not None else load_segments()
    if seg is None or not len(df):
        return df
    simple = seg.groupby("symbol").filter(lambda g: len(g) == 1 and g["keep"].all())["symbol"]
    special = seg[~seg["symbol"].isin(set(simple))]
    mask = df["ticker"].isin(set(special["symbol"]))
    if not mask.any():
        return df
    part = df[mask].merge(special[["symbol", "ticker", "start", "end", "keep"]]
                          .rename(columns={"ticker": "seg_ticker", "symbol": "ticker"}), on="ticker")
    part = part[(part["Date"] >= part["start"]) & (part["Date"] <= part["end"]) & part["keep"]]
    part = part.assign(ticker=part["seg_ticker"]).drop(columns=["seg_ticker", "start", "end", "keep"])
    return pd.concat([df[~mask], part], ignore_index=True)


# =========================================================================== #
# 7. Point-in-time eligibility
# =========================================================================== #
def eligibility_details(panel: pd.DataFrame, min_price: float = 3.0, min_adv_usd: float = 2e6,
                        min_bars: int = 120, adv_window: int = 20) -> pd.DataFrame:
    """Per (ticker, Date): n_bars, price, adv_usd, eligible -- each computed from
    rows up to and including that date only (rolling/cumulative per ticker),
    aligned to ``panel``'s index.

    * price = ``close_raw`` (actual printed close) when present, else
      ``company_close``. Adjusted closes are *not* PIT: a later 10:1 split
      makes 2016 prices look 10x smaller.
    * adv_usd = mean of company_close x company_volume over the last
      ``adv_window`` sessions (split-invariant), NaN until a full window exists.
    * n_bars = number of bars of the ticker so far (history starts 2016-01-01).
    """
    p = panel[["ticker", "Date"]].copy()
    p["_c"] = panel["company_close"].astype(float)
    p["_dv"] = p["_c"] * panel["company_volume"].astype(float)
    p["price"] = panel["close_raw"].astype(float).fillna(p["_c"]) if "close_raw" in panel else p["_c"]
    order = p.sort_values(["ticker", "Date"], kind="mergesort").index
    q = p.loc[order]
    g = q.groupby("ticker", sort=False)
    q["n_bars"] = g.cumcount() + 1
    q["adv_usd"] = g["_dv"].transform(lambda s: s.rolling(adv_window, min_periods=adv_window).mean())
    q["eligible"] = ((q["price"] >= min_price) & (q["adv_usd"] >= min_adv_usd)
                     & (q["n_bars"] >= min_bars)).fillna(False)
    return q.loc[panel.index, ["ticker", "Date", "n_bars", "price", "adv_usd", "eligible"]]


def eligible(panel: pd.DataFrame, min_price: float = 3.0, min_adv_usd: float = 2e6,
             min_bars: int = 120) -> pd.Series:
    """Boolean Series aligned to ``panel``: is (ticker, Date) investable using
    only information available at that date's close? Same rule for training
    rows and live scoring. See ``eligibility_details``."""
    return eligibility_details(panel, min_price, min_adv_usd, min_bars)["eligible"].astype(bool)


def eligible_counts(write: bool = True, **rule) -> pd.DataFrame:
    """Universe size per session under ``eligible`` on the cleaned store."""
    from .prices import load_prices
    px = load_prices(columns=["company_close", "company_volume"], with_raw=True)
    det = eligibility_details(px, **rule)
    out = (det.groupby("Date").agg(n_with_bar=("ticker", "size"), n_eligible=("eligible", "sum"))
              .reset_index())
    if write:
        out.to_parquet(os.path.join(data_dir("universe"), "eligible_counts.parquet"), index=False)
        yr = det[det["eligible"]].groupby(det["Date"].str[:4])["ticker"].nunique()
        yr.rename("n_unique_eligible").to_frame().to_parquet(
            os.path.join(data_dir("universe"), "eligible_unique_per_year.parquet"))
    return out


# =========================================================================== #
# 8. Report
# =========================================================================== #
def write_report(sample_n: int = 8, seed: int = 0) -> str:
    from ._alpaca import md_table
    cl = load_classification()
    ud = data_dir("universe")
    calib = pd.read_parquet(os.path.join(ud, "etf_test_calibration.parquet"))
    seg = load_segments()
    rng = np.random.default_rng(seed)
    listed = cl[(cl["exchange"].isin(LISTED_EXCHANGES)) | cl["exchange"].isna()]
    L = ["# v3 universe report (WS1)", "",
         f"Built from `{ud}`. Candidates: {len(cl):,} rows "
         f"({', '.join(f'{k}: {v:,}' for k, v in cl['source'].value_counts().items())}).", "",
         "## Universe: common stocks with SIP bars since 2016", ""]
    u = cl[cl["in_universe"]]
    by = u.groupby(u["source"] + ":" + u["status"].fillna("?"))["fetch_symbol"].nunique()
    L += [md_table(by.rename("symbols")), "",
          f"**{u['fetch_symbol'].nunique():,} distinct symbols** are downloaded "
          f"(spac-like names among them: {int(u.drop_duplicates('fetch_symbol')['spac_like'].sum()):,}, kept "
          "but flagged `spac_like`). `alpaca_assets:active` are currently listed; everything else "
          "(Alpaca inactive, the 2018 NASDAQ list, symbols found only by discovery) is a delisted "
          "or renamed security — the survivorship-bias correction.", "",
          "## Classification counts (NASDAQ/NYSE/AMEX + unknown-exchange candidates)", "",
          md_table(pd.crosstab(listed["cls"], listed["source"] + ":" + listed["status"].fillna("?"),
                               margins=True)), "",
          f"Other exchanges (ARCA, BATS, OTC, TXSE) are excluded: "
          f"{int((cl['cls'] == 'non_listed_exchange').sum()):,} asset rows. Exception: "
          f"{int((cl['reason'] == 'exchange_now_OTC;sip_bars_while_listed').sum()):,} securities Alpaca now "
          "reports as OTC whose name classifies as common *and* that have SIP bars — companies delisted "
          "to the OTC market; SIP only carries exchange-listed trading, so their bars are the listed "
          "period. They are included (reason `exchange_now_OTC;sip_bars_while_listed`).", "",
          "Rules (first match wins, `universe.classify_name`): exchange filter → decisive symbol "
          "patterns (`.WS`, `.U`, `.RT`, `.PR*`, `-P`, `.WI`, CUSIP-like, NASDAQ 5th letter W/U/R/X) → "
          "name keywords (ETF/ETN sponsors and wording; preferred/depositary/cumulative/perpetual; "
          "warrant; unit (not LP common units); right (not ADS 'right to receive'); notes/debentures/"
          "bonds 'due YYYY'; royalty/grantor/commodity/crypto/liquidating trusts; closed-end funds by "
          "'Fund'/'Municipal'/sponsor+fund wording) → NASDAQ 5th letter P/O/N/M (preferred) or G/H/I/L "
          "(notes) unless the name says common/class/ordinary (GOOGL) → common. ADRs, REIT 'shares of "
          "beneficial interest' and MLP common units stay common.", ""]
    for c in [c for c in CLASSES if c in set(listed["cls"])]:
        sub = listed[(listed["cls"] == c) & listed["has_bars"]]
        if not len(sub):
            sub = listed[listed["cls"] == c]
        smp = sub.iloc[rng.choice(len(sub), size=min(sample_n, len(sub)), replace=False)]
        L += [f"### Spot-check: {c} ({len(listed[listed['cls'] == c]):,} rows)", "",
              md_table(smp[["symbol", "name", "exchange", "status", "source", "reason"]]
                       .assign(name=lambda d: d["name"].fillna("").str.slice(0, 80)), index=False), ""]
    un = cl[~cl["name_verified"] & (cl["exchange"].isna() | cl["exchange"].isin(LISTED_EXCHANGES))]
    L += ["## Unnamed symbols (discovery / blank Alpaca names): ETF-likeness test", "",
          "SIP only carries exchange-listed securities, so every discovered symbol was listed at some "
          "point since 2016, but without a name it could be a (dead) ETF. Test: weekly log returns, max "
          "|corr| with any *named* ETF over >= 26 common weeks (single-stock trackers removed from the "
          "references). A dead ETF almost always has a live twin; a company does not. Calibration on "
          "named securities (self excluded):", "", md_table(calib, index=False), "",
          f"Rule used: corr >= {ETF_CORR_THRESHOLD} or (corr >= {ETF_CORR_LOW_VOL} and annualised vol < "
          f"{ETF_MAX_ANN_VOL:.0%}). ETFs the rule misses are mostly illiquid (median weekly dollar volume "
          "~10x below stocks), so the $2M ADV eligibility filter removes most of the residue. Overrides: acquiree of a cash merger / worthless "
          "removal (Alpaca corporate actions) → company, unless corr >= 0.95 (ETF reorganisations also "
          "appear as mergers); acquiree of a stock merger into an ETF/fund → "
          "fund. Result for unnamed rows:", "",
          md_table(un.groupby(["source", "cls"]).size().rename("rows")), ""]
    for c in ("common", "etf"):
        sub = un[(un["cls"] == c) & un["etf_corr"].notna()]
        if len(sub):
            smp = sub.iloc[rng.choice(len(sub), size=min(12, len(sub)), replace=False)]
            L += [f"Sample unnamed → {c}:", "",
                  md_table(smp[["fetch_symbol", "first_week", "last_week", "n_weeks", "etf_corr",
                                "etf_match", "ann_vol", "ca_evidence"]], index=False), ""]
    if seg is not None:
        ms = seg[seg["n_seg"] > 1]
        L += ["## Symbol reuse", "",
              "Rule (`universe.detect_symbol_reuse`): only traded bars count (Alpaca glues reused "
              "symbols with zero-volume filler bars); a new segment starts after > 90 calendar days "
              "without a traded bar; when Alpaca lists more than one security under the symbol "
              "(active+inactive duplicates or `SYM_DELISTED`) after > 30 days, or > 7 days with a > 2x "
              "price-level change; such a multi-security symbol "
              "without a gap is split at a one-day move > ln 3 (price-level break). The latest segment "
              "keeps the symbol, earlier ones become `SYM~1`, `SYM~2`… Segment class: the current "
              "segment takes the active security's class, past segments the class shared by the "
              "symbol's other securities (disagreement → `ambiguous`, excluded). Kept = common and "
              ">= 5 traded bars. Note the earlier segments' *adjusted* price level can be wrong (Alpaca "
              "applies the newest company's split factors to the whole symbol), but returns within a "
              "segment are unaffected and `close_raw` stays correct.", "",
              f"* symbols in the store: {seg['symbol'].nunique():,}; with > 1 segment: "
              f"{ms['symbol'].nunique():,} ({int((ms['seg'] > 1).sum()):,} splits: "
              f"{', '.join(f'{k} {v}' for k, v in ms.loc[ms['seg'] > 1, 'split_reason'].value_counts().items())});",
              f"* multi-security symbols: {seg.loc[seg['multi_security'], 'symbol'].nunique():,};",
              f"* segments excluded: {int((~seg['keep']).sum()):,} "
              f"({', '.join(f'{k} {v}' for k, v in seg.loc[~seg['keep'], 'cls'].value_counts().items())}).", ""]
        ex = ms.sort_values(["multi_security", "symbol"], ascending=[False, True])
        exs = ex[ex["symbol"].isin(ex["symbol"].drop_duplicates().head(14))]
        L += ["Examples:", "", md_table(exs[["ticker", "start", "end", "n_bars", "gap_before_days",
                                             "split_reason", "multi_security", "cls", "keep"]], index=False), ""]
    dp = os.path.join(ud, "duplicates.parquet")
    if os.path.exists(dp):
        dup = pd.read_parquet(dp)
        L += ["## Duplicate histories (same security under two symbols)", "",
              "Alpaca re-keys a renamed company's history to its new symbol, but the old symbol (when "
              "nobody reused it) still returns the same bars (SQ/XYZ, SIVB/SIVBQ); inside one request the "
              "API returns the series once, across requests twice. Two symbols sharing >= 20 traded bars "
              "with identical (Date, close, volume, trade_count) are the same security. The one with more "
              "bars (tie: later last date) is canonical; the other is dropped entirely when the shared bars "
              "cover > 90% of it (`alias`), otherwise only the shared rows are dropped (`rows`). Applied by "
              "`load_prices(clean=True)` before the reuse segmentation.", "",
              f"* pairs: {len(dup):,} ({', '.join(f'{k} {v}' for k, v in dup['mode'].value_counts().items())}); "
              f"rows dropped: {int(dup.loc[dup['mode'] == 'alias', 'n_alias'].sum()):,} (aliases) + "
              f"{len(pd.read_parquet(os.path.join(ud, 'duplicate_rows.parquet'))):,} (partial).", "",
              md_table(dup.head(15).round(3), index=False), ""]
    p = os.path.join(ud, "eligible_counts.parquet")
    if os.path.exists(p):
        ec = pd.read_parquet(p)
        yr = ec.groupby(ec["Date"].str[:4]).agg(sessions=("Date", "size"), mean_with_bar=("n_with_bar", "mean"),
                                                mean_eligible=("n_eligible", "mean"),
                                                min_eligible=("n_eligible", "min"),
                                                max_eligible=("n_eligible", "max"))
        up = os.path.join(ud, "eligible_unique_per_year.parquet")
        if os.path.exists(up):
            yr = yr.join(pd.read_parquet(up))
        L += ["## Universe size under `eligible(min_price=3, min_adv_usd=2e6, min_bars=120)`", "",
              "Per session on the cleaned store (price = raw close, ADV = 20-session mean of "
              "close×volume, history counted from 2016-01-01 so 2016 ramps up):", "",
              md_table(yr.round(0).astype(int)), "", "Per-date counts: `eligible_counts.parquet`.", ""]
    txt = "\n".join(L)
    with open(os.path.join(ud, "universe_report.md"), "w") as f:
        f.write(txt)
    return txt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="v3 universe builder")
    ap.add_argument("step", choices=["snapshot", "discover", "classify", "reuse", "eligibility", "report"])
    a = ap.parse_args()
    if a.step == "snapshot":
        c = AlpacaClient()
        snapshot_assets(c)
        snapshot_corporate_actions(c)
    elif a.step == "discover":
        discover_symbols(extra_symbols=candidate_extra_symbols(load_assets(), load_corporate_actions(),
                                                               load_nasdaq_2018()))
    elif a.step == "classify":
        build_classification()
    elif a.step == "reuse":
        from .prices import load_prices
        px = load_prices(clean=False, columns=["company_close", "company_volume", "trade_count"], with_raw=False)
        detect_duplicates(px)
        detect_symbol_reuse(apply_duplicates(px))
    elif a.step == "eligibility":
        eligible_counts()
    else:
        print(write_report())
