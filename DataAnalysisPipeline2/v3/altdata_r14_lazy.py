"""Round 14 ``lazy`` block ("Lazy Prices", Cohen, Malloy & Nguyen 2020): change
in a firm's 10-K / 10-Q text vs its previous comparable filing. Pre-registered in
``results/round14_preregistration.md``; the definitional details are fixed in
``results/round14_addendum_lazy.md`` (committed before any forward-return
statistic).

Source: EDGAR, one GET per filing for its *primary document* (``primaryDocument``
of the bulk ``submissions.zip`` already in the store), ``Archives/edgar/data/
<cik>/<accn>/<doc>``; <= ``MAX_RPS`` requests/s (shared thread-safe limiter, the
``form4.SECClient``), resumable. No free bulk corpus covers 2016-2026 10-K *and*
10-Q text without registration (EDGAR-CORPUS stops in 2020 and is 10-K only;
the Notre Dame SRAF parsed 10-X files are Google-Drive archives that end before
2026), so the per-filing route is the cheapest complete one.

Per filing the raw HTML is discarded after reduction to (``parse_document``):
* text: inline-XBRL header, scripts/styles and *numeric tables* removed
  (a ``<table>`` whose digits are > ``TABLE_DIGIT_SHARE`` of its alphanumerics;
  layout tables holding prose are kept), tags stripped, entities unescaped;
* tokens ``[a-z]{2,}`` of the lower-cased text (no stop-word removal, no stemming);
* a hashed bag of words (``crc32(word) mod 2**HASH_BITS``, counts), the token
  count, and Loughran-McDonald (master dictionary 1993-2025) negative /
  uncertainty / litigious counts;
* for 10-Ks, the same hashed vector of "Item 1. Business" (``item1_span``) when
  it is separable -> ``sec_lazy/item1_vectors.parquet`` (firm-year business
  description, a fallback for text-peer edges).

Features (``LAZY_COLUMNS``) per filing vs the same firm's previous comparable
filing (10-K: previous 10-K; 10-Q: the 10-Q for the same fiscal quarter one
year earlier) -- ``filing_features``; carried forward weekly from the session
after the acceptance date, ``lazy_days`` = age, missing after 400 days
(``weekly_features``).

    python -m DataAnalysisPipeline2.v3.altdata_r14_lazy index | download | features | audit | ics
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
import time
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from ._alpaca import data_dir

FORMS = ("10-K", "10-Q")
START = "2016-01-01"
MAX_RPS = 3.9                      # task cap: <= 4 requests/s to EDGAR
HASH_BITS = 18
HASH_MOD = 1 << HASH_BITS
TABLE_DIGIT_SHARE = 0.15
MIN_TOKENS = 200                   # shorter documents (cover pages / exhibits only) -> no features
ITEM1_MIN_TOKENS = 150
MAX_DOC_BYTES = 40_000_000
MAX_AGE_DAYS = 400
PRIOR_WINDOW = {"10-K": (300, 430), "10-Q": (335, 395)}   # days between report periods
LM_CATS = ("neg", "unc", "lit")
LAZY_COLUMNS = ["lazy_cos", "lazy_jac", "lazy_dlen", "lazy_dneg", "lazy_dunc", "lazy_dlit",
                "lazy_is10k", "lazy_days"]
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
SHARD_SIZE = 200


def lazy_dir(*parts: str) -> str:
    return data_dir("sec_lazy", *parts)


# --------------------------------------------------------------------------- #
# Loughran-McDonald word lists
# --------------------------------------------------------------------------- #
LM_CSV = "LM_MasterDictionary_1993-2025.csv"
_LM_COLS = {"neg": "Negative", "unc": "Uncertainty", "lit": "Litigious"}


def load_lm(path: str | None = None) -> dict:
    """{'neg'|'unc'|'lit': set of lower-case words}. A category flag > 0 is the
    year the word was added; 0 / negative = not (or no longer) in the list."""
    path = path or os.path.join(lazy_dir("lm"), LM_CSV)
    d = pd.read_csv(path, usecols=["Word"] + list(_LM_COLS.values()), keep_default_na=False)
    w = d["Word"].astype(str).str.lower()
    return {k: set(w[d[c].astype(int) > 0]) for k, c in _LM_COLS.items()}


# --------------------------------------------------------------------------- #
# Document -> compact representation
# --------------------------------------------------------------------------- #
_RE_DROP = re.compile(r"<(ix:header|script|style|head|xbrl|title)\b.*?</\1\s*>", re.I | re.S)
_RE_HIDDEN = re.compile(r"<div[^>]*display\s*:\s*none[^>]*>.*?</div\s*>", re.I | re.S)
_RE_TABLE = re.compile(r"<table\b.*?</table\s*>", re.I | re.S)
_RE_BLOCK = re.compile(r"<(?:br|p|div|tr|li|h[1-6])\b[^>]*>|</(?:p|div|tr|li|h[1-6])\s*>", re.I)
_RE_TAG = re.compile(r"<[^>]+>", re.S)
_RE_WS = re.compile(r"[ \t\r\f\v\xa0]+")
_RE_TOKEN = re.compile(r"[a-z]{2,}")
_RE_DIGIT = re.compile(r"\d")
_RE_ALNUM = re.compile(r"[A-Za-z0-9]")


def _strip_tags(s: str) -> str:
    s = _RE_BLOCK.sub("\n", s)
    s = _RE_TAG.sub(" ", s)
    return _html.unescape(s)


def _table_repl(m: re.Match) -> str:
    txt = _strip_tags(m.group(0))
    d = len(_RE_DIGIT.findall(txt))
    a = len(_RE_ALNUM.findall(txt))
    if a == 0 or d / a > TABLE_DIGIT_SHARE:
        return "\n"
    return "\n" + txt + "\n"


def clean_text(doc: str) -> str:
    """Plain text of a primary document (HTML / iXBRL or plain text)."""
    if "<" in doc[:20000] and re.search(r"<(html|body|div|p|table|font)\b", doc[:200000], re.I):
        s = _RE_DROP.sub(" ", doc)
        s = _RE_HIDDEN.sub(" ", s)
        s = _RE_TABLE.sub(_table_repl, s)
        s = _strip_tags(s)
    else:
        s = _RE_TAG.sub(" ", doc)
    return _RE_WS.sub(" ", s)


def tokenize(text: str) -> list[str]:
    return _RE_TOKEN.findall(text.lower())


_HCACHE: dict = {}


def hash_word(w: str) -> int:
    h = _HCACHE.get(w)
    if h is None:
        h = zlib.crc32(w.encode()) & (HASH_MOD - 1)
        if len(_HCACHE) < 2_000_000:
            _HCACHE[w] = h
    return h


def hashed_bow(tokens) -> tuple[np.ndarray, np.ndarray]:
    """Sorted bucket indices (uint32) and counts (uint32) of a token list."""
    c = Counter(hash_word(t) for t in tokens)
    if not c:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    idx = np.fromiter(c.keys(), np.int64, len(c))
    cnt = np.fromiter(c.values(), np.int64, len(c))
    o = np.argsort(idx)
    return idx[o].astype(np.uint32), cnt[o].astype(np.uint32)


_ITEM1_START = re.compile(r"\bitem\s*1\s*[\.\:\-–—]?\s*(?:\(?\s*[a-z]?\s*\)?\s*)?business\b", re.I)
_ITEM1_END = re.compile(r"\bitem\s*(?:1a\s*[\.\:\-–—]?\s*risk\s*factors|1b\b|2\s*[\.\:\-–—]?\s*"
                        r"(?:description\s*of\s*)?propert)", re.I)


def item1_span(text: str) -> str | None:
    """'Item 1. Business' section of a 10-K text: among all Item-1 headings, the
    one with the longest run to the next Item 1A / 1B / 2 heading (skips the
    table of contents). None if no run has >= ITEM1_MIN_TOKENS tokens."""
    ends = [m.start() for m in _ITEM1_END.finditer(text)]
    if not ends:
        return None
    best = None
    for m in _ITEM1_START.finditer(text):
        nxt = next((e for e in ends if e > m.end()), None)
        if nxt is None:
            continue
        if best is None or nxt - m.end() > best[1] - best[0]:
            best = (m.end(), nxt)
    if best is None:
        return None
    seg = text[best[0]:best[1]]
    return seg if len(_RE_TOKEN.findall(seg.lower())) >= ITEM1_MIN_TOKENS else None


def parse_document(doc: str, form: str, lm: dict) -> dict:
    text = clean_text(doc)
    toks = tokenize(text)
    idx, cnt = hashed_bow(toks)
    out = {"n_tok": len(toks), "idx": idx, "cnt": cnt}
    for k in LM_CATS:
        s = lm[k]
        out[f"n_{k}"] = sum(1 for t in toks if t in s)
    if form == "10-K":
        seg = item1_span(text)
        if seg is not None:
            t1 = tokenize(seg)
            i1, c1 = hashed_bow(t1)
            out.update(i1_idx=i1, i1_cnt=c1, i1_tok=len(t1))
    return out


# --------------------------------------------------------------------------- #
# Filing index (local submissions.zip)
# --------------------------------------------------------------------------- #
def _rows(block: dict, cik: int) -> list[tuple]:
    out = []
    forms = block.get("form") or []
    for i, f in enumerate(forms):
        if f in FORMS and block["filingDate"][i] >= START:
            out.append((cik, block["accessionNumber"][i], f, block["filingDate"][i],
                        block.get("reportDate", [""] * len(forms))[i] or "",
                        block["acceptanceDateTime"][i], block["primaryDocument"][i] or ""))
    return out


def build_index(log=print) -> pd.DataFrame:
    """10-K / 10-Q originals filed >= START by the CIK-map issuers, with
    primary document, report period and acceptance time (UTC)."""
    from ._alpaca import data_dir as dd
    try:
        import orjson
        loads = orjson.loads
    except ImportError:  # pragma: no cover
        loads = json.loads
    cm = pd.read_parquet(os.path.join(dd("fundamentals"), "cik_map.parquet"))
    ciks = sorted({int(c) for c in cm["cik"].dropna().unique()})
    rows = []
    with zipfile.ZipFile(os.path.join(dd("edgar"), "submissions.zip")) as z:
        names = set(z.namelist())
        for cik in ciks:
            main = f"CIK{cik:010d}.json"
            if main not in names:
                continue
            d = loads(z.read(main))
            fl = d.get("filings", {}) or {}
            rows += _rows(fl.get("recent", {}) or {}, cik)
            for extra in fl.get("files", []) or []:
                if extra.get("name") in names:
                    rows += _rows(loads(z.read(extra["name"])), cik)
    f = pd.DataFrame(rows, columns=["cik", "accn", "form", "filing_date", "report_date", "acceptance", "doc"])
    f["acceptance"] = pd.to_datetime(f["acceptance"], utc=True, errors="coerce")
    f = f.drop_duplicates("accn").sort_values(["cik", "acceptance"]).reset_index(drop=True)
    f = f.assign(tier=priority_tiers(f))
    f.to_parquet(os.path.join(lazy_dir(), "index.parquet"), index=False)
    log(f"[lazy] index: {len(f):,} filings, {f['cik'].nunique():,} CIKs; tiers {f['tier'].value_counts().sort_index().to_dict()}")
    return f


def priority_tiers(f: pd.DataFrame) -> np.ndarray:
    """Download order: 1 = CIKs ever in the weekly top-1000 (all forms),
    2 = other ever-eligible CIKs' 10-Ks, 3 = their 10-Qs, 4 = never eligible."""
    from . import dataset as ds
    from . import round6 as r6
    a = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"), columns=["ticker", "Date", "eligible", "cik"])
    top = pd.read_parquet(r6.topn_path())
    top_c = set(a.merge(top, on=["ticker", "Date"])["cik"].dropna().astype(int))
    el_c = set(a.loc[a["eligible"].astype(bool), "cik"].dropna().astype(int))
    t = np.full(len(f), 4)
    el = f["cik"].isin(el_c).to_numpy()
    t[el & (f["form"] == "10-Q").to_numpy()] = 3
    t[el & (f["form"] == "10-K").to_numpy()] = 2
    t[f["cik"].isin(top_c).to_numpy()] = 1
    return t


# --------------------------------------------------------------------------- #
# Resumable downloader
# --------------------------------------------------------------------------- #
def _shards(kind: str = "docs") -> list[str]:
    d = lazy_dir(kind)
    return sorted(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".parquet"))


def done_accns() -> set:
    s = set()
    for p in _shards("docs"):
        s |= set(pd.read_parquet(p, columns=["accn"])["accn"])
    return s


_LM = None


def _parse_job(args):
    global _LM
    if _LM is None:
        _LM = load_lm()
    raw, form = args
    try:
        doc = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        return parse_document(doc, form, _LM)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def _write_shard(recs: list[dict], seq: int) -> None:
    if not recs:
        return
    base = ["cik", "accn", "form", "filing_date", "report_date", "acceptance", "status", "n_tok",
            "n_neg", "n_unc", "n_lit"]
    d = pd.DataFrame({c: [r.get(c) for r in recs] for c in base})
    d["idx"] = [r.get("idx", np.zeros(0, np.uint32)) for r in recs]
    d["cnt"] = [r.get("cnt", np.zeros(0, np.uint32)) for r in recs]
    tag = f"{int(time.time())}_{seq:05d}"
    d.to_parquet(os.path.join(lazy_dir("docs"), f"docs_{tag}.parquet"), index=False, compression="zstd")
    i1 = [r for r in recs if "i1_idx" in r]
    if i1:
        pd.DataFrame({"cik": [r["cik"] for r in i1], "accn": [r["accn"] for r in i1],
                      "filing_date": [r["filing_date"] for r in i1], "report_date": [r["report_date"] for r in i1],
                      "acceptance": [r["acceptance"] for r in i1], "i1_tok": [r["i1_tok"] for r in i1],
                      "idx": [r["i1_idx"] for r in i1], "cnt": [r["i1_cnt"] for r in i1]}
                     ).to_parquet(os.path.join(lazy_dir("item1"), f"item1_{tag}.parquet"), index=False,
                                  compression="zstd")


def download(max_tier: int = 3, max_hours: float | None = None, threads: int = 4, workers: int = 4,
             log=print) -> None:
    """Fetch + reduce every not-yet-done filing of tiers <= max_tier, in tier
    order (within a tier: by CIK, then date). Safe to kill and restart."""
    import faulthandler
    import signal
    import requests
    from . import altdata as ad
    from .form4 import Limiter, SECClient
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    codes: Counter = Counter()
    idx = pd.read_parquet(os.path.join(lazy_dir(), "index.parquet"))
    done = done_accns()
    todo = idx[(idx["tier"] <= max_tier) & ~idx["accn"].isin(done)].sort_values(["tier", "cik", "acceptance"])
    log(f"[lazy] download: {len(todo):,} to do ({len(done):,} done)")
    sess = requests.Session()
    sess.headers.update(ad._sec_headers())
    sess.headers["Accept-Encoding"] = "gzip, deflate"
    adapter = requests.adapters.HTTPAdapter(pool_connections=threads, pool_maxsize=threads)
    sess.mount("https://", adapter)
    lim = Limiter(MAX_RPS)
    t0 = time.time()

    class _Client(SECClient):
        def get_bytes(self, url):
            for k in range(self.retries):
                self.lim.wait()
                self.n += 1
                try:
                    r = self.s.get(url, timeout=90)
                except Exception as e:  # noqa: BLE001
                    codes[type(e).__name__] += 1
                    self.sleep(2 ** k)
                    continue
                codes[r.status_code] += 1
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return None
                self.sleep(2 ** k * (5 if r.status_code in (403, 429) else 1))
            return b"__FAILED__"

    cl = _Client(session=sess, limiter=lim)
    rows = todo.to_dict("records")

    def fetch(r):
        if not r["doc"]:
            return r, None
        acc = r["accn"].replace("-", "")
        return r, cl.get_bytes(ARCHIVES.format(cik=int(r["cik"]), acc=acc, doc=r["doc"]))

    from collections import deque
    state = {"recs": [], "seq": 0, "ok": 0, "bad": 0, "n": 0}

    def record(rec):
        state["recs"].append(rec)
        state["n"] += 1
        state["ok"] += rec["status"] == "ok"
        if len(state["recs"]) >= SHARD_SIZE:
            _write_shard(state["recs"], state["seq"])
            state["seq"] += 1
            state["recs"] = []
            el = time.time() - t0
            k = state["n"] + state["bad"]
            log(f"[lazy] {k:,}/{len(rows):,} ok {state['ok']:,} failed {state['bad']} | {cl.n / max(el, 1):.2f} req/s | "
                f"{el / 60:.0f} min, ETA {(len(rows) - k) * el / max(k, 1) / 3600:.1f} h | tier {rec_tier[0]} | "
                f"http {dict(codes)}")

    rec_tier = [None]

    def drain(pq, block):
        while pq and (block or pq[0][1].done()):
            base, fut = pq.popleft()
            res = fut.result()
            record({**base, "status": "parse_error"} if "error" in res else {**base, **res, "status": "ok"})

    with ThreadPoolExecutor(threads) as tp, ProcessPoolExecutor(workers) as pp:
        fq, pq, it = deque(), deque(), iter(rows)
        stop = False
        while True:
            while not stop and len(fq) < 8 * threads:
                r = next(it, None)
                if r is None or (max_hours is not None and time.time() - t0 > max_hours * 3600):
                    if r is not None:
                        log("[lazy] time budget reached, stopping")
                    stop = True
                    break
                fq.append(tp.submit(fetch, r))
            if not fq:
                break
            r, raw = fq.popleft().result()
            rec_tier[0] = r["tier"]
            base = {k: r[k] for k in ("cik", "accn", "form", "filing_date", "report_date", "acceptance")}
            if raw is None:
                record({**base, "status": "missing"})
            elif raw == b"__FAILED__":
                state["bad"] += 1            # not recorded -> retried on the next run
            elif len(raw) > MAX_DOC_BYTES:
                record({**base, "status": "too_big"})
            else:
                pq.append((base, pp.submit(_parse_job, (raw, r["form"]))))
            drain(pq, block=len(pq) > 64)
        drain(pq, block=True)
        _write_shard(state["recs"], state["seq"])
        log(f"[lazy] download finished: ok {state['ok']:,}, failed {state['bad']}")


def load_docs(columns=None) -> pd.DataFrame:
    parts = [pd.read_parquet(p, columns=columns) for p in _shards("docs")]
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates("accn", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-filing features
# --------------------------------------------------------------------------- #
def sparse_sims(i1, c1, i2, c2) -> tuple[float, float]:
    """(cosine of the count vectors, Jaccard of the bucket sets)."""
    if len(i1) == 0 or len(i2) == 0:
        return np.nan, np.nan
    common, a, b = np.intersect1d(i1, i2, assume_unique=True, return_indices=True)
    x1, x2 = c1.astype(np.float64), c2.astype(np.float64)
    dot = float(np.dot(x1[a], x2[b]))
    cos = dot / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))
    jac = len(common) / (len(i1) + len(i2) - len(common))
    return cos, jac


def match_prior(f: pd.DataFrame) -> pd.Series:
    """Index (into ``f``) of each filing's comparable prior filing, or -1:
    same CIK and form, accepted earlier, report period PRIOR_WINDOW days
    earlier (filing date if a report date is missing); the latest such one."""
    f = f.copy()
    per = pd.to_datetime(f["report_date"].replace("", np.nan), errors="coerce")
    f["_per"] = per.fillna(pd.to_datetime(f["filing_date"]))
    f["_i"] = np.arange(len(f))
    out = np.full(len(f), -1, dtype=np.int64)
    for (cik, form), g in f.groupby(["cik", "form"], sort=False):
        lo, hi = PRIOR_WINDOW[form]
        g = g.sort_values("acceptance")
        per_v, acc_v, ii = g["_per"].to_numpy(), g["acceptance"].to_numpy(), g["_i"].to_numpy()
        for k in range(len(g)):
            gap = (per_v[k] - per_v[:k]).astype("timedelta64[D]").astype(np.int64)
            ok = np.where((gap >= lo) & (gap <= hi) & (acc_v[:k] < acc_v[k]))[0]
            if len(ok):
                out[ii[k]] = ii[ok[-1]]
    return pd.Series(out, index=f.index)


def filing_features(docs: pd.DataFrame) -> pd.DataFrame:
    """One row per usable filing: cik, accn, form, acceptance, prior_accn and
    the raw lazy features (NaN when there is no comparable prior)."""
    d = docs[(docs["status"] == "ok") & (docs["n_tok"] >= MIN_TOKENS)].reset_index(drop=True)
    d["acceptance"] = pd.to_datetime(d["acceptance"], utc=True)
    pr = match_prior(d).to_numpy()
    n = len(d)
    cos, jac = np.full(n, np.nan), np.full(n, np.nan)
    I, C = d["idx"].to_numpy(), d["cnt"].to_numpy()
    for k in range(n):
        j = pr[k]
        if j >= 0:
            cos[k], jac[k] = sparse_sims(np.asarray(I[k]), np.asarray(C[k]), np.asarray(I[j]), np.asarray(C[j]))
    ntok = d["n_tok"].to_numpy(dtype=float)
    has = pr >= 0
    pj = np.where(has, pr, 0)
    out = d[["cik", "accn", "form", "filing_date", "report_date", "acceptance", "n_tok"]].copy()
    out["prior_accn"] = np.where(has, d["accn"].to_numpy()[pj], None)
    out["lazy_cos"], out["lazy_jac"] = cos, jac
    out["lazy_dlen"] = np.where(has, np.log(ntok / ntok[pj]), np.nan)
    for k in LM_CATS:
        sh = d[f"n_{k}"].to_numpy(dtype=float) / ntok
        out[f"lazy_d{k}"] = np.where(has, (sh - sh[pj]) * 100.0, np.nan)   # percentage points
    out["lazy_is10k"] = (out["form"] == "10-K").astype(float)
    return out


def avail_dates(acceptance: pd.Series, calendar) -> pd.Series:
    """First session strictly after the acceptance's New York calendar date."""
    cal = np.asarray(calendar).astype(str)
    day = pd.DatetimeIndex(acceptance).tz_convert("America/New_York").strftime("%Y-%m-%d")
    pos = np.searchsorted(cal, np.asarray(day).astype(str), side="right")
    v = np.where(pos < len(cal), cal[np.minimum(pos, len(cal) - 1)], None)
    return pd.Series(pd.to_datetime(v), index=acceptance.index)


def weekly_join(rows: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Latest event (``avail_date`` <= Date, finite lazy_cos) per (cik, Date)
    row -> LAZY_COLUMNS + lazy_accn; block missing when older than MAX_AGE_DAYS."""
    r = rows[["cik", "Date"]].copy()
    r["_i"] = np.arange(len(r))
    r["_d"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
    m = r[r["cik"].notna()].assign(cik=lambda x: x["cik"].astype("int64")).sort_values("_d")
    e = ev[np.isfinite(ev["lazy_cos"])].assign(
        _a=pd.to_datetime(ev["avail_date"]).astype("datetime64[ns]"), cik=lambda x: x["cik"].astype("int64"))
    e = e.dropna(subset=["_a"]).sort_values(["_a", "acceptance"])
    vals = LAZY_COLUMNS[:-1]
    j = pd.merge_asof(m, e[["cik", "_a", "accn"] + vals], left_on="_d", right_on="_a", by="cik",
                      direction="backward")
    j["lazy_days"] = (j["_d"] - j["_a"]).dt.days.astype(float)
    stale = (j["lazy_days"] > MAX_AGE_DAYS).to_numpy()
    j.loc[stale, LAZY_COLUMNS] = np.nan
    j.loc[stale, "accn"] = None
    out = pd.DataFrame(np.nan, index=np.arange(len(r)), columns=LAZY_COLUMNS)
    out.loc[j["_i"].to_numpy(), LAZY_COLUMNS] = j[LAZY_COLUMNS].to_numpy(dtype=float)
    out["lazy_accn"] = None
    out.loc[j["_i"].to_numpy(), "lazy_accn"] = j["accn"].to_numpy()
    return out


def weekly_path() -> str:
    from . import dataset as ds
    return os.path.join(ds.v3_dir(), "alt14_lazy_weekly.parquet")


def events_path() -> str:
    return os.path.join(lazy_dir(), "lazy_events.parquet")


def build_features(log=print) -> pd.DataFrame:
    """Filing events -> ``sec_lazy/lazy_events.parquet``; weekly join on the
    Round-8 ``alt_weekly`` rows -> ``v3model/alt14_lazy_weekly.parquet``;
    firm-year Item 1 vectors -> ``sec_lazy/item1_vectors.parquet``."""
    from . import dataset as ds
    from .altdata_r10 import price_matrices  # noqa: F401  (calendar source documented there)
    t0 = time.time()
    docs = load_docs()
    ev = filing_features(docs)
    files = sorted(__import__("glob").glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    cal = ds.session_calendar(pd.concat([pd.read_parquet(f, columns=["Date"]) for f in files])["Date"])
    ev["avail_date"] = avail_dates(ev["acceptance"], cal).dt.strftime("%Y-%m-%d")
    ev.to_parquet(events_path(), index=False)
    log(f"[lazy] events: {len(ev):,} filings, {ev['cik'].nunique():,} CIKs, with prior "
        f"{np.isfinite(ev['lazy_cos']).mean():.1%} ({time.time() - t0:.0f}s)")
    rows = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"),
                           columns=["ticker", "Date", "eligible", "fwd_ret_30", "cik"])
    w = weekly_join(rows, ev)
    for c in LAZY_COLUMNS:
        rows[c] = w[c].astype("float32").to_numpy()
    rows["lazy_accn"] = w["lazy_accn"].to_numpy()
    rows.to_parquet(weekly_path(), index=False)
    el = rows["eligible"].astype(bool)
    log(f"[lazy] weekly: {len(rows):,} rows, lazy_cos finite {rows['lazy_cos'].notna().mean():.1%} "
        f"(eligible {rows.loc[el, 'lazy_cos'].notna().mean():.1%})")
    i1 = [pd.read_parquet(p) for p in _shards("item1")]
    if i1:
        i1 = pd.concat(i1, ignore_index=True).drop_duplicates("accn", keep="last")
        i1["acceptance"] = pd.to_datetime(i1["acceptance"], utc=True)
        i1["avail_date"] = avail_dates(i1["acceptance"], cal).dt.strftime("%Y-%m-%d")
        i1["fiscal_year"] = pd.to_datetime(i1["report_date"].replace("", np.nan), errors="coerce").dt.year
        i1.sort_values(["cik", "acceptance"]).to_parquet(os.path.join(lazy_dir(), "item1_vectors.parquet"),
                                                         index=False, compression="zstd")
        log(f"[lazy] item1 vectors: {len(i1):,} 10-Ks, {i1['cik'].nunique():,} CIKs")
    return rows


# --------------------------------------------------------------------------- #
# Audit, coverage, standalone ICs
# --------------------------------------------------------------------------- #
def asof_audit(rows: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Every weekly value must come from a filing of the row's CIK whose
    acceptance (New York date) is strictly before the row's Date, not older
    than MAX_AGE_DAYS, and the latest such filing with a finite feature."""
    r = rows[rows["lazy_accn"].notna()].merge(
        ev[["accn", "cik", "acceptance", "avail_date"]].rename(columns={"cik": "ev_cik"}),
        left_on="lazy_accn", right_on="accn", how="left")
    acc_ny = pd.DatetimeIndex(r["acceptance"]).tz_convert("America/New_York").tz_localize(None).normalize()
    d = pd.to_datetime(r["Date"])
    return {"rows_with_value": int(len(r)),
            "missing_source": int(r["acceptance"].isna().sum()),
            "cik_mismatch": int((r["ev_cik"].astype("int64") != r["cik"].astype("int64")).sum()),
            "accepted_on_or_after_date": int((acc_ny >= d).sum()),
            "avail_after_date": int((pd.to_datetime(r["avail_date"]) > d).sum()),
            "older_than_max_age": int(((d - pd.to_datetime(r["avail_date"])).dt.days > MAX_AGE_DAYS).sum()),
            "min_gap_days_accept_to_date": int((d - acc_ny).dt.days.min()) if len(r) else None}


def coverage_by_year(rows: pd.DataFrame, ev: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    y = rows["Date"].astype(str).str[:4]
    el = rows["eligible"].astype(bool)
    has = rows["lazy_cos"].notna()
    t = pd.DataFrame({"weekly_all": has.groupby(y).mean(), "weekly_eligible": has[el].groupby(y[el]).mean(),
                      "eligible_with_cik": has[el & rows["cik"].notna()].groupby(y[el & rows["cik"].notna()]).mean()})
    fy = ev["filing_date"].str[:4]
    t2 = pd.DataFrame({"filings_parsed": ev.groupby(fy).size(),
                       "with_prior": ev[np.isfinite(ev["lazy_cos"])].groupby(fy[np.isfinite(ev["lazy_cos"])]).size(),
                       "index_filings": idx.groupby(idx["filing_date"].str[:4]).size()})
    return t, t2


def feature_ics(rows: pd.DataFrame, log=print) -> dict:
    """Round 8-10 standalone weekly Spearman ICs vs fwd_ret_30 (all / tradable /
    top-1000), NW t, coverage and decay on tradable names."""
    from . import evaluation as evm
    from . import gate
    from . import round6 as r6
    alt = rows[rows["fwd_ret_30"].notna()].copy()
    top = pd.read_parquet(r6.topn_path()).assign(_top=True)
    alt = alt.merge(top, on=["ticker", "Date"], how="left")
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date"] + [f"fwd_ret_{h}" for h in r6.DECAY])
    out = {}
    for c in LAZY_COLUMNS:
        row = {}
        for uname, m in (("all", np.ones(len(alt), bool)), ("tradable", alt["eligible"].astype(bool).to_numpy()),
                         ("top1000", alt["_top"].eq(True).to_numpy(dtype=bool))):
            sub = alt.loc[m, ["ticker", "Date", c, "fwd_ret_30"]].dropna()
            sub = sub[sub.groupby("Date")[c].transform("nunique") > 1]
            s = sub.groupby("Date").apply(lambda g: g[c].corr(g["fwd_ret_30"], method="spearman") if len(g) > 30
                                          else np.nan, include_groups=False).dropna()
            row[uname] = {"ic": float(s.mean()) if len(s) else float("nan"),
                          "nw_t": evm.newey_west_t(s, gate.NW_LAGS) if len(s) > 3 else float("nan"),
                          "coverage": float(alt.loc[m, c].notna().mean()),
                          "by_year": s.groupby(s.index.astype(str).str[:4]).mean().round(4).to_dict()}
        p = alt.loc[alt["eligible"].astype(bool), ["ticker", "Date", c]].rename(columns={c: "pred"}).dropna()
        row["decay"] = {h: float(r6.per_date_ic(p, lab, f"fwd_ret_{h}").mean()) for h in r6.DECAY}
        out[c] = row
        log(f"  {c}: IC all {row['all']['ic']:.4f} tradable {row['tradable']['ic']:.4f} "
            f"top1000 {row['top1000']['ic']:.4f} cov {row['tradable']['coverage']:.0%}")
    return out


def report(path: str | None = None, runtime_note: str = "", log=print) -> str:
    """As-of audit, coverage, standalone ICs -> results/round14_lazy_data.md
    (+ the raw numbers in sec_lazy/report.json)."""
    rows = pd.read_parquet(weekly_path())
    ev = pd.read_parquet(events_path())
    idx = pd.read_parquet(os.path.join(lazy_dir(), "index.parquet"))
    docs = load_docs(columns=["accn", "form", "status", "n_tok"])
    audit = asof_audit(rows, ev)
    cov_w, cov_f = coverage_by_year(rows, ev, idx)
    ics = feature_ics(rows, log=log)
    tiers = idx.assign(done=idx["accn"].isin(set(docs["accn"]))).groupby("tier")["done"].agg(["size", "sum"])
    json.dump({"audit": audit, "ics": ics, "tiers": tiers.to_dict()},
              open(os.path.join(lazy_dir(), "report.json"), "w"), indent=1, default=str)
    i1p = os.path.join(lazy_dir(), "item1_vectors.parquet")
    n_i1 = len(pd.read_parquet(i1p, columns=["accn"])) if os.path.exists(i1p) else 0
    L_ = ["# Round 14 `lazy` block: data, as-of audit, coverage, standalone ICs", "",
          "Built by `v3/altdata_r14_lazy.py` under the definitions in `results/round14_addendum_lazy.md` "
          "(committed before any forward-return statistic). No model trained, no book run.", ""]
    if runtime_note:
        L_ += [runtime_note, ""]
    L_ += ["## Download coverage (filings 2016-01-01 .. 2026-11-05, submissions snapshot 2026-11-06)", "",
           "| tier | content | filings in index | downloaded | share |", "|---|---|---|---|---|"]
    lab = {1: "CIKs ever in weekly top-1000, 10-K + 10-Q", 2: "other ever-eligible CIKs, 10-K",
           3: "other ever-eligible CIKs, 10-Q", 4: "never-eligible CIKs (not downloaded)"}
    for t, r in tiers.iterrows():
        L_.append(f"| {t} | {lab.get(t, '')} | {int(r['size']):,} | {int(r['sum']):,} | {r['sum'] / r['size']:.1%} |")
    st = docs["status"].value_counts().to_dict()
    L_ += ["", f"Documents processed: {len(docs):,} (status {st}); usable (>= {MIN_TOKENS} tokens): "
           f"{len(ev):,} filings of {ev['cik'].nunique():,} CIKs; with a comparable prior: "
           f"{int(np.isfinite(ev['lazy_cos']).sum()):,}. Item-1 vectors: {n_i1:,} 10-Ks.", "",
           "Median tokens: 10-K " + f"{ev.loc[ev['form'] == '10-K', 'n_tok'].median():,.0f}, 10-Q "
           f"{ev.loc[ev['form'] == '10-Q', 'n_tok'].median():,.0f}. Median `lazy_cos`: 10-K "
           f"{ev.loc[ev['form'] == '10-K', 'lazy_cos'].median():.3f}, 10-Q {ev.loc[ev['form'] == '10-Q', 'lazy_cos'].median():.3f}.",
           "", "### By filing year", "", "| year | index filings | parsed | with prior |", "|---|---|---|---|"]
    for y, r in cov_f.fillna(0).iterrows():
        L_.append(f"| {y} | {int(r['index_filings']):,} | {int(r['filings_parsed']):,} | {int(r['with_prior']):,} |")
    L_ += ["", "### Weekly rows with a value (`lazy_cos` finite)", "",
           "| year | all rows | eligible | eligible with CIK |", "|---|---|---|---|"]
    for y, r in cov_w.iterrows():
        L_.append(f"| {y} | {r['weekly_all']:.1%} | {r['weekly_eligible']:.1%} | {r['eligible_with_cik']:.1%} |")
    L_ += ["", "## As-of audit", "", "Every weekly value traced back to its source filing (`lazy_accn`):", ""]
    for k, v in audit.items():
        L_.append(f"* {k}: {v}")
    ok = audit["accepted_on_or_after_date"] == 0 and audit["avail_after_date"] == 0 and audit["cik_mismatch"] == 0 \
        and audit["missing_source"] == 0 and audit["older_than_max_age"] == 0
    L_ += ["", f"**{'PASS' if ok else 'FAIL'}**: no value is used on or before its filing's acceptance date "
           "(the earliest use is the next session).", "",
           "## Standalone weekly ICs (raw features vs 30-session return)", "",
           "| feature | IC all | t | IC tradable | t | IC top-1000 | t | coverage tradable | decay 5d | 21d | 63d |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in ics.items():
        L_.append(f"| {c} | {r['all']['ic']:.4f} | {r['all']['nw_t']:.2f} | {r['tradable']['ic']:.4f} | "
                  f"{r['tradable']['nw_t']:.2f} | {r['top1000']['ic']:.4f} | {r['top1000']['nw_t']:.2f} | "
                  f"{r['tradable']['coverage']:.0%} | {r['decay'][5]:.3f} | {r['decay'][21]:.3f} | {r['decay'][63]:.3f} |")
    yrs = sorted({y for r in ics.values() for y in r["tradable"]["by_year"]})
    L_ += ["", "Per-year IC, tradable names:", "", "| feature | " + " | ".join(yrs) + " |",
           "|---|" + "---|" * len(yrs)]
    for c, r in ics.items():
        L_.append(f"| {c} | " + " | ".join(f"{r['tradable']['by_year'].get(y, float('nan')):.3f}" for y in yrs) + " |")
    md = "\n".join(L_) + "\n"
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                                "round14_lazy_data.md")
    open(path, "w").write(md)
    log(f"[lazy] report -> {path}")
    return md


if __name__ == "__main__":
    a = sys.argv[1:]
    if "index" in a:
        build_index()
    if "download" in a:
        tier = int(next((x.split("=")[1] for x in a if x.startswith("--tier=")), 3))
        hours = next((float(x.split("=")[1]) for x in a if x.startswith("--hours=")), None)
        download(max_tier=tier, max_hours=hours)
    if "features" in a:
        build_features()
    if "report" in a:
        report()
