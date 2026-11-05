"""SEC EDGAR bulk downloads + ticker<->CIK and SIC->sector mappings (WS2).

Downloads (resumable, size + sha256 logged to ``<data>/edgar/manifest.json``):
  * companyfacts.zip   - every filer's XBRL facts (input of v3.fundamentals)
  * submissions.zip    - every filer's metadata (tickers, exchanges, SIC, names, filings)
  * company_tickers_exchange.json - current ticker/CIK/exchange list

SEC fair-access policy: a User-Agent with a real contact is mandatory (otherwise
HTTP 403) and at most 10 requests/second.  The User-Agent is read ONLY from the
env var ``SEC_USER_AGENT`` (or the same key in the gitignored
``trading_agent/.env``); nothing is ever hardcoded.

Outputs (under ``$BDA_DATA_V3``, default ``.../BDA/bda_data_v3``):
  * fundamentals/cik_map.parquet   one row per (cik, ticker)
  * fundamentals/cik_names.parquet one row per (cik, name) incl. former names (name matching)
  * fundamentals/sectors.parquet   one row per cik: sic -> Fama-French 12 industry

CLI (run from DataAnalysisPipeline2/):
    python -m v3.edgar download          # the three files above
    python -m v3.edgar maps              # cik_map / cik_names / sectors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("v3.edgar")

# --------------------------------------------------------------------------- paths
DEFAULT_DATA_DIR = "/Users/manuelruckerabella/Workspace/UNI/Q6/BDA/bda_data_v3"
_AGENT_ENV = Path(__file__).resolve().parent.parent / "trading_agent" / ".env"

COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def data_dir() -> Path:
    return Path(os.environ.get("BDA_DATA_V3", DEFAULT_DATA_DIR))


def edgar_dir() -> Path:
    d = data_dir() / "edgar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fundamentals_dir() -> Path:
    d = data_dir() / "fundamentals"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- user agent
class SECUserAgentMissing(RuntimeError):
    pass


_UA_HELP = (
    "SEC EDGAR requires a User-Agent that identifies you with a real contact, e.g.\n"
    "    export SEC_USER_AGENT='Jane Doe jane.doe@example.edu'\n"
    "or add the line  SEC_USER_AGENT=Jane Doe jane.doe@example.edu  to "
    f"{_AGENT_ENV} (gitignored).  Requests without it are rejected with HTTP 403. "
    "See https://www.sec.gov/os/accessing-edgar-data"
)


def _read_env_file(path: Path, key: str) -> str | None:
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def sec_user_agent() -> str:
    """Return the SEC User-Agent from env / .env, or raise with instructions."""
    ua = os.environ.get("SEC_USER_AGENT") or _read_env_file(_AGENT_ENV, "SEC_USER_AGENT")
    ua = (ua or "").strip()
    if not ua:
        raise SECUserAgentMissing("SEC_USER_AGENT is not set.\n" + _UA_HELP)
    if "@" not in ua or len(ua.split()) < 2:
        raise SECUserAgentMissing(
            "SEC_USER_AGENT must contain a name and a contact e-mail.\n" + _UA_HELP)
    return ua


# --------------------------------------------------------------------------- http
class _RateLimiter:
    """Global <= max_per_sec request throttle (SEC fair access: 10 req/s)."""

    def __init__(self, max_per_sec: float = 8.0):
        self.min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_RATE = _RateLimiter()


def _session():
    import requests

    s = requests.Session()
    # identity encoding: byte ranges must refer to the stored bytes for resume to work
    s.headers.update({"User-Agent": sec_user_agent(), "Accept-Encoding": "identity"})
    return s


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _update_manifest(name: str, info: dict):
    mpath = edgar_dir() / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[name] = info
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def download(url: str, dest: Path | None = None, *, max_retries: int = 5,
             force: bool = False) -> Path:
    """Resumable download (HTTP Range on a ``.part`` file); logs size + sha256.

    If ``dest`` already exists with the server's Content-Length (and the server
    Last-Modified is not newer) it is kept unless ``force``.
    """
    sess = _session()  # raises SECUserAgentMissing before any network access
    dest = Path(dest) if dest else edgar_dir() / url.rsplit("/", 1)[-1]
    part = dest.with_suffix(dest.suffix + ".part")

    _RATE.wait()
    head = sess.head(url, allow_redirects=True, timeout=60)
    if head.status_code == 403:
        raise SECUserAgentMissing(f"HTTP 403 from SEC for {url}; check SEC_USER_AGENT.\n" + _UA_HELP)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0)) or None
    last_mod = head.headers.get("Last-Modified")

    if dest.exists() and not force and total and dest.stat().st_size == total:
        log.info("%s up to date (%d bytes)", dest.name, total)
        return dest

    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have > total:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        _RATE.wait()
        try:
            with sess.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 403:
                    raise SECUserAgentMissing(f"HTTP 403 from SEC for {url}.\n" + _UA_HELP)
                if have and r.status_code == 200:  # server ignored Range: restart
                    have = 0
                    mode = "wb"
                elif r.status_code in (200, 206):
                    mode = "ab" if have else "wb"
                else:
                    r.raise_for_status()
                    mode = "wb"
                with open(part, mode) as f:
                    t0, done = time.time(), have
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t0 > 15:
                            log.info("%s: %.1f MB%s", dest.name, done / 1e6,
                                     f" / {total / 1e6:.1f} MB" if total else "")
                            t0 = time.time()
            size = part.stat().st_size
            if total and size < total:
                raise IOError(f"short read {size}/{total}")
            break
        except SECUserAgentMissing:
            raise
        except Exception as e:  # network hiccup -> resume
            log.warning("download %s attempt %d failed: %s", url, attempt, e)
            if attempt == max_retries:
                raise
            time.sleep(min(60, 2 ** attempt))

    part.replace(dest)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:  # integrity check (central directory)
            n = len(z.namelist())
    else:
        n = None
    info = {"url": url, "bytes": dest.stat().st_size, "sha256": _sha256(dest),
            "server_last_modified": last_mod, "downloaded_utc": pd.Timestamp.utcnow().isoformat(),
            "zip_members": n}
    _update_manifest(dest.name, info)
    log.info("downloaded %s: %d bytes sha256=%s members=%s", dest.name, info["bytes"],
             info["sha256"][:16], n)
    return dest


def download_all(force: bool = False) -> dict[str, Path]:
    sec_user_agent()  # fail fast with instructions
    return {
        "tickers": download(TICKERS_EXCHANGE_URL, force=force),
        "companyfacts": download(COMPANYFACTS_URL, force=force),
        "submissions": download(SUBMISSIONS_URL, force=force),
    }


# --------------------------------------------------------------------------- tickers
def norm_ticker(t) -> str | None:
    """Canonical ticker: upper-case, share-class separator '.' (SEC 'BRK-B' == Alpaca 'BRK.B')."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    t = str(t).strip().upper().replace("-", ".").replace("/", ".")
    return t or None


_NAME_DROP = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LLC|LP|L P|HOLDINGS?|"
    r"GROUP|THE|NV|N V|SA|S A|AG|SE|TRUST|BANCORP)\b")


def norm_name(name) -> str | None:
    if not isinstance(name, str):
        return None
    s = name.upper().replace("&", " AND ")
    s = re.sub(r"/[A-Z]{2,3}/?$", " ", s)          # EDGAR state suffix e.g. 'ACME CORP /DE/'
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = _NAME_DROP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


PERIODIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT", "20-F", "40-F",
                  "20-F/A", "40-F/A", "10-KT/A", "10-QT/A"}


def _parse_submission(raw: bytes) -> dict | None:
    try:
        import orjson as _j
        d = _j.loads(raw)
    except ImportError:  # pragma: no cover
        d = json.loads(raw)
    rec = d.get("filings", {}).get("recent", {}) or {}
    dates = rec.get("filingDate", []) or []
    forms = rec.get("form", []) or []
    first = min(dates) if dates else None
    last = max(dates) if dates else None
    periodic = [dt for dt, f in zip(dates, forms) if f in PERIODIC_FORMS]
    for f in d.get("filings", {}).get("files", []) or []:  # older filing pages
        if f.get("filingFrom"):
            first = min(first, f["filingFrom"]) if first else f["filingFrom"]
    return {
        "cik": int(d.get("cik")),
        "name": d.get("name"),
        "entity_type": d.get("entityType"),
        "sic": d.get("sic") or None,
        "sic_description": d.get("sicDescription") or None,
        "tickers": list(d.get("tickers") or []),
        "exchanges": list(d.get("exchanges") or []),
        "former_names": list(d.get("formerNames") or []),
        "first_filing": first,
        "last_filing": last,
        "last_periodic_filing": max(periodic) if periodic else None,
        "n_periodic_recent": len(periodic),
    }


def _parse_submission_chunk(args) -> list[dict]:
    zpath, names = args
    out = []
    with zipfile.ZipFile(zpath) as z:
        for n in names:
            try:
                r = _parse_submission(z.read(n))
            except Exception as e:  # malformed member: skip, keep going
                log.warning("skip %s: %s", n, e)
                continue
            if r is not None:
                out.append(r)
    return out


def load_submissions(zpath: Path, *, keep_ciks: set[int] | None = None,
                     workers: int = 8) -> list[dict]:
    """Parse primary ``CIK##########.json`` members of submissions.zip.

    Keeps entities that have a ticker, any periodic filing, or are in ``keep_ciks``
    (e.g. every CIK of companyfacts) -- individuals / funds are dropped.
    """
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"CIK\d{10}\.json", n)]
    chunks = [(str(zpath), names[i:i + 5000]) for i in range(0, len(names), 5000)]
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_parse_submission_chunk, chunks):
            for r in part:
                if r["tickers"] or r["n_periodic_recent"] or (keep_ciks and r["cik"] in keep_ciks):
                    rows.append(r)
    return rows


def build_cik_map(submissions: list[dict], tickers_exchange: dict | None,
                  *, snapshot_date=None, delist_grace_days: int = 120) -> pd.DataFrame:
    """One row per (cik, ticker).

    Columns: cik, ticker (canonical, '.' class separator), ticker_raw, exchange, name,
    sic, sic_description, entity_type, first_filing, last_filing, last_periodic_filing,
    in_current_list, valid_from, valid_to, ticker_ambiguous, n_ciks_for_ticker.

    Validity (SEC does not publish ticker history, so this is approximate):
      * valid_from = first EDGAR filing of the CIK (the ticker may have changed since;
        not derivable)
      * valid_to   = NaT (open) if the (cik, ticker) pair is in the current
        company_tickers_exchange.json; otherwise last periodic filing + grace days
        (a delisted company stops filing 10-K/10-Q shortly after delisting).
    ``ticker_ambiguous``: same canonical ticker claimed by >1 CIK (reused symbols or
    multiple share-class entities); resolve with :func:`resolve_cik` (date-aware).
    """
    current: dict[tuple[int, str], str | None] = {}
    if tickers_exchange:
        f = tickers_exchange["fields"]
        for row in tickers_exchange["data"]:
            r = dict(zip(f, row))
            current[(int(r["cik"]), norm_ticker(r["ticker"]))] = r.get("exchange")

    rows = []
    for s in submissions:
        exch = s["exchanges"]
        for i, t in enumerate(s["tickers"]):
            nt = norm_ticker(t)
            if not nt:
                continue
            rows.append({"cik": s["cik"], "ticker": nt, "ticker_raw": t,
                         "exchange": exch[i] if i < len(exch) else None})
    known = {(r["cik"], r["ticker"]) for r in rows}
    for (cik, t), ex in current.items():  # current list entries missing from submissions
        if (cik, t) not in known:
            rows.append({"cik": cik, "ticker": t, "ticker_raw": t, "exchange": ex})
    df = pd.DataFrame(rows, columns=["cik", "ticker", "ticker_raw", "exchange"])
    meta = pd.DataFrame([{k: s[k] for k in ("cik", "name", "sic", "sic_description", "entity_type",
                                            "first_filing", "last_filing", "last_periodic_filing")}
                         for s in submissions])
    if meta.empty:
        meta = pd.DataFrame(columns=["cik", "name", "sic", "sic_description", "entity_type",
                                     "first_filing", "last_filing", "last_periodic_filing"])
    df = df.merge(meta, on="cik", how="left")
    for c in ("first_filing", "last_filing", "last_periodic_filing"):
        df[c] = pd.to_datetime(df[c])
    df["in_current_list"] = [(c, t) in current for c, t in zip(df["cik"], df["ticker"])]
    df["valid_from"] = df["first_filing"]
    end = df["last_periodic_filing"].fillna(df["last_filing"]) + pd.Timedelta(days=delist_grace_days)
    df["valid_to"] = end.where(~df["in_current_list"], pd.NaT)
    df["n_ciks_for_ticker"] = df.groupby("ticker")["cik"].transform("nunique")
    df["ticker_ambiguous"] = df["n_ciks_for_ticker"] > 1
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce").astype("Int64")
    return df.drop_duplicates(["cik", "ticker"]).sort_values(["ticker", "valid_from"]).reset_index(drop=True)


def build_cik_names(submissions: list[dict]) -> pd.DataFrame:
    """(cik, name, name_norm, name_from, name_to, is_current) incl. formerNames."""
    rows = []
    for s in submissions:
        rows.append({"cik": s["cik"], "name": s["name"], "name_from": None, "name_to": None,
                     "is_current": True})
        for fn in s["former_names"]:
            rows.append({"cik": s["cik"], "name": fn.get("name"), "name_from": fn.get("from"),
                         "name_to": fn.get("to"), "is_current": False})
    df = pd.DataFrame(rows, columns=["cik", "name", "name_from", "name_to", "is_current"])
    df["name_norm"] = df["name"].map(norm_name)
    for c in ("name_from", "name_to"):
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def match_assets_to_cik(assets: pd.DataFrame, cik_map: pd.DataFrame,
                        cik_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map broker assets (columns ``ticker`` and optionally ``name``) to CIKs.

    1. exact canonical-ticker match (unique CIK) -> method 'ticker'
    2. ambiguous ticker: prefer a CIK whose normalized (current/former) name matches
    3. no ticker match: unique normalized-name match -> method 'name'
       (this is how delisted names, whose SEC ``tickers`` list is usually empty, get mapped)
    Returns assets + cik, map_method.
    """
    a = assets.copy()
    a["tkey"] = a["ticker"].map(norm_ticker)
    a["nkey"] = a["name"].map(norm_name) if "name" in a else None
    by_t = cik_map.groupby("ticker")["cik"].agg(lambda s: sorted(set(s))).to_dict()
    by_n: dict[str, set] = {}
    names_of: dict[int, set] = {}
    if cik_names is not None:
        for c, n in zip(cik_names["cik"], cik_names["name_norm"]):
            if n:
                by_n.setdefault(n, set()).add(int(c))
                names_of.setdefault(int(c), set()).add(n)
    ciks, methods = [], []
    for t, n in zip(a["tkey"], a["nkey"]):
        cands = by_t.get(t, [])
        if len(cands) == 1:
            ciks.append(cands[0]); methods.append("ticker"); continue
        if len(cands) > 1:
            hit = [c for c in cands if n and n in names_of.get(c, ())]
            if len(hit) == 1:
                ciks.append(hit[0]); methods.append("ticker+name"); continue
            ciks.append(None); methods.append("ambiguous"); continue
        hit = by_n.get(n, set()) if n else set()
        if len(hit) == 1:
            ciks.append(next(iter(hit))); methods.append("name"); continue
        ciks.append(None); methods.append("unmapped" if not hit else "ambiguous_name")
    a["cik"] = pd.array(ciks, dtype="Int64")
    a["map_method"] = methods
    return a.drop(columns=["tkey", "nkey"])


# --------------------------------------------------------------------------- sectors
# Fama-French 12 industries (Ken French, "Siccodes12"); inclusive SIC ranges.
# Anything not listed -> 12 "Other".
FF12_NAMES = {
    1: ("NoDur", "Consumer Non-Durables"),
    2: ("Durbl", "Consumer Durables"),
    3: ("Manuf", "Manufacturing"),
    4: ("Enrgy", "Energy (Oil, Gas, Coal)"),
    5: ("Chems", "Chemicals"),
    6: ("BusEq", "Business Equipment (Computers, Software, Electronics)"),
    7: ("Telcm", "Telecommunications"),
    8: ("Utils", "Utilities"),
    9: ("Shops", "Wholesale, Retail, Services"),
    10: ("Hlth", "Healthcare, Medical Equipment, Drugs"),
    11: ("Money", "Finance"),
    12: ("Other", "Other"),
}
FF12_SIC_RANGES: dict[int, list[tuple[int, int]]] = {
    1: [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    2: [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
        (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    3: [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
        (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
        (3830, 3839), (3860, 3899)],
    4: [(1200, 1399), (2900, 2999)],
    5: [(2800, 2829), (2840, 2899)],
    6: [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    7: [(4800, 4899)],
    8: [(4900, 4949)],
    9: [(5000, 5999), (7200, 7299), (7600, 7699)],
    10: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    11: [(6000, 6999)],
}


def sic_to_ff12(sic) -> int | None:
    """SIC code -> FF12 industry id (1..12); None if SIC unknown/0."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    for k, ranges in FF12_SIC_RANGES.items():
        for lo, hi in ranges:
            if lo <= s <= hi:
                return k
    return 12


def build_sectors(submissions: list[dict]) -> pd.DataFrame:
    """One row per CIK (active or delisted): cik, sic, sic_description, ff12, ff12_code, sector_name.

    NOTE: EDGAR SIC is the *current* assignment (not point-in-time); SIC changes are rare.
    """
    rows = []
    for s in submissions:
        sic = pd.to_numeric(s.get("sic"), errors="coerce")
        k = sic_to_ff12(sic) if pd.notna(sic) else None
        rows.append({"cik": s["cik"], "sic": None if pd.isna(sic) else int(sic),
                     "sic_description": s.get("sic_description"), "ff12": k,
                     "ff12_code": FF12_NAMES[k][0] if k else None,
                     "sector_name": FF12_NAMES[k][1] if k else None})
    df = pd.DataFrame(rows, columns=["cik", "sic", "sic_description", "ff12", "ff12_code", "sector_name"])
    df["sic"] = df["sic"].astype("Int64")
    df["ff12"] = df["ff12"].astype("Int64")
    return df.drop_duplicates("cik").reset_index(drop=True)


def companyfacts_ciks(zpath: Path) -> set[int]:
    with zipfile.ZipFile(zpath) as z:
        return {int(m.group(1)) for n in z.namelist() if (m := re.fullmatch(r"CIK(\d{10})\.json", n))}


def build_maps(workers: int = 8) -> dict[str, Path]:
    ed, fd = edgar_dir(), fundamentals_dir()
    sub_zip = ed / "submissions.zip"
    cf_zip = ed / "companyfacts.zip"
    tick_json = ed / "company_tickers_exchange.json"
    if not sub_zip.exists():
        raise FileNotFoundError(f"{sub_zip} missing - run `python -m v3.edgar download` first")
    keep = companyfacts_ciks(cf_zip) if cf_zip.exists() else None
    subs = load_submissions(sub_zip, keep_ciks=keep, workers=workers)
    tex = json.loads(tick_json.read_text()) if tick_json.exists() else None
    cmap = build_cik_map(subs, tex)
    names = build_cik_names(subs)
    sectors = build_sectors(subs)
    out = {"cik_map": fd / "cik_map.parquet", "cik_names": fd / "cik_names.parquet",
           "sectors": fd / "sectors.parquet"}
    cmap.to_parquet(out["cik_map"], index=False)
    names.to_parquet(out["cik_names"], index=False)
    sectors.to_parquet(out["sectors"], index=False)
    log.info("cik_map: %d rows, %d tickers (%d ambiguous), %d ciks; sectors: %d ciks",
             len(cmap), cmap.ticker.nunique(), cmap.loc[cmap.ticker_ambiguous, "ticker"].nunique(),
             cmap.cik.nunique(), len(sectors))
    return out


def main(argv: Iterable[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "maps", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd in ("download", "all"):
        download_all(force=a.force)
    if a.cmd in ("maps", "all"):
        build_maps(workers=a.workers)


if __name__ == "__main__":
    main()
