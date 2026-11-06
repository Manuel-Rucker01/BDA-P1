"""Daily Form 4 feed: index + XML parser, rate limiter, retries, schema equality
with the bulk parser (synthetic); parity with the bulk files and live == offline
features on real data (skipped when the data store is absent)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import altdata as ad
from DataAnalysisPipeline2.v3 import form4 as f4
from DataAnalysisPipeline2.v3.tests.test_altdata import _insider_zip

XML = """<SEC-DOCUMENT><DOCUMENT><TYPE>4<TEXT><XML>
<?xml version="1.0"?>
<ownershipDocument><documentType>{doc}</documentType>
 <issuer><issuerCik>0000000001</issuerCik></issuer>
 <reportingOwner><reportingOwnerId><rptOwnerCik>0000000011</rptOwnerCik></reportingOwnerId>
  <reportingOwnerRelationship><isDirector>{d}</isDirector><isOfficer>0</isOfficer></reportingOwnerRelationship></reportingOwner>
 <reportingOwner><reportingOwnerId><rptOwnerCik>0000000099</rptOwnerCik></reportingOwnerId></reportingOwner>
 <nonDerivativeTable>
  <nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
   <transactionAmounts><transactionShares><value>100</value></transactionShares>
   <transactionPricePerShare><value>10</value></transactionPricePerShare></transactionAmounts></nonDerivativeTransaction>
  <nonDerivativeTransaction><transactionCoding><transactionCode>M</transactionCode></transactionCoding>
   <transactionAmounts><transactionShares><value>50</value></transactionShares></transactionAmounts></nonDerivativeTransaction>
  <nonDerivativeTransaction><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
   <transactionAmounts><transactionShares><value>1,000</value></transactionShares></transactionAmounts></nonDerivativeTransaction>
 </nonDerivativeTable></ownershipDocument>
</XML></TEXT></DOCUMENT></SEC-DOCUMENT>"""


def test_parse_form4_schema_codes_owner_and_lag():
    ev = f4.parse_form4(XML.format(doc="4", d="true"), "acc-1", "2025-01-03")
    assert list(ev.columns) == f4.EVENT_COLUMNS
    assert list(ev["code"]) == ["P", "S"]                          # M dropped
    assert (ev["owner_cik"] == 11).all() and ev["is_od"].all()     # primary (first) owner
    assert ev["value"].tolist() == [1000.0, 0.0]                   # missing price -> 0 dollars
    assert ev["shares"].iloc[1] == 1000.0                          # thousands separator
    assert (ev["avail_date"] == pd.Timestamp("2025-01-06")).all()  # filed Fri -> Mon
    assert f4.parse_form4(XML.format(doc="4/A", d="1"), "a", "2025-01-03").empty   # amendment excluded
    assert f4.parse_form4("<html>broken", "a", "2025-01-03").empty


def test_bulk_and_daily_parsers_agree_on_the_same_filing():
    bulk = ad.parse_insider_zip(_insider_zip())
    b = bulk[bulk["accession"] == "a1"].reset_index(drop=True)
    xml = XML.format(doc="4", d="true").replace("<transactionCode>S</transactionCode>", "<transactionCode>X</transactionCode>")
    d = f4.parse_form4(xml, "a1", "2025-01-02")
    cols = ["accession", "issuer_cik", "owner_cik", "is_od", "code", "shares", "price", "value", "filing_date",
            "avail_date"]
    num = ["issuer_cik", "owner_cik", "shares", "price", "value"]
    pd.testing.assert_frame_equal(b[num].astype(float), d[num].astype(float))
    other = [c for c in cols if c not in num]
    pd.testing.assert_frame_equal(b[other].astype(str), d[other].astype(str))


def test_index_parsing_and_issuer_filter():
    txt = ("Description: Master Index\n\nCIK|Company Name|Form Type|Date Filed|File Name\n"
           "--------\n"
           "1|ISSUER A|4|20250103|edgar/data/1/0000000011-25-000001.txt\n"
           "11|OWNER X|4|20250103|edgar/data/11/0000000011-25-000001.txt\n"
           "2|ISSUER B|4/A|20250103|edgar/data/2/0000000012-25-000002.txt\n"
           "3|ISSUER C|8-K|20250103|edgar/data/3/0000000013-25-000003.txt\n"
           "77|OWNER Y|5|20250103|edgar/data/77/0000000077-25-000004.txt\n")
    idx = f4.parse_index(txt)
    assert set(idx["form"]) == {"4", "5"}
    s = f4.select_filings(idx, {1, 2, 3})
    assert list(s["accession"]) == ["0000000011-25-000001"]       # once, despite two index lines
    assert len(f4.select_filings(idx, None)) == 2


def test_limiter_spacing():
    t = {"now": 0.0}
    sleeps = []
    lim = f4.Limiter(rps=8, clock=lambda: t["now"], sleep=lambda s: sleeps.append(s))
    for _ in range(4):
        lim.wait()
    assert np.allclose(sleeps, [0.125, 0.25, 0.375])               # <= 8 requests / s


class _R:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


class _S:
    def __init__(self, codes):
        self.codes = list(codes)

    def get(self, url, timeout=None):
        return _R(self.codes.pop(0), "ok")


def test_client_retries_then_succeeds_and_404_is_none():
    c = f4.SECClient(session=_S([429, 503, 200]), limiter=f4.Limiter(1e6), sleep=lambda s: None)
    assert c.get("u") == "ok" and c.n == 3
    c = f4.SECClient(session=_S([404]), limiter=f4.Limiter(1e6), sleep=lambda s: None)
    assert c.get("u") is None
    c = f4.SECClient(session=_S([500] * 5), limiter=f4.Limiter(1e6), sleep=lambda s: None, retries=5)
    with pytest.raises(RuntimeError):
        c.get("u")


# --------------------------------------------------------------- real data
PARITY_DAYS = ("2026-03-02", "2026-03-03")


def _have(*paths):
    return all(os.path.exists(p) for p in paths)


@pytest.mark.skipif(not _have(*[os.path.join(f4.cache_dir(), f"{d}.parquet") for d in PARITY_DAYS]),
                    reason="parity days not ingested")
def test_daily_feed_matches_bulk_on_overlap():
    from DataAnalysisPipeline2.v3 import altlive
    rep = altlive.parity(PARITY_DAYS)
    assert rep["bulk_rows"] > 100
    # documented tolerance: identical rows (after the bulk rounding convention);
    # only the primary-owner choice of multi-owner filings may differ
    assert rep["share_bulk_found_in_daily"] >= 0.995
    assert rep["share_daily_found_in_bulk"] >= 0.995
    assert rep["field_mismatch_share"] == 0.0
    assert rep["owner_mismatch_share"] <= 0.05 and rep["officer_director_flag_mismatch_share"] <= 0.01


@pytest.mark.skipif(not os.path.exists(os.path.join(ad.data_dir(), "v3model", "weekly_dataset_r8.parquet")),
                    reason="round-8 frame not built")
def test_live_features_equal_offline_frame():
    from DataAnalysisPipeline2.v3 import altlive
    d = "2025-06-27"
    live, off = altlive.offline_vs_live(d)
    pd.testing.assert_frame_equal(live, off, check_exact=False, rtol=1e-5, atol=1e-5)


def test_ingest_day_survives_unavailable_index(monkeypatch, tmp_path):
    monkeypatch.setattr(f4, "cache_dir", lambda: str(tmp_path))
    c = f4.SECClient(session=_S([403] * 5), limiter=f4.Limiter(1e6), sleep=lambda s: None, retries=5)
    assert f4.ingest_day("2026-05-25", c, None, log=lambda *a: None) == -3
    assert not list(tmp_path.iterdir())                            # nothing cached -> retried later


def test_ingest_day_retries_only_failed_filings(monkeypatch, tmp_path):
    monkeypatch.setattr(f4, "cache_dir", lambda: str(tmp_path))
    idx_txt = ("CIK|Company Name|Form Type|Date Filed|File Name\n"
               "1|A|4|20250103|edgar/data/1/0000000011-25-000001.txt\n"
               "1|A|4|20250103|edgar/data/1/0000000011-25-000002.txt\n")
    calls = {"n": 0, "fail": True}

    class Sess:
        def get(self, url, timeout=None):
            calls["n"] += 1
            if url.endswith(".idx"):
                return _R(200, idx_txt)
            if url.endswith("000002.txt") and calls["fail"]:
                return _R(500)
            return _R(200, XML.format(doc="4", d="true"))

    c = f4.SECClient(session=Sess(), limiter=f4.Limiter(1e6), sleep=lambda s: None, retries=2)
    assert f4.ingest_day("2025-01-03", c, None, workers=1, log=lambda *a: None) == 2
    assert (tmp_path / "2025-01-03.failed.json").exists()
    calls["fail"], before = False, calls["n"]
    assert f4.ingest_day("2025-01-03", c, None, workers=1, log=lambda *a: None) == 4
    assert calls["n"] - before == 1                                  # only the failed filing re-fetched
    assert not (tmp_path / "2025-01-03.failed.json").exists()
    assert f4.ingest_day("2025-01-03", c, None, log=lambda *a: None) == -1
