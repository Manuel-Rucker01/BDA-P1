"""SEC insider / FINRA short-interest parsers and point-in-time lags (synthetic, no network)."""
import io
import zipfile

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import altdata as ad


def _zip(sub, own, trans) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, df in (("SUBMISSION.tsv", sub), ("REPORTINGOWNER.tsv", own), ("NONDERIV_TRANS.tsv", trans)):
            z.writestr(name, df.to_csv(sep="\t", index=False))
    buf.seek(0)
    return buf


def _insider_zip():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2", "a3", "a4", "a5"],
                        "FILING_DATE": ["02-JAN-2025", "03-JAN-2025", "03-JAN-2025", "10-JAN-2025", "11-JAN-2025"],
                        "ISSUERCIK": ["0000000001"] * 4 + ["0000000002"], "DOCUMENT_TYPE": ["4", "4", "4", "4/A", "4"]})
    own = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a2", "a3", "a4", "a5", "a1"],
                        "RPTOWNERCIK": ["11", "12", "13", "11", "14", "99"],
                        "RPTOWNER_RELATIONSHIP": ["Director", "Officer", "TenPercentOwner", "Director", "Other",
                                                  "Director"]})
    tr = pd.DataFrame({"ACCESSION_NUMBER": ["a1", "a1", "a2", "a3", "a4", "a5"],
                       "TRANS_CODE": ["P", "M", "P", "S", "P", "P"],
                       "TRANS_SHARES": ["100", "50", "200", "1000", "100", "10"],
                       "TRANS_PRICEPERSHARE": ["10", "1", "10", "20", "10", ""],
                       "TRANS_ACQUIRED_DISP_CD": ["A", "A", "A", "D", "A", "A"]})
    return _zip(sub, own, tr)


def test_parse_insider_zip_codes_owners_amendments_and_lag():
    e = ad.parse_insider_zip(_insider_zip())
    assert set(e["code"]) == {"P", "S"}                              # M (option exercise) dropped
    assert "a4" not in set(e["accession"])                           # 4/A amendment excluded
    a1 = e[e["accession"] == "a1"].iloc[0]
    assert a1["owner_cik"] == 11 and bool(a1["is_od"])               # primary owner only
    assert a1["value"] == 1000.0
    # filed Thu 2025-01-02 -> available Fri 01-03; filed Fri 01-03 -> Mon 01-06
    assert (e.set_index("accession").loc["a1", "avail_date"] == pd.Timestamp("2025-01-03"))
    assert (e.set_index("accession").loc["a2", "avail_date"] == pd.Timestamp("2025-01-06"))
    assert e.set_index("accession").loc["a5", "value"] == 0.0         # missing price -> 0 dollars


def test_insider_features_windows_and_pit():
    e = ad.parse_insider_zip(_insider_zip())
    f = ad.insider_features(e, ["2025-01-03", "2025-01-06", "2025-03-20", "2025-09-01"]).set_index(["issuer_cik", "Date"])
    # 01-03: only a1 (filed 01-02) is available; a2/a3 filed 01-03 are not yet
    r = f.loc[(1, "2025-01-03")]
    assert r["ins_buy_n_30"] == 1 and r["ins_sell_n_30"] == 0
    r = f.loc[(1, "2025-01-06")]
    assert r["ins_buy_n_30"] == 2 and r["ins_sell_n_30"] == 1 and r["ins_n_buyers_90"] == 2
    assert np.isclose(r["ins_net_ratio_90"], (3000 - 20000) / 23000)
    assert r["ins_od_buy_90"] == 1 and r["ins_cluster_30"] == 0
    r = f.loc[(1, "2025-03-20")]                                     # > 30 days later: only 90/180 windows
    assert r["ins_buy_n_30"] == 0 and r["ins_buy_n_90"] == 2
    assert (1, "2025-09-01") not in f.index                         # everything aged out of 180 days


def test_cluster_buy():
    ev = pd.DataFrame({"accession": ["x1", "x2", "x3"], "issuer_cik": [5, 5, 5],
                       "avail_date": pd.to_datetime(["2025-02-03", "2025-02-10", "2025-02-20"]),
                       "owner_cik": [1, 2, 3], "is_od": [False] * 3, "code": ["P"] * 3, "value": [1.0] * 3})
    f = ad.insider_features(ev, ["2025-02-19", "2025-02-21"]).set_index("Date")
    assert f.loc["2025-02-19", "ins_cluster_30"] == 0 and f.loc["2025-02-21", "ins_cluster_30"] == 1


def _si_csv(settle, rows):
    head = ("accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|"
            "currentShortPositionQuantity|previousShortPositionQuantity|stockSplitFlag|averageDailyVolumeQuantity|"
            "daysToCoverQuantity|revisionFlag|changePercent|changePreviousNumber|settlementDate")
    lines = [f"{settle.replace('-', '')}|{s}|x|A|{m}|{si}|{prev}||1000|{dtc}||0|0|{settle}" for s, m, si, prev, dtc in rows]
    return io.StringIO("\n".join([head] + lines))


def test_parse_si_lag_and_exchange_filter():
    s = ad.parse_si_csv(_si_csv("2026-01-15", [("AAA", "NYSE", 500, 400, 2.5), ("OTCX", "OTC", 9, 9, 1),
                                                ("BRK/B", "NYSE", 10, 10, 1)]))
    assert set(s["symbol"]) == {"AAA", "BRK.B"}
    assert s["avail_date"].iloc[0] == pd.Timestamp("2026-01-28")    # +8 NYSE business days (MLK 01-19)


def test_si_features_pit_and_symbol_reuse_guard():
    si = pd.concat([ad.parse_si_csv(_si_csv("2026-01-15", [("AAA", "NYSE", 500, 400, 2.5)])),
                    ad.parse_si_csv(_si_csv("2026-01-30", [("AAA", "NYSE", 800, 500, 3.0)]))])
    rows = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "AAA~1"],
                         "Date": ["2026-01-27", "2026-01-28", "2026-02-13", "2026-02-13"],
                         "first_date": ["2020-01-02", "2020-01-02", "2020-01-02", "2026-02-01"],
                         "shares": [1e4, 1e4, 1e4, 1e4]})
    f = ad.si_features(si, rows)
    assert f.iloc[0].isna().all()                                    # nothing published yet
    assert np.isclose(f.iloc[1]["si_ratio"], 0.05) and np.isclose(f.iloc[1]["si_chg"], 0.25)
    assert np.isclose(f.iloc[2]["si_ratio"], 0.08) and f.iloc[2]["si_dtc"] == 3.0   # 01-30 settle avail 02-11
    assert f.iloc[3].isna().all()                                    # settled before this listing began
