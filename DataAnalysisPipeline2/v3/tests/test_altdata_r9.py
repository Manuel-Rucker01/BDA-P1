"""SEC 13F / fails-to-deliver parsers, point-in-time rules and no-lookahead
checks (synthetic, no network)."""
import io
import zipfile

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import altdata_r9 as a9


# --------------------------------------------------------------------------- #
# Fails-to-deliver
# --------------------------------------------------------------------------- #
def _ftd_zip(lines, inner="cnsfails202501a"):
    buf = io.BytesIO()
    head = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(inner, "\n".join([head] + lines + ["Trailer record count 2", "Trailer total quantity of shares 9"]))
    buf.seek(0)
    return buf


def test_parse_ftd_zip_trailer_and_symbols():
    f = a9.parse_ftd_zip(_ftd_zip(["20250102|084670702|BRKB|95|BERKSHIRE B|309.9",
                                   "20250103|12345678X|AAA|1000|AAA INC|1.0",
                                   "20250103|BADCUSIP|BBB|5|bad|1"]), "cnsfails202501a.zip")
    assert list(f["symbol"]) == ["BRKB", "AAA"]                       # trailer + 8-char CUSIP dropped
    assert f["qty"].sum() == 1095 and f["file"].iloc[0] == "cnsfails202501a.zip"
    assert list(a9.sym_key(pd.Series(["BRK.B", "AAA~1", "brkb"]))) == ["BRKB", "AAA", "BRKB"]


def test_ftd_period_and_availability():
    assert a9.ftd_period("cnsfails202502b.zip") == (pd.Timestamp("2025-02-16"), pd.Timestamp("2025-02-28"))
    # 2025-01-15 + 21 days = Wed 02-05 -> available Thu 02-06
    assert a9.ftd_avail("cnsfails202501a.zip") == pd.Timestamp("2025-02-06")
    # a later observed posting date moves it later; earlier one does not move it earlier
    assert a9.ftd_avail("cnsfails202501a.zip", "Mon, 10 Feb 2025 14:00:00 GMT") == pd.Timestamp("2025-02-11")
    assert a9.ftd_avail("cnsfails202501a.zip", "Thu, 30 Jan 2025 14:00:00 GMT") == pd.Timestamp("2025-02-06")
    # Last-Modified long after the period = re-upload -> ignored
    assert a9.ftd_avail("cnsfails202501a.zip", "Sat, 19 Dec 2026 09:31:24 GMT") == pd.Timestamp("2025-02-06")


def _ftd_records():
    rows = []
    # file a: Jan 1-15 2025 (3 settlement days), file b: Jan 16-31 (2 days), file c: Feb 1-14 (2 days)
    for d in ("2025-01-02", "2025-01-03", "2025-01-06"):
        rows.append((d, "C0000000A", "AAA", 100.0, "cnsfails202501a.zip"))
    rows.append(("2025-01-06", "C0000000Z", "ZZZ", 5.0, "cnsfails202501a.zip"))
    for d in ("2025-01-21", "2025-01-22"):
        rows.append((d, "C0000000B", "BBB", 50.0, "cnsfails202501b.zip"))
    rows.append(("2025-01-22", "C0000000A", "AAA", 400.0, "cnsfails202501b.zip"))
    for d in ("2025-02-03", "2025-02-04"):
        rows.append((d, "C0000000A", "AAA", 1000.0, "cnsfails202502a.zip"))
    f = pd.DataFrame(rows, columns=["settlement_date", "cusip", "symbol", "qty", "file"])
    f["settlement_date"] = pd.to_datetime(f["settlement_date"])
    f["price"] = 1.0
    f["avail_date"] = f["file"].map(a9.ftd_avail)
    return f


def test_ftd_weekly_pit_and_values():
    ftd = _ftd_records()
    # availability: a -> 2025-02-06, b -> 2025-02-24 (01-31 + 21 = 02-21 Fri -> Mon 02-24), c -> 03-10
    rows = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "BBB", "QQQ", "AAA~1"],
                         "Date": ["2025-02-05", "2025-02-06", "2025-02-24", "2025-02-24", "2025-02-24", "2025-02-24"],
                         "first_date": ["2010-01-04"] * 5 + ["2025-02-01"], "shares": [1e4] * 6})
    f = a9.ftd_weekly(rows, ftd)
    assert f.iloc[0].isna().all()                                    # nothing published yet
    assert np.isclose(f.iloc[1]["ftd_ratio"], 100 / 1e4)             # 300 shares / 3 days
    assert np.isclose(f.iloc[1]["ftd_persist_3m"], 1.0)
    # 02-24: files a+b available; latest = b (2 settlement days: 400 on one day)
    assert np.isclose(f.iloc[2]["ftd_ratio"], 400 / 2 / 1e4)
    assert np.isclose(f.iloc[2]["ftd_ratio_3m"], 700 / 5 / 1e4)
    assert np.isclose(f.iloc[2]["ftd_persist_3m"], 4 / 5)
    assert np.isclose(f.iloc[3]["ftd_ratio"], 100 / 2 / 1e4) and f.iloc[3]["ftd_mapped"] == 1
    assert f.iloc[4]["ftd_mapped"] == 0 and np.isnan(f.iloc[4]["ftd_ratio"])   # never failed -> unmapped
    assert f.iloc[5]["ftd_mapped"] == 0                              # records predate this listing (reuse guard)


def test_ftd_weekly_no_lookahead():
    """Changing or adding records in files not yet available on t leaves t's features unchanged."""
    ftd = _ftd_records()
    rows = pd.DataFrame({"ticker": ["AAA", "BBB", "ZZZ"], "Date": ["2025-02-24"] * 3,
                         "first_date": ["2010-01-04"] * 3, "shares": [1e4] * 3})
    f0 = a9.ftd_weekly(rows, ftd)
    fut = ftd.copy()
    fut.loc[fut["file"] == "cnsfails202502a.zip", "qty"] *= 1000
    extra = pd.DataFrame({"settlement_date": pd.to_datetime(["2025-02-05"]), "cusip": ["C0000000Q"],
                          "symbol": ["ZZZ"], "qty": [1e9], "file": ["cnsfails202502a.zip"], "price": [1.0]})
    extra["avail_date"] = extra["file"].map(a9.ftd_avail)
    f1 = a9.ftd_weekly(rows, pd.concat([fut, extra], ignore_index=True))
    pd.testing.assert_frame_equal(f0, f1)


# --------------------------------------------------------------------------- #
# 13F
# --------------------------------------------------------------------------- #
def _f13_zip(sub, cov, info, folder=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, df in (("SUBMISSION.tsv", sub), ("COVERPAGE.tsv", cov), ("INFOTABLE.tsv", info)):
            z.writestr(folder + name, df.to_csv(sep="\t", index=False))
    buf.seek(0)
    return buf


def test_parse_13f_zip_filters():
    sub = pd.DataFrame({"ACCESSION_NUMBER": ["h1", "h2", "a1", "n1"],
                        "FILING_DATE": ["10-FEB-2025", "14-FEB-2025", "20-FEB-2025", "10-FEB-2025"],
                        "SUBMISSIONTYPE": ["13F-HR", "13F-HR", "13F-HR/A", "13F-NT"],
                        "CIK": ["0000000011", "0000000012", "0000000011", "0000000013"],
                        "PERIODOFREPORT": ["31-DEC-2024"] * 4})
    cov = pd.DataFrame({"ACCESSION_NUMBER": ["h1", "h2", "a1", "n1"], "ISAMENDMENT": ["", "", "Y", ""],
                        "REPORTTYPE": ["13F HOLDINGS REPORT", "13F COMBINATION REPORT", "13F HOLDINGS REPORT",
                                       "13F NOTICE"]})
    info = pd.DataFrame({"ACCESSION_NUMBER": ["h1", "h1", "h1", "h1", "h2", "a1"],
                         "CUSIP": ["C0000000A", "C0000000A", "C0000000A", "C0000000B", "c0000000a", "C0000000A"],
                         "SSHPRNAMT": ["100", "50", "999", "10", "7", "5000"],
                         "SSHPRNAMTTYPE": ["SH", "SH", "SH", "PRN", "SH", "SH"],
                         "PUTCALL": ["", "", "Put", "", "", ""]})
    f, h = a9.parse_13f_zip(_f13_zip(sub, cov, info, folder="nested/"))
    assert set(f.loc[f["is_original"], "accession"]) == {"h1", "h2"}     # amendment + notice excluded
    h = h.set_index(["accession", "cusip"])["shares"]
    assert h.loc[("h1", "C0000000A")] == 150                            # put + PRN rows excluded, rows summed
    assert h.loc[("h2", "C0000000A")] == 7 and len(h) == 2


def test_deadline_and_select_filings():
    # Q4 2025: 12-31 + 45 = Sat 2026-02-14 -> Mon 02-16 is Presidents' Day -> deadline Tue 02-17, usable 02-18
    assert a9.f13_deadline(pd.Timestamp("2025-12-31")).iloc[0] == pd.Timestamp("2026-02-17")
    assert a9.f13_avail(pd.Timestamp("2025-12-31")).iloc[0] == pd.Timestamp("2026-02-18")
    assert a9.f13_avail(pd.Timestamp("2025-03-31")).iloc[0] == pd.Timestamp("2025-05-16")   # Thu deadline
    f = pd.DataFrame({"accession": ["x1", "x2", "x3", "x4"], "cik": [1, 1, 2, 3],
                      "filing_date": pd.to_datetime(["2025-04-20", "2025-05-01", "2025-05-15", "2025-05-16"]),
                      "period": pd.to_datetime(["2025-03-31"] * 4), "is_original": [True] * 4})
    s = a9.select_filings(f)
    assert list(s["accession"]) == ["x1", "x3"]                          # earliest per filer; late filer out
    assert (s["avail_date"] == pd.Timestamp("2025-05-16")).all()


def _links_ftd():
    f = pd.DataFrame({"settlement_date": pd.to_datetime(["2024-11-04", "2025-03-03", "2025-04-01", "2025-06-02",
                                                        "2025-03-03", "2025-03-03"]),
                      "cusip": ["OLDCUSIP1", "NEWCUSIP1", "NEWCUSIP1", "FUTCUSIP1", "BBBCUSIP1", "BBBCUSIP1"],
                      "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB"], "qty": 1.0, "price": 1.0, "file": "x"})
    f["avail_date"] = f["settlement_date"] + pd.Timedelta(days=25)
    return f


def test_cusip_links_point_in_time():
    ftd = _links_ftd()
    tick = pd.DataFrame({"ticker": ["AAA", "BBB~1"], "first_date": ["2000-01-03", "2025-04-01"]})
    lk = a9.cusip_links(ftd, tick, as_of=pd.Timestamp("2025-05-16"), ref=pd.Timestamp("2025-03-31")).set_index("ticker")
    assert lk.loc["AAA", "cusip"] == "NEWCUSIP1"                        # latest record <= quarter end
    assert "BBB~1" not in lk.index                                      # records predate this listing
    lk2 = a9.cusip_links(ftd, tick, as_of=pd.Timestamp("2025-03-01"), ref=pd.Timestamp("2024-12-31")).set_index("ticker")
    assert lk2.loc["AAA", "cusip"] == "OLDCUSIP1"                       # NEW/FUT not yet published


def _holdings_setup():
    # ticker AAA (cusip CA), BBB (cusip CB), CCC linked but nobody holds it; DDD never in FTD (unlinked)
    ftd = pd.DataFrame({"settlement_date": pd.to_datetime(["2024-06-03"] * 3 + ["2025-03-03"]),
                        "cusip": ["CA0000001", "CB0000001", "CC0000001", "CA0000001"],
                        "symbol": ["AAA", "BBB", "CCC", "AAA"], "qty": 1.0, "price": 1.0, "file": "x"})
    ftd["avail_date"] = ftd["settlement_date"] + pd.Timedelta(days=20)
    filings = pd.DataFrame({"accession": ["p1", "p2", "p3", "q1", "q2", "q4"], "cik": [1, 2, 3, 1, 2, 4],
                            "filing_date": pd.to_datetime(["2024-11-01"] * 3 + ["2025-02-01"] * 3),
                            "period": pd.to_datetime(["2024-09-30"] * 3 + ["2024-12-31"] * 3),
                            "is_original": True})
    filings = a9.select_filings(filings)
    hold = pd.DataFrame({"accession": ["p1", "p2", "p3", "p1", "q1", "q2", "q4", "q4"],
                         "cusip": ["CA0000001", "CA0000001", "CA0000001", "CB0000001",
                                   "CA0000001", "CB0000001", "CA0000001", "CB0000001"],
                         "shares": [100.0, 100.0, 100.0, 50.0, 300.0, 80.0, 10.0, 20.0]})
    rows = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC", "DDD"] * 4,
                         "Date": ["2024-09-27"] * 4 + ["2024-12-27"] * 4 + ["2025-02-14"] * 4 + ["2025-02-18"] * 4,
                         "first_date": "2010-01-04", "shares": 1000.0})
    return ftd, filings, hold, rows


def test_inst_weekly_values_and_availability():
    ftd, filings, hold, rows = _holdings_setup()
    f = a9.inst_weekly(rows, filings, hold, ftd, log=lambda *a: None)
    f.index = pd.MultiIndex.from_frame(rows[["ticker", "Date"]])
    # Q3 2024 usable from 2024-11-15 (deadline Thu 11-14); Q4 from 2025-02-18 (deadline Fri 02-14)
    assert f.loc[("AAA", "2024-09-27"), a9.INST_COLUMNS].isna().all()      # no quarter usable yet
    r = f.loc[("AAA", "2025-02-14")]                                  # Q4 deadline day: still Q3
    assert r["inst_period"] == pd.Timestamp("2024-09-30")
    assert np.isclose(r["inst_breadth"], 3 / 3) and np.isclose(r["inst_io"], 300 / 1000)
    r = f.loc[("AAA", "2025-02-18")]                                  # Q4 available
    assert r["inst_period"] == pd.Timestamp("2024-12-31")
    assert np.isclose(r["inst_breadth"], 2 / 3)                       # filers 1, 4 hold AAA among {1, 2, 4}
    # continuing filers {1, 2}: AAA held by 1,2 in Q3 -> by 1 in Q4: one exit, CHS dbreadth = (1 - 2) / 2
    assert np.isclose(r["inst_dbreadth"], -0.5) and np.isclose(r["inst_exit_n"], np.log1p(1))
    assert np.isclose(r["inst_new_n"], 0.0)
    assert np.isclose(r["inst_io"], 310 / 1000) and np.isclose(r["inst_dio"], 0.01)
    b = f.loc[("BBB", "2025-02-18")]                                  # filer 1 exits, filer 2 enters
    assert np.isclose(b["inst_new_n"], np.log1p(1)) and np.isclose(b["inst_exit_n"], np.log1p(1))
    assert np.isclose(b["inst_dbreadth"], 0.0)
    c = f.loc[("CCC", "2025-02-18")]
    assert c["inst_mapped"] == 1 and c["inst_breadth"] == 0           # linked, nobody holds it
    d = f.loc[("DDD", "2025-02-18")]
    assert d["inst_mapped"] == 0 and np.isnan(d["inst_breadth"])      # unlinked -> missing (hot-deck later)


def test_inst_weekly_no_lookahead():
    """Holdings of a quarter not yet available, and late filings, cannot move earlier features."""
    ftd, filings, hold, rows = _holdings_setup()
    early = rows[rows["Date"] <= "2025-02-14"].reset_index(drop=True)
    f0 = a9.inst_weekly(early, filings, hold, ftd, log=lambda *a: None)
    h1 = hold.copy()
    h1.loc[h1["accession"].str.startswith("q"), "shares"] *= 100
    h1 = pd.concat([h1, pd.DataFrame({"accession": ["q4"], "cusip": ["CC0000001"], "shares": [5.0]})])
    f1 = a9.inst_weekly(early, filings, h1, ftd, log=lambda *a: None)
    pd.testing.assert_frame_equal(f0, f1)
    late = pd.DataFrame({"accession": ["late"], "cik": [9], "filing_date": pd.to_datetime(["2024-11-20"]),
                         "period": pd.to_datetime(["2024-09-30"]), "is_original": True})
    raw = pd.concat([filings.drop(columns=["deadline", "avail_date"]), late], ignore_index=True)
    f2 = a9.inst_weekly(early, a9.select_filings(raw),
                        pd.concat([hold, pd.DataFrame({"accession": ["late"], "cusip": ["CA0000001"],
                                                       "shares": [1e6]})]), ftd, log=lambda *a: None)
    pd.testing.assert_frame_equal(f0, f2)                             # late filer never enters


def test_block_hotdeck_symmetric():
    from DataAnalysisPipeline2.v3 import round8 as r8
    from DataAnalysisPipeline2.v3 import round9 as r9
    rng = np.random.default_rng(1)
    n = 10
    w = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "Date": ["2025-01-03"] * n})
    for c in r8.HD_MATCH:
        w[c] = rng.normal(size=n)
    w["x"] = np.r_[rng.normal(size=6), np.full(4, np.nan)]
    w["y"] = np.r_[rng.normal(size=6), np.full(4, np.nan)]
    has = np.r_[np.ones(6, bool), np.zeros(4, bool)]
    hd = r9.block_hotdeck(w, ["x", "y"], has)
    donors = w.loc[has, ["x", "y"]].to_numpy()
    assert np.allclose(hd.loc[has].to_numpy(), donors)
    for v in hd.loc[~has].to_numpy():
        assert any(np.allclose(v, d) for d in donors)
