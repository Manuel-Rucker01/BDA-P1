"""Round 14 ``lazy`` block: document reduction, prior matching, similarity
features, availability timing and the weekly as-of join (synthetic, no network)."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import altdata_r14_lazy as L

LM = {"neg": {"loss", "adverse"}, "unc": {"may", "uncertain"}, "lit": {"lawsuit"}}

DOC = """<html><head><title>x</title></head><body>
<ix:header><ix:hidden>dei hidden facts should vanish</ix:hidden></ix:header>
<div style="display:none">hidden div text</div>
<p>Item 1. Business</p><p>Table of contents line</p><p>Item 1A. Risk Factors</p>
<p>ITEM 1. BUSINESS</p><p>""" + " ".join(["widgets"] * 160) + """ we may face an adverse lawsuit</p>
<table><tr><td>Revenue</td><td>1,234</td><td>5,678</td><td>9,012</td></tr></table>
<table><tr><td>This layout table holds a long prose sentence about our company strategy</td></tr></table>
<p>ITEM 1A. RISK FACTORS</p><p>Loss of customers is uncertain &amp; material.</p>
</body></html>"""


def test_clean_text_drops_xbrl_hidden_and_numeric_tables():
    t = L.clean_text(DOC).lower()
    assert "dei hidden facts" not in t and "hidden div text" not in t
    assert "revenue" not in t                        # numeric table removed
    assert "layout table holds a long prose" in t    # prose table kept
    assert "& material" in t                         # entities unescaped


def test_item1_skips_table_of_contents():
    seg = L.item1_span(L.clean_text(DOC))
    assert seg is not None and "widgets" in seg and "table of contents" not in seg.lower()
    assert "loss of customers" not in seg.lower()


def test_parse_document_counts_and_hash():
    r = L.parse_document(DOC, "10-K", LM)
    toks = L.tokenize(L.clean_text(DOC))
    assert r["n_tok"] == len(toks)
    assert r["n_neg"] == 2 and r["n_unc"] == 2 and r["n_lit"] == 1     # adverse, loss / may, uncertain / lawsuit
    assert int(r["cnt"].sum()) == len(toks) and np.all(np.diff(r["idx"].astype(np.int64)) > 0)
    assert r["idx"].max() < L.HASH_MOD and "i1_idx" in r
    assert "i1_idx" not in L.parse_document(DOC, "10-Q", LM)
    assert L.hash_word("widgets") == L.hash_word("widgets")           # crc32: stable across processes


def test_sparse_sims():
    i, c = np.array([1, 5, 9], np.uint32), np.array([1, 2, 3], np.uint32)
    cos, jac = L.sparse_sims(i, c, i, c)
    assert np.isclose(cos, 1.0) and jac == 1.0
    cos, jac = L.sparse_sims(i, c, np.array([5, 7], np.uint32), np.array([2, 1], np.uint32))
    assert np.isclose(cos, 4 / (np.sqrt(14) * np.sqrt(5))) and np.isclose(jac, 1 / 4)
    assert np.isnan(L.sparse_sims(i, c, np.zeros(0, np.uint32), np.zeros(0, np.uint32))[0])


def _docs():
    rows = [  # cik, accn, form, filing_date, report_date, acceptance
        (1, "a1", "10-K", "2017-03-01", "2016-12-31", "2017-03-01T21:00:00Z"),
        (1, "a2", "10-Q", "2017-05-01", "2017-03-31", "2017-05-01T12:00:00Z"),
        (1, "a3", "10-Q", "2017-08-01", "2017-06-30", "2017-08-01T12:00:00Z"),
        (1, "a4", "10-K", "2018-03-01", "2017-12-31", "2018-03-01T21:00:00Z"),
        (1, "a5", "10-Q", "2018-05-01", "2018-03-31", "2018-05-01T12:00:00Z"),
        (1, "a6", "10-K", "2020-03-01", "2019-12-31", "2020-03-01T21:00:00Z"),   # 2018 10-K missing -> no prior
    ]
    d = pd.DataFrame(rows, columns=["cik", "accn", "form", "filing_date", "report_date", "acceptance"])
    d["status"], d["n_tok"] = "ok", [1000, 500, 500, 2000, 500, 1000]
    d["n_neg"], d["n_unc"], d["n_lit"] = [10, 5, 5, 40, 10, 10], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]
    v = (np.array([1, 2, 3], np.uint32), np.array([5, 5, 5], np.uint32))
    d["idx"] = [v[0]] * 6
    d["cnt"] = [v[1]] * 6
    return d


def test_prior_matching_same_form_one_year():
    d = _docs()
    d["acceptance"] = pd.to_datetime(d["acceptance"], utc=True)
    pr = L.match_prior(d).to_numpy()
    assert list(pr) == [-1, -1, -1, 0, 1, -1]      # 10-K vs 10-K, 10-Q vs same quarter a year earlier


def test_filing_features_values():
    ev = L.filing_features(_docs()).set_index("accn")
    assert np.isclose(ev.loc["a4", "lazy_cos"], 1.0) and ev.loc["a4", "prior_accn"] == "a1"
    assert np.isclose(ev.loc["a4", "lazy_dlen"], np.log(2.0))
    assert np.isclose(ev.loc["a4", "lazy_dneg"], (40 / 2000 - 10 / 1000) * 100)
    assert np.isclose(ev.loc["a5", "lazy_dneg"], (10 / 500 - 5 / 500) * 100)
    assert np.isnan(ev.loc["a6", "lazy_cos"]) and ev.loc["a4", "lazy_is10k"] == 1.0


def test_avail_is_next_session_after_ny_acceptance_date():
    cal = np.array([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2018-02-26", "2018-03-09")])
    acc = pd.Series(pd.to_datetime(["2018-03-01T14:00:00Z",      # Thu 09:00 ET -> Fri
                                    "2018-03-02T03:00:00Z",      # Thu 22:00 ET (UTC Fri) -> Fri
                                    "2018-03-02T22:00:00Z"],     # Fri 17:00 ET -> Mon
                                   utc=True))
    got = L.avail_dates(acc, cal).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2018-03-02", "2018-03-02", "2018-03-05"]


def test_weekly_join_no_lookahead_and_staleness():
    ev = L.filing_features(_docs())
    cal = np.array([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2016-01-01", "2021-12-31")])
    ev["avail_date"] = L.avail_dates(ev["acceptance"], cal).dt.strftime("%Y-%m-%d")
    rows = pd.DataFrame({"cik": [1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
                         "Date": ["2018-03-01", "2018-03-02", "2018-05-04", "2019-06-10", "2020-03-06", "2018-05-04"]})
    w = L.weekly_join(rows, ev)
    assert np.isnan(w.loc[0, "lazy_cos"])                 # a4 accepted 2018-03-01 16:00 ET -> not usable that day
    assert w.loc[1, "lazy_accn"] == "a4" and w.loc[1, "lazy_days"] == 0
    assert w.loc[2, "lazy_accn"] == "a5" and w.loc[2, "lazy_is10k"] == 0
    assert np.isnan(w.loc[3, "lazy_cos"]) and w.loc[3, "lazy_accn"] is None     # > 400 days since a5
    assert np.isnan(w.loc[4, "lazy_cos"])                 # a6 has no prior -> nothing finite within 400 days
    assert np.isnan(w.loc[5, "lazy_cos"])                 # no CIK
    rows["lazy_accn"] = w["lazy_accn"]
    rows["lazy_cos"] = w["lazy_cos"]
    a = L.asof_audit(rows.assign(cik=rows["cik"]), ev)
    assert a["accepted_on_or_after_date"] == 0 and a["avail_after_date"] == 0 and a["cik_mismatch"] == 0
