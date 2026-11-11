import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import altdata_r14_spill as sp


def test_tnic_valid_from_is_first_session_on_or_after_july_1():
    cal = np.array(["2024-06-28", "2024-07-01", "2024-07-02", "2025-06-30", "2025-07-01"])
    assert sp.tnic_valid_from(2023, cal) == "2024-07-01"
    cal2 = np.array(["2024-06-28", "2024-07-02"])
    assert sp.tnic_valid_from(2023, cal2) == "2024-07-02"
    assert sp.tnic_valid_from(2030, cal) is None


def test_topk_ties_broken_by_dst():
    df = pd.DataFrame({"s": [1, 1, 1, 1, 2], "d": [9, 3, 5, 4, 1], "w": [0.5, 0.5, 0.9, 0.1, 0.2]})
    out = sp.topk(df, "s", "d", "w", k=3)
    assert out[out["s"] == 1]["d"].tolist() == [5, 3, 9]
    assert out[out["s"] == 2]["d"].tolist() == [1]


def test_map_gvkey_cik_overlap_and_latest():
    links = pd.DataFrame({"gvkey": ["000001", "000001", "000002", "000003"],
                          "cik": [10, 11, 20, 30],
                          "first_date": ["2000-01-01", "2015-01-01", "2000-01-01", "2020-01-01"],
                          "last_date": ["2016-06-30", "2025-01-01", "2010-01-01", "2025-01-01"]})
    m = sp.map_gvkey_cik(links, 2016)
    assert m["000001"] == 11            # both overlap -> latest last_date
    assert "000002" not in m.index      # link ended before 2016
    assert "000003" not in m.index      # starts after 2017-12-31


def test_trailing_returns_needs_both_closes_and_clips():
    close = np.array([[1.0, 10.0, np.nan], [2.0, 10.0, 5.0], [8.0, np.nan, 5.0]])
    r = sp.trailing_returns(close, 2, 1)
    assert r[0] == pytest.approx(2.0)   # 8/2-1 = 3 -> clipped to 2
    assert np.isnan(r[1])
    assert r[2] == pytest.approx(0.0)
    assert np.isnan(sp.trailing_returns(close, 0, 1)).all()


def test_neighbour_means_weighting_and_missing():
    src = np.array([0, 0, 0, 1])
    w = np.array([1.0, 3.0, 2.0, 1.0])
    rd = np.array([[0.1, 0.2, 0.0], [0.3, np.nan, 0.0], [np.nan, 0.5, 0.0], [np.nan, np.nan, np.nan]])
    m, n = sp.neighbour_means(src, w, rd, 3)
    assert m[0, 0] == pytest.approx((0.1 * 1 + 0.3 * 3) / 4)
    assert m[0, 1] == pytest.approx((0.2 * 1 + 0.5 * 2) / 3)
    assert n.tolist() == [2, 0, 0]
    assert np.isnan(m[1]).all() and np.isnan(m[2]).all()


def _holdings():
    # filers 1..6 broad holders of A,B,C ; filers 7,8,9 specialists holding A and B only
    rows = []
    for f in range(1, 7):
        for t in ("A", "B", "C", "D"):
            rows.append((t, f))
    for f in (7, 8, 9):
        rows += [("A", f), ("B", f)]
    rows += [("E", 1), ("E", 2)]                     # E: fewer than min holders
    return pd.DataFrame(rows, columns=["ticker", "cik"])


def test_idf_cosine_prefers_specialist_overlap_and_filters():
    e = sp.idf_cosine_edges(_holdings(), k=2, min_holders=5, min_common=3)
    assert "E" not in set(e["src"]) | set(e["dst"])
    a = e[e["src"] == "A"]
    assert a["dst"].iloc[0] == "B"                   # shares the specialists
    assert (e["src"] != e["dst"]).all()
    assert e.groupby("src").size().max() <= 2
    assert (e["n_common"] >= 3).all()


def test_idf_cosine_same_cik_excluded():
    e = sp.idf_cosine_edges(_holdings(), k=3, same_cik={"A": 100, "B": 100})
    assert not ((e["src"] == "A") & (e["dst"] == "B")).any()
    assert not ((e["src"] == "B") & (e["dst"] == "A")).any()


def test_version_for_valid_to_exclusive():
    e = pd.DataFrame({"src": ["A", "A"], "dst": ["B", "C"], "weight": [1.0, 1.0],
                      "valid_from": ["2020-01-01", "2020-06-01"], "valid_to": ["2020-06-01", None]})
    assert sp._version_for(e, "2020-05-29")["dst"].tolist() == ["B"]
    assert sp._version_for(e, "2020-06-01")["dst"].tolist() == ["C"]
    assert sp._version_for(e, "2019-12-31").empty


def _date_setup():
    # store tickers: X (cik 1), Y (cik 2), Z (cik 3), Y2 (cik 2, second class), W (no cik)
    col = {"X": 0, "Y": 1, "Y2": 2, "Z": 3, "W": 4}
    R = np.array([[0.01, 0.10, 0.20], [0.02, 0.20, 0.30], [0.05, 0.50, 0.50], [0.03, 0.30, np.nan],
                  [0.04, 0.40, 0.40]])
    rt = pd.DataFrame({"ticker": ["X", "Y", "Y2", "Z", "W"], "cik": [1.0, 2.0, 2.0, 3.0, np.nan],
                       "eligible": [True, True, False, True, True], "n_bars": [100, 100, 500, 100, 100]})
    rt.index = [10, 11, 12, 13, 14]
    et = pd.DataFrame({"src_cik": [1, 1, 2], "dst_cik": [2, 3, 1], "weight": [1.0, 3.0, 1.0]})
    eo = pd.DataFrame({"src": ["X", "W"], "dst": ["W", "Z"], "weight": [1.0, 1.0]})
    sic = pd.Series({1: 1000, 2: 1000, 3: 2000})
    return rt, R, col, et, eo, sic


def _get(out, c):
    return out[:, sp.ALL_COLUMNS.index(c)]


def test_date_features_tnic_own_sic():
    rt, R, col, et, eo, sic = _date_setup()
    out = sp.date_features(rt, R, col, et, eo, sic)
    # tnic for X: neighbours cik2 (primary ticker = Y, eligible) w1 and cik3 (Z) w3
    assert _get(out, "spill_tnic_r21")[0] == pytest.approx((0.20 * 1 + 0.30 * 3) / 4)
    assert _get(out, "spill_tnic_r63")[0] == pytest.approx(0.30)          # Z missing at 63
    assert _get(out, "spill_tnic_x21")[0] == pytest.approx((0.20 + 0.90) / 4 - 0.10)
    assert _get(out, "spill_tnic_n")[0] == 2
    # both tickers of cik 2 get cik 2's tnic features (neighbour X)
    assert _get(out, "spill_tnic_r21")[1] == pytest.approx(0.10)
    assert _get(out, "spill_tnic_r21")[2] == pytest.approx(0.10)
    assert _get(out, "spill_tnic_x21")[2] == pytest.approx(0.10 - 0.50)  # own return of Y2 itself
    # Z: cik3 has no TNIC edges; W has no cik -> missing
    assert np.isnan(_get(out, "spill_tnic_n")[3]) and np.isnan(_get(out, "spill_tnic_n")[4])
    # own (ticker nodes)
    assert _get(out, "spill_own_r21")[0] == pytest.approx(0.40)
    assert _get(out, "spill_own_r21")[4] == pytest.approx(0.30)
    assert np.isnan(_get(out, "spill_own_n")[1])
    # sic: cik1 and cik2 share 1000 -> X sees Y's return, cik 3 alone -> n = 0, r missing
    assert _get(out, "spill_sic_r21")[0] == pytest.approx(0.20)
    assert _get(out, "spill_sic_r21")[1] == pytest.approx(0.10)
    assert _get(out, "spill_sic_n")[3] == 0 and np.isnan(_get(out, "spill_sic_r21")[3])
    assert np.isnan(_get(out, "spill_sic_n")[4])


def test_date_features_placebo_uses_permuted_neighbour_returns_only():
    rt, R, col, et, eo, sic = _date_setup()
    Rn = R[::-1].copy()
    out = sp.date_features(rt, R, col, et, eo, sic, Rn)
    # X's own-edge neighbour W now reads row 0 (X's own return); x uses X's true own return
    assert _get(out, "spill_own_r21")[0] == pytest.approx(0.10)
    assert _get(out, "spill_own_x21")[0] == pytest.approx(0.0)


def test_primary_tickers():
    rt = pd.DataFrame({"ticker": ["B", "A", "C"], "cik": [5.0, 5.0, 6.0], "eligible": [False, False, True],
                       "n_bars": [10, 10, 1]})
    p = sp.primary_tickers(rt).set_index("cik")["ticker"]
    assert p[5] == "A" and p[6] == "C"


def test_ttl_lines_parse():
    rdflib = pytest.importorskip("rdflib")
    e = pd.DataFrame({"src": ["BRK.B", "AAC~1"], "dst": ["AAPL", "X"], "weight": [0.5, 0.25]})
    txt = sp.TTL_HEADER + "\n".join(sp.ttl_lines(e, "own", "2020-01-02", None, "2019-09-30")) + "\n" \
        + "\n".join(sp.ttl_lines(e, "tnic", "2020-07-01", "2021-07-01", "2019"))
    g = rdflib.Graph().parse(data=txt, format="turtle")
    onto = rdflib.Namespace(sp.ONTO)
    assert len(list(g.triples((None, onto.sharesOwnersWith, None)))) == 2
    assert len(list(g.triples((None, onto.productPeerOf, None)))) == 2
    assert len(list(g.triples((None, onto.validTo, None)))) == 2
