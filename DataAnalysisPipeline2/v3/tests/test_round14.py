"""Round 14 trial runner: block handling, placebo permutation, graph normalisation,
GNN mechanics, blend, selection and adoption (toy inputs, no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round8 as r8
from DataAnalysisPipeline2.v3 import round14 as r14


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
def _toy_frame(n_dates=3, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dates):
        d = f"2020-01-{k + 1:02d}"
        for i in range(n):
            rows.append({"ticker": f"T{i}", "Date": d, "cik": float(1000 + i) if i % 4 else np.nan,
                         **{c: rng.normal() for c in r8.HD_MATCH}})
    return pd.DataFrame(rows)


def _toy_blocks(w, seed=1):
    rng = np.random.default_rng(seed)
    sp = w[["ticker", "Date"]].copy()
    for c in r14.SPILL_COLS:
        sp[c] = rng.normal(size=len(sp))
    # E1 exists only for rows with a CIK and even ticker number; E2 for most rows
    tn = sp["ticker"].str[1:].astype(int)
    e1 = w["cik"].notna().to_numpy() & (tn % 2 == 0).to_numpy()
    sp.loc[~e1, r14.TNIC_COLS] = np.nan
    sp["spill_tnic_has"] = e1.astype(float)
    e2 = (tn % 7 != 0).to_numpy()
    sp.loc[~e2, r14.OWN_COLS] = np.nan
    sp["spill_own_has"] = e2.astype(float)
    lz = w[["ticker", "Date"]].copy()
    for c in r14.LAZY_COLS:
        lz[c] = rng.normal(size=len(lz))
    lz.loc[~w["cik"].notna().to_numpy() | (tn % 3 == 0).to_numpy(), r14.LAZY_COLS] = np.nan
    return sp, lz


def test_lazy_block_excludes_is10k():
    assert "lazy_is10k" not in r14.LAZY_COLS
    assert "lazy_days" in r14.LAZY_COLS and len(r14.LAZY_COLS) == 7
    assert len(r14.SPILL_COLS) == 14 and not any("sic" in c for c in r14.SPILL_COLS)


def test_prepare_blocks_hotdeck_rules():
    w = _toy_frame()
    sp, lz = _toy_blocks(w)
    f = r14.prepare_blocks(w, sp, lz)
    # E1 and lazy: every row gets a value (hot-deck), E2 keeps its holes (z = 0)
    assert np.isfinite(f[r14.TNIC_COLS + r14.LAZY_COLS].to_numpy()).all()
    own_raw = f[r14.raw(r14.OWN_COLS)].to_numpy()
    own_hd = f[r14.OWN_COLS].to_numpy()
    holes = sp[r14.OWN_COLS].isna().to_numpy()
    assert (own_hd[holes] == 0).all() and (own_raw[holes] == 0).all()
    # has flags: E1 needs a CIK; lazy needs CIK and lazy_cos
    assert ((f[r14.HAS["tnic"]] > 0) <= f["cik"].notna()).all()
    assert ((f[r14.HAS["lazy"]] > 0) == (f["cik"].notna() & lz["lazy_cos"].notna())).all()
    # rows with their own E1 value keep it (before z-scoring, check rank order within a date)
    d0 = f["Date"] == "2020-01-01"
    m = d0 & (f[r14.HAS["tnic"]] > 0)
    a = f.loc[m, "spill_tnic_r5"].rank()
    b = sp.loc[m.to_numpy(), "spill_tnic_r5"].rank()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_prepare_blocks_lazy_restriction():
    w = _toy_frame()
    sp, lz = _toy_blocks(w)
    keep = {1001, 1002, 1005}
    f = r14.prepare_blocks(w, None, lz, lazy_ciks=keep)
    has = f[r14.HAS["lazy"]] > 0
    assert set(f.loc[has, "cik"].astype(int)) <= keep
    assert has.any()


def test_placebo_permutation_moves_values_within_ticker():
    w = _toy_frame(n_dates=5)
    sp, _ = _toy_blocks(w)
    perm = r14.date_permutation(w["Date"].unique(), seed=3)
    assert sorted(perm) == sorted(perm.values())
    q = r14.permute_block_dates(sp, ["spill_own_r5"], perm)
    m = q.merge(sp[["ticker", "Date", "spill_own_r5"]], left_on=["ticker", q["Date"].map(perm)],
                right_on=["ticker", "Date"], suffixes=("", "_src"))
    assert len(m) == len(sp)
    assert np.allclose(m["spill_own_r5"].fillna(-99), m["spill_own_r5_src"].fillna(-99))


# --------------------------------------------------------------------------- #
# Graph normalisation
# --------------------------------------------------------------------------- #
def _dense(n, csr):
    indptr, idx, data = csr
    A = np.zeros((n, n))
    for i in range(n):
        A[i, idx[indptr[i]:indptr[i + 1]]] = data[indptr[i]:indptr[i + 1]]
    return A


def test_normalise_graph_symmetric_rownorm_and_mix():
    n = 5
    e1 = (np.array([0, 1, 2, 2]), np.array([1, 0, 3, 2]), np.array([0.2, 0.1, 0.3, 0.9]))   # 2->2 self loop
    e2 = (np.array([0, 4]), np.array([2, 0]), np.array([1.0, 0.5]))
    A = _dense(n, r14.normalise_graph(n, [e1, e2]))
    assert A[2, 2] == 0
    np.testing.assert_allclose(A.sum(1), [1, 1, 1, 1, 1], atol=1e-6)
    # node 0: E1 neighbour 1 (max(0.2, 0.1) -> 1.0 after row-norm), E2 neighbours 2 and 4
    np.testing.assert_allclose(A[0], [0, 0.5, 0.5 * 1.0 / 1.5, 0, 0.5 * 0.5 / 1.5], atol=1e-6)
    # symmetrised: node 3 gets node 2 from E1 only (k = 1)
    np.testing.assert_allclose(A[3], [0, 0, 1, 0, 0], atol=1e-6)


def test_normalise_graph_isolated_rows_zero():
    A = _dense(4, r14.normalise_graph(4, [(np.array([0]), np.array([1]), np.array([1.0]))]))
    assert A[2].sum() == 0 and A[3].sum() == 0


# --------------------------------------------------------------------------- #
# GNN
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def _toy_T(n_dates=12, n=60, d_in=5, seed=0):
    rng = np.random.default_rng(seed)
    T = {}
    for k in range(n_dates):
        s = rng.integers(0, n, 200)
        d = rng.integers(0, n, 200)
        g = dict(zip(("indptr", "indices", "data"), r14.normalise_graph(n, [(s, d, rng.random(200))])))
        g["tickers"] = np.array([f"T{i}" for i in range(n)])
        x = rng.normal(size=(n, d_in)).astype(np.float32)
        A = r14._sparse(g)
        nb = torch.sparse.mm(A, torch.from_numpy(x)).numpy()
        y = (0.5 * x[:, 0] + 0.8 * nb[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
        T[f"2020-{k + 1:02d}-01"] = {"x": torch.from_numpy(x), "y": torch.from_numpy(y),
                                     "m": torch.ones(n, dtype=torch.bool), "A": A, "tickers": g["tickers"], "g": g}
    return T


def test_sage_architecture_matches_preregistration():
    m = r14.make_sage(7)
    assert m.s1.out_features == gate.R14_GNN["hidden"] == 64
    assert m.drop.p == gate.R14_GNN["dropout"] == 0.3
    assert sum(1 for n in ("s1", "s2") if hasattr(m, n)) == gate.R14_GNN["layers"]


def test_gnn_training_deterministic_and_uses_graph():
    torch.set_num_threads(1)
    T = _toy_T()
    d = sorted(T)
    m1, i1 = r14.train_gnn(T, d[:9], d[9:], 5, max_epochs=8, patience=3)
    m2, i2 = r14.train_gnn(T, d[:9], d[9:], 5, max_epochs=8, patience=3)
    t = T[d[-1]]
    with torch.no_grad():
        p1, p2 = m1(t["x"], t["A"]), m2(t["x"], t["A"])
        p0 = m1(t["x"], r14._sparse(t["g"], zero=True))
    assert torch.equal(p1, p2) and i1["best_epoch"] == i2["best_epoch"]
    assert 1 <= i1["best_epoch"] <= i1["epochs_run"] <= 8
    assert not torch.allclose(p1, p0)          # the graph term matters on a graph-driven target


def test_sparse_permutation_relabels_nodes():
    g = dict(zip(("indptr", "indices", "data"),
                 r14.normalise_graph(3, [(np.array([0]), np.array([1]), np.array([1.0]))])))
    g["tickers"] = np.array(["a", "b", "c"])
    perm = np.array([2, 0, 1])
    A = r14._sparse(g).to_dense().numpy()
    Ap = r14._sparse(g, perm=perm).to_dense().numpy()
    np.testing.assert_allclose(Ap, A[np.ix_(perm, perm)])


# --------------------------------------------------------------------------- #
# Blend, selection, adoption
# --------------------------------------------------------------------------- #
def test_rank_blend_half_half():
    a = pd.DataFrame({"ticker": list("abcd"), "Date": "2020-01-01", "pred": [0.1, 0.2, 0.3, 0.4], "fold": 2020})
    b = pd.DataFrame({"ticker": list("abcd"), "Date": "2020-01-01", "pred": [4.0, 3.0, 2.0, 1.0]})
    out = r14.rank_blend(a, b)
    assert gate.R14_GNN["blend"] == 0.5
    assert np.allclose(out["pred"], out["pred"].iloc[0])          # exact opposites -> all tied
    out2 = r14.rank_blend(a, a.drop(columns="fold"))
    assert list(out2["pred"]) == [0.25, 0.5, 0.75, 1.0]


def test_choose_config_r14():
    perf = {"sn30_spill": {"ir_spy": 0.1, "max_dd": -0.30}, "sn30_lazy": {"ir_spy": 0.3, "max_dd": -0.40},
            "gnn_spill": {"ir_spy": 0.2, "max_dd": -0.20}, "earn_blackout": {"ir_spy": 9.0, "max_dd": -0.1}}
    assert gate.choose_config_r14(perf) == "gnn_spill"
    assert gate.N_TRIALS_R14 == 110


def test_adopt_r14_rule():
    inc = {"sharpe": 1.0, "ir_spy": -0.10, "max_dd": -0.28, "ic_tradable": 0.073, "ic_top1000": 0.020}
    good = {"sharpe": 1.03, "ir_spy": -0.05, "max_dd": -0.29, "ic_tradable": 0.074, "ic_top1000": 0.020}
    assert gate.adopt_r14(good, inc, False)["rule_met"]
    for k, v in (("sharpe", 1.02), ("ir_spy", -0.06), ("max_dd", -0.2901), ("ic_tradable", 0.073),
                 ("ic_top1000", 0.0199)):
        assert not gate.adopt_r14({**good, k: v}, inc, False)["rule_met"], k
    assert gate.adopt_r14({**good, "sharpe": 0.0}, inc, True)["rule_met"]
    assert gate.adopt_r14(good, inc, False)["needs_live_data_path"]
