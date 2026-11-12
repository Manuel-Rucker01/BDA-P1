"""Round 14 (pre-registered in ``results/round14_preregistration.md``; runner
decisions in ``results/round14_addendum_trials.md``; constants ``gate.R14_*``):
firm-to-firm graph spillover, filing-text change ("Lazy Prices") and a GNN, on
top of the incumbent ``sn30_altall_hd`` signal and the unlevered
``earn_blackout`` book.

    sn30_spill      incumbent 65 features + spill block (E1 TNIC + E2 13F)
    sn30_lazy       incumbent + lazy block
    sn30_graphtext  incumbent + both blocks
    gnn_spill       2-layer GraphSAGE (mean aggregation) over E1+E2, incumbent
                    features as node inputs, rank-blended 50/50 with the
                    incumbent ensemble

    python -m DataAnalysisPipeline2.v3.round14 frame probe wf check_sig graphs gnn blend \
        books check_book summarize signals placebo_wf placebo_sig importance gate

Run with OMP_NUM_THREADS=1 for the books (parallel processes).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import altdata_r14_lazy as a14l
from . import altdata_r14_spill as a14s
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import round9 as r9
from . import round10 as r10
from . import round11 as r11
from . import round13 as r13
from . import stress as st
from . import train as tr

PPY = 252
R14_DATASET = "dataset_r14"
INC = gate.R9_BASE                       # sn30_altall_hd (stored Round-8 walk-forward)
INC_RERUN = "r14_sn30_altall_hd"         # the same recipe re-run under the Round-14 code
GNN = "gnn_spill"
GNN_RAW = "r14_gnn_raw"                  # the GNN alone (before the blend), information only
TNIC_COLS = a14s.cols_for("tnic")
OWN_COLS = a14s.cols_for("own")
SPILL_COLS = list(a14s.SPILL_COLUMNS)
LAZY_COLS = [c for c in a14l.LAZY_COLUMNS if c != "lazy_is10k"]     # addendum (a)/(b)
BLOCKS = {"spill": SPILL_COLS, "lazy": LAZY_COLS}
TRIAL_BLOCKS = {"sn30_spill": ("spill",), "sn30_lazy": ("lazy",), "sn30_graphtext": ("spill", "lazy")}
HAS = {"tnic": "r14_tnic_has", "own": "r14_own_has", "lazy": "r14_lazy_has"}
BOOKS = (gate.R14_BASE_TRIAL,) + gate.R14_TRIALS
SIGNAL_OF = {gate.R14_BASE_TRIAL: INC_RERUN, **{t: t for t in gate.R14_TRIALS}}
PLACEBO_SEED = 14
PLACEBO = {"sn30_spill": "r14_placebo_spill", "sn30_lazy": "r14_placebo_lazy"}
GNN_MAX_EPOCHS = 100
GNN_PATIENCE = 10
GNN_LR = 1e-3
_G: dict = {}


def frame_path() -> str:
    return tr.weekly_path(R14_DATASET)


def meta_path() -> str:
    return ex.eval_dir("r14_frame_meta.json")


def raw(cols) -> list:
    return [f"{c}_raw" for c in cols]


def feats_for(name: str) -> list:
    base = r8.feats_for(INC)
    if name in (INC, INC_RERUN):
        return base
    return base + [c for b in TRIAL_BLOCKS[name] for c in BLOCKS[b]]


# --------------------------------------------------------------------------- #
# Lazy corpus coverage (addendum decision (c))
# --------------------------------------------------------------------------- #
def lazy_coverage() -> dict:
    """Download share per tier and the CIKs with every tier-1..3 filing downloaded."""
    idx = pd.read_parquet(os.path.join(a14l.lazy_dir(), "index.parquet"), columns=["cik", "accn", "tier"])
    docs = a14l.load_docs(columns=["accn"])
    idx = idx.assign(done=idx["accn"].isin(set(docs["accn"])))
    t3 = idx[idx["tier"] <= 3]
    by = t3.groupby("tier")["done"].mean().to_dict()
    full = t3.groupby("cik")["done"].all()
    return {"share_by_tier": {int(k): float(v) for k, v in by.items()},
            "share_tiers_1_3": float(t3["done"].mean()),
            "complete": bool(t3["done"].mean() >= 0.999),
            "full_ciks": set(int(c) for c in full[full].index)}


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
def prepare_blocks(w: pd.DataFrame, sp: pd.DataFrame | None, lz: pd.DataFrame | None,
                   lazy_ciks: set | None = None) -> pd.DataFrame:
    """Merge the blocks onto the evaluation frame ``w`` and apply the addendum's
    handling: raw copies; E1 hot-decked on has = TNIC node & CIK; E2 not
    hot-decked; lazy hot-decked on has = CIK & finite lazy_cos (& CIK in
    ``lazy_ciks`` when given, decision (c)); then per-date z-scores."""
    w = w.copy()
    cik = w["cik"].notna().to_numpy()
    new = []
    if sp is not None:
        w = w.merge(sp[["ticker", "Date", "spill_tnic_has", "spill_own_has"] + SPILL_COLS],
                    on=["ticker", "Date"], how="left")
        w[raw(SPILL_COLS)] = w[SPILL_COLS].to_numpy()
        has_t = (w["spill_tnic_has"].fillna(0).to_numpy() > 0) & cik
        w[HAS["tnic"]] = has_t.astype(float)
        w[HAS["own"]] = (w["spill_own_has"].fillna(0).to_numpy() > 0).astype(float)
        w[TNIC_COLS] = r9.block_hotdeck(w, TNIC_COLS, has_t)
        w = w.drop(columns=["spill_tnic_has", "spill_own_has"])
        new += SPILL_COLS
    if lz is not None:
        w = w.merge(lz[["ticker", "Date"] + LAZY_COLS], on=["ticker", "Date"], how="left")
        w[raw(LAZY_COLS)] = w[LAZY_COLS].to_numpy()
        has_l = np.isfinite(w["lazy_cos"].to_numpy(dtype=float)) & cik
        if lazy_ciks is not None:
            ck = pd.to_numeric(w["cik"], errors="coerce")
            has_l &= ck.isin(lazy_ciks).to_numpy()
        w[HAS["lazy"]] = has_l.astype(float)
        w[LAZY_COLS] = r9.block_hotdeck(w, LAZY_COLS, has_l)
        new += LAZY_COLS
    if new:
        w[new + raw(new)] = core.cross_sectional_zscore(w, new + raw(new))
    return w


def build_frame(log=print, with_lazy: bool | None = None) -> pd.DataFrame:
    w = pd.read_parquet(r8.frame_path())
    sp = pd.read_parquet(a14s.out_path())
    cov = lazy_coverage()
    if with_lazy is None:
        with_lazy = True
    lz, restrict = None, None
    if with_lazy:
        lz = pd.read_parquet(a14l.weekly_path())
        restrict = None if cov["complete"] else cov["full_ciks"]
    w = prepare_blocks(w, sp, lz, lazy_ciks=restrict)
    w.to_parquet(frame_path(), index=False)
    meta = {"rows": int(len(w)), "with_lazy": bool(with_lazy),
            "lazy_tier_share": cov["share_by_tier"], "lazy_complete": cov["complete"],
            "lazy_restricted_to_full_coverage": restrict is not None,
            "n_full_coverage_ciks": len(cov["full_ciks"]),
            "has_share": {k: float(w[v].mean()) for k, v in HAS.items() if v in w}}
    json.dump(meta, open(meta_path(), "w"), indent=1)
    log(f"[r14] frame {len(w):,} rows -> {frame_path()} | {meta}")
    return w


# --------------------------------------------------------------------------- #
# Leak probes (Round 9-10 design)
# --------------------------------------------------------------------------- #
def leak_probe(log=print, end: str = "2022-12-31", seed: int = 0, blocks=("spill", "lazy")) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    w = pd.read_parquet(frame_path())
    w = w[w["Date"] <= end].reset_index(drop=True)
    targets = {"no_edgar": w["cik"].isna().astype(int).to_numpy(),
               "delisted": r10._delisted(w["ticker"]).astype(int).to_numpy()}
    dates = np.array(sorted(w["Date"].unique()))
    rng = np.random.default_rng(seed)
    te = w["Date"].isin(set(dates[rng.random(len(dates)) < 0.3])).to_numpy()
    base = r8.feats_for(INC)
    sets = {"base (incumbent 65)": base}
    blocks = [b for b in blocks if BLOCKS[b][0] in w]
    for b in blocks:
        sets[f"+{b} raw"] = base + raw(BLOCKS[b])
        sets[f"+{b} hd"] = base + BLOCKS[b]
        sets[f"{b} hd only"] = BLOCKS[b]
    if "spill" in blocks:
        sets["+E1 tnic hd"] = base + TNIC_COLS
        sets["+E2 own (no hd)"] = base + OWN_COLS
        sets["E1 has-flag only"] = [HAS["tnic"]]
        sets["E2 has-flag only"] = [HAS["own"]]
    if "lazy" in blocks:
        sets["lazy has-flag only"] = [HAS["lazy"]]
    if len(blocks) == 2:
        sets["+both hd"] = base + SPILL_COLS + LAZY_COLS
    p = ex.eval_dir("r14_leak.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    out.update({"period": f"2017..{end}", "rows": int(len(w)),
                "base_rate": {k: float(v.mean()) for k, v in targets.items()}})
    for tname, y in targets.items():
        out.setdefault(tname, {})
        for sname, cols in sets.items():
            if sname in out[tname]:
                continue
            m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbose=-1, n_jobs=8)
            m.fit(w.loc[~te, cols], y[~te])
            out[tname][sname] = float(roc_auc_score(y[te], m.predict_proba(w.loc[te, cols])[:, 1]))
            log(f"  [{tname}] {sname}: AUC {out[tname][sname]:.3f}")
    json.dump(out, open(p, "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward (trees), with feature importances
# --------------------------------------------------------------------------- #
def importances(models: dict, feats: list) -> dict:
    """Per model: importance share per feature (sums to 1). LightGBM gain,
    XGBoost total gain, CatBoost PredictionValuesChange."""
    out = {}
    for name, m in models.items():
        if name == "lgbm":
            v = np.asarray(m.booster_.feature_importance("gain"), float)
        elif name == "xgb":
            sc = m.get_booster().get_score(importance_type="total_gain")
            v = np.array([sc.get(f, 0.0) for f in feats], float)
        else:
            v = np.asarray(m.get_feature_importance(), float)
        s = v.sum()
        out[name] = dict(zip(feats, (v / s if s > 0 else v).tolist()))
    return out


def _fit_imp(frame, feats, target, calendar, log=print):
    ms, info = tr.fit_ensemble(frame, feats, target, calendar, log=log)
    info["importance"] = importances(ms, feats)
    return ms, info


def run_wf(log=print, names=None, frame: pd.DataFrame | None = None, feats_of=None) -> None:
    w = pd.read_parquet(frame_path()) if frame is None else frame
    cal = ds.calendar_from_dataset()
    for name in names or [INC_RERUN] + list(TRIAL_BLOCKS):
        feats = (feats_of or feats_for)(name)
        missing = [c for c in feats if c not in w]
        if missing:
            raise KeyError(f"{name}: frame lacks {missing[:5]}")
        t0 = time.time()
        preds, folds = tr.walk_forward(w, feats, "target_sn_30", cal, log=log, fit_fn=_fit_imp)
        preds.to_parquet(os.path.join(ds.v3_dir("wf"), f"{name}.parquet"), index=False)
        json.dump({"variant": name, "features": feats, "folds": folds, "seconds": round(time.time() - t0)},
                  open(os.path.join(ds.v3_dir("wf"), f"{name}.json"), "w"), indent=1, default=str)
        log(f"[r14] {name} done in {time.time() - t0:.0f}s")


def check_signal(log=print) -> dict:
    """Engineering check: the Round-14 rerun of sn30_altall_hd == the stored one."""
    a = ex.load_preds(INC_RERUN)
    b = ex.load_preds(INC)
    m = a.merge(b, on=["ticker", "Date"], suffixes=("_r14", "_r8"))
    d = (m["pred_r14"] - m["pred_r8"]).abs()
    out = {"n_r14": int(len(a)), "n_r8": int(len(b)), "n_matched": int(len(m)), "max_abs_diff": float(d.max()),
           "mean_abs_diff": float(d.mean()), "share_identical": float((d < 1e-9).mean()),
           "corr": float(m["pred_r14"].corr(m["pred_r8"]))}
    json.dump(out, open(ex.eval_dir("r14_check_signal.json"), "w"), indent=1)
    log(f"[r14 check] signal rerun vs stored: {out}")
    return out


# --------------------------------------------------------------------------- #
# Placebos
# --------------------------------------------------------------------------- #
def date_permutation(dates, seed: int = PLACEBO_SEED) -> dict:
    """{d: pi(d)}: a fixed random permutation of the decision dates."""
    d = np.array(sorted(set(dates)))
    return dict(zip(d, d[np.random.default_rng(seed).permutation(len(d))]))


def permute_block_dates(alt: pd.DataFrame, cols: list, perm: dict) -> pd.DataFrame:
    """Block values of row (ticker, d) := the same ticker's values on perm[d]
    (missing if the ticker has no row then)."""
    inv = {v: k for k, v in perm.items()}
    out = alt[["ticker", "Date"] + cols].copy()
    out["Date"] = out["Date"].map(inv)
    return out.dropna(subset=["Date"])


def build_placebo_frame(block: str, log=print) -> pd.DataFrame:
    w = pd.read_parquet(r8.frame_path())
    perm = date_permutation(w["Date"].unique())
    if block == "spill":
        sp = pd.read_parquet(a14s.out_path(), columns=["ticker", "Date", "spill_tnic_has", "spill_own_has"]
                             + SPILL_COLS)
        f = prepare_blocks(w, permute_block_dates(sp, ["spill_tnic_has", "spill_own_has"] + SPILL_COLS, perm),
                           None)
    else:
        lz = pd.read_parquet(a14l.weekly_path(), columns=["ticker", "Date"] + LAZY_COLS)
        cov = lazy_coverage()
        f = prepare_blocks(w, None, permute_block_dates(lz, LAZY_COLS, perm),
                           lazy_ciks=None if cov["complete"] else cov["full_ciks"])
    log(f"[r14 placebo] {block} frame built")
    return f


def run_placebo_wf(log=print, blocks=("spill", "lazy")) -> None:
    for b in blocks:
        trial = f"sn30_{b}"
        f = build_placebo_frame(b, log=log)
        run_wf(log=log, names=[PLACEBO[trial]], frame=f, feats_of=lambda _n, t=trial: feats_for(t))
        del f


def shuffled_date_ic(p: pd.DataFrame, lab: pd.DataFrame, n_perm: int = 20, seed: int = 0) -> dict:
    """Signal of date t vs ``fwd_ret_30`` of a randomly permuted date (same ticker)."""
    rng = np.random.default_rng(seed)
    d = np.array(sorted(p["Date"].unique()))
    vals = []
    for _ in range(n_perm):
        pm = dict(zip(d, d[rng.permutation(len(d))]))
        q = p.assign(Date=p["Date"].map(pm))
        vals.append(float(r6.per_date_ic(q, lab, "fwd_ret_30").dropna().mean()))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "max_abs": float(np.max(np.abs(vals)))}


# --------------------------------------------------------------------------- #
# GNN: graph snapshots
# --------------------------------------------------------------------------- #
def graphs_path() -> str:
    return os.path.join(a14s.spill_dir(), "gnn_graphs_r14.pkl")


def normalise_graph(n: int, edge_sets: list) -> tuple:
    """Aggregation matrix for one date as CSR arrays (indptr, indices, data).
    ``edge_sets``: list of (src_idx, dst_idx, weight). Each set: self-loops
    dropped, duplicates -> max, symmetrised w_ij = max(w_ij, w_ji),
    row-normalised; A = sum of the sets / k_i (k_i = sets in which i has a
    neighbour; isolated rows stay 0)."""
    from scipy import sparse
    acc = sparse.csr_matrix((n, n), dtype=np.float64)
    k = np.zeros(n)
    for s, d, wt in edge_sets:
        s, d, wt = np.asarray(s, np.int64), np.asarray(d, np.int64), np.asarray(wt, float)
        ok = (s != d) & np.isfinite(wt) & (wt > 0)
        s, d, wt = s[ok], d[ok], wt[ok]
        if not len(s):
            continue
        a = pd.DataFrame({"s": np.r_[s, d], "d": np.r_[d, s], "w": np.r_[wt, wt]}).groupby(["s", "d"])["w"].max()
        si = a.index.get_level_values(0).to_numpy()
        di = a.index.get_level_values(1).to_numpy()
        m = sparse.csr_matrix((a.to_numpy(), (si, di)), shape=(n, n))
        rs = np.asarray(m.sum(axis=1)).ravel()
        inv = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1.0), 0.0)
        acc = acc + sparse.diags(inv) @ m
        k += rs > 0
    acc = (sparse.diags(np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)) @ acc).tocsr()
    acc.sort_indices()
    return acc.indptr.astype(np.int64), acc.indices.astype(np.int64), acc.data.astype(np.float32)


def _nbars(dates: set) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    parts = []
    for f in files:
        x = pd.read_parquet(f, columns=["ticker", "Date", "n_bars"])
        parts.append(x[x["Date"].isin(dates)])
    return pd.concat(parts, ignore_index=True)


def _versions(edges: pd.DataFrame) -> list:
    vf = edges["valid_from"].astype(str).to_numpy()
    vt = edges["valid_to"].fillna("9999-12-31").astype(str).to_numpy()
    key = pd.Series(list(zip(vf, vt)))
    return [(a, b, edges.iloc[np.asarray(ix)]) for (a, b), ix in key.groupby(key).groups.items()]


def build_graphs(log=print) -> dict:
    """One aggregation matrix per weekly decision date over the frame's rows
    (v3-eligible names), from E1 + E2 as of that date."""
    t0 = time.time()
    w = pd.read_parquet(r8.frame_path(), columns=["ticker", "Date"])
    dates = sorted(w["Date"].unique())
    rows = a14s.weekly_rows()
    nb = _nbars(set(dates))
    vers = {k: _versions(pd.read_parquet(a14s.edges_path(k))) for k in ("tnic", "own")}
    rows_by = dict(tuple(rows.groupby("Date")))
    nb_by = dict(tuple(nb.groupby("Date")))
    out = {}
    stats = []
    for d, g in w.groupby("Date", sort=True):
        tick = g["ticker"].to_numpy()
        pos = pd.Series(np.arange(len(tick)), index=tick)
        sets = []
        nE = {}
        for kind in ("tnic", "own"):
            sub = [e for a, b, e in vers[kind] if a <= d < b]
            if not sub:
                nE[kind] = 0
                continue
            e = a14s.edges_asof(d, kind, edges=pd.concat(sub), rows=rows_by.get(d, rows.iloc[:0]),
                                nbars=nb_by.get(d))
            si, di = e["src"].map(pos), e["dst"].map(pos)
            ok = si.notna() & di.notna()
            sets.append((si[ok].to_numpy(np.int64), di[ok].to_numpy(np.int64), e.loc[ok, "weight"].to_numpy(float)))
            nE[kind] = int(ok.sum())
        indptr, indices, data = normalise_graph(len(tick), sets)
        out[d] = {"tickers": tick, "indptr": indptr, "indices": indices, "data": data}
        stats.append({"Date": d, "n": len(tick), "e_tnic": nE.get("tnic", 0), "e_own": nE.get("own", 0),
                      "nnz": int(len(data)), "isolated": float((np.diff(indptr) == 0).mean())})
    pickle.dump(out, open(graphs_path(), "wb"), protocol=4)
    s = pd.DataFrame(stats)
    s.to_parquet(ex.eval_dir("r14_graph_stats.parquet"), index=False)
    log(f"[r14 graphs] {len(out)} dates, mean nodes {s['n'].mean():.0f}, nnz {s['nnz'].mean():.0f}, "
        f"isolated {s['isolated'].mean():.1%} ({time.time() - t0:.0f}s)")
    return out


# --------------------------------------------------------------------------- #
# GNN: model + walk-forward
# --------------------------------------------------------------------------- #
def make_sage(d_in: int, hidden: int = gate.R14_GNN["hidden"], dropout: float = gate.R14_GNN["dropout"]):
    import torch
    from torch import nn

    class SAGE(nn.Module):
        """2-layer GraphSAGE with (weighted) mean aggregation and a linear read-out."""

        def __init__(self):
            super().__init__()
            self.s1, self.n1 = nn.Linear(d_in, hidden), nn.Linear(d_in, hidden, bias=False)
            self.s2, self.n2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden, bias=False)
            self.out = nn.Linear(hidden, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, A):
            h = torch.relu(self.s1(x) + self.n1(torch.sparse.mm(A, x)))
            h = self.drop(h)
            h = torch.relu(self.s2(h) + self.n2(torch.sparse.mm(A, h)))
            h = self.drop(h)
            return self.out(h).squeeze(-1)

    assert gate.R14_GNN["layers"] == 2
    return SAGE()


def _sparse(g: dict, perm: np.ndarray | None = None, zero: bool = False):
    import torch
    n = len(g["tickers"])
    indptr, idx, data = g["indptr"], g["indices"], g["data"]
    rows = np.repeat(np.arange(n), np.diff(indptr))
    cols = idx
    if perm is not None:                          # node identities permuted: A'[i, j] = A[pi(i), pi(j)]
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        rows, cols = inv[rows], inv[cols]
    if zero:
        data = np.zeros(0, np.float32)
        rows = cols = np.zeros(0, np.int64)
    i = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    return torch.sparse_coo_tensor(i, torch.from_numpy(np.asarray(data, np.float32)), (n, n)).coalesce()


def _date_tensors(w: pd.DataFrame, feats: list, graphs: dict) -> dict:
    import torch
    out = {}
    for d, g in w.groupby("Date", sort=True):
        gr = graphs[d]
        g = g.set_index("ticker").loc[gr["tickers"]]
        y = g["target_sn_30"].to_numpy(dtype=np.float32)
        out[d] = {"x": torch.from_numpy(np.nan_to_num(g[feats].to_numpy(dtype=np.float32))),
                  "y": torch.from_numpy(np.nan_to_num(y)), "m": torch.from_numpy(np.isfinite(y)),
                  "A": _sparse(gr), "tickers": gr["tickers"]}
    return out


def train_gnn(T: dict, fit_d: list, val_d: list, d_in: int, seed: int = gate.R14_GNN["seed"],
              max_epochs: int = GNN_MAX_EPOCHS, patience: int = GNN_PATIENCE, lr: float = GNN_LR, log=print):
    """Adam on per-date MSE, one date per step, early stopping on the
    validation-tail MSE (best epoch kept)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_sage(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fit_d = [d for d in fit_d if bool(T[d]["m"].any())]
    val_d = [d for d in val_d if bool(T[d]["m"].any())]

    def val_mse():
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for d in val_d:
                t = T[d]
                p = model(t["x"], t["A"])
                se += float(((p - t["y"])[t["m"]] ** 2).sum())
                n += int(t["m"].sum())
        return se / max(n, 1)

    best, best_state, best_ep, bad = np.inf, None, 0, 0
    hist = []
    for ep in range(max_epochs):
        model.train()
        for k in rng.permutation(len(fit_d)):
            t = T[fit_d[k]]
            opt.zero_grad()
            p = model(t["x"], t["A"])
            loss = ((p - t["y"])[t["m"]] ** 2).mean()
            loss.backward()
            opt.step()
        v = val_mse()
        hist.append(v)
        if v < best - 1e-9:
            best, best_ep, bad = v, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_ep, "epochs_run": len(hist), "best_val_mse": best, "val_hist": hist}


def run_gnn(log=print, years=tr.TEST_YEARS) -> pd.DataFrame:
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    feats = r8.feats_for(INC)
    w = pd.read_parquet(r8.frame_path(), columns=["ticker", "Date", "target_sn_30"] + feats)
    graphs = pickle.load(open(graphs_path(), "rb"))
    T = _date_tensors(w, feats, graphs)
    cal = ds.calendar_from_dataset()
    all_d = sorted(T)
    labelled = sorted(w.loc[w["target_sn_30"].notna(), "Date"].unique())
    preds, folds = [], []
    for y in years:
        start = f"{y}-01-01"
        tr_d = tr.train_window_dates(labelled, cal, start)
        te_d = [d for d in all_d if start <= d <= f"{y}-12-31"]
        if not te_d or not tr_d:
            continue
        fit_d, val_d = tr.split_fit_val(tr_d, cal)
        t1 = time.time()
        model, info = train_gnn(T, fit_d, val_d, len(feats), log=log)
        with torch.no_grad():
            u1, w1 = float(model.n1.weight.norm()), float(model.s1.weight.norm())
            for d in te_d:
                t = T[d]
                n = len(t["tickers"])
                perm = np.random.default_rng(int(d.replace("-", ""))).permutation(n)
                p = model(t["x"], t["A"]).numpy()
                p0 = model(t["x"], _sparse(graphs[d], zero=True)).numpy()
                ps = model(t["x"], _sparse(graphs[d], perm=perm)).numpy()
                preds.append(pd.DataFrame({"ticker": t["tickers"], "Date": d, "pred": p, "pred_nograph": p0,
                                           "pred_shufnb": ps, "fold": y}))
        folds.append({"year": y, "fit_range": [fit_d[0], fit_d[-1]], "val_range": [val_d[0], val_d[-1]],
                      "n_fit_dates": len(fit_d), "n_val_dates": len(val_d), **info,
                      "nb_weight_share_l1": u1 / (u1 + w1), "seconds": round(time.time() - t1)})
        log(f"  [gnn] fold {y}: best epoch {info['best_epoch']}/{info['epochs_run']} val MSE "
            f"{info['best_val_mse']:.5f}, nb share {u1 / (u1 + w1):.2f} ({time.time() - t1:.0f}s)")
    p = pd.concat(preds, ignore_index=True)
    for c in ("pred", "pred_nograph", "pred_shufnb"):
        p[c] = p.groupby("Date")[c].rank(pct=True)
    p.to_parquet(os.path.join(ds.v3_dir("wf"), f"{GNN_RAW}.parquet"), index=False)
    json.dump({"variant": GNN_RAW, "features": feats, "arch": gate.R14_GNN, "lr": GNN_LR,
               "max_epochs": GNN_MAX_EPOCHS, "patience": GNN_PATIENCE, "folds": folds,
               "seconds": round(time.time() - t0)},
              open(os.path.join(ds.v3_dir("wf"), f"{GNN_RAW}.json"), "w"), indent=1, default=float)
    log(f"[r14] GNN done in {time.time() - t0:.0f}s")
    return p


def rank_blend(a: pd.DataFrame, b: pd.DataFrame, wa: float = 1 - gate.R14_GNN["blend"],
               col_b: str = "pred") -> pd.DataFrame:
    """Per-date rank of wa * rank(a) + (1 - wa) * rank(b) on the common rows."""
    m = a[["ticker", "Date", "pred", "fold"]].merge(b[["ticker", "Date", col_b]].rename(columns={col_b: "_b"}),
                                                    on=["ticker", "Date"], how="inner")
    ra = m.groupby("Date")["pred"].rank(pct=True)
    rb = m.groupby("Date")["_b"].rank(pct=True)
    m["pred"] = (wa * ra + (1 - wa) * rb).groupby(m["Date"]).rank(pct=True)
    return m[["ticker", "Date", "pred", "fold"]]


def blend(log=print) -> pd.DataFrame:
    inc = ex.load_preds(INC_RERUN)
    g = ex.load_preds(GNN_RAW)
    out = rank_blend(inc, g)
    assert len(out) == len(inc), (len(out), len(inc))
    out.to_parquet(os.path.join(ds.v3_dir("wf"), f"{GNN}.parquet"), index=False)
    for c, nm in (("pred_nograph", "r14_gnn_blend_nograph"), ("pred_shufnb", "r14_gnn_blend_shufnb")):
        rank_blend(inc, g, col_b=c).to_parquet(os.path.join(ds.v3_dir("wf"), f"{nm}.parquet"), index=False)
    log(f"[r14] {GNN}: {len(out):,} rows")
    return out


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def _worker(book):
    t0 = time.time()
    prep = r8.prep_with(ex.load_preds(SIGNAL_OF[book]), _G["base"])
    res = r13.run_book(gate.R14_BASE_TRIAL, prep, bl=_G["bl"])
    res["daily"].assign(ret=res["ret"]).to_parquet(ex.eval_dir(f"r14_{book}_daily.parquet"))
    res["log"].to_parquet(ex.eval_dir(f"r14_{book}_log.parquet"))
    lg = res["log"]
    print(f"  [r14] {book}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((lg['status'] != 'solved').sum())})", flush=True)
    return book


def run_books(log=print, books=BOOKS, workers: int = 3) -> None:
    import multiprocessing as mp
    t0 = time.time()
    _G["base"] = r5.load_prep()
    _G["bl"] = r11.blackout_inputs(log)
    log(f"[r14] inputs ready ({time.time() - t0:.0f}s)")
    books = list(books)
    with mp.get_context("fork").Pool(min(workers, len(books))) as pool:
        for b in pool.imap_unordered(_worker, books):
            log(f"[r14] {b} saved ({time.time() - t0:.0f}s)")


def check_book(log=print) -> dict:
    """Engineering check: the Round-14 incumbent book == Round 11's earn_blackout."""
    a = pd.read_parquet(ex.eval_dir(f"r14_{gate.R14_BASE_TRIAL}_daily.parquet"))["ret"]
    b = pd.read_parquet(ex.eval_dir(f"r11_{gate.R14_BASE_TRIAL}_daily.parquet"))["ret"]
    a.index, b.index = a.index.astype(str), b.index.astype(str)
    out = {"max_abs_diff": float((a - b.reindex(a.index)).abs().max()), "n": int(len(a)), "n_r11": int(len(b)),
           "same_index": bool(a.index.equals(b.index)), "sharpe_r14": ev.sharpe(a), "sharpe_r11": ev.sharpe(b)}
    json.dump(out, open(ex.eval_dir("r14_check_book.json"), "w"), indent=1)
    log(f"[r14 check] earn_blackout vs r11: {out}")
    return out


def _load(book):
    d = pd.read_parquet(ex.eval_dir(f"r14_{book}_daily.parquet"))
    d.index = d.index.astype(str)
    return d, pd.read_parquet(ex.eval_dir(f"r14_{book}_log.parquet"))


def summarize(log=print, books=BOOKS) -> dict:
    spy = ex.spy_returns()
    spy.index = spy.index.astype(str)
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    res = {"configs": {}, "trials": {}}
    for b in books:
        if not os.path.exists(ex.eval_dir(f"r14_{b}_daily.parquet")):
            continue
        d, lg = _load(b)
        r = d["ret"]
        m = r5.metrics(r, lg.assign(ex_ante_te=np.nan), spy, ew, cw)
        a = m["all"]
        a["realized_over_exante_vol"] = float(a["vol"] / lg["ex_ante_vol"].mean())
        a["beta_spy"] = r11.realized_beta(r, spy)
        a["avg_n_blackout"] = float(lg["n_blackout"].mean())
        cov = st.window_stats(r, spy, *gate.R13_COVID)
        a.update({"covid_dd": cov["max_dd"], "covid_ret": cov["book"]})
        a["holdout_window"] = r11._window(r, spy, gate.C_START, gate.C_END)
        a["ytd_2026"] = r11._window(r, spy, "2026-01-01", "2026-12-31")
        res["configs"][b] = m
        if b in gate.R14_TRIALS:
            res["trials"][f"r14:{b}"] = float(r.mean() / r.std(ddof=1))
        log(f"  {b}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} beta {a['beta_spy']:.2f} "
            f"TO {a.get('turnover_m', float('nan')):.0%} cost {a['cost_drag_ann']:.2%}")
    json.dump(res, open(ex.eval_dir("results_r14.json"), "w"), indent=1, default=float)
    return res


# --------------------------------------------------------------------------- #
# Signal tables
# --------------------------------------------------------------------------- #
def own_value_rows() -> dict:
    """Rows with a CIK and the block's own value (Round-10 decisive check)."""
    import pyarrow.parquet as pq
    have = [c for c in HAS.values() if c in pq.read_schema(frame_path()).names]
    fr = pd.read_parquet(frame_path(), columns=["ticker", "Date"] + have)
    m = {"sn30_spill": fr[HAS["tnic"]] > 0, GNN: fr[HAS["tnic"]] > 0}
    if HAS["lazy"] in fr:
        m["sn30_lazy"] = fr[HAS["lazy"]] > 0
        m["sn30_graphtext"] = (fr[HAS["tnic"]] > 0) & (fr[HAS["lazy"]] > 0)
    return {k: fr.loc[v, ["ticker", "Date"]] for k, v in m.items()}


def _ic(p, lab):
    return r6.per_date_ic(p, lab, "fwd_ret_30").dropna()


def signal_row(p: pd.DataFrame, lab: pd.DataFrame, top: pd.DataFrame) -> dict:
    s = _ic(p, lab)
    by = s.groupby(s.index.str[:4]).mean()
    t1 = r6.ic_within(p, lab, top)["all"]
    hp = _ic(p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)], lab)
    return {"ic_tradable": float(s.mean()), "nw_t": ev.newey_west_t(s, gate.NW_LAGS),
            "years_pos": int(sum(by.get(y, np.nan) > 0 for y in gate.A_YEARS)),
            "by_year": by.round(4).to_dict(), "ic_top1000": t1["mean_ic"], "top1000_t": t1["nw_t"],
            "holdout_proxy": float(hp.mean())}


def signal_tables(log=print, with_placebo: bool = True) -> dict:
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30"])
    top = pd.read_parquet(r6.topn_path())
    own = own_value_rows()
    inc = ex.load_preds(INC_RERUN)
    names = [INC, INC_RERUN] + [t for t in gate.R14_TRIALS
                                if os.path.exists(os.path.join(ds.v3_dir("wf"), f"{t}.parquet"))]
    extra = [n for n in (GNN_RAW, "r14_gnn_blend_nograph", "r14_gnn_blend_shufnb") + tuple(PLACEBO.values())
             if os.path.exists(os.path.join(ds.v3_dir("wf"), f"{n}.parquet"))]
    out = {"signals": {}, "own_value": {}, "placebo_signal": {}}
    for v in names + extra:
        p = ex.load_preds(v)
        out["signals"][v] = signal_row(p, lab, top)
        if with_placebo and v in names:
            out["placebo_signal"][v] = shuffled_date_ic(p, lab)
        log(f"  {v}: IC {out['signals'][v]['ic_tradable']:.4f} (t {out['signals'][v]['nw_t']:.2f}) "
            f"top1000 {out['signals'][v]['ic_top1000']:.4f} holdout {out['signals'][v]['holdout_proxy']:.4f}")
    for v, sub in own.items():
        if v not in out["signals"]:
            continue
        icv = float(_ic(ex.load_preds(v).merge(sub, on=["ticker", "Date"]), lab).mean())
        ib = inc.merge(sub, on=["ticker", "Date"])
        icb = float(_ic(ib, lab).mean())
        out["own_value"][v] = {"ic": icv, "ic_inc": icb, "gain": icv - icb,
                               "rows_share": float(len(ib) / len(inc))}
    for t, pl in PLACEBO.items():
        if pl in out["signals"] and t in out["signals"]:
            out.setdefault("placebo_retrain", {})[t] = {
                "trial_gain": out["signals"][t]["ic_tradable"] - out["signals"][INC_RERUN]["ic_tradable"],
                "placebo_gain": out["signals"][pl]["ic_tradable"] - out["signals"][INC_RERUN]["ic_tradable"]}
    json.dump(out, open(ex.eval_dir("r14_signal_ics.json"), "w"), indent=1, default=float)
    return out


def importance_share(log=print) -> dict:
    out = {}
    for t, blocks in TRIAL_BLOCKS.items():
        p = os.path.join(ds.v3_dir("wf"), f"{t}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        sh = {b: [] for b in blocks}
        per_model = {b: {} for b in blocks}
        for f in j["folds"]:
            for mname, imp in f["importance"].items():
                for b in blocks:
                    v = float(sum(imp.get(c, 0.0) for c in BLOCKS[b]))
                    sh[b].append(v)
                    per_model[b].setdefault(mname, []).append(v)
        out[t] = {b: {"mean": float(np.mean(v)), "by_model": {m: float(np.mean(x)) for m, x in per_model[b].items()},
                      "n_cols": len(BLOCKS[b]), "uniform_share": len(BLOCKS[b]) / len(j["features"])}
                  for b, v in sh.items()}
        if "spill" in blocks:
            f0 = j["folds"]
            for sub, cols in (("E1_tnic", TNIC_COLS), ("E2_own", OWN_COLS)):
                out[t][sub] = float(np.mean([sum(imp.get(c, 0.0) for c in cols)
                                             for f in f0 for imp in f["importance"].values()]))
        top = pd.DataFrame([imp for f in j["folds"] for imp in f["importance"].values()]).mean()
        new = [c for b in blocks for c in BLOCKS[b]]
        out[t]["top_new_features"] = top[new].sort_values(ascending=False).head(5).round(4).to_dict()
        out[t]["rank_of_best_new"] = int((top.rank(ascending=False)[new]).min())
    pg = os.path.join(ds.v3_dir("wf"), f"{GNN_RAW}.json")
    if os.path.exists(pg):
        j = json.load(open(pg))
        out[GNN] = {"blend_weight": gate.R14_GNN["blend"],
                    "nb_weight_share_l1": float(np.mean([f["nb_weight_share_l1"] for f in j["folds"]])),
                    "best_epochs": [f["best_epoch"] for f in j["folds"]]}
    json.dump(out, open(ex.eval_dir("r14_importance.json"), "w"), indent=1)
    log(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- #
# Gate + adoption
# --------------------------------------------------------------------------- #
def _prior_trials() -> dict:
    trials = r13._prior_trials()
    trials.update(json.load(open(ex.eval_dir("results_r13.json")))["trials"])
    return trials


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r14.json")))
    trials = _prior_trials()
    n_prior = gate.N_TRIALS_R8 + len(gate.R11_TRIALS) + len(gate.R12_TRIALS) + len(gate.R13_TRIALS)
    assert len(trials) == n_prior, f"expected {n_prior} booked prior trials, got {len(trials)}"
    assert len(res["trials"]) == len(gate.R14_TRIALS), res["trials"].keys()
    trials.update(res["trials"])
    perf = {k: v["all"] for k, v in res["configs"].items()}
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sic = json.load(open(ex.eval_dir("r14_signal_ics.json")))["signals"]
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    spy_all = ex.spy_returns()
    spy_all.index = spy_all.index.astype(str)
    e_all = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    e_all.index = e_all.index.astype(str)
    out = {"n_trials": gate.N_TRIALS_R14, "n_sharpes_for_dispersion": len(trials), "per_book": {}}
    for b in BOOKS:
        r = _load(b)[0]["ret"]
        spy = spy_all.reindex(r.index).fillna(0.0)
        e = e_all.reindex(r.index).fillna(0.0)
        dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R14)
        boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
                "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, c: ev.info_ratio(a, c)),
                "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, c: ev.info_ratio(a, c))}
        sg = r11.signal_stats(SIGNAL_OF[b], lab)
        sm = sg["summary"]
        rows = gate.evaluate_gate({"mean_ic": sm["all"]["mean_ic"], "nw_t": sm["all"]["nw_t"]},
                                  {y: sm[y]["mean_ic"] for y in gate.A_YEARS if y in sm}, perf[b], ew_sh,
                                  dsr["dsr"], sg["holdout_proxy"])
        rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R14})"
        rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
        rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. "
                                              f"{boot['ir_spy']['hi']:.2f})",
                     "value": perf[b]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                     "pass": bool(perf[b]["ir_spy"] > gate.R5_MIN_IR_SPY)})
        out["per_book"][b] = {"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot,
                              "failed": [x["id"] for x in rows if not x["pass"]]}
        log(f"  {b}: DSR {dsr['dsr']:.3f} IR_spy {perf[b]['ir_spy']:.2f} [{boot['ir_spy']['lo']:.2f}, "
            f"{boot['ir_spy']['hi']:.2f}] passed={out['per_book'][b]['passed']} failed={out['per_book'][b]['failed']}")
    choice = gate.choose_config_r14(perf)
    out["selected"] = choice
    keys = ("sharpe", "ir_spy", "max_dd")
    tv = {**{k: perf[choice][k] for k in keys}, "ic_tradable": sic[choice]["ic_tradable"],
          "ic_top1000": sic[choice]["ic_top1000"]}
    iv = {**{k: perf[gate.R14_BASE_TRIAL][k] for k in keys}, "ic_tradable": sic[INC_RERUN]["ic_tradable"],
          "ic_top1000": sic[INC_RERUN]["ic_top1000"]}
    out["passed"] = out["per_book"][choice]["passed"]
    out["adoption"] = gate.adopt_r14(tv, iv, out["passed"])
    out["adoption"].update({"trial": tv, "incumbent": iv})
    json.dump(out, open(ex.eval_dir("gate_r14.json"), "w"), indent=1, default=float)
    log(f"[r14 gate] selected={choice} passed={out['passed']} adoption={out['adoption']}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"frame": build_frame, "frame_spill": lambda: build_frame(with_lazy=False),
             "probe": leak_probe, "probe_spill": lambda: leak_probe(blocks=("spill",)),
             "wf": run_wf, "wf_inc": lambda: run_wf(names=[INC_RERUN]),
             "wf_spill": lambda: run_wf(names=["sn30_spill"]),
             "wf_lazy": lambda: run_wf(names=["sn30_lazy", "sn30_graphtext"]),
             "check_sig": check_signal, "graphs": build_graphs, "gnn": run_gnn, "blend": blend,
             "books": run_books,
             "books_a": lambda: run_books(books=(gate.R14_BASE_TRIAL, "sn30_spill", GNN)),
             "books_b": lambda: run_books(books=("sn30_lazy", "sn30_graphtext"), workers=2),
             "check_book": check_book, "summarize": summarize, "signals": signal_tables,
             "placebo_wf": run_placebo_wf, "placebo_wf_spill": lambda: run_placebo_wf(blocks=("spill",)),
             "placebo_wf_lazy": lambda: run_placebo_wf(blocks=("lazy",)),
             "importance": importance_share, "gate": apply_gate}
    for a_ in sys.argv[1:]:
        steps[a_]()
