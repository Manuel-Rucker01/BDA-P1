"""
Data Analysis Pipeline 2: Knowledge Graph Embeddings + Multi-Model Comparison.

Pipeline:
  1. Extract structural triples from the financial KG (Company, Sector,
     Industry, Size, Volatility, Country, Region, SubRegion, sharesBorderWith,
     madeAcquisition, acquisitionCountry). Observation and literal-valued
     predicates are excluded — per-day numeric features are joined from DuckDB
     instead, avoiding geometric distortion in the embedding space.
  2. Train a **RotatE** model (PyTorch, 128-dim complex) on these triples with:
       - Self-adversarial negative sampling (multiple negs per positive,
         re-weighted by their current difficulty — Sun et al. 2019)
       - Sigmoid+log loss (more stable than margin-ranking)
       - Vectorised sampling, GPU/MPS auto-detect, early stopping
     RotatE handles symmetric (sharesBorderWith), antisymmetric and
     compositional relations — a strict superset of what TransE can model.
  3. Export the learned company embeddings to parquet.
  4. Join embeddings with per-observation features computed by DuckDB window
     functions, including volatility-adjusted return and sector momentum.
  5. Compare five base classifiers + a stacking ensemble under three feature
     configurations (tabular / embedding / combined). Report accuracy, F1 and
     ROC-AUC for every (model, feature-set) combination.
"""

import os
import random
import time
import warnings

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdflib import Graph, Namespace, URIRef
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
FIN_KG_PATH = os.path.join(EXPLOITATION_DIR, "financial_knowledge_graph.ttl")
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")
EMBED_OUT_PATH = os.path.join(EXPLOITATION_DIR, "company_embeddings.parquet")

# RotatE hyperparameters — tuned for ~10k structural triples / ~3k entities.
EMBED_DIM = 128          # Complex dim — total real params per entity = 2 × dim
ROTATE_LR = 5e-4
ROTATE_EPOCHS = 250
ROTATE_BATCH = 1024
ROTATE_NEG_PER_POS = 8   # Multiple negatives per positive (RotatE paper uses 64)
ROTATE_GAMMA = 12.0      # Margin in the sigmoid-log loss
ROTATE_ADV_TEMP = 1.0    # Self-adversarial sampling temperature
ROTATE_PATIENCE = 20

# Stacking ensemble CV folds — kept small because the dataset is large.
STACK_CV = 2

# Compress the 2 × EMBED_DIM real embedding down to PCA_DIM features before
# joining with tabular signal. With ~10 tabular columns, feeding 256 raw
# embedding dims swamps the tabular signal — PCA condenses the structural
# information into a handful of orthogonal axes.
PCA_DIM = 16

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# MPS sometimes underperforms CPU on small KGE models due to per-op kernel
# launch overhead — let users opt in via env var if they want to try it.
_USE_MPS = os.environ.get("KGE_USE_MPS", "0") == "1"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps" if _USE_MPS and torch.backends.mps.is_available() else "cpu")
)

FIN_ONTO = Namespace("http://bda.upc.edu/finance/ontology#")
FIN_ENT = Namespace("http://bda.upc.edu/finance/resource/")

STRUCTURAL_RELATIONS = {
    FIN_ONTO.operatesInSector,
    FIN_ONTO.belongsToIndustry,
    FIN_ONTO.hasSize,
    FIN_ONTO.hasVolatilityProfile,
    FIN_ONTO.headquarteredIn,
    FIN_ONTO.locatedInRegion,
    FIN_ONTO.partOfSubRegion,
    FIN_ONTO.sharesBorderWith,
    FIN_ONTO.madeAcquisition,
    FIN_ONTO.acquisitionCountry,
}


# ── Step 1: Extract structural triples ────────────────────────────────────────

def extract_structural_triples(kg_path: str):
    print(f"[1/5] Parsing knowledge graph (device: {DEVICE})...")
    t0 = time.time()
    g = Graph()
    g.parse(kg_path, format="turtle")
    print(f"  -> Parsed {len(g)} triples in {time.time()-t0:.1f}s")

    triples = [
        (str(h), str(r), str(t))
        for h, r, t in g
        if r in STRUCTURAL_RELATIONS and isinstance(t, URIRef)
    ]
    print(f"  -> Retained {len(triples)} structural triples (URI–URI only)")

    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    print(f"  -> Entities: {len(entities)}, Relations: {len(relations)}")
    return triples, ent2id, rel2id


# ── Step 2: RotatE Model ──────────────────────────────────────────────────────

class RotatE(nn.Module):
    """
    RotatE (Sun et al. 2019): entities live in a complex space, each relation is
    a rotation in that space. For a triple (h, r, t):
        score = -|| h ∘ r − t ||
    where ∘ is element-wise complex multiplication and r has unit modulus.
    Relation embeddings are parameterised by phases in [-π, π] — guaranteed unit
    modulus by construction.
    """
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.dim = dim
        # Entities: 2 * dim real values = `dim` complex numbers
        self.ent_emb = nn.Embedding(n_entities, 2 * dim)
        # Relations: stored as phases, dim real values each
        self.rel_phase = nn.Embedding(n_relations, dim)
        bound = 6 / dim**0.5
        nn.init.uniform_(self.ent_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -np.pi, np.pi)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        phase = self.rel_phase(r_idx)

        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        r_re, r_im = torch.cos(phase), torch.sin(phase)

        # Complex multiply: (h_re + i h_im)(r_re + i r_im)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Distance to t in complex space
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        # || · ||_2 over the complex magnitudes
        dist = torch.sqrt((diff_re ** 2 + diff_im ** 2) + 1e-12).sum(dim=-1)
        return dist


def train_rotate(triples, ent2id, rel2id):
    print("\n[2/5] Training RotatE model with self-adversarial negative sampling...")
    n_ent = len(ent2id)
    n_rel = len(rel2id)
    model = RotatE(n_ent, n_rel, EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=ROTATE_LR)

    h_all = torch.tensor([ent2id[h] for h, _, _ in triples], device=DEVICE)
    r_all = torch.tensor([rel2id[r] for _, r, _ in triples], device=DEVICE)
    t_all = torch.tensor([ent2id[t] for _, _, t in triples], device=DEVICE)
    n_triples = len(triples)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, ROTATE_EPOCHS + 1):
        perm = torch.randperm(n_triples, device=DEVICE)
        total_loss = 0.0
        for i in range(0, n_triples, ROTATE_BATCH):
            idx = perm[i : i + ROTATE_BATCH]
            bs = len(idx)
            h_pos, r_pos, t_pos = h_all[idx], r_all[idx], t_all[idx]

            # Positive distance — lower is better
            pos_dist = model(h_pos, r_pos, t_pos)
            pos_score = ROTATE_GAMMA - pos_dist  # higher is better

            # Multiple negatives per positive — corrupt heads OR tails
            corrupt_tail = (torch.rand(bs, device=DEVICE) < 0.5)
            neg_ents = torch.randint(0, n_ent, (bs, ROTATE_NEG_PER_POS), device=DEVICE)

            h_neg = torch.where(
                corrupt_tail.unsqueeze(1),
                h_pos.unsqueeze(1).expand(-1, ROTATE_NEG_PER_POS),
                neg_ents,
            )
            t_neg = torch.where(
                corrupt_tail.unsqueeze(1),
                neg_ents,
                t_pos.unsqueeze(1).expand(-1, ROTATE_NEG_PER_POS),
            )
            r_neg = r_pos.unsqueeze(1).expand(-1, ROTATE_NEG_PER_POS)

            neg_dist = model(h_neg.reshape(-1), r_neg.reshape(-1), t_neg.reshape(-1))
            neg_dist = neg_dist.view(bs, ROTATE_NEG_PER_POS)
            neg_score = ROTATE_GAMMA - neg_dist

            # Self-adversarial negative weights (Sun et al. 2019, eq. 5):
            #   weights ∝ softmax(α · negative_score)
            # — harder negatives (higher score) get more weight. Detached to
            # avoid gradients flowing through the weights.
            with torch.no_grad():
                neg_weights = torch.softmax(neg_score * ROTATE_ADV_TEMP, dim=-1)

            # Sigmoid log-loss (RotatE): margin-style but smoother than hinge
            pos_loss = -nn.functional.logsigmoid(pos_score).mean()
            neg_loss = -(neg_weights * nn.functional.logsigmoid(-neg_score)).sum(dim=-1).mean()
            loss = (pos_loss + neg_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 25 == 0:
            print(f"  Epoch {epoch:>3}/{ROTATE_EPOCHS}  loss={total_loss:.4f}")

        if total_loss < best_loss - 1e-4:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ROTATE_PATIENCE:
                print(f"  -> Early stop at epoch {epoch} (best loss {best_loss:.4f})")
                break

    print("  -> RotatE training complete.")
    return model


# ── Step 3: Extract company embeddings ────────────────────────────────────────

def extract_company_embeddings(model, ent2id):
    print("\n[3/5] Extracting company embeddings...")
    model.eval()
    company_prefix = str(FIN_ENT)
    results = {}
    weights = model.ent_emb.weight.detach().cpu().numpy()
    embed_dim_real = weights.shape[1]  # 2 * EMBED_DIM
    for uri, idx in ent2id.items():
        local = uri.replace(company_prefix, "")
        # Tickers have no underscores; category nodes do (Sector_X, Acq_<id>...)
        if "_" not in local:
            results[local] = weights[idx]
    print(f"  -> Extracted embeddings for {len(results)} company nodes "
          f"(dim={embed_dim_real} real values = {EMBED_DIM} complex)")

    emb_df = pd.DataFrame.from_dict(
        results, orient="index",
        columns=[f"emb_{i}" for i in range(embed_dim_real)]
    ).reset_index().rename(columns={"index": "ticker"})
    try:
        emb_df.to_parquet(EMBED_OUT_PATH, index=False)
        print(f"  -> Embeddings saved to {EMBED_OUT_PATH}")
    except Exception as e:
        print(f"  -> [WARN] Could not save parquet ({e})")
    return results, embed_dim_real


# ── Step 4: Join with DuckDB observation data ─────────────────────────────────

def load_observation_features(db_path: str):
    print("\n[4/5] Loading per-observation features from DuckDB...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("""
        WITH base AS (
            SELECT
                Symbol AS ticker, Date, target_7d_up, Sector,
                LOG(NULLIF(MarketCap, 0)) AS log_market_cap,
                eur_rate, jpy_rate, company_close, company_volume,

                -- Lagged price / momentum
                (company_close - LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS daily_return,

                (company_close - AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)) /
                    NULLIF(AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS price_vs_ma5,

                company_volume / NULLIF(AVG(company_volume) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,

                STDDEV(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS rolling_volatility_10d
            FROM master_dataset
            WHERE target_7d_up IS NOT NULL
        )
        SELECT
            base.*,

            -- Volatility-adjusted return: pure-alpha proxy, removes the
            -- mechanical scaling of vol with return.
            daily_return / NULLIF(rolling_volatility_10d, 0) AS vol_adjusted_return,

            -- Volume z-score on a 20-day window
            (company_volume - AVG(company_volume) OVER (PARTITION BY ticker ORDER BY Date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)) /
                NULLIF(STDDEV(company_volume) OVER (PARTITION BY ticker ORDER BY Date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) AS volume_zscore_20d,

            -- Sector momentum: average daily_return across same-sector peers
            -- on the same date (uses only same-day data, no future leak).
            AVG(daily_return) OVER (PARTITION BY Sector, Date) AS sector_daily_return
        FROM base
        ORDER BY ticker, Date
    """).df()
    conn.close()
    print(f"  -> Loaded {len(df)} observations for {df['ticker'].nunique()} tickers")
    return df


def build_feature_matrices(df_obs, company_embeddings, embed_dim_real):
    print("\n[5/5] Building feature matrices...")
    embed_cols = [f"emb_{i}" for i in range(embed_dim_real)]
    emb_df = pd.DataFrame.from_dict(
        company_embeddings, orient="index", columns=embed_cols
    ).reset_index().rename(columns={"index": "ticker"})

    # PCA-reduce the embedding axis to balance the signal against the small
    # tabular feature set. PCA is fit on the unique company embeddings (not on
    # the row-replicated observation matrix) to avoid biasing toward
    # heavy-history tickers.
    raw_emb = emb_df[embed_cols].values
    pca = PCA(n_components=min(PCA_DIM, raw_emb.shape[1]), random_state=SEED)
    reduced_emb = pca.fit_transform(raw_emb)
    pca_cols = [f"kg_pc{i}" for i in range(reduced_emb.shape[1])]
    emb_df_reduced = pd.DataFrame(reduced_emb, columns=pca_cols)
    emb_df_reduced["ticker"] = emb_df["ticker"].values
    print(f"  -> PCA: {raw_emb.shape[1]} → {reduced_emb.shape[1]} dims, "
          f"variance retained = {pca.explained_variance_ratio_.sum():.3f}")

    merged = df_obs.merge(emb_df_reduced, on="ticker", how="inner")
    print(f"  -> {len(merged)} observations after embedding join "
          f"({len(df_obs) - len(merged)} dropped — no structural data)")

    control_cols = {"ticker", "Date", "target_7d_up", "Sector",
                    "company_close", "company_volume"}
    tabular_cols = [c for c in merged.columns
                    if c not in control_cols and c not in pca_cols]

    X_tab = merged[tabular_cols].fillna(0).values.astype(np.float32)
    X_emb = merged[pca_cols].fillna(0).values.astype(np.float32)
    X_full = np.concatenate([X_tab, X_emb], axis=1)
    y = merged["target_7d_up"].values.astype(int)

    feature_sets = {
        "tabular_only":      X_tab,
        "embedding_only":    X_emb,
        "tabular+embedding": X_full,
    }
    print(f"  -> Tabular cols ({len(tabular_cols)}): {tabular_cols}")
    print(f"  -> KG-PCA cols ({len(pca_cols)}): {pca_cols}")
    return feature_sets, y, merged


# ── Step 6: Multi-model comparison + stacking ─────────────────────────────────

def make_base_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1, class_weight="balanced",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            alpha=1e-4, learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, random_state=SEED,
        ),
    }
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=SEED, verbose=0, auto_class_weights="Balanced",
        )
    except ImportError:
        print("  [skip] catboost not installed")
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            random_state=SEED, n_jobs=-1, eval_metric="logloss",
            tree_method="hist",
        )
    except ImportError:
        print("  [skip] xgboost not installed")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, max_depth=-1, num_leaves=63,
            learning_rate=0.05, random_state=SEED, n_jobs=-1,
            class_weight="balanced", verbose=-1,
        )
    except ImportError:
        print("  [skip] lightgbm not installed")
    return models


def make_stacking_model(base_models):
    """Stack the three boosted models with a logistic meta-learner."""
    estimators = []
    for name in ("CatBoost", "XGBoost", "LightGBM"):
        if name in base_models:
            estimators.append((name.lower(), base_models[name]))
    if len(estimators) < 2:
        return None
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=SEED),
        stack_method="predict_proba",
        passthrough=False,
        cv=STACK_CV,
        n_jobs=-1,
    )


def _eval_one(model, X_train_s, y_train, X_test_s, y_test):
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        y_proba = y_pred.astype(float)
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="binary", zero_division=0),
        roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan"),
    )


def evaluate_all(feature_sets, y, merged_df):
    print("\n" + "=" * 72)
    print("  Multi-Model Comparison (Tabular vs KG-Embedding vs Combined)")
    print("=" * 72)

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    split = int(len(merged_df) * 0.8)
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {len(y_train)}  Test: {len(y_test)}  "
          f"Test positive rate: {y_test.mean():.3f}")

    results = []

    for feat_name, X_full in feature_sets.items():
        X_train, X_test = X_full[:split], X_full[split:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base_models = make_base_models()
        # Add a stacking ensemble — only meaningful with enough base learners
        stack = make_stacking_model(base_models)
        if stack is not None:
            base_models["Stack"] = stack

        for model_name, model in base_models.items():
            t0 = time.time()
            try:
                acc, f1, auc = _eval_one(model, X_train_s, y_train, X_test_s, y_test)
                elapsed = time.time() - t0
                results.append({
                    "feature_set": feat_name, "model": model_name,
                    "accuracy": acc, "f1": f1, "roc_auc": auc,
                    "time_s": elapsed,
                })
                print(f"  [{feat_name:>17}] {model_name:<13}  "
                      f"acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [{feat_name:>17}] {model_name:<13}  FAILED: {e}")

    print("\n" + "=" * 72)
    print("Summary (sorted by ROC-AUC):")
    print("=" * 72)
    df_res = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(df_res.to_string(index=False, float_format="{:.4f}".format))

    best_combined = df_res[df_res.feature_set == "tabular+embedding"].head(1)
    if len(best_combined):
        b = best_combined.iloc[0]
        print(f"\nBest combined model: {b['model']}  "
              f"(AUC={b['roc_auc']:.4f}, F1={b['f1']:.4f})")
    return df_res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    triples, ent2id, rel2id = extract_structural_triples(FIN_KG_PATH)
    model = train_rotate(triples, ent2id, rel2id)
    company_embeddings, embed_dim_real = extract_company_embeddings(model, ent2id)
    df_obs = load_observation_features(DB_PATH)
    feature_sets, y, merged = build_feature_matrices(df_obs, company_embeddings, embed_dim_real)
    evaluate_all(feature_sets, y, merged)


if __name__ == "__main__":
    main()
