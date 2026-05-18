"""
Data Analysis Pipeline 2: Knowledge Graph Embeddings + ML Classifier.

Pipeline:
  1. Extract structural triples from the financial KG (Company, Sector,
     Industry, Size, Volatility, Country, Region, SubRegion, sharesBorderWith).
     Observation nodes are excluded — they carry the per-day numeric features
     that will be joined separately from DuckDB.
  2. Train a TransE model (PyTorch) on these structural triples to learn a
     dense vector representation for every entity and relation.
  3. Export the learned company embeddings.
  4. Join embeddings with per-observation features and the binary target
     (target_7d_up) from ExploitationZone.duckdb.
  5. Train and evaluate a RandomForest classifier and compare against the
     tabular baseline from stock_prediction_random_forest.py.

When the ownership subgraph (being added by the third team member) is merged
into financial_knowledge_graph.ttl, the TransE model will automatically learn
richer representations without any code change — just re-run this script.
"""

import os
import random
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
FIN_KG_PATH = os.path.join(EXPLOITATION_DIR, "financial_knowledge_graph.ttl")
DB_PATH = os.path.join(EXPLOITATION_DIR, "ExploitationZone.duckdb")

EMBED_DIM = 64
TRANSE_LR = 0.01
TRANSE_EPOCHS = 100
TRANSE_BATCH = 512
TRANSE_MARGIN = 1.0

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FIN_ONTO = Namespace("http://bda.upc.edu/finance/ontology#")
FIN_ENT = Namespace("http://bda.upc.edu/finance/resource/")

# Relations that form the structural backbone (no literal-valued predicates,
# no per-day Observation nodes — those inflate the graph and are already
# covered by the DuckDB join).
STRUCTURAL_RELATIONS = {
    FIN_ONTO.operatesInSector,
    FIN_ONTO.belongsToIndustry,
    FIN_ONTO.hasSize,
    FIN_ONTO.hasVolatilityProfile,
    FIN_ONTO.headquarteredIn,
    FIN_ONTO.locatedInRegion,
    FIN_ONTO.partOfSubRegion,
    FIN_ONTO.sharesBorderWith,
    # Acquisition edges (added in P2 via company_acquisitions dataset)
    FIN_ONTO.madeAcquisition,
    FIN_ONTO.acquisitionCountry,
}


# ── Step 1: Extract structural triples ────────────────────────────────────────

def extract_structural_triples(kg_path: str):
    print("[1/5] Parsing knowledge graph...")
    t0 = time.time()
    g = Graph()
    g.parse(kg_path, format="turtle")
    print(f"  -> Parsed {len(g)} triples in {time.time()-t0:.1f}s")

    triples = []
    for h, r, t in g:
        if r in STRUCTURAL_RELATIONS and isinstance(t, URIRef):
            triples.append((str(h), str(r), str(t)))

    print(f"  -> Retained {len(triples)} structural triples (URI–URI only)")

    # Build entity / relation index
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    print(f"  -> Entities: {len(entities)}, Relations: {len(relations)}")
    return triples, ent2id, rel2id


# ── Step 2: TransE Model ───────────────────────────────────────────────────────

class TransE(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.ent_emb = nn.Embedding(n_entities, dim)
        self.rel_emb = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.ent_emb.weight, -6 / dim**0.5, 6 / dim**0.5)
        nn.init.uniform_(self.rel_emb.weight, -6 / dim**0.5, 6 / dim**0.5)

    def forward(self, h_idx, r_idx, t_idx):
        h = nn.functional.normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = nn.functional.normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)
        return torch.norm(h + r - t, p=2, dim=1)

    def score(self, h_idx, r_idx, t_idx):
        return -self.forward(h_idx, r_idx, t_idx)


def corrupt_triple(h_id, t_id, n_entities):
    if random.random() < 0.5:
        return random.randint(0, n_entities - 1), t_id
    return h_id, random.randint(0, n_entities - 1)


def train_transe(triples, ent2id, rel2id):
    print("\n[2/5] Training TransE model...")
    n_ent = len(ent2id)
    n_rel = len(rel2id)
    model = TransE(n_ent, n_rel, EMBED_DIM)
    optimizer = optim.Adam(model.parameters(), lr=TRANSE_LR)
    margin_loss = nn.MarginRankingLoss(margin=TRANSE_MARGIN)

    triple_ids = [
        (ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples
    ]

    for epoch in range(1, TRANSE_EPOCHS + 1):
        random.shuffle(triple_ids)
        total_loss = 0.0
        for i in range(0, len(triple_ids), TRANSE_BATCH):
            batch = triple_ids[i : i + TRANSE_BATCH]
            h_ids = torch.tensor([b[0] for b in batch])
            r_ids = torch.tensor([b[1] for b in batch])
            t_ids = torch.tensor([b[2] for b in batch])

            # Corrupt each positive triple once
            neg_h_ids, neg_t_ids = zip(
                *[corrupt_triple(h, t, n_ent) for h, t in zip(h_ids.tolist(), t_ids.tolist())]
            )
            neg_h_ids = torch.tensor(neg_h_ids)
            neg_t_ids = torch.tensor(neg_t_ids)

            pos_dist = model(h_ids, r_ids, t_ids)
            neg_dist = model(neg_h_ids, r_ids, neg_t_ids)

            target = -torch.ones(len(batch))
            loss = margin_loss(pos_dist, neg_dist, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3}/{TRANSE_EPOCHS}  loss={total_loss:.4f}")

    print("  -> TransE training complete.")
    return model


# ── Step 3: Extract company embeddings ────────────────────────────────────────

def extract_company_embeddings(model, ent2id):
    print("\n[3/5] Extracting company embeddings...")
    model.eval()
    company_prefix = str(FIN_ENT)
    results = {}
    with torch.no_grad():
        for uri, idx in ent2id.items():
            # Company URIs are short tickers (e.g. http://.../AAPL)
            local = uri.replace(company_prefix, "")
            # Exclude structural category nodes (e.g. Sector_Technology)
            if "_" not in local:
                emb = model.ent_emb.weight[idx].numpy()
                results[local] = emb
    print(f"  -> Extracted embeddings for {len(results)} company nodes")
    return results


# ── Step 4: Join with DuckDB observation data ─────────────────────────────────

def load_observation_features(db_path: str):
    print("\n[4/5] Loading per-observation features from DuckDB...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("""
        SELECT Symbol AS ticker,
               Date,
               target_7d_up,
               company_close,
               company_volume,
               eur_rate,
               jpy_rate,
               MarketCap
        FROM master_dataset
        WHERE target_7d_up IS NOT NULL
    """).df()
    conn.close()
    print(f"  -> Loaded {len(df)} observations for {df['ticker'].nunique()} tickers")
    return df


def build_feature_matrix(df_obs, company_embeddings):
    print("\n[5/5] Building feature matrix...")
    embed_cols = [f"emb_{i}" for i in range(EMBED_DIM)]

    # Convert embedding dict to DataFrame
    emb_df = pd.DataFrame.from_dict(
        company_embeddings, orient="index", columns=embed_cols
    ).reset_index().rename(columns={"index": "ticker"})

    merged = df_obs.merge(emb_df, on="ticker", how="inner")
    print(f"  -> {len(merged)} observations after embedding join "
          f"({len(df_obs) - len(merged)} dropped — no structural data)")

    tabular_features = ["company_close", "company_volume", "eur_rate", "jpy_rate", "MarketCap"]
    all_features = tabular_features + embed_cols

    X = merged[all_features].fillna(0).values
    y = merged["target_7d_up"].values
    return X, y, merged


# ── Step 5: Train and evaluate the classifier ─────────────────────────────────

def train_and_evaluate(X, y, merged_df):
    print("\n" + "=" * 55)
    print("  KG-Embedding + Random Forest Classifier")
    print("=" * 55)

    # Temporal split — same protocol as the baseline RF in P1
    merged_df = merged_df.sort_values("Date")
    split = int(len(merged_df) * 0.8)
    train_idx = merged_df.index[:split]
    test_idx = merged_df.index[split:]

    X_train, X_test = X[: split], X[split:]
    y_train, y_test = y[: split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1
    )
    print(f"Training on {len(X_train)} records (tabular + {EMBED_DIM}-dim KG embedding)...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return clf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    triples, ent2id, rel2id = extract_structural_triples(FIN_KG_PATH)
    model = train_transe(triples, ent2id, rel2id)
    company_embeddings = extract_company_embeddings(model, ent2id)
    df_obs = load_observation_features(DB_PATH)
    X, y, merged = build_feature_matrix(df_obs, company_embeddings)
    train_and_evaluate(X, y, merged)


if __name__ == "__main__":
    main()
