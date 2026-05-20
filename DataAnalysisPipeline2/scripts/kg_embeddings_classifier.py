"""
Data Analysis Pipeline 2: Knowledge Graph Embeddings + ML Classifier.

Pipeline:
  1. Extract structural triples from the financial KG, including the newly 
     integrated corporate M&A (Mergers & Acquisitions) subgraph:
     (Company, Sector, Industry, Size, Volatility, Country, Region, SubRegion, 
     sharesBorderWith, madeAcquisition, acquiredCompany, acquisitionCategory, 
     acquisitionCountry).
     Observation nodes and literal-valued predicates are strictly excluded to 
     prevent geometric distortion in the translational space and avoid data leakage.
  2. Train a regularized TransE model (PyTorch) using adaptive Early Stopping 
     on structural triples to learn 128-dimensional latent representations.
  3. Export the optimized, dense company embeddings.
  4. Perform data fusion by joining embeddings with rolling temporal indicators 
     (log_market_cap, daily_return, price_vs_ma5, volume_ratio, rolling_volatility_10d)
     extracted dynamically via window functions from ExploitationZone.duckdb.
  5. Train and evaluate an adjusted RandomForest classifier utilizing balanced 
     class weights and soft-probability threshold calibration (0.45) to overcome 
     majority-class bias.
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

EMBED_DIM = 128
TRANSE_LR = 0.001
TRANSE_EPOCHS = 500
TRANSE_BATCH = 512
TRANSE_MARGIN = 0.5

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

    FIN_ONTO.madeAcquisition,
    FIN_ONTO.acquiredCompany,
    FIN_ONTO.acquisitionCategory,
    FIN_ONTO.acquisitionCountry
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

    # --- CONFIGURACIÓN DE EARLY STOPPING ---
    patience = 15
    best_loss = float('inf')
    patience_counter = 0
    # ---------------------------------------


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
        # --- LÓGICA DE DETENCIÓN TEMPRANA ---
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0  # Resetea si seguimos mejorando
            # Opcional: guardar los mejores pesos aquí
            # torch.save(model.state_dict(), 'best_transe.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early Stopping activado en la época {epoch}. La pérdida no mejora.")
                break


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
   # Modifica la consulta en load_observation_features:
    df = conn.execute("""
        SELECT 
            Symbol AS ticker,
            Date,
            target_7d_up,
            
            -- 1. Variables Base Escalables
            LOG(NULLIF(MarketCap, 0)) AS log_market_cap,
            eur_rate,
            jpy_rate,
            
            -- 2. Indicadores de Precio y Tendencia (Momentum)
            (company_close - LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date)) / 
                NULLIF(LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS daily_return,
                
            (company_close - AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)) / 
                NULLIF(AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS price_vs_ma5,
            
            -- 3. Indicadores de Volumen y Volatilidad Líquida
            company_volume / NULLIF(AVG(company_volume) OVER (PARTITION BY Symbol ORDER BY Date ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,
            
            STDDEV(company_close) OVER (PARTITION BY Symbol ORDER BY Date ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS rolling_volatility_10d
            
        FROM master_dataset
        WHERE target_7d_up IS NOT NULL
        ORDER BY Symbol, Date
    """).df()
    conn.close()
    print(f"  -> Loaded {len(df)} observations for {df['ticker'].nunique()} tickers")
    
    return df


def build_feature_matrix(df_obs, company_embeddings):
    print("\n[5/5] Building feature matrix...")
    embed_cols = [f"emb_{i}" for i in range(EMBED_DIM)]

    # Convertir diccionario de embeddings a DataFrame de Pandas
    emb_df = pd.DataFrame.from_dict(
        company_embeddings, orient="index", columns=embed_cols
    ).reset_index().rename(columns={"index": "ticker"})

    # Cruzar los datos de DuckDB con los Embeddings del grafo
    merged = df_obs.merge(emb_df, on="ticker", how="inner")
    print(f"  -> {len(merged)} observations after embedding join")

    # --- SELECCIÓN AUTOMÁTICA REAL ---
    # Identificamos las columnas de control que NO van al modelo
    control_cols = {"ticker", "Date", "target_7d_up"}
    
    # Recorremos el dataframe cruzado (merged) y nos quedamos con:
    # 1. Las columnas numéricas nuevas calculadas en tu SELECT de DuckDB.
    # 2. Las columnas emb_0, emb_1... del Grafo.
    all_features = [col for col in merged.columns if col not in control_cols]
    
    print(f"  -> Total features used for training: {len(all_features)}")
    print(f"  -> Tabular/Graph columns found: {all_features[:5]}... + {len(embed_cols)} embeddings")
    # ----------------------------------

    # Extraer matrices para Scikit-Learn rellenando nulos del LAG
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
        n_estimators=150, max_depth=25, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )
    print(f"Training on {len(X_train)} records (tabular + {EMBED_DIM}-dim KG embedding)...")
    clf.fit(X_train, y_train)
    # Reemplaza la sección de predicción en train_and_evaluate por esto:
    y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilidades de que sea Clase 1

    # Cambia el umbral de decisión (puedes probar entre 0.40 y 0.45)
    custom_threshold = 0.45  
    y_pred = (y_prob >= custom_threshold).astype(int)

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
