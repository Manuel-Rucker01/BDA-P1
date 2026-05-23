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
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
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
    RotatE (Sun et al. 2019) integrated with a Relational Graph Convolutional Network (R-GCN)
    refiner. Entities live in a complex space, and are refined by propagating messages over
    the semantic graph relations via R-GCN before the RotatE scoring distance is computed:
        score = -|| h_refined ∘ r − t_refined ||
    """
    def __init__(self, n_entities: int, n_relations: int, dim: int, edge_index_list):
        super().__init__()
        self.dim = dim
        # Entities: 2 * dim real values = `dim` complex numbers
        self.ent_emb = nn.Embedding(n_entities, 2 * dim)
        # Relations: stored as phases, dim real values each
        self.rel_phase = nn.Embedding(n_relations, dim)
        
        # R-GCN Refiner
        from rgcn_layer import RGCNRefiner
        self.refiner = RGCNRefiner(n_entities, n_relations, 2 * dim, 2 * dim)
        self.edge_index_list = edge_index_list
        
        bound = 6 / dim**0.5
        nn.init.uniform_(self.ent_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -np.pi, np.pi)

    def get_refined_embeddings(self):
        return self.refiner(self.ent_emb.weight, self.edge_index_list)

    def forward(self, h_idx, r_idx, t_idx):
        refined_emb = self.get_refined_embeddings()
        h = refined_emb[h_idx]
        t = refined_emb[t_idx]
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
    from rgcn_layer import build_rgcn_adjacencies
    n_ent = len(ent2id)
    n_rel = len(rel2id)
    edge_index_list = build_rgcn_adjacencies(triples, ent2id, rel2id, n_ent, n_rel, DEVICE)
    model = RotatE(n_ent, n_rel, EMBED_DIM, edge_index_list).to(DEVICE)
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
    with torch.no_grad():
        refined_weights = model.get_refined_embeddings()
    weights = refined_weights.detach().cpu().numpy()
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

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, span_fast=12, span_slow=26):
    ema_fast = series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = series.ewm(span=span_slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_macd_signal(macd_series, span_signal=9):
    return macd_series.ewm(span=span_signal, adjust=False).mean()

def load_observation_features(db_path: str):
    """Per-observation feature builder. Every feature is *strictly past* — uses
    only ROWS BETWEEN N PRECEDING AND CURRENT ROW and PARTITION BY Date for
    cross-sectional rank, so there is no leak from the LEAD(close, 7)
    that builds target_7d_up."""
    print("\n[4/5] Loading per-observation features from DuckDB...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("""
        WITH base AS (
            SELECT
                Symbol AS ticker, Date, target_7d_up, Sector, Industry,
                LOG(NULLIF(MarketCap, 0)) AS log_market_cap,
                eur_rate, jpy_rate, company_close, company_volume,

                -- 1-day return (reversal signal)
                (company_close - LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 1) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS daily_return,

                -- Multi-window cumulative returns (momentum signal)
                (company_close - LAG(company_close, 5)  OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 5)  OVER (PARTITION BY Symbol ORDER BY Date), 0) AS return_5d,
                (company_close - LAG(company_close, 10) OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 10) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS return_10d,
                (company_close - LAG(company_close, 20) OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 20) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS return_20d,
                (company_close - LAG(company_close, 50) OVER (PARTITION BY Symbol ORDER BY Date)) /
                    NULLIF(LAG(company_close, 50) OVER (PARTITION BY Symbol ORDER BY Date), 0) AS return_50d,

                -- Price vs short / long moving averages
                (company_close - AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)) /
                    NULLIF(AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS price_vs_ma5,
                (company_close - AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)) /
                    NULLIF(AVG(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) AS price_vs_ma20,

                -- Stochastic-style position-in-range: where in the past 20-day
                -- high-low band is today's close? 0 = at the low, 1 = at the
                -- high.
                (company_close - MIN(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)) /
                    NULLIF(MAX(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
                         - MIN(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW), 0) AS stoch_20d,

                -- Volume features
                company_volume / NULLIF(AVG(company_volume) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS volume_ratio,

                -- Multi-window rolling volatility
                STDDEV(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)  AS rolling_volatility_5d,
                STDDEV(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS rolling_volatility_10d,
                STDDEV(company_close) OVER (PARTITION BY Symbol ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS rolling_volatility_20d,

                -- Calendar features (DuckDB EXTRACT)
                EXTRACT(DOW   FROM Date) AS day_of_week,
                EXTRACT(MONTH FROM Date) AS month_of_year
            FROM master_dataset
            WHERE target_7d_up IS NOT NULL
        ),
        enriched AS (
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

                -- Cross-sectional sector momentum (same Date, same Sector)
                AVG(daily_return) OVER (PARTITION BY Sector, Date) AS sector_daily_return,
                AVG(return_5d)    OVER (PARTITION BY Sector, Date) AS sector_return_5d,

                -- Cross-sectional ranks WITHIN each Date — robust to scale
                -- shifts and often dominates raw values for stock prediction.
                -- PERCENT_RANK is in [0, 1], 0 = lowest, 1 = highest.
                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY daily_return)    AS rank_daily_return,
                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY return_5d)       AS rank_return_5d,
                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY return_20d)      AS rank_return_20d,
                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY rolling_volatility_10d) AS rank_volatility,
                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY volume_ratio)    AS rank_volume_ratio
            FROM base
        )
        SELECT * FROM enriched ORDER BY ticker, Date
    """).df()
    conn.close()

    # --- Pandas-based advanced technical indicators ---
    print("  -> Computing advanced technical indicators (RSI-14, MACD, BB Width, Lags)...")
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    
    df['rsi_14'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_rsi(x, 14)).fillna(50)
    
    df['macd'] = df.groupby('ticker')['company_close'].transform(lambda x: compute_macd(x, 12, 26)).fillna(0)
    df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: compute_macd_signal(x, 9)).fillna(0)
    
    df['bb_mean'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['bb_std'] = df.groupby('ticker')['company_close'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
    df['bb_width'] = (4 * df['bb_std']) / df['bb_mean'].replace(0, 1e-9)
    df['bb_width'] = df['bb_width'].fillna(0)
    df = df.drop(columns=['bb_mean', 'bb_std'])
    
    # Lagged features (daily_return and volume_ratio lags)
    for lag in [1, 2, 5]:
        df[f'daily_return_lag_{lag}'] = df.groupby('ticker')['daily_return'].shift(lag).fillna(0)
        df[f'volume_ratio_lag_{lag}'] = df.groupby('ticker')['volume_ratio'].shift(lag).fillna(1.0)

    print(f"  -> Loaded {len(df)} observations for {df['ticker'].nunique()} tickers")
    return df


def load_macro_features(macro_ttl_path: str):
    """Read GDP, growth, inflation, and trade per country from the macroeconomic graph and
    return a DataFrame of features. Used to attach macroeconomic context to each ticker via its HQ country."""
    from rdflib import Graph as RdfGraph, Namespace
    g = RdfGraph()
    g.parse(macro_ttl_path, format="turtle")
    macro_onto = Namespace("http://bda.upc.edu/macro/ontology#")
    macro_ent = Namespace("http://bda.upc.edu/macro/resource/")

    rows = []
    for s in set(g.subjects()):
        if not str(s).startswith(str(macro_ent)):
            continue
        country = str(s).replace(str(macro_ent), "").replace("_", " ")
        gdp = g.value(s, macro_onto.gdpUSD)
        growth = g.value(s, macro_onto.gdpGrowthPercent)
        inflation = g.value(s, macro_onto.inflationPercent)
        trade = g.value(s, macro_onto.tradePercentOfGDP)
        interest = g.value(s, macro_onto.interestRatePercent)
        if gdp is not None or growth is not None or inflation is not None or trade is not None or interest is not None:
            rows.append({
                "country": country,
                "gdp_usd": float(gdp) if gdp is not None else None,
                "gdp_growth_pct": float(growth) if growth is not None else None,
                "inflation_pct": float(inflation) if inflation is not None else None,
                "trade_pct": float(trade) if trade is not None else None,
                "interest_rate_pct": float(interest) if interest is not None else None,
            })
    return pd.DataFrame(rows)


def attach_macro_features(df_obs, db_path: str, macro_ttl_path: str):
    """Join GDP and GDP growth onto each row via the TrustedZone `companies`
    table (which carries each ticker's resolved HQ country)."""
    conn = duckdb.connect(os.path.abspath(os.path.join(
        os.path.dirname(db_path), "..", "TrustedZone", "TrustedZone.duckdb")),
        read_only=True)
    companies = conn.execute("SELECT Symbol AS ticker, country FROM companies").df()
    conn.close()
    macro = load_macro_features(macro_ttl_path)

    df = df_obs.merge(companies, on="ticker", how="left")
    df = df.merge(macro, on="country", how="left")
    # Drop the country string — categorical, encoded indirectly via gdp
    df = df.drop(columns=["country"])
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
    # STRICT TEMPORAL ALIGNMENT BUG FIX: Sort by Date to align all datasets before extracting arrays
    merged = merged.sort_values("Date").reset_index(drop=True)
    print(f"  -> {len(merged)} observations after embedding join "
          f"({len(df_obs) - len(merged)} dropped — no structural data)")

    control_cols = {"ticker", "Date", "target_7d_up", "Sector", "Industry",
                    "company_close", "company_volume"}
    tabular_cols = [c for c in merged.columns
                    if c not in control_cols and c not in pca_cols
                    and pd.api.types.is_numeric_dtype(merged[c])]

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
    return feature_sets, y, merged, pca, tabular_cols, pca_cols


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
        # Reduced depth to 4 to prevent overfitting on noisy stock data
        models["CatBoost"] = CatBoostClassifier(
            iterations=1000, depth=4, learning_rate=0.02,
            random_seed=SEED, verbose=0, auto_class_weights="Balanced",
        )
    except ImportError:
        print("  [skip] catboost not installed")
    try:
        from xgboost import XGBClassifier
        # Reduced max_depth to 4 and added subsampling to control variance
        models["XGBoost"] = XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, eval_metric="logloss",
            tree_method="hist",
        )
    except ImportError:
        print("  [skip] xgboost not installed")
    try:
        from lightgbm import LGBMClassifier
        # Reduced num_leaves/max_depth and added bagging/subsampling
        models["LightGBM"] = LGBMClassifier(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1,
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


def fit_with_early_stopping(model, X_train_s, y_train, model_name):
    # For early stopping on boosting models, split training into 90% train and 10% validation temporally
    val_split = int(len(X_train_s) * 0.9)
    X_tr, X_val = X_train_s[:val_split], X_train_s[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]
    
    if model_name == "CatBoost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    elif model_name == "XGBoost":
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            try:
                import xgboost as xgb
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)], 
                    verbose=False
                )
            except Exception:
                model.fit(X_train_s, y_train)
    elif model_name == "LightGBM":
        try:
            import lightgbm
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_val, y_val)], 
                callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)]
            )
        except Exception:
            model.fit(X_train_s, y_train)
    else:
        # Standard fit for RF, MLP
        model.fit(X_train_s, y_train)


def evaluate_all(feature_sets, y, merged_df, pca=None, tabular_cols=None, pca_cols=None, company_embeddings=None):
    print("\n" + "=" * 72)
    print("  Multi-Model Comparison with Advanced Ensembling & Leakage Prevention")
    print("=" * 72)

    # 1. TEMPORAL split WITH EMBARGO (leakage prevention)
    # Target 7d horizon means we must discard the first 7 days of the test set
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    
    # 80/20 index split point
    split_idx = int(len(merged_df) * 0.8)
    split_date = merged_df.iloc[split_idx]["Date"]
    split_date_dt = pd.to_datetime(split_date)
    
    # Embargo date: split_date + 7 days
    embargo_date = split_date_dt + pd.Timedelta(days=7)
    
    # Define masks
    train_mask = merged_df["Date"] < split_date
    test_mask = pd.to_datetime(merged_df["Date"]) >= embargo_date
    
    print(f"Split Date: {split_date}  -> Test Embargo Date: {embargo_date.strftime('%Y-%m-%d')}")
    print(f"Initial split counts - Train: {train_mask.sum()}  Test (Un-embargoed): {len(merged_df) - train_mask.sum()}")
    print(f"Final split counts   - Train: {train_mask.sum()}  Test (With 7d Embargo): {test_mask.sum()}  "
          f"(Dropped {len(merged_df) - train_mask.sum() - test_mask.sum()} overlapping boundary rows)")

    results = []

    for feat_name, X_full in feature_sets.items():
        X_train = X_full[train_mask]
        X_test = X_full[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base_models = make_base_models()
        # Add standard Stack ensemble
        stack = make_stacking_model(base_models)
        if stack is not None:
            base_models["Stack"] = stack

        # Store test probability predictions for custom ensembles
        test_probas = {}

        for model_name, model in base_models.items():
            t0 = time.time()
            try:
                # Fit model (boosting models use early stopping via temporal validation)
                fit_with_early_stopping(model, X_train_s, y_train, model_name)
                
                # Predict
                y_pred = model.predict(X_test_s)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_s)[:, 1]
                    test_probas[model_name] = y_proba
                else:
                    y_proba = y_pred.astype(float)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
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

        # --- 2. Advanced Ensemble Mixtures ---
        # Only mix if we have at least 2 successful boosting models/RandomForest
        mix_models = [m for m in ["CatBoost", "XGBoost", "LightGBM", "RandomForest"] if m in test_probas]
        if len(mix_models) >= 2:
            # A. Soft-Voting (Raw Probability Average)
            soft_proba = np.mean([test_probas[m] for m in mix_models], axis=0)
            soft_pred = (soft_proba >= 0.5).astype(int)
            soft_acc = accuracy_score(y_test, soft_pred)
            soft_f1 = f1_score(y_test, soft_pred, average="binary", zero_division=0)
            soft_auc = roc_auc_score(y_test, soft_proba)
            
            results.append({
                "feature_set": feat_name, "model": "SoftVote_Ensemble",
                "accuracy": soft_acc, "f1": soft_f1, "roc_auc": soft_auc,
                "time_s": 0.0,
            })
            print(f"  [{feat_name:>17}] SoftVote_Ensemble   acc={soft_acc:.4f}  f1={soft_f1:.4f}  auc={soft_auc:.4f}  (mix of {mix_models})")
            
            # Export predictions for backtesting on the combined feature set
            if feat_name == "tabular+embedding":
                try:
                    test_df = merged_df[test_mask].copy()
                    test_df["pred_proba"] = soft_proba
                    pred_out_path = os.path.join(EXPLOITATION_DIR, "test_predictions.parquet")
                    test_df[["ticker", "Date", "company_close", "target_7d_up", "pred_proba"]].to_parquet(pred_out_path, index=False)
                    print(f"  -> [EXPORT] Saved SoftVote_Ensemble predictions to {pred_out_path} for backtesting")
                except Exception as e:
                    print(f"  -> [WARN] Could not export predictions: {e}")
                
                # Pickle best model ensemble, scaler, PCA, and metadata
                try:
                    import pickle
                    best_model_path = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
                    serial_data = {
                        "trained_models": base_models,
                        "mix_models": mix_models,
                        "scaler": scaler,
                        "pca": pca,
                        "tabular_cols": tabular_cols,
                        "pca_cols": pca_cols,
                        "company_embeddings": company_embeddings,
                    }
                    with open(best_model_path, "wb") as f:
                        pickle.dump(serial_data, f)
                    print(f"  -> [EXPORT] Saved best ensemble model & metadata to {best_model_path}")
                except Exception as e:
                    print(f"  -> [WARN] Could not pickle best ensemble model: {e}")
            
            # B. Rank-Average Ensemble (Percentile Rank Average)
            # Extremely robust to model calibration and scale shifts in financial datasets!
            rank_matrix = np.column_stack([pd.Series(test_probas[m]).rank(pct=True).values for m in mix_models])
            rank_avg = np.mean(rank_matrix, axis=1)
            # Threshold rank at median (0.5) to make binary prediction
            rank_pred = (rank_avg >= 0.5).astype(int)
            rank_acc = accuracy_score(y_test, rank_pred)
            rank_f1 = f1_score(y_test, rank_pred, average="binary", zero_division=0)
            rank_auc = roc_auc_score(y_test, rank_avg)
            
            results.append({
                "feature_set": feat_name, "model": "RankAvg_Ensemble",
                "accuracy": rank_acc, "f1": rank_f1, "roc_auc": rank_auc,
                "time_s": 0.0,
            })
            print(f"  [{feat_name:>17}] RankAvg_Ensemble    acc={rank_acc:.4f}  f1={rank_f1:.4f}  auc={rank_auc:.4f}  (mix of {mix_models})")

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
    df_obs = attach_macro_features(df_obs, DB_PATH, MACRO_KG_PATH)
    feature_sets, y, merged, pca, tabular_cols, pca_cols = build_feature_matrices(df_obs, company_embeddings, embed_dim_real)
    evaluate_all(feature_sets, y, merged, pca=pca, tabular_cols=tabular_cols, pca_cols=pca_cols, company_embeddings=company_embeddings)


if __name__ == "__main__":
    main()
