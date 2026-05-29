"""
Data Analysis Pipeline 2: Knowledge Graph Embeddings + Multi-Model Bake-Off
                          for 30-DAY CROSS-SECTIONAL RANK PREDICTION.

Why 30d-rank and not 7d-up?
---------------------------
The structural KG (sector, industry, size, country, region, borders,
acquisitions) is dominated by near-static categorical labels.  Short-horizon
direction (target_7d_up) is driven by microstructure — momentum, volume
shocks, vol regime — none of which the KG encodes.  At a 30-day horizon, the
**relative ordering** of stocks (which is what a portfolio actually needs)
becomes more strongly tied to sector, country and corporate-structure
factors.  So we regress `target_30d_rank ∈ [0,1]` — the cross-sectional
PERCENT_RANK of 30-day forward return within each Date — and score the
models with metrics that match portfolio construction:

  * Spearman rank IC  — correlation between predicted rank and realised rank
  * Hit rate          — sign agreement between (pred − 0.5) and (rank − 0.5)
  * Decile spread     — mean realised return of the top decile minus the
                        bottom decile of predicted ranks (the "long/short"
                        spread you would actually capture)
  * RMSE              — sanity check on calibration

Pipeline:
  1. Extract structural triples from the financial KG (URI–URI only).
  2. Train a RotatE model (Sun et al. 2019) refined by an R-GCN message-
     passing layer.  Self-adversarial negative sampling, sigmoid-log loss,
     early stopping.
  3. Extract refined company embeddings and compress to PCA_DIM axes.
  4. Join with per-observation features computed by DuckDB window functions
     (multi-window momentum + volatility, calendar, cross-sectional ranks
     within Date, RSI/MACD/Bollinger).  Macro features attached via the
     ticker's resolved HQ country.
  5. Walk-forward CV (expanding window, 5 folds, 30-day embargo between
     train and test).  Each fold trains the bake-off on data up to fold-end
     and evaluates on the next slice.  Per-fold metrics are aggregated as
     mean ± stdev to give an honest confidence interval on the IC.
  6. Bake-off across three feature configurations × four base regressors
     (RandomForest, CatBoost, XGBoost, LightGBM) + Stack (Ridge meta).
     MLP dropped — it was consistently the slowest and weakest model.
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
from scipy.stats import binom, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
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

# ── Part-4 toggles (Priorities 3, 4, 5) ──────────────────────────────────────
# P3: cross-sectional Z standardisation within Date — neutralises level effects
#     across the cross-section so the model has to learn relative ranking
#     instead of absolute magnitudes. Replaces the global StandardScaler when
#     enabled. Defaults to ON.
CS_Z_STANDARDIZE = os.environ.get("CS_Z", "1") == "1"
# P4: small MLP regressor (64-64, BN, dropout, AdamW, temporal 90/10 ES).
ENABLE_MLP = os.environ.get("ENABLE_MLP", "1") == "1"
MLP_EPOCHS = 80
MLP_BATCH = 512
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_DROPOUT = 0.2
MLP_HIDDEN = 64
MLP_PATIENCE = 8
# P5: diverse-ensemble selection threshold on OOF Pearson correlation.
#     Pairs above this threshold are considered redundant.
DIVERSE_CORR_THRESHOLD = float(os.environ.get("DIVERSE_CORR_THRESHOLD", "0.85"))

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
    """Per-observation feature builder.

    Two leakage concerns and how they're handled:

    1. Time-series window features ALWAYS use
           ROWS BETWEEN N PRECEDING AND CURRENT ROW
       i.e. strictly past-and-present data per ticker.  None of them peek at
       LEAD prices that build the target.

    2. PERCENT_RANK() OVER (PARTITION BY Date ORDER BY <feature>) ranks
       across all tickers on the *same* date.  These are SAFE because:
         (a) the feature being ranked is itself past-only — it does not
             use any forward look,
         (b) the cross-section of EOD prices for the same day is available
             live (every other ticker's close on the same date is published
             at the same EOD),
         (c) the rank target's future window (30d) never enters any feature
             computation — only its own forward return defines it.
       Conclusion: the rank features are usable in production by reading
       every ticker's close at EOD before the next session.
    """
    print("\n[4/5] Loading per-observation features from DuckDB...")
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("""
        WITH base AS (
            SELECT
                Symbol AS ticker, Date,
                target_7d_up, target_30d_return, target_30d_rank,
                Sector, Industry,
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
            WHERE target_30d_rank IS NOT NULL
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
    """Join GDP and GDP growth onto each row via the Exploitation-Zone
    `companies` lookup (Symbol→resolved HQ country). The country reconciliation
    lives in the Exploitation Zone (materialised by graph_generation.py); the
    Trusted Zone holds only per-source cleaned tables (schema parity)."""
    conn = duckdb.connect(db_path, read_only=True)
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
    # Sort by Date so the temporal split / walk-forward folds operate on
    # strictly time-ordered rows.
    merged = merged.sort_values("Date").reset_index(drop=True)
    print(f"  -> {len(merged)} observations after embedding join "
          f"({len(df_obs) - len(merged)} dropped — no structural data)")

    # Targets and bookkeeping columns are excluded from the feature matrix.
    control_cols = {"ticker", "Date", "Sector", "Industry",
                    "company_close", "company_volume",
                    "target_7d_up", "target_30d_return", "target_30d_rank"}
    tabular_cols = [c for c in merged.columns
                    if c not in control_cols and c not in pca_cols
                    and pd.api.types.is_numeric_dtype(merged[c])]

    X_tab = merged[tabular_cols].fillna(0).values.astype(np.float32)
    X_emb = merged[pca_cols].fillna(0).values.astype(np.float32)
    X_full = np.concatenate([X_tab, X_emb], axis=1)

    # Primary target: cross-sectional 30-day rank (regression target in [0,1])
    y_rank = merged["target_30d_rank"].astype(float).values
    # Realised 30-day return — used to compute decile spread for the
    # predicted ranks
    y_ret = merged["target_30d_return"].astype(float).values

    feature_sets = {
        "tabular_only":      X_tab,
        "embedding_only":    X_emb,
        "tabular+embedding": X_full,
    }
    print(f"  -> Target: target_30d_rank "
          f"(mean={y_rank.mean():.3f}, std={y_rank.std():.3f})")
    return feature_sets, y_rank, y_ret, merged, pca, tabular_cols, pca_cols


# ── Step 6: Multi-model regression bake-off + Walk-forward CV ─────────────────

# Walk-forward CV — chronological expanding-window folds.  Each fold trains
# on all data up to a cut date and evaluates on the following window.  An
# embargo of EMBARGO_DAYS sits between train and test to prevent the
# 30-day forward-return target from leaking across the boundary.
N_FOLDS = 5
EMBARGO_DAYS = 30


# ── Part 4 P3: cross-sectional Z standardisation ──────────────────────────────

def cross_sectional_zscore(X, dates, eps=1e-8):
    """Z-standardise each column **within each date** using only that date's
    cross-section.

    Critical: each date's stats are computed from that date alone, so this
    transformation is strictly point-in-time — no future information leaks
    into either the train or test slab. Cross-section stats on the test slab
    are taken from the test slab's own per-date cross-section (this is the
    standard quant-finance "neutralisation" treatment; not a leak because
    the cross-section on date d is observable at d's close).

    Parameters
    ----------
    X : (N, F) float array
    dates : (N,) datetime-like array, aligned with X
    eps : safety floor on std to avoid div-by-zero on constant columns

    Returns
    -------
    Z : (N, F) float array, same shape as X
    """
    X = np.asarray(X, dtype=np.float32)
    df = pd.DataFrame(X)
    df["__date__"] = pd.to_datetime(dates)
    grp = df.groupby("__date__")
    means = grp.transform("mean")
    stds = grp.transform("std").fillna(0.0) + eps
    Z = ((df.drop(columns="__date__") - means) / stds).values.astype(np.float32)
    # Clip extreme z-scores so a single thin-cross-section date can't blow up
    # the input distribution. ±6 σ is generous; anything beyond is almost
    # certainly a degenerate cross-section.
    return np.clip(Z, -6.0, 6.0)


# ── Part 4 P4: small MLP regressor ────────────────────────────────────────────

class TorchMLPRegressor:
    """Sklearn-compatible wrapper around a small (64-64) MLP regressor.

    Architecture: Linear(F → 64) → BN → ReLU → Dropout(0.2)
                  → Linear(64 → 64) → BN → ReLU → Dropout(0.2)
                  → Linear(64 → 1)

    Training: AdamW, MSE loss, temporal 90/10 split inside `fit` for early
    stopping on val MSE (patience=8). The temporal split is the natural one
    given X is already sorted chronologically by the caller.
    """

    def __init__(self, in_dim, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT,
                 lr=MLP_LR, weight_decay=MLP_WD,
                 epochs=MLP_EPOCHS, batch=MLP_BATCH,
                 patience=MLP_PATIENCE, device=None, seed=SEED):
        self.in_dim = in_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch = batch
        self.patience = patience
        self.device = device or DEVICE
        self.seed = seed
        self._build()

    def _build(self):
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        ).to(self.device)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n = len(X)
        if n < 50:
            # Too small for ES; just do a plain fit
            self._fit_loop(X, y, X, y)
            return self

        cut = int(n * 0.9)
        X_tr, X_val = X[:cut], X[cut:]
        y_tr, y_val = y[:cut], y[cut:]
        self._fit_loop(X_tr, y_tr, X_val, y_val)
        return self

    def _fit_loop(self, X_tr, y_tr, X_val, y_val):
        opt = optim.AdamW(self.net.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        n_tr = len(X_tr_t)
        best_val = float("inf")
        best_state = None
        bad = 0

        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n_tr, device=self.device)
            for i in range(0, n_tr, self.batch):
                idx = perm[i:i + self.batch]
                xb = X_tr_t[idx]
                yb = y_tr_t[idx]
                # BatchNorm1d needs >1 sample in train mode
                if xb.size(0) < 2:
                    continue
                opt.zero_grad()
                pred = self.net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val_t).squeeze(-1)
                val_loss = float(loss_fn(val_pred, y_val_t).item())
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            out = self.net(xb).squeeze(-1).cpu().numpy()
        return out


def make_base_models():
    """Base regressors and rankers for the bake-off.

    Ridge is included as Part-4-priority-1: the linear baseline.  If the
    tree ensembles fail to beat Ridge by more than ~0.02 IC, the paper's
    narrative has to acknowledge that linearity is doing most of the work
    and the non-linear models are mostly cost without benefit.

    Priority 2: Rank-aware loss on LightGBM, XGBoost, and CatBoost (lambdarank,
    rank:pairwise, YetiRankPairwise). These optimize list-wise/pairwise ordering
    rather than absolute MSE.
    """
    models = {
        # Ridge — alpha is CV-tuned per fold over a wide grid.
        "Ridge": RidgeCV(
            alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
    }
    try:
        from catboost import CatBoostRegressor, CatBoostRanker
        models["CatBoost"] = CatBoostRegressor(
            iterations=1000, depth=4, learning_rate=0.02,
            random_seed=SEED, verbose=0, loss_function="RMSE",
        )
        models["CatBoost_Rank"] = CatBoostRanker(
            iterations=1000, depth=4, learning_rate=0.02,
            random_seed=SEED, verbose=0, loss_function="YetiRankPairwise",
        )
    except ImportError:
        print("  [skip] catboost not installed")
    try:
        from xgboost import XGBRegressor, XGBRanker
        models["XGBoost"] = XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1,
            tree_method="hist", objective="reg:squarederror",
        )
        models["XGBoost_Rank"] = XGBRanker(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1,
            tree_method="hist", objective="rank:pairwise",
            early_stopping_rounds=50,
        )
    except ImportError:
        print("  [skip] xgboost not installed")
    try:
        from lightgbm import LGBMRegressor, LGBMRanker
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        models["LightGBM_Rank"] = LGBMRanker(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
            objective="lambdarank",
        )
    except ImportError:
        print("  [skip] lightgbm not installed")
    # MLP (Part 4 P4) — instantiated lazily because it needs in_dim from X.
    # Caller adds it via make_mlp(in_dim) once it knows the feature width.
    return models


def make_mlp(in_dim):
    """Instantiate a fresh TorchMLPRegressor for a given input width."""
    return TorchMLPRegressor(in_dim=in_dim)


def make_stacking_model(base_models):
    """Stack the three gradient-boosted regressors with a RidgeCV meta-learner.
    Stacking is restricted to standard regressors because StackingRegressor
    does not support group-wise CV split queries required by rankers."""
    estimators = []
    for name in ("CatBoost", "XGBoost", "LightGBM"):
        if name in base_models:
            estimators.append((name.lower(), base_models[name]))
    if len(estimators) < 2:
        return None
    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        passthrough=False,
        cv=STACK_CV,
        n_jobs=-1,
    )


def fit_with_early_stopping(model, X_train_s, y_train, model_name, train_dates=None):
    """Train a regressor/ranker with a 90/10 temporal validation split for early
    stopping. RandomForest and Stack just use a plain fit."""
    val_split = int(len(X_train_s) * 0.9)
    X_tr, X_val = X_train_s[:val_split], X_train_s[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    if model_name.endswith("_Rank"):
        # Map targets in [0, 1] to integer relevance grades in [0, 30] for rankers
        y_tr = np.clip(y_tr * 30, 0, 30).astype(int)
        y_val = np.clip(y_val * 30, 0, 30).astype(int)
        y_train = np.clip(y_train * 30, 0, 30).astype(int)

        if train_dates is not None:
            train_dates_tr = train_dates[:val_split]
            train_dates_val = train_dates[val_split:]

            # Compute date query groups for temporal validation splits.
            # Since merged_df is sorted by Date chronologically, np.unique preserves the order.
            _, groups_tr = np.unique(train_dates_tr, return_counts=True)
            _, groups_val = np.unique(train_dates_val, return_counts=True)
            _, groups_full = np.unique(train_dates, return_counts=True)

            # Map unique dates to group IDs for CatBoost YetiRank
            _, group_ids_tr = np.unique(train_dates_tr, return_inverse=True)
            _, group_ids_val = np.unique(train_dates_val, return_inverse=True)
            _, group_ids_full = np.unique(train_dates, return_inverse=True)
        else:
            groups_tr, groups_val, groups_full = None, None, None
            group_ids_tr, group_ids_val, group_ids_full = None, None, None

        if model_name == "CatBoost_Rank":
            from catboost import Pool
            train_pool = Pool(X_tr, y_tr, group_id=group_ids_tr)
            val_pool = Pool(X_val, y_val, group_id=group_ids_val)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)
        elif model_name == "XGBoost_Rank":
            try:
                model.fit(X_tr, y_tr, group=groups_tr, eval_set=[(X_val, y_val)],
                          eval_group=[groups_val], verbose=False)
            except Exception:
                model.fit(X_train_s, y_train, group=groups_full, verbose=False)
        elif model_name == "LightGBM_Rank":
            try:
                import lightgbm
                model.fit(X_tr, y_tr, group=groups_tr, eval_set=[(X_val, y_val)],
                          eval_group=[groups_val],
                          callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)])
            except Exception:
                model.fit(X_train_s, y_train, group=groups_full)
    else:
        if model_name == "CatBoost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=50, verbose=0)
        elif model_name == "XGBoost":
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50, verbose=False)
            except TypeError:
                try:
                    import xgboost as xgb
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              callbacks=[xgb.callback.EarlyStopping(
                                  rounds=50, save_best=True)],
                              verbose=False)
                except Exception:
                    model.fit(X_train_s, y_train)
        elif model_name == "LightGBM":
            try:
                import lightgbm
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          callbacks=[lightgbm.early_stopping(
                              stopping_rounds=50, verbose=False)])
            except Exception:
                model.fit(X_train_s, y_train)
        elif model_name == "MLP":
            # TorchMLPRegressor.fit does its own internal 90/10 ES, so just
            # hand it the full training tensor.
            model.fit(X_train_s, y_train)
        else:
            # RandomForest, Stack — no early stopping support
            model.fit(X_train_s, y_train)


def regression_metrics(y_true_rank, y_pred_rank, y_true_return,
                       dates=None, trim_days_end=30):
    """Portfolio-relevant metrics for the rank-prediction task.

    All metrics are computed on the test slab passed in.  Decile spread is
    the only one that requires per-date grouping (because what we care about
    is "given the cross-section on date d, the top-decile of predicted ranks
    outperformed the bottom decile by X% over the next 30 days, averaged
    across d").

    Parameters
    ----------
    y_true_rank, y_pred_rank, y_true_return : 1-D arrays, same length
    dates : pd.Series or pd.DatetimeIndex aligned with the arrays.  If None,
            decile_spread_pct is computed on a single pooled bucket (legacy
            behaviour — only used by call-sites that don't pass dates yet).
    trim_days_end : int
            Skip the last `trim_days_end` calendar days of the slab when
            computing decile spread.  Necessary because the 30-day forward
            target on those dates extends past the fold boundary into either
            in-sample data of the next fold or out-of-data.

    Returns dict with:
      ic                : Spearman rank correlation between pred and realised rank
      hit_rate          : fraction of rows where sign(pred-0.5) == sign(rank-0.5)
      decile_spread_pct : mean (over dates) of (top-decile realised 30d ret)
                          minus (bottom-decile realised 30d ret), %
      rmse              : RMSE of [0,1] rank prediction (calibration sanity)
      n_dates           : number of dates that contributed to decile_spread
                          (informational — useful for debugging trim/qcut)
    """
    if len(y_true_rank) < 10 or np.std(y_pred_rank) < 1e-9:
        return dict(ic=np.nan, hit_rate=np.nan,
                    decile_spread_pct=np.nan, rmse=np.nan, n_dates=0)

    ic, _ = spearmanr(y_pred_rank, y_true_rank)
    hit = ((y_pred_rank - 0.5) * (y_true_rank - 0.5) > 0).mean()
    rmse = float(np.sqrt(mean_squared_error(y_true_rank, y_pred_rank)))

    # ── Per-date decile spread (the portfolio-relevant metric) ──────────────
    decile_spread = float("nan")
    n_dates_used = 0
    if dates is not None and len(dates) == len(y_pred_rank):
        d = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "pred": y_pred_rank,
            "ret":  y_true_return,
        })
        # Trim: drop the last `trim_days_end` days of the slab so the 30-day
        # forward return is fully observable inside (or at the edge of) the
        # data window rather than peeking into a future fold.
        max_date = d["date"].max()
        cutoff = max_date - pd.Timedelta(days=trim_days_end)
        d = d[d["date"] <= cutoff]
        n_dates_used = d["date"].nunique()

        per_date_spreads = []
        for _, sub in d.groupby("date"):
            if len(sub) < 10:
                continue
            try:
                deciles = pd.qcut(sub["pred"], q=10, labels=False, duplicates="drop")
            except ValueError:
                continue
            # Need at least the top and bottom buckets to be present
            if deciles.nunique() < 2:
                continue
            top_label = int(deciles.max())
            bot_label = int(deciles.min())
            top = sub.loc[deciles == top_label, "ret"].mean()
            bot = sub.loc[deciles == bot_label, "ret"].mean()
            per_date_spreads.append(top - bot)
        if per_date_spreads:
            decile_spread = float(np.mean(per_date_spreads))
            n_dates_used = len(per_date_spreads)
    elif len(y_pred_rank) >= 20:
        # Legacy fallback: pooled-bucket spread (kept so call-sites without
        # dates still produce *something* rather than NaN).
        cut = max(1, len(y_pred_rank) // 10)
        order = np.argsort(y_pred_rank)
        bot = y_true_return[order[:cut]].mean()
        top = y_true_return[order[-cut:]].mean()
        decile_spread = float(top - bot)
        n_dates_used = -1  # sentinel: pooled, not per-date

    return dict(ic=float(ic), hit_rate=float(hit),
                decile_spread_pct=decile_spread, rmse=rmse,
                n_dates=int(n_dates_used))


# ── Statistical aggregation helpers (Part 3 protocol) ─────────────────────────

def _summarize_with_se(fold_df: pd.DataFrame, group_cols=("feature_set", "model")):
    """For each (feature_set, model), return mean and SE of each metric.
    SE bounds the mean; sigma is fold-to-fold variation — both are reported
    so the paper can distinguish "we don't know the mean" from "the mean
    varies across folds"."""
    metrics = ["ic", "hit_rate", "decile_spread_pct", "rmse"]
    grouped = fold_df.groupby(list(group_cols))
    rows = []
    for keys, sub in grouped:
        K = int(sub["fold"].nunique())
        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_folds"] = K
        for m in metrics:
            mean = sub[m].mean()
            std = sub[m].std(ddof=1) if K > 1 else np.nan
            se = (std / np.sqrt(K)) if K > 1 else np.nan
            row[f"{m}_mean"] = float(mean) if pd.notna(mean) else np.nan
            row[f"{m}_std"] = float(std) if pd.notna(std) else np.nan
            row[f"{m}_se"] = float(se) if pd.notna(se) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False).reset_index(drop=True)


def _sign_tests(fold_df: pd.DataFrame, group_cols=("feature_set", "model")):
    """Per (feature_set, model): count folds with positive IC and positive
    decile spread.  Report one-sided binomial p-value against null=50%."""
    rows = []
    grouped = fold_df.groupby(list(group_cols))
    for keys, sub in grouped:
        K = int(sub["fold"].nunique())
        n_pos_ic = int((sub["ic"] > 0).sum())
        n_pos_ds = int((sub["decile_spread_pct"] > 0).sum())
        # One-sided p-value for at-least-this-many-successes under H0=0.5
        p_ic = float(1 - binom.cdf(n_pos_ic - 1, K, 0.5)) if K > 0 else np.nan
        p_ds = float(1 - binom.cdf(n_pos_ds - 1, K, 0.5)) if K > 0 else np.nan
        rows.append({
            **{c: v for c, v in zip(group_cols, keys)},
            "K": K,
            "n_pos_ic": n_pos_ic,
            "sign_p_ic": p_ic,
            "n_pos_decile_spread": n_pos_ds,
            "sign_p_decile_spread": p_ds,
        })
    return pd.DataFrame(rows)


def _bootstrap_paired_diff_ci(deltas: np.ndarray, n_boot: int = 10_000,
                              seed: int = SEED, ci=(2.5, 97.5)):
    """Bootstrap a CI for the mean of `deltas`.  `deltas` is K paired
    fold-level differences (e.g. IC_modelA - IC_modelB across folds).

    Returns (mean, lo, hi).  At K=5 the CI is genuinely noisy — by design
    we use the sign test as the primary inference and report the bootstrap
    CI as a complementary range estimate.
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    K = len(deltas)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        boot_means[b] = deltas[idx].mean()
    lo, hi = np.percentile(boot_means, ci)
    return float(deltas.mean()), float(lo), float(hi)


def _paired_sign_test_pvalue(deltas: np.ndarray, alternative: str = "two-sided"):
    """Sign test on paired fold-level differences.  Counts positives, runs
    one- or two-sided binomial test against 0.5.  Zeros are dropped."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    deltas = deltas[deltas != 0]
    n = len(deltas)
    if n == 0:
        return float("nan"), 0, 0
    n_pos = int((deltas > 0).sum())
    # Two-sided p-value: 2 * P(X >= max(n_pos, n-n_pos))
    k = max(n_pos, n - n_pos)
    one_sided = float(1 - binom.cdf(k - 1, n, 0.5))
    p = min(1.0, 2.0 * one_sided) if alternative == "two-sided" else one_sided
    return float(p), n_pos, n


def _pairwise_comparisons(fold_df: pd.DataFrame, metric: str,
                          comparisons):
    """For each comparison (e.g. combined - tabular) over the same model,
    compute per-fold paired diffs, sign-test p-value, and bootstrap 95% CI.

    `comparisons` is a list of (config_a, config_b) tuples — the diff is
    a - b.  Model is held constant inside each comparison.
    """
    rows = []
    models = sorted(fold_df["model"].unique())
    for cfg_a, cfg_b in comparisons:
        for model in models:
            a = fold_df[(fold_df.feature_set == cfg_a) & (fold_df.model == model)]
            b = fold_df[(fold_df.feature_set == cfg_b) & (fold_df.model == model)]
            if a.empty or b.empty:
                continue
            merged = a[["fold", metric]].rename(columns={metric: "a"}).merge(
                b[["fold", metric]].rename(columns={metric: "b"}), on="fold")
            deltas = (merged["a"] - merged["b"]).values
            p_sign, n_pos, n = _paired_sign_test_pvalue(deltas)
            mean_d, lo, hi = _bootstrap_paired_diff_ci(deltas)
            rows.append({
                "metric": metric,
                "comparison": f"{cfg_a} - {cfg_b}",
                "model": model,
                "K": n,
                "mean_diff": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_pos": n_pos,
                "sign_p": p_sign,
                "ci_includes_zero": (lo <= 0 <= hi) if not np.isnan(lo) else None,
            })
    return pd.DataFrame(rows)


def make_walk_forward_folds(merged_df, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS):
    """Expanding-window walk-forward folds.

    The time axis is divided into (n_folds + 1) chronological slabs.  Fold k
    (0-indexed) trains on slabs [0..k] and evaluates on slab k+1, with an
    EMBARGO_DAYS gap between the last training day and the first test day.

    Yields tuples (fold_idx, train_mask, test_mask, train_end_date, test_start_date).
    """
    dates = pd.to_datetime(merged_df["Date"].values)
    unique_dates = np.array(sorted(set(dates)))
    n_blocks = n_folds + 1
    block_size = max(1, len(unique_dates) // n_blocks)

    for k in range(n_folds):
        # Train on data up to the end of block k
        train_end_idx = (k + 1) * block_size - 1
        if train_end_idx >= len(unique_dates):
            break
        train_end_date = unique_dates[train_end_idx]
        test_start_date = train_end_date + pd.Timedelta(days=embargo_days)
        # Test ends at the end of block k+1 (or the data, whichever first)
        test_end_idx = min((k + 2) * block_size - 1, len(unique_dates) - 1)
        test_end_date = unique_dates[test_end_idx]

        train_mask = dates <= train_end_date
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)
        if train_mask.sum() < 100 or test_mask.sum() < 100:
            continue
        yield k, train_mask, test_mask, train_end_date, test_start_date


def evaluate_all(feature_sets, y_rank, y_ret, merged_df,
                 pca=None, tabular_cols=None, pca_cols=None,
                 company_embeddings=None):
    print("\n" + "=" * 72)
    print(f"  Walk-Forward Bake-Off — target_30d_rank, {N_FOLDS} expanding folds, "
          f"{EMBARGO_DAYS}-day embargo")
    print("=" * 72)

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    # Collect per-fold metrics keyed by (feature_set, model_name)
    fold_records = []
    # Per (fold, feature_set, model, ticker, date) prediction rows — used to
    # compute pairwise OOF correlations across models and to drive Part 3
    # statistical tests
    oof_rows = []
    # Predictions from the LAST fold (most recent test slab) — used as the
    # bake-off's "production" prediction set for backtesting + bot loading
    last_fold_predictions = {}
    last_fold_artefacts = {}

    folds = list(make_walk_forward_folds(merged_df))
    print(f"  -> Built {len(folds)} walk-forward folds")

    # Pre-extract a date series we can index by mask for per-fold qcut
    all_dates = pd.to_datetime(merged_df["Date"]).values
    all_tickers = merged_df["ticker"].values

    for fold_idx, train_mask, test_mask, train_end, test_start in folds:
        print(f"\n--- Fold {fold_idx+1}/{len(folds)}: "
              f"train ≤ {pd.Timestamp(train_end).strftime('%Y-%m-%d')}  "
              f"test ≥ {pd.Timestamp(test_start).strftime('%Y-%m-%d')}  "
              f"({train_mask.sum()} train / {test_mask.sum()} test)")

        # Test-slab date and ticker vectors used for per-date decile spread
        # and OOF persistence.
        test_dates = all_dates[test_mask]
        test_tickers = all_tickers[test_mask]

        train_dates = all_dates[train_mask]

        for feat_name, X_full in feature_sets.items():
            X_train = X_full[train_mask]
            X_test = X_full[test_mask]
            y_train = y_rank[train_mask]
            y_test_rank = y_rank[test_mask]
            y_test_ret = y_ret[test_mask]

            # ── Preprocessing ────────────────────────────────────────────
            # When CS_Z_STANDARDIZE is on, replace global StandardScaler with
            # cross-sectional z-scoring within each Date. This is the
            # standard quant-finance neutralisation that removes
            # market-wide level effects, forcing the model to learn the
            # *relative* ordering of names — which is exactly what the
            # target_30d_rank measures. We still pass a (fit-train)
            # StandardScaler to the artefacts dict so live inference can
            # mirror the same transform on a single observation if needed.
            scaler = StandardScaler()
            scaler.fit(X_train)
            if CS_Z_STANDARDIZE:
                X_train_s = cross_sectional_zscore(X_train, train_dates)
                X_test_s  = cross_sectional_zscore(X_test, test_dates)
            else:
                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

            base_models = make_base_models()
            # MLP needs in_dim and is feature-set / fold-specific
            if ENABLE_MLP:
                base_models["MLP"] = make_mlp(X_train_s.shape[1])
            stack = make_stacking_model(base_models)
            if stack is not None:
                base_models["Stack"] = stack

            preds_this_fold = {}

            for model_name, model in base_models.items():
                t0 = time.time()
                try:
                    fit_with_early_stopping(model, X_train_s, y_train, model_name, train_dates=train_dates)
                    y_pred = model.predict(X_test_s)
                    # Predictions are continuous regression outputs.  Squash
                    # them into [0,1] for IC/decile metrics that interpret
                    # predictions as ranks (any monotone transform is fine
                    # for IC, but [0,1] keeps the RMSE comparable).
                    pred_rank = pd.Series(y_pred).rank(pct=True).values
                    preds_this_fold[model_name] = pred_rank

                    m = regression_metrics(
                        y_test_rank, pred_rank, y_test_ret,
                        dates=test_dates, trim_days_end=0,
                    )
                    fold_records.append(dict(
                        fold=fold_idx, feature_set=feat_name,
                        model=model_name, time_s=time.time()-t0, **m,
                    ))
                    # Persist OOF predictions for downstream correlation +
                    # statistical analysis.
                    oof_rows.append(pd.DataFrame({
                        "fold": fold_idx,
                        "feature_set": feat_name,
                        "model": model_name,
                        "ticker": test_tickers,
                        "date": test_dates,
                        "pred_rank": pred_rank,
                        "y_true_rank": y_test_rank,
                        "y_true_return": y_test_ret,
                    }))
                    print(f"    [{feat_name:>17}] {model_name:<13}  "
                          f"IC={m['ic']:+.4f}  hit={m['hit_rate']:.3f}  "
                          f"dec_spread={m['decile_spread_pct']:+.2f}%  "
                          f"rmse={m['rmse']:.3f}  "
                          f"ndates={m['n_dates']}  ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"    [{feat_name:>17}] {model_name:<13}  FAILED: {e}")

            # Soft-vote ensemble of the tree models for this fold (Ridge
            # excluded — it's the linear baseline, not a member of the
            # tree-ensemble soft-vote).
            mix_models = [m for m in ("RandomForest", "CatBoost", "XGBoost", "LightGBM")
                          if m in preds_this_fold]
            if len(mix_models) >= 2:
                soft = np.mean([preds_this_fold[m] for m in mix_models], axis=0)
                # Re-rank after averaging for a clean [0,1] prediction
                soft = pd.Series(soft).rank(pct=True).values
                m_soft = regression_metrics(
                    y_test_rank, soft, y_test_ret,
                    dates=test_dates, trim_days_end=0,
                )
                fold_records.append(dict(
                    fold=fold_idx, feature_set=feat_name,
                    model="SoftVote", time_s=0.0, **m_soft,
                ))
                oof_rows.append(pd.DataFrame({
                    "fold": fold_idx,
                    "feature_set": feat_name,
                    "model": "SoftVote",
                    "ticker": test_tickers,
                    "date": test_dates,
                    "pred_rank": soft,
                    "y_true_rank": y_test_rank,
                    "y_true_return": y_test_ret,
                }))
                print(f"    [{feat_name:>17}] {'SoftVote':<13}  "
                      f"IC={m_soft['ic']:+.4f}  hit={m_soft['hit_rate']:.3f}  "
                      f"dec_spread={m_soft['decile_spread_pct']:+.2f}%  "
                      f"rmse={m_soft['rmse']:.3f}")
                preds_this_fold["SoftVote"] = soft

            # Soft-vote ensemble of the ranking tree models for this fold
            mix_rank_models = [m for m in ("CatBoost_Rank", "XGBoost_Rank", "LightGBM_Rank")
                               if m in preds_this_fold]
            if len(mix_rank_models) >= 2:
                soft_rank = np.mean([preds_this_fold[m] for m in mix_rank_models], axis=0)
                soft_rank = pd.Series(soft_rank).rank(pct=True).values
                m_soft_rank = regression_metrics(
                    y_test_rank, soft_rank, y_test_ret,
                    dates=test_dates, trim_days_end=0,
                )
                fold_records.append(dict(
                    fold=fold_idx, feature_set=feat_name,
                    model="SoftVote_Rank", time_s=0.0, **m_soft_rank,
                ))
                oof_rows.append(pd.DataFrame({
                    "fold": fold_idx,
                    "feature_set": feat_name,
                    "model": "SoftVote_Rank",
                    "ticker": test_tickers,
                    "date": test_dates,
                    "pred_rank": soft_rank,
                    "y_true_rank": y_test_rank,
                    "y_true_return": y_test_ret,
                }))
                print(f"    [{feat_name:>17}] {'SoftVote_Rank':<13}  "
                      f"IC={m_soft_rank['ic']:+.4f}  hit={m_soft_rank['hit_rate']:.3f}  "
                      f"dec_spread={m_soft_rank['decile_spread_pct']:+.2f}%  "
                      f"rmse={m_soft_rank['rmse']:.3f}")
                preds_this_fold["SoftVote_Rank"] = soft_rank

            # On the LAST fold + combined feature set, freeze artefacts so the
            # backtester / trading bot loads "the model trained on the most
            # recent data prior to live deployment."
            is_last_fold = (fold_idx == folds[-1][0])
            if is_last_fold and feat_name == "tabular+embedding":
                # Choose the best ensembling members for deployment based on this fold's out-of-sample IC
                deployed_mix = mix_models
                if "SoftVote_Rank" in preds_this_fold and "SoftVote" in preds_this_fold:
                    ic_reg = m_soft["ic"]
                    ic_rank = m_soft_rank["ic"]
                    if ic_rank > ic_reg:
                        deployed_mix = mix_rank_models
                        print(f"\n[DEPLOY] Rank-aware SoftVote outperforms standard SoftVote ({ic_rank:+.4f} vs {ic_reg:+.4f} IC).")
                        print(f"         Deploying rankers: {deployed_mix}")
                    else:
                        print(f"\n[DEPLOY] Standard SoftVote outperforms rank-aware SoftVote ({ic_reg:+.4f} vs {ic_rank:+.4f} IC).")
                        print(f"         Deploying regressors: {deployed_mix}")
                else:
                    print(f"\n[DEPLOY] Defaulting to standard regressors: {deployed_mix}")

                last_fold_predictions[feat_name] = {
                    "test_mask": test_mask,
                    "preds": preds_this_fold,
                }
                last_fold_artefacts = {
                    "trained_models": base_models,
                    "mix_models": deployed_mix,
                    "scaler": scaler,
                    "pca": pca,
                    "tabular_cols": tabular_cols,
                    "pca_cols": pca_cols,
                    "company_embeddings": company_embeddings,
                    # Part-4 P3 flag: when True, downstream consumers must
                    # apply per-date cross-sectional z-scoring instead of
                    # `scaler.transform(...)`. Backtests / live bot honour
                    # this flag via the same `cross_sectional_zscore`
                    # function exported from this module.
                    "cs_z_standardize": CS_Z_STANDARDIZE,
                }

    # ── Part 4 P5: diverse-ensemble selector ────────────────────────────────
    # Greedy pick: start from the highest-IC base model and walk down the
    # IC-ranked list, admitting each next candidate iff its OOF Pearson
    # correlation with every already-admitted member is <= DIVERSE_CORR_THRESHOLD.
    # The resulting "SoftVote_Diverse" row is appended both to fold_records
    # (so it shows up in every summary table) and to oof_rows (so it shows up
    # in the OOF correlation matrix).
    # We exclude SoftVote* / Stack / Ridge from the candidate pool — those are
    # themselves ensembles or the linear baseline, not independent learners
    # whose diversity we're trying to leverage.
    EXCLUDE_FROM_DIVERSE = {"SoftVote", "SoftVote_Rank", "Stack"}
    print("\n" + "=" * 86)
    print(f"Part-4 P5 — Diverse-ensemble selector  (OOF Pearson threshold ≤ {DIVERSE_CORR_THRESHOLD})")
    print("=" * 86)

    if oof_rows:
        try:
            tmp_fold_df = pd.DataFrame(fold_records)
            tmp_oof_df = pd.concat(oof_rows, ignore_index=True)
            for feat_name in feature_sets.keys():
                ic_rank = (tmp_fold_df[tmp_fold_df.feature_set == feat_name]
                           .groupby("model")["ic"].mean()
                           .sort_values(ascending=False))
                # Candidate pool: base learners only (exclude ensembles + Ridge)
                candidates = [m for m in ic_rank.index
                              if m not in EXCLUDE_FROM_DIVERSE
                              and m != "Ridge"
                              and not m.endswith("_Rank")]
                if not candidates:
                    print(f"  [{feat_name}] no candidates — skipping")
                    continue

                # OOF correlation pivot for THIS feature_set's candidates
                feat_oof = tmp_oof_df[
                    (tmp_oof_df.feature_set == feat_name)
                    & (tmp_oof_df.model.isin(candidates))
                ]
                pivot_feat = feat_oof.pivot_table(
                    index=["fold", "ticker", "date"],
                    columns="model", values="pred_rank",
                )
                corr_feat = pivot_feat.corr(method="pearson")

                selected = []
                rejected = []
                for cand in candidates:
                    if cand not in corr_feat.columns:
                        continue
                    if not selected:
                        selected.append(cand)
                        continue
                    max_corr = max(abs(corr_feat.loc[cand, s]) for s in selected)
                    if max_corr <= DIVERSE_CORR_THRESHOLD:
                        selected.append(cand)
                    else:
                        rejected.append((cand, float(max_corr)))

                rej_str = (", ".join(f"{m}(ρ={r:.2f})" for m, r in rejected)
                           if rejected else "none")
                print(f"  [{feat_name}] selected: {selected}  |  rejected: {rej_str}")

                if len(selected) < 2:
                    print(f"  [{feat_name}] only {len(selected)} member — no ensemble formed")
                    continue

                # Build per-fold SoftVote_Diverse predictions from the
                # already-computed OOF pred_ranks of the selected members.
                for fold_idx in sorted(tmp_oof_df["fold"].unique()):
                    sub = tmp_oof_df[
                        (tmp_oof_df.feature_set == feat_name)
                        & (tmp_oof_df.fold == fold_idx)
                        & (tmp_oof_df.model.isin(selected))
                    ]
                    if sub.empty:
                        continue
                    wide = sub.pivot_table(
                        index=["ticker", "date"],
                        columns="model", values="pred_rank",
                    )
                    avg = wide.mean(axis=1)
                    diverse_rank = avg.rank(pct=True).values
                    # Re-attach true targets, keyed by the same (ticker, date)
                    truth = (sub.drop_duplicates(subset=["ticker", "date"])
                             .set_index(["ticker", "date"])
                             [["y_true_rank", "y_true_return"]]
                             .loc[wide.index])
                    y_t_rank = truth["y_true_rank"].values
                    y_t_ret = truth["y_true_return"].values
                    dates_arr = np.array([d for (_, d) in wide.index])
                    tickers_arr = np.array([t for (t, _) in wide.index])

                    m_div = regression_metrics(
                        y_t_rank, diverse_rank, y_t_ret,
                        dates=dates_arr, trim_days_end=0,
                    )
                    fold_records.append(dict(
                        fold=int(fold_idx), feature_set=feat_name,
                        model="SoftVote_Diverse", time_s=0.0, **m_div,
                    ))
                    oof_rows.append(pd.DataFrame({
                        "fold": int(fold_idx),
                        "feature_set": feat_name,
                        "model": "SoftVote_Diverse",
                        "ticker": tickers_arr,
                        "date": dates_arr,
                        "pred_rank": diverse_rank,
                        "y_true_rank": y_t_rank,
                        "y_true_return": y_t_ret,
                    }))
        except Exception as e:
            print(f"[WARN] Diverse-ensemble selection failed: {e}")

    # ── Persist raw per-fold metrics + OOF predictions ──────────────────────
    fold_df = pd.DataFrame(fold_records)
    per_fold_path = os.path.join(EXPLOITATION_DIR, "per_fold_results.csv")
    try:
        fold_df.to_csv(per_fold_path, index=False)
        print(f"\n[EXPORT] per-fold metrics → {per_fold_path} ({len(fold_df)} rows)")
    except Exception as e:
        print(f"[WARN] Could not write per-fold CSV: {e}")

    if oof_rows:
        try:
            oof_df = pd.concat(oof_rows, ignore_index=True)
            oof_path = os.path.join(EXPLOITATION_DIR, "oof_predictions.parquet")
            oof_df.to_parquet(oof_path, index=False)
            print(f"[EXPORT] OOF predictions → {oof_path} "
                  f"({len(oof_df):,} rows across {oof_df['model'].nunique()} models)")
        except Exception as e:
            print(f"[WARN] Could not write OOF parquet: {e}")

    # ── Summary table: mean (SE = σ/√K) ─────────────────────────────────────
    print("\n" + "=" * 86)
    print("Walk-Forward Summary  —  mean (SE = σ/√K) across folds, sorted by IC mean")
    print("=" * 86)
    summary = _summarize_with_se(fold_df)
    # Pretty-print: mean (SE) for IC, decile spread, hit rate, plus RMSE
    def _fmt_mean_se(row, m):
        v = row[f"{m}_mean"]
        s = row[f"{m}_se"]
        if pd.isna(v):
            return "    n/a       "
        if pd.isna(s):
            return f"{v:+8.4f}        "
        return f"{v:+8.4f} ({s:6.4f})"
    print(f"{'feature_set':<19} {'model':<13} {'IC mean (SE)':<22} "
          f"{'decile% mean (SE)':<22} {'hit mean (SE)':<22} "
          f"{'RMSE':<8} K")
    print("-" * 122)
    for _, r in summary.iterrows():
        print(f"{r['feature_set']:<19} {r['model']:<13} "
              f"{_fmt_mean_se(r,'ic'):<22} "
              f"{_fmt_mean_se(r,'decile_spread_pct'):<22} "
              f"{_fmt_mean_se(r,'hit_rate'):<22} "
              f"{r['rmse_mean']:6.4f}  "
              f"{int(r['n_folds'])}")

    # ── Sign tests on IC and decile spread ───────────────────────────────────
    print("\n" + "=" * 86)
    print("Sign tests  —  H0: per-fold metric has equal chance of being positive")
    print("=" * 86)
    sign_df = _sign_tests(fold_df)
    sign_df = sign_df.merge(
        summary[["feature_set", "model", "ic_mean", "decile_spread_pct_mean"]],
        on=["feature_set", "model"],
    ).sort_values("ic_mean", ascending=False)
    print(f"{'feature_set':<19} {'model':<13}  K  "
          f"IC+/K  p_IC    decile+/K  p_decile")
    print("-" * 86)
    for _, r in sign_df.iterrows():
        print(f"{r['feature_set']:<19} {r['model']:<13} {r['K']:>2}  "
              f"{r['n_pos_ic']}/{r['K']}    {r['sign_p_ic']:.3f}     "
              f"{r['n_pos_decile_spread']}/{r['K']}        "
              f"{r['sign_p_decile_spread']:.3f}")
    try:
        sign_path = os.path.join(EXPLOITATION_DIR, "sign_tests.csv")
        sign_df.to_csv(sign_path, index=False)
        print(f"\n[EXPORT] sign tests → {sign_path}")
    except Exception as e:
        print(f"[WARN] Could not write sign-test CSV: {e}")

    # ── Paired pairwise comparisons + bootstrap CIs ──────────────────────────
    print("\n" + "=" * 86)
    print("Paired pairwise comparisons  —  mean diff [95% bootstrap CI], sign-test p")
    print("=" * 86)
    comparisons = [
        ("tabular+embedding", "tabular_only"),
        ("embedding_only",    "tabular_only"),
        ("tabular+embedding", "embedding_only"),
    ]
    pairwise_blocks = []
    for metric in ("ic", "decile_spread_pct"):
        block = _pairwise_comparisons(fold_df, metric, comparisons)
        block["metric"] = metric
        pairwise_blocks.append(block)
    pairwise_df = pd.concat(pairwise_blocks, ignore_index=True)
    print(f"{'metric':<18} {'comparison':<40} {'model':<13}  "
          f"mean_diff [CI lo, CI hi]      p_sign  CI∋0?")
    print("-" * 110)
    for _, r in pairwise_df.iterrows():
        ci_inc_zero = "yes" if r["ci_includes_zero"] else "no"
        ci_inc_zero = "n/a" if r["ci_includes_zero"] is None else ci_inc_zero
        print(f"{r['metric']:<18} {r['comparison']:<40} {r['model']:<13} "
              f"{r['mean_diff']:+8.4f}  [{r['ci_lo']:+7.4f}, {r['ci_hi']:+7.4f}]   "
              f"{r['sign_p']:.3f}   {ci_inc_zero}")
    try:
        pw_path = os.path.join(EXPLOITATION_DIR, "pairwise_comparisons.csv")
        pairwise_df.to_csv(pw_path, index=False)
        print(f"\n[EXPORT] pairwise comparisons → {pw_path}")
    except Exception as e:
        print(f"[WARN] Could not write pairwise CSV: {e}")

    # ── OOF correlation matrix (combined feature set only) ───────────────────
    if oof_rows:
        try:
            oof_combined = oof_df[
                (oof_df.feature_set == "tabular+embedding")
                & (~oof_df.model.isin(["Stack"]))
            ].copy()
            pivot = oof_combined.pivot_table(
                index=["fold", "ticker", "date"],
                columns="model", values="pred_rank",
            )
            corr = pivot.corr(method="pearson").round(4)
            print("\nOOF pairwise Pearson correlation  (combined feature set):")
            print(corr.to_string(float_format="{:.3f}".format))
            corr_path = os.path.join(EXPLOITATION_DIR, "oof_correlation_matrix.csv")
            corr.to_csv(corr_path)
            print(f"[EXPORT] OOF correlation matrix → {corr_path}")
        except Exception as e:
            print(f"[WARN] OOF correlation matrix failed: {e}")

    # Best (model, feature_set) under the combined config (kept for prose use)
    best = summary[summary.feature_set == "tabular+embedding"].head(1)
    if len(best):
        b = best.iloc[0]
        print(f"\nBest combined model:  {b['model']}  "
              f"IC = {b['ic_mean']:+.4f} (SE {b['ic_se']:.4f}, σ {b['ic_std']:.4f})  "
              f"decile spread = {b['decile_spread_pct_mean']:+.2f}% "
              f"(SE {b['decile_spread_pct_se']:.2f}, σ {b['decile_spread_pct_std']:.2f})")

    # ── Export last-fold predictions + frozen model artefacts ───────────────
    if last_fold_predictions and last_fold_artefacts:
        try:
            test_mask = last_fold_predictions["tabular+embedding"]["test_mask"]
            preds = last_fold_predictions["tabular+embedding"]["preds"]
            export_df = merged_df[test_mask].copy()
            for mname, pvec in preds.items():
                export_df[f"pred_rank_{mname}"] = pvec
            if "SoftVote" in preds:
                export_df["pred_rank"] = preds["SoftVote"]
            elif preds:
                export_df["pred_rank"] = next(iter(preds.values()))
            keep = ["ticker", "Date", "company_close",
                    "target_30d_rank", "target_30d_return", "target_7d_up",
                    "pred_rank"] + [f"pred_rank_{m}" for m in preds.keys()]
            keep = [c for c in keep if c in export_df.columns]
            export_path = os.path.join(EXPLOITATION_DIR, "test_predictions.parquet")
            export_df[keep].to_parquet(export_path, index=False)
            print(f"\n[EXPORT] Saved last-fold predictions to {export_path}")
        except Exception as e:
            print(f"[WARN] Could not export predictions: {e}")

        try:
            import pickle
            best_model_path = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
            with open(best_model_path, "wb") as f:
                pickle.dump(last_fold_artefacts, f)
            print(f"[EXPORT] Saved last-fold artefacts to {best_model_path}")
        except Exception as e:
            print(f"[WARN] Could not pickle artefacts: {e}")

    return summary, fold_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if os.path.exists(EMBED_OUT_PATH):
        print(f"\n[CACHE] Loading pre-trained company embeddings from {EMBED_OUT_PATH}...")
        df_emb = pd.read_parquet(EMBED_OUT_PATH)
        embed_cols = [c for c in df_emb.columns if c.startswith("emb_")]
        embed_dim_real = len(embed_cols)
        company_embeddings = {
            row["ticker"]: row[embed_cols].values.astype(np.float32)
            for _, row in df_emb.iterrows()
        }
        print(f"  -> Loaded embeddings for {len(company_embeddings)} companies (dim={embed_dim_real})")
    else:
        triples, ent2id, rel2id = extract_structural_triples(FIN_KG_PATH)
        model = train_rotate(triples, ent2id, rel2id)
        company_embeddings, embed_dim_real = extract_company_embeddings(model, ent2id)

    df_obs = load_observation_features(DB_PATH)
    df_obs = attach_macro_features(df_obs, DB_PATH, MACRO_KG_PATH)
    feature_sets, y_rank, y_ret, merged, pca, tabular_cols, pca_cols = (
        build_feature_matrices(df_obs, company_embeddings, embed_dim_real)
    )
    evaluate_all(feature_sets, y_rank, y_ret, merged,
                 pca=pca, tabular_cols=tabular_cols, pca_cols=pca_cols,
                 company_embeddings=company_embeddings)


if __name__ == "__main__":
    main()
