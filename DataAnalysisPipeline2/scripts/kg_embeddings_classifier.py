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
from scipy.stats import spearmanr
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
    print(f"  -> Tabular cols ({len(tabular_cols)}): {tabular_cols}")
    print(f"  -> KG-PCA cols ({len(pca_cols)}): {pca_cols}")
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


def make_base_models():
    """Four base regressors for the bake-off.  MLP dropped (was consistently
    the weakest and slowest in the classification bake-off — and the new
    target is regression on a tight [0,1] range, where tree ensembles
    dominate)."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
    }
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=1000, depth=4, learning_rate=0.02,
            random_seed=SEED, verbose=0, loss_function="RMSE",
        )
    except ImportError:
        print("  [skip] catboost not installed")
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1,
            tree_method="hist", objective="reg:squarederror",
        )
    except ImportError:
        print("  [skip] xgboost not installed")
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1000, max_depth=4, num_leaves=15,
            learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        print("  [skip] lightgbm not installed")
    return models


def make_stacking_model(base_models):
    """Stack the three gradient-boosted regressors with a RidgeCV meta-learner."""
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


def fit_with_early_stopping(model, X_train_s, y_train, model_name):
    """Train a regressor with a 90/10 temporal validation split for early
    stopping on the gradient-boosted family.  RandomForest and Stack just
    use a plain fit."""
    val_split = int(len(X_train_s) * 0.9)
    X_tr, X_val = X_train_s[:val_split], X_train_s[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

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
    else:
        # RandomForest, Stack — no early stopping support
        model.fit(X_train_s, y_train)


def regression_metrics(y_true_rank, y_pred_rank, y_true_return):
    """Portfolio-relevant metrics for the rank-prediction task.

    Returns dict with:
      ic        : Spearman correlation between predicted and realised rank
      hit_rate  : fraction of rows where sign(pred-0.5) == sign(rank-0.5)
      decile_spread_pct : mean realised 30d-return of top decile of predicted
                          ranks minus that of the bottom decile (the
                          long/short spread an idealised portfolio captures)
      rmse      : root mean squared error of the [0,1] rank prediction
                  — calibration sanity check
    """
    if len(y_true_rank) < 10 or np.std(y_pred_rank) < 1e-9:
        return dict(ic=np.nan, hit_rate=np.nan,
                    decile_spread_pct=np.nan, rmse=np.nan)

    ic, _ = spearmanr(y_pred_rank, y_true_rank)
    hit = ((y_pred_rank - 0.5) * (y_true_rank - 0.5) > 0).mean()
    rmse = float(np.sqrt(mean_squared_error(y_true_rank, y_pred_rank)))

    # Decile spread: rank the predictions, take top/bottom 10%, compute
    # mean realised RETURN difference (in % units, since target_30d_return
    # is already percent-of-price).
    n = len(y_pred_rank)
    if n >= 20:
        cut = max(1, n // 10)
        order = np.argsort(y_pred_rank)
        bot = y_true_return[order[:cut]].mean()
        top = y_true_return[order[-cut:]].mean()
        decile_spread = float(top - bot)
    else:
        decile_spread = float("nan")

    return dict(ic=float(ic), hit_rate=float(hit),
                decile_spread_pct=decile_spread, rmse=rmse)


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
    # Predictions from the LAST fold (most recent test slab) — used as the
    # bake-off's "production" prediction set for backtesting + bot loading
    last_fold_predictions = {}
    last_fold_artefacts = {}

    folds = list(make_walk_forward_folds(merged_df))
    print(f"  -> Built {len(folds)} walk-forward folds")

    for fold_idx, train_mask, test_mask, train_end, test_start in folds:
        print(f"\n--- Fold {fold_idx+1}/{len(folds)}: "
              f"train ≤ {pd.Timestamp(train_end).strftime('%Y-%m-%d')}  "
              f"test ≥ {pd.Timestamp(test_start).strftime('%Y-%m-%d')}  "
              f"({train_mask.sum()} train / {test_mask.sum()} test)")

        for feat_name, X_full in feature_sets.items():
            X_train = X_full[train_mask]
            X_test = X_full[test_mask]
            y_train = y_rank[train_mask]
            y_test_rank = y_rank[test_mask]
            y_test_ret = y_ret[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            base_models = make_base_models()
            stack = make_stacking_model(base_models)
            if stack is not None:
                base_models["Stack"] = stack

            preds_this_fold = {}

            for model_name, model in base_models.items():
                t0 = time.time()
                try:
                    fit_with_early_stopping(model, X_train_s, y_train, model_name)
                    y_pred = model.predict(X_test_s)
                    # Predictions are continuous regression outputs.  Squash
                    # them into [0,1] for IC/decile metrics that interpret
                    # predictions as ranks (any monotone transform is fine
                    # for IC, but [0,1] keeps the RMSE comparable).
                    pred_rank = pd.Series(y_pred).rank(pct=True).values
                    preds_this_fold[model_name] = pred_rank

                    m = regression_metrics(y_test_rank, pred_rank, y_test_ret)
                    fold_records.append(dict(
                        fold=fold_idx, feature_set=feat_name,
                        model=model_name, time_s=time.time()-t0, **m,
                    ))
                    print(f"    [{feat_name:>17}] {model_name:<13}  "
                          f"IC={m['ic']:+.4f}  hit={m['hit_rate']:.3f}  "
                          f"dec_spread={m['decile_spread_pct']:+.2f}%  "
                          f"rmse={m['rmse']:.3f}  ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"    [{feat_name:>17}] {model_name:<13}  FAILED: {e}")

            # Soft-vote ensemble of the tree models for this fold
            mix_models = [m for m in ("RandomForest", "CatBoost", "XGBoost", "LightGBM")
                          if m in preds_this_fold]
            if len(mix_models) >= 2:
                soft = np.mean([preds_this_fold[m] for m in mix_models], axis=0)
                # Re-rank after averaging for a clean [0,1] prediction
                soft = pd.Series(soft).rank(pct=True).values
                m_soft = regression_metrics(y_test_rank, soft, y_test_ret)
                fold_records.append(dict(
                    fold=fold_idx, feature_set=feat_name,
                    model="SoftVote", time_s=0.0, **m_soft,
                ))
                print(f"    [{feat_name:>17}] {'SoftVote':<13}  "
                      f"IC={m_soft['ic']:+.4f}  hit={m_soft['hit_rate']:.3f}  "
                      f"dec_spread={m_soft['decile_spread_pct']:+.2f}%  "
                      f"rmse={m_soft['rmse']:.3f}")
                preds_this_fold["SoftVote"] = soft

            # On the LAST fold + combined feature set, freeze artefacts so the
            # backtester / trading bot loads "the model trained on the most
            # recent data prior to live deployment."
            is_last_fold = (fold_idx == folds[-1][0])
            if is_last_fold and feat_name == "tabular+embedding":
                last_fold_predictions[feat_name] = {
                    "test_mask": test_mask,
                    "preds": preds_this_fold,
                }
                last_fold_artefacts = {
                    "trained_models": base_models,
                    "mix_models": mix_models,
                    "scaler": scaler,
                    "pca": pca,
                    "tabular_cols": tabular_cols,
                    "pca_cols": pca_cols,
                    "company_embeddings": company_embeddings,
                }

    # ── Aggregate per-fold metrics into mean ± std ──────────────────────────
    print("\n" + "=" * 72)
    print("Walk-Forward Summary  (mean ± std across folds, sorted by IC desc)")
    print("=" * 72)
    fold_df = pd.DataFrame(fold_records)
    agg = (fold_df.groupby(["feature_set", "model"])
           .agg(ic_mean=("ic", "mean"),       ic_std=("ic", "std"),
                hit_mean=("hit_rate", "mean"), hit_std=("hit_rate", "std"),
                ds_mean=("decile_spread_pct", "mean"),
                ds_std=("decile_spread_pct", "std"),
                rmse_mean=("rmse", "mean"),
                n_folds=("ic", "count"))
           .reset_index()
           .sort_values("ic_mean", ascending=False))
    print(agg.to_string(index=False, float_format="{:+.4f}".format))

    # Best (model, feature_set) under the combined config
    best = agg[agg.feature_set == "tabular+embedding"].head(1)
    if len(best):
        b = best.iloc[0]
        print(f"\nBest combined model:  {b['model']}  "
              f"IC = {b['ic_mean']:+.4f} ± {b['ic_std']:.4f}  "
              f"hit = {b['hit_mean']:.3f}  "
              f"decile spread = {b['ds_mean']:+.2f}% ± {b['ds_std']:.2f}%")

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

    return agg, fold_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
