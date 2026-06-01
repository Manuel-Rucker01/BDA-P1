# v2: PIT News-Sentiment Features in Training — CV Findings

Walk-forward CV (5 folds, 30d embargo), identical embeddings/data/folds; the
only change is +5 point-in-time news features (news_sent_mean_7d, news_count_7d,
news_sent_mean_30d, news_sent_momentum, news_sent_std_7d) joined strictly as-of.

Coverage: 82,618 articles, 1,518/1,890 tickers; 61,496/404,948 modeled
observations (~15%) carry news.

## Combined (tabular+embedding) mean IC across folds

| Model            | baseline (news off) | news on | Δ        |
|------------------|--------------------:|--------:|---------:|
| SoftVote_Diverse |             +0.1053 | +0.1135 | +0.0081  |
| XGBoost          |             +0.1031 | +0.1092 | +0.0062  |
| Stack            |             +0.1010 | +0.1077 | +0.0068  |
| SoftVote         |             +0.1007 | +0.1061 | +0.0054  |
| RandomForest     |             +0.0999 | +0.1018 | +0.0019  |
| LightGBM         |             +0.0893 | +0.0962 | +0.0069  |
| MLP              |             +0.0893 | +0.1033 | +0.0140  |
| CatBoost         |             +0.0676 | +0.0767 | +0.0091  |

SoftVote_Diverse per-fold Δ: [+0.005, -0.008, +0.005, +0.019, +0.019]
→ 4/5 folds improved; lift concentrated in later folds (more data + denser news).

## Honest verdict
Directionally positive on **all 8 models** and **4/5 folds**, but the +0.008 IC
lift is small vs the ~0.043 fold-to-fold std and a 4/5 sign test is p≈0.19 —
**promising, not statistically conclusive** at n=5 folds. Caveats: news coverage
is skewed to large caps; PIT integrity assumed from Alpaca timestamps;
entity-resolution uses current symbols.

## Portfolio-level clean-OOS validation (post-training window, full universe)

Same backtest harness; news model vs news-free baseline on 2026-03-20→2026-05-15:

| Strategy / model        | Cum Return | Sharpe | Max DD  | IR vs B&H |
|-------------------------|-----------:|-------:|--------:|----------:|
| Buy & Hold              |    +22.18% |   5.22 |  -2.41% |        -- |
| Top-K=10, news-free     |    +47.28% |   6.66 |  -2.40% |    +4.37  |
| **Top-K=10, news model**|  **+57.49%** | 6.55 | -3.39% |  **+4.52** |

Verdict: the CV IC lift converts to portfolio P&L — **+10.2 pp return and higher IR**,
at slightly higher volatility. Adopted into the deployed model (PIT features
computed live; v1 tilt disabled). Caveat: a single 2-month / 9-rebalance window.
