"""
Live News-Sentiment Knowledge-Graph Feature  (LIVE-ONLY — never backtested)
============================================================================

This module adds a real-time news-sentiment signal to the trading agent. It is
deliberately confined to the LIVE execution path and is **never** used in the
walk-forward bake-off, the backtests, or the structural RotatE embeddings.

WHY LIVE-ONLY:
    A leak-free backtest of news sentiment would require a point-in-time (PIT)
    historical news archive — each article timestamped at its true public
    release, scored with only-then-available information. Current-news APIs
    (Alpaca, Finnhub, ...) expose *today's* news, not a clean PIT archive, so
    backtesting against them silently injects look-ahead bias. Rather than
    contaminate the validated methodology, we use news strictly as a live tilt
    and document it as a forward-looking enhancement / future work.

PIPELINE (at live decision time):
    Alpaca News API  ->  per-article sentiment score
                     ->  ephemeral RDF "news-event" graph
                         (NewsEvent nodes linked to Company nodes,
                          carrying sentimentScore + publishedAt literals)
                     ->  SPARQL aggregation  ->  per-ticker mean sentiment

The RDF + SPARQL step keeps the feature thematically a Knowledge-Graph
extension (consistent with the project's semantic-data framing) while staying
completely separate from the structural KG that trains the embeddings.

SENTIMENT BACKEND (pluggable, graceful degradation):
    1. FinBERT  (transformers) — finance-tuned, best quality   [optional]
    2. VADER    (vaderSentiment) — general lexicon            [optional]
    3. Built-in finance lexicon — zero-dependency fallback    [always present]
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, List

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import XSD, RDF

# ── Namespaces for the ephemeral news-event graph ────────────────────────────
NEWS = Namespace("http://bda.upc.edu/finance/news#")
ENT = Namespace("http://bda.upc.edu/finance/resource/")


# ── Sentiment backend ─────────────────────────────────────────────────────────

# Finance-oriented sentiment lexicon. Deliberately small but high-precision for
# market-moving headline vocabulary. Scores are in [-1, +1] per matched term.
_POS_TERMS = {
    "beats": 1.0, "beat": 1.0, "surges": 1.0, "surge": 0.9, "soars": 1.0,
    "jumps": 0.8, "rally": 0.8, "rallies": 0.8, "upgrade": 1.0, "upgraded": 1.0,
    "outperform": 0.9, "raises": 0.8, "raised": 0.7, "record": 0.7, "profit": 0.6,
    "growth": 0.6, "strong": 0.6, "tops": 0.8, "approval": 0.9, "approved": 0.9,
    "wins": 0.7, "win": 0.6, "expands": 0.5, "boost": 0.7, "bullish": 0.9,
    "gains": 0.6, "gain": 0.5, "exceeds": 0.9, "positive": 0.6, "buyback": 0.7,
    "dividend": 0.4, "partnership": 0.5, "breakthrough": 0.9, "rebound": 0.7,
}
_NEG_TERMS = {
    "misses": -1.0, "miss": -0.9, "plunges": -1.0, "plunge": -0.9, "tumbles": -1.0,
    "falls": -0.6, "drops": -0.6, "slumps": -0.9, "downgrade": -1.0, "downgraded": -1.0,
    "underperform": -0.9, "cuts": -0.7, "cut": -0.6, "loss": -0.7, "losses": -0.7,
    "weak": -0.6, "warning": -0.8, "warns": -0.8, "probe": -0.9, "investigation": -0.9,
    "lawsuit": -0.8, "sued": -0.7, "bankruptcy": -1.0, "fraud": -1.0, "recall": -0.8,
    "halts": -0.7, "halt": -0.6, "bearish": -0.9, "slashes": -0.9, "slashed": -0.9,
    "decline": -0.6, "declines": -0.6, "disappoints": -0.9, "fail": -0.8, "fails": -0.8,
    "delays": -0.6, "negative": -0.6, "default": -1.0, "scandal": -1.0, "layoffs": -0.7,
}
_NEGATORS = {"no", "not", "never", "without", "fails", "failed", "avoids", "denies"}

_token_re = re.compile(r"[a-z']+")


def _lexicon_score(text: str) -> float | None:
    """Finance-lexicon sentiment in [-1, 1]; None if no sentiment terms found."""
    if not text:
        return None
    toks = _token_re.findall(text.lower())
    if not toks:
        return None
    score, hits = 0.0, 0
    for i, tok in enumerate(toks):
        val = _POS_TERMS.get(tok) or _NEG_TERMS.get(tok)
        if val is None:
            continue
        # crude negation flip if a negator appears in the preceding 2 tokens
        window = toks[max(0, i - 2):i]
        if any(w in _NEGATORS for w in window):
            val = -val
        score += val
        hits += 1
    if hits == 0:
        return None
    return max(-1.0, min(1.0, score / hits))


_FINBERT = None
_VADER = None


def _get_finbert():
    global _FINBERT
    if _FINBERT is None:
        try:
            from transformers import pipeline
            _FINBERT = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            _FINBERT = False
    return _FINBERT


def _get_vader():
    global _VADER
    if _VADER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER = False
    return _VADER


def score_text(text: str) -> float | None:
    """Sentiment of a headline/summary in [-1, 1]. Tries FinBERT -> VADER ->
    finance lexicon. Returns None when no signal could be extracted."""
    fb = _get_finbert()
    if fb:
        try:
            r = fb(text[:512])[0]
            lab, sc = r["label"].lower(), float(r["score"])
            if lab == "positive":
                return sc
            if lab == "negative":
                return -sc
            return 0.0
        except Exception:
            pass
    vd = _get_vader()
    if vd:
        try:
            return float(vd.polarity_scores(text)["compound"])
        except Exception:
            pass
    return _lexicon_score(text)


def sentiment_backend_name() -> str:
    if _get_finbert():
        return "FinBERT"
    if _get_vader():
        return "VADER"
    return "finance-lexicon"


# ── Alpaca news fetch ─────────────────────────────────────────────────────────

def fetch_news(tickers: List[str], api_key: str, secret_key: str,
               lookback_days: int = 7, per_symbol_limit: int = 50) -> List[dict]:
    """Pull recent news for the given tickers from the Alpaca News API.

    Returns a list of dicts: {ticker, headline, summary, created_at}. Each
    article may map to several symbols; we emit one record per (article, ticker)
    intersection with the requested universe. Fails soft (returns []).
    """
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
    except Exception as e:
        print(f"[News] alpaca news client unavailable: {e}")
        return []
    if not api_key or not secret_key:
        print("[News] no Alpaca credentials — skipping news fetch.")
        return []

    client = NewsClient(api_key, secret_key)
    start = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=lookback_days)
    requested = set(tickers)
    out: List[dict] = []
    # Alpaca caps symbols per request; chunk to be safe.
    CHUNK = 50
    tickers = list(tickers)
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            req = NewsRequest(symbols=",".join(chunk), start=start,
                              limit=per_symbol_limit, include_content=False)
            resp = client.get_news(req)
            articles = getattr(resp, "data", None)
            if isinstance(articles, dict):
                articles = articles.get("news", [])
            elif articles is None:
                articles = getattr(resp, "news", []) or []
            for a in articles:
                headline = getattr(a, "headline", "") or ""
                summary = getattr(a, "summary", "") or ""
                created = getattr(a, "created_at", None)
                syms = getattr(a, "symbols", []) or []
                for s in syms:
                    if s in requested:
                        out.append({"ticker": s, "headline": headline,
                                    "summary": summary, "created_at": str(created)})
        except Exception as e:
            print(f"[News] fetch failed for chunk {i//CHUNK}: {e}")
            continue
    print(f"[News] fetched {len(out)} (article,ticker) records for "
          f"{len(requested)} tickers over {lookback_days}d.")
    return out


# ── Ephemeral RDF news-event graph + SPARQL aggregation ──────────────────────

def build_news_event_graph(news_items: List[dict]) -> Graph:
    """Construct an ephemeral RDF graph of NewsEvent nodes linked to Company
    nodes, each carrying a sentimentScore and publishedAt literal. This graph
    is built fresh at decision time and discarded — it is NOT the structural
    KG and never trains embeddings."""
    g = Graph()
    g.bind("news", NEWS)
    g.bind("ent", ENT)
    for idx, item in enumerate(news_items):
        text = f"{item.get('headline','')}. {item.get('summary','')}".strip()
        s = score_text(text)
        if s is None:
            continue
        ev = URIRef(f"{NEWS}event_{idx}")
        comp = URIRef(f"{ENT}{item['ticker']}")
        g.add((ev, RDF.type, NEWS.NewsEvent))
        g.add((ev, NEWS.aboutCompany, comp))
        g.add((ev, NEWS.sentimentScore, Literal(round(float(s), 4), datatype=XSD.double)))
        if item.get("created_at"):
            g.add((ev, NEWS.publishedAt, Literal(item["created_at"])))
    return g


def aggregate_sentiment_via_sparql(g: Graph) -> Dict[str, dict]:
    """SPARQL: mean sentiment + article count per company in the news graph."""
    q = """
    PREFIX news: <http://bda.upc.edu/finance/news#>
    SELECT ?company (AVG(?s) AS ?mean_sentiment) (COUNT(?ev) AS ?n)
    WHERE {
        ?ev a news:NewsEvent ;
            news:aboutCompany ?company ;
            news:sentimentScore ?s .
    }
    GROUP BY ?company
    """
    res = {}
    for row in g.query(q):
        ticker = str(row.company).split("/")[-1]
        res[ticker] = {"sentiment": float(row.mean_sentiment), "n": int(row.n)}
    return res


def get_live_sentiment(tickers: List[str], api_key: str, secret_key: str,
                       lookback_days: int = 7) -> Dict[str, dict]:
    """End-to-end: fetch news -> RDF news-event graph -> SPARQL aggregation.
    Returns {ticker: {sentiment in [-1,1], n_articles}}. Fails soft to {}."""
    items = fetch_news(tickers, api_key, secret_key, lookback_days=lookback_days)
    if not items:
        return {}
    g = build_news_event_graph(items)
    agg = aggregate_sentiment_via_sparql(g)
    print(f"[News] sentiment via {sentiment_backend_name()} for "
          f"{len(agg)} tickers (of {len(tickers)} requested).")
    return agg
