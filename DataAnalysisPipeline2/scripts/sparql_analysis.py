"""
Data Analysis Pipeline 2: SPARQL Pattern Matching on the Financial Knowledge Graph.

This script runs analytical SPARQL queries over the financial and macroeconomic
knowledge graphs to extract business insights that would be hard to express in
standard SQL.

Queries:
  Q1 - Target success rate by sector and market-cap class
  Q2 - Companies with high volatility in geopolitically tense countries (cross-graph)
  Q3 - Intra-sector peer identification (structural graph similarity)
  Q4 - Country economic profile enrichment for US-headquartered sectors
  Q5 - Acquisition volume and total spend per acquiring company
"""

import os
import time
import pandas as pd
from rdflib import ConjunctiveGraph, Namespace, Graph
from rdflib.namespace import RDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
FIN_KG_PATH = os.path.join(EXPLOITATION_DIR, "financial_knowledge_graph.ttl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")

FIN_ONTO = "http://bda.upc.edu/finance/ontology#"
FIN_ENT = "http://bda.upc.edu/finance/resource/"
MACRO_ONTO = "http://bda.upc.edu/macro/ontology#"
MACRO_ENT = "http://bda.upc.edu/macro/resource/"


def load_graphs():
    print("[INFO] Loading financial knowledge graph (this may take ~30s)...")
    t0 = time.time()
    fin_g = Graph()
    fin_g.parse(FIN_KG_PATH, format="turtle")
    print(f"  -> Financial KG loaded: {len(fin_g)} triples ({time.time()-t0:.1f}s)")

    print("[INFO] Loading macroeconomic knowledge graph...")
    macro_g = Graph()
    macro_g.parse(MACRO_KG_PATH, format="turtle")
    print(f"  -> Macroeconomic KG loaded: {len(macro_g)} triples")

    combined = ConjunctiveGraph()
    for triple in fin_g:
        combined.add(triple)
    for triple in macro_g:
        combined.add(triple)
    return combined, fin_g, macro_g


# ── Q1 ────────────────────────────────────────────────────────────────────────

Q1 = f"""
PREFIX onto: <{FIN_ONTO}>

SELECT ?sector ?size (AVG(?target) AS ?avg_target) (COUNT(?obs) AS ?n_obs)
WHERE {{
    ?company a onto:Company ;
             onto:operatesInSector ?sectorNode ;
             onto:hasSize         ?sizeNode ;
             onto:hasObservation  ?obs .
    ?obs onto:target7dUp ?target .
    BIND(STRAFTER(STR(?sectorNode), "Sector_") AS ?sector)
    BIND(STRAFTER(STR(?sizeNode),   "Size_")   AS ?size)
}}
GROUP BY ?sector ?size
ORDER BY DESC(?avg_target)
"""

# ── Q2 ────────────────────────────────────────────────────────────────────────
# Links the financial graph (company → country) with the macro graph
# (country → tension) using URI rewriting, since the two graphs share country
# names but use different namespaces.

Q2 = f"""
PREFIX fin:  <{FIN_ONTO}>
PREFIX macro: <{MACRO_ONTO}>
PREFIX ent_m: <{MACRO_ENT}>

SELECT DISTINCT ?ticker ?country ?rival_country ?gdp_usd
WHERE {{
    ?company a fin:Company ;
             fin:hasVolatilityProfile ?volNode ;
             fin:headquarteredIn      ?finCountry .
    FILTER(STRAFTER(STR(?volNode), "Volatility_") = "High_Volatility")

    # Rewrite financial country URI → macro country URI
    BIND(
        IRI(CONCAT("{MACRO_ENT}", STRAFTER(STR(?finCountry), "Country_")))
        AS ?macroCountry
    )

    ?macroCountry macro:hasTensionWith ?rivalNode ;
                  macro:gdpUSD         ?gdp_usd .

    BIND(STRAFTER(STR(?company),    "{FIN_ENT}")      AS ?ticker)
    BIND(STRAFTER(STR(?finCountry), "Country_")        AS ?country)
    BIND(STRAFTER(STR(?rivalNode),  "{MACRO_ENT}")     AS ?rival_country)
}}
ORDER BY DESC(?gdp_usd)
LIMIT 200
"""

# ── Q3 ────────────────────────────────────────────────────────────────────────
# For each company find its "peers": companies in the SAME sector AND size class.
# Once the ownership subgraph is added by the colleague, this query can be
# extended with OPTIONAL { ?companyA fin:ownedBy ?companyB } to find structural
# similarities between owner and owned peers.

Q3 = f"""
PREFIX onto: <{FIN_ONTO}>

SELECT ?tickerA ?tickerB ?sector ?size
WHERE {{
    ?companyA a onto:Company ;
              onto:operatesInSector ?sectorNode ;
              onto:hasSize          ?sizeNode .
    ?companyB a onto:Company ;
              onto:operatesInSector ?sectorNode ;
              onto:hasSize          ?sizeNode .
    FILTER(?companyA != ?companyB)
    BIND(STRAFTER(STR(?companyA),   "{FIN_ENT}") AS ?tickerA)
    BIND(STRAFTER(STR(?companyB),   "{FIN_ENT}") AS ?tickerB)
    BIND(STRAFTER(STR(?sectorNode), "Sector_")   AS ?sector)
    BIND(STRAFTER(STR(?sizeNode),   "Size_")     AS ?size)
}}
LIMIT 500
"""

# ── Q4 ────────────────────────────────────────────────────────────────────────
# For each sector, how many Mega-Cap companies are based in the US?
# Cross-graph enrichment: attach the US GDP figure as economic context.

Q4 = f"""
PREFIX fin:   <{FIN_ONTO}>
PREFIX macro: <{MACRO_ONTO}>
PREFIX ent_m: <{MACRO_ENT}>

SELECT ?sector (COUNT(DISTINCT ?company) AS ?mega_cap_count) ?us_gdp
WHERE {{
    ?company a fin:Company ;
             fin:operatesInSector ?sectorNode ;
             fin:hasSize          <{FIN_ENT}Size_Mega_Cap> ;
             fin:headquarteredIn  <{FIN_ENT}Country_United_States> .

    ent_m:United_States macro:gdpUSD ?us_gdp .

    BIND(STRAFTER(STR(?sectorNode), "Sector_") AS ?sector)
}}
GROUP BY ?sector ?us_gdp
ORDER BY DESC(?mega_cap_count)
"""


# ── Q5 ────────────────────────────────────────────────────────────────────────
# For each acquiring company: total number of acquisitions, number with a known
# price, and total spend.  Gives a quick M&A activity fingerprint per ticker.

Q5 = f"""
PREFIX fin: <{FIN_ONTO}>
PREFIX ent: <{FIN_ENT}>

SELECT ?ticker
       (COUNT(?acq) AS ?total_acquisitions)
       (COUNT(?price) AS ?priced_acquisitions)
       (SUM(?price) AS ?total_spend_usd)
WHERE {{
    ?company a fin:Company ;
             fin:madeAcquisition ?acq .
    OPTIONAL {{ ?acq fin:acquisitionPrice ?price }}
    BIND(STRAFTER(STR(?company), "{FIN_ENT}") AS ?ticker)
}}
GROUP BY ?ticker
ORDER BY DESC(?total_acquisitions)
"""


def run_query(graph, label, sparql, max_rows=20):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    results = list(graph.query(sparql))
    elapsed = time.time() - t0
    print(f"  ({len(results)} rows in {elapsed:.2f}s)")
    if results:
        cols = [str(v) for v in results[0].labels]
        rows = [[str(cell) if cell is not None else "N/A" for cell in row] for row in results[:max_rows]]
        df = pd.DataFrame(rows, columns=cols)
        print(df.to_string(index=False))
    return results


def main():
    combined, fin_g, macro_g = load_graphs()

    run_query(
        fin_g, "Q1: 7-day upward rate by sector and size class", Q1
    )
    run_query(
        combined,
        "Q2: High-volatility companies in countries with geopolitical tensions",
        Q2,
    )
    run_query(
        fin_g, "Q3: Intra-sector peer pairs (sector + size match)", Q3
    )
    run_query(
        combined,
        "Q4: Mega-cap US companies per sector with US GDP context",
        Q4,
    )
    run_query(
        fin_g,
        "Q5: Acquisition activity per company (count + total spend)",
        Q5,
    )


if __name__ == "__main__":
    main()
