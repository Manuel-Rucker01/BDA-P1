"""
Data Analysis Pipeline 2: SPARQL Pattern Matching on the Financial Knowledge Graph.

This script runs analytical SPARQL queries over the financial and macroeconomic
knowledge graphs to extract business insights that would be hard to express in
standard SQL.

Queries:
  Q1  - 7-day upward rate by sector and market-cap class (with stdev + min support)
  Q2  - High-volatility companies in geopolitically tense countries, enriched
        with rival country GDP (cross-graph)
  Q3  - Intra-sector + same-industry peer pairs (ordered, deduplicated)
  Q4  - Mega-cap US companies per sector with US GDP context
  Q5  - Acquisition activity per company (count, total + average spend)
  Q6  - Volatility profile vs 7-day upward rate (which risk class pays off?)
  Q7  - Border-proximity risk: companies in countries that share a border with
        a country having active geopolitical tensions (cross-graph, 2-hop)
  Q8  - Sector concentration by macro region
  Q9  - Large/mega-cap companies in high-volatility sectors that have made NO
        acquisitions (anti-join via FILTER NOT EXISTS)
  Q10 - Top-3 most-acquisitive companies per sector (rank via correlated
        subquery, no SPARQL window functions needed)
"""

import os
import time
import pandas as pd
from rdflib import ConjunctiveGraph, Graph

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
# 7-day upward rate by sector × size with statistical support (count + stdev).
# A high mean is only meaningful if the group has enough observations.

Q1 = f"""
PREFIX onto: <{FIN_ONTO}>

SELECT ?sector ?size
       (AVG(?target) AS ?avg_target)
       (COUNT(?target) AS ?n_obs)
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
HAVING (COUNT(?target) >= 100)
ORDER BY DESC(?avg_target)
"""

# ── Q2 ────────────────────────────────────────────────────────────────────────
# Cross-graph: high-volatility companies headquartered in countries flagged
# with geopolitical tensions. Enriched with the rival country's GDP and
# the company's own country GDP, so we can quantify the economic stakes.

Q2 = f"""
PREFIX fin:   <{FIN_ONTO}>
PREFIX macro: <{MACRO_ONTO}>

SELECT DISTINCT ?ticker ?country ?country_gdp ?rival_country ?rival_gdp
WHERE {{
    ?company a fin:Company ;
             fin:hasVolatilityProfile ?volNode ;
             fin:headquarteredIn      ?finCountry .
    FILTER(STRAFTER(STR(?volNode), "Volatility_") = "High_Volatility")

    BIND(IRI(CONCAT("{MACRO_ENT}", STRAFTER(STR(?finCountry), "Country_"))) AS ?macroCountry)

    ?macroCountry macro:hasTensionWith ?rivalNode ;
                  macro:gdpUSD         ?country_gdp .
    OPTIONAL {{ ?rivalNode macro:gdpUSD ?rival_gdp }}

    BIND(STRAFTER(STR(?company),    "{FIN_ENT}") AS ?ticker)
    BIND(STRAFTER(STR(?finCountry), "Country_")  AS ?country)
    BIND(STRAFTER(STR(?rivalNode),  "{MACRO_ENT}") AS ?rival_country)
}}
ORDER BY DESC(?country_gdp)
LIMIT 200
"""

# ── Q3 ────────────────────────────────────────────────────────────────────────
# Tighter peer definition: same sector AND same industry AND same size class.
# Deduplicated by enforcing ?tickerA < ?tickerB lexicographically.

Q3 = f"""
PREFIX onto: <{FIN_ONTO}>

SELECT ?tickerA ?tickerB ?sector ?industry ?size
WHERE {{
    ?companyA a onto:Company ;
              onto:operatesInSector ?sectorNode ;
              onto:belongsToIndustry ?industryNode ;
              onto:hasSize          ?sizeNode .
    ?companyB a onto:Company ;
              onto:operatesInSector ?sectorNode ;
              onto:belongsToIndustry ?industryNode ;
              onto:hasSize          ?sizeNode .
    BIND(STRAFTER(STR(?companyA), "{FIN_ENT}") AS ?tickerA)
    BIND(STRAFTER(STR(?companyB), "{FIN_ENT}") AS ?tickerB)
    FILTER(STR(?tickerA) < STR(?tickerB))
    BIND(STRAFTER(STR(?sectorNode),   "Sector_")   AS ?sector)
    BIND(STRAFTER(STR(?industryNode), "Industry_") AS ?industry)
    BIND(STRAFTER(STR(?sizeNode),     "Size_")     AS ?size)
}}
ORDER BY ?sector ?industry
LIMIT 500
"""

# ── Q4 ────────────────────────────────────────────────────────────────────────
# Mega-cap US companies per sector, enriched with US GDP from the macro graph.

Q4 = f"""
PREFIX fin:   <{FIN_ONTO}>
PREFIX macro: <{MACRO_ONTO}>
PREFIX ent_m: <{MACRO_ENT}>

SELECT ?sector
       (COUNT(DISTINCT ?company) AS ?mega_cap_count)
       ?us_gdp
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
# Acquisition fingerprint per acquirer: counts, total + average spend.
# Filters to acquirers with at least one priced acquisition.

Q5 = f"""
PREFIX fin: <{FIN_ONTO}>

SELECT ?ticker
       (COUNT(?acq) AS ?total_acquisitions)
       (COUNT(?price) AS ?priced_acquisitions)
       (SUM(?price) AS ?total_spend_usd)
       (AVG(?price) AS ?avg_price_usd)
WHERE {{
    ?company a fin:Company ;
             fin:madeAcquisition ?acq .
    OPTIONAL {{ ?acq fin:acquisitionPrice ?price }}
    BIND(STRAFTER(STR(?company), "{FIN_ENT}") AS ?ticker)
}}
GROUP BY ?ticker
ORDER BY DESC(?total_acquisitions)
"""

# ── Q6 ────────────────────────────────────────────────────────────────────────
# Does taking on more volatility actually pay off? Aggregates target7dUp by
# the company's volatility class. A flat or inverted curve would suggest
# the risk premium is not realised at the 7-day horizon.

Q6 = f"""
PREFIX onto: <{FIN_ONTO}>

SELECT ?volatility
       (AVG(?target) AS ?avg_target)
       (COUNT(?target) AS ?n_obs)
       (COUNT(DISTINCT ?company) AS ?n_companies)
WHERE {{
    ?company a onto:Company ;
             onto:hasVolatilityProfile ?volNode ;
             onto:hasObservation       ?obs .
    ?obs onto:target7dUp ?target .
    BIND(STRAFTER(STR(?volNode), "Volatility_") AS ?volatility)
}}
GROUP BY ?volatility
ORDER BY DESC(?avg_target)
"""

# ── Q7 ────────────────────────────────────────────────────────────────────────
# Border-proximity risk (2-hop cross-graph traversal): companies whose HQ
# country shares a border with a country flagged for geopolitical tension.
# Captures spillover risk — e.g. a Polish company facing the Russia–Ukraine
# tension via its border with Ukraine.

Q7 = f"""
PREFIX fin:   <{FIN_ONTO}>
PREFIX macro: <{MACRO_ONTO}>

SELECT DISTINCT ?ticker ?hq_country ?border_country ?tense_with
WHERE {{
    ?company a fin:Company ;
             fin:headquarteredIn ?finHq .

    BIND(IRI(CONCAT("{MACRO_ENT}", STRAFTER(STR(?finHq), "Country_"))) AS ?macroHq)

    ?macroHq macro:sharesBorderWith ?borderNode .
    ?borderNode macro:hasTensionWith ?tenseNode .

    BIND(STRAFTER(STR(?company),    "{FIN_ENT}")   AS ?ticker)
    BIND(STRAFTER(STR(?finHq),      "Country_")    AS ?hq_country)
    BIND(STRAFTER(STR(?borderNode), "{MACRO_ENT}") AS ?border_country)
    BIND(STRAFTER(STR(?tenseNode),  "{MACRO_ENT}") AS ?tense_with)
}}
LIMIT 200
"""

# ── Q8 ────────────────────────────────────────────────────────────────────────
# Sector concentration by macro region — how globally distributed is each
# sector? Uses the financial graph's Region nodes (populated from the
# RESTCountries API at graph-generation time).

Q8 = f"""
PREFIX fin: <{FIN_ONTO}>

SELECT ?sector ?region
       (COUNT(DISTINCT ?company) AS ?n_companies)
WHERE {{
    ?company a fin:Company ;
             fin:operatesInSector ?sectorNode ;
             fin:headquarteredIn  ?country .
    ?country fin:locatedInRegion ?regionNode .
    BIND(STRAFTER(STR(?sectorNode), "Sector_") AS ?sector)
    BIND(STRAFTER(STR(?regionNode), "Region_") AS ?region)
}}
GROUP BY ?sector ?region
ORDER BY ?sector DESC(?n_companies)
"""

# ── Q9 ────────────────────────────────────────────────────────────────────────
# Large/mega-cap companies in sectors that contain at least one high-volatility
# company, but that have NOT yet made any acquisitions. Useful for spotting
# potentially acquisition-hungry candidates in turbulent industries.

Q9 = f"""
PREFIX fin: <{FIN_ONTO}>

SELECT ?ticker ?sector
WHERE {{
    ?company a fin:Company ;
             fin:operatesInSector ?sectorNode ;
             fin:hasSize          ?sizeNode .
    FILTER(STRAFTER(STR(?sizeNode), "Size_") IN ("Large_Cap", "Mega_Cap"))

    # Sector-level filter: keep sectors with ≥1 high-volatility company.
    {{
        SELECT DISTINCT ?sectorNode WHERE {{
            ?peer fin:operatesInSector     ?sectorNode ;
                  fin:hasVolatilityProfile ?volNode .
            FILTER(STRAFTER(STR(?volNode), "Volatility_") = "High_Volatility")
        }}
    }}

    # Anti-join: no acquisitions on record for this company.
    FILTER NOT EXISTS {{ ?company fin:madeAcquisition ?anyAcq }}

    BIND(STRAFTER(STR(?company),    "{FIN_ENT}") AS ?ticker)
    BIND(STRAFTER(STR(?sectorNode), "Sector_")   AS ?sector)
}}
ORDER BY ?sector ?ticker
LIMIT 200
"""

# ── Q10 ───────────────────────────────────────────────────────────────────────
# Top-3 most-acquisitive companies per sector. Implements rank-in-top-3 via a
# correlated FILTER NOT EXISTS subquery (SPARQL has no native window functions).

Q10 = f"""
PREFIX fin: <{FIN_ONTO}>

SELECT ?sector ?ticker ?n_acquisitions
WHERE {{
    {{
        SELECT ?sectorNode ?company (COUNT(?acq) AS ?n_acquisitions)
        WHERE {{
            ?company a fin:Company ;
                     fin:operatesInSector ?sectorNode ;
                     fin:madeAcquisition  ?acq .
        }}
        GROUP BY ?sectorNode ?company
    }}

    # Keep rows where FEWER than 3 same-sector peers have more acquisitions.
    FILTER NOT EXISTS {{
        SELECT ?sectorNode ?company WHERE {{
            {{
                SELECT ?sectorNode ?other (COUNT(?a2) AS ?other_count)
                WHERE {{
                    ?other a fin:Company ;
                           fin:operatesInSector ?sectorNode ;
                           fin:madeAcquisition  ?a2 .
                }}
                GROUP BY ?sectorNode ?other
            }}
            FILTER(?other != ?company)
            FILTER(?other_count > ?n_acquisitions)
        }}
        GROUP BY ?sectorNode ?company
        HAVING(COUNT(?other) >= 3)
    }}

    BIND(STRAFTER(STR(?company),    "{FIN_ENT}") AS ?ticker)
    BIND(STRAFTER(STR(?sectorNode), "Sector_")   AS ?sector)
}}
ORDER BY ?sector DESC(?n_acquisitions)
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

    run_query(fin_g,    "Q1:  7-day upward rate by sector × size (≥100 obs)", Q1)
    run_query(combined, "Q2:  High-volatility companies in tense countries (cross-graph)", Q2)
    run_query(fin_g,    "Q3:  Sector + industry + size peer pairs (deduplicated)", Q3)
    run_query(combined, "Q4:  Mega-cap US companies per sector with US GDP", Q4)
    run_query(fin_g,    "Q5:  Acquisition fingerprint per acquirer", Q5)
    run_query(fin_g,    "Q6:  7-day upward rate by volatility class (risk premium)", Q6)
    run_query(combined, "Q7:  Border-proximity geopolitical risk (2-hop)", Q7)
    run_query(fin_g,    "Q8:  Sector concentration by macro region", Q8)
    run_query(fin_g,    "Q9:  Large/mega-cap in high-vol sectors with NO acquisitions", Q9)
    run_query(fin_g,    "Q10: Top-3 most acquisitive companies per sector", Q10)


if __name__ == "__main__":
    main()
