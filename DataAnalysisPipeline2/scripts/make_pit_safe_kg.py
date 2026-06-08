#!/usr/bin/env python3
"""
make_pit_safe_kg.py -- Produce a point-in-time (PIT) SAFE candidate knowledge graph.

WHY THIS EXISTS (the leak)
--------------------------
The production KG (ExploitationZone/financial_knowledge_graph.ttl) attaches a
static structural edge to every company:

    <Company> onto:hasVolatilityProfile ent:Volatility_{Low,Medium,High}_Volatility .

That volatility class is computed in ExploitationZone/graph_generation.py
(~lines 128-178) as ``STDDEV((Close - Open) / Open)`` over the ENTIRE
``company_history`` table per company -- with NO date partition. It is therefore
a single global bucket derived from a company's FULL price history, including
the FUTURE relative to any given training/backtest date. Baking it into the
company's static graph embedding makes future information available to the model
at every past date: a classic look-ahead leak.

THE FIX
-------
Remove the leaked structural edge entirely:
  * all triples with predicate onto:hasVolatilityProfile, and
  * the now-orphaned onto:VolatilityClass nodes (ent:Volatility_*).

No information is truly lost: point-in-time volatility is already available
cleanly as per-date TABULAR features (rolling_volatility_5d / 10d / 20d),
computed causally elsewhere in the pipeline. Size / sector / country / industry /
acquisition edges are static-by-nature and are KEPT.

This is a STANDALONE post-processor: it does NOT require the TrustedZone DuckDB.
It reads the produced .ttl, strips the leak, and writes a NEW candidate .ttl.
The production .ttl is never modified.

USAGE
-----
    python DataAnalysisPipeline2/scripts/make_pit_safe_kg.py
    # or override paths:
    python DataAnalysisPipeline2/scripts/make_pit_safe_kg.py IN.ttl OUT.ttl
    # or via env:
    PIT_KG_INPUT=... PIT_KG_OUTPUT=... python .../make_pit_safe_kg.py

Idempotent: re-running on an already-clean graph removes 0 edges/nodes and
still emits a valid output file.
"""

import os
import sys
import time

from rdflib import Graph, RDF, URIRef

# --- Namespaces (match graph_generation.py) ---
ONTO = "http://bda.upc.edu/finance/ontology#"
ENT = "http://bda.upc.edu/finance/resource/"

HAS_VOLATILITY_PROFILE = URIRef(ONTO + "hasVolatilityProfile")
VOLATILITY_CLASS = URIRef(ONTO + "VolatilityClass")

# --- Default paths (relative to repo root, resolved from this file's location) ---
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
DEFAULT_INPUT = os.path.join(
    _REPO_ROOT, "ExploitationZone", "financial_knowledge_graph.ttl"
)
DEFAULT_OUTPUT = os.path.join(
    _REPO_ROOT, "ExploitationZone", "financial_knowledge_graph_pitsafe.ttl"
)


def resolve_paths():
    """CLI args take precedence, then env vars, then defaults."""
    in_path = DEFAULT_INPUT
    out_path = DEFAULT_OUTPUT
    in_path = os.environ.get("PIT_KG_INPUT", in_path)
    out_path = os.environ.get("PIT_KG_OUTPUT", out_path)
    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    return os.path.abspath(in_path), os.path.abspath(out_path)


def main():
    in_path, out_path = resolve_paths()

    if os.path.abspath(in_path) == os.path.abspath(out_path):
        sys.exit("ERROR: refusing to overwrite the input file. Use a distinct output path.")
    if os.path.abspath(out_path) == os.path.abspath(DEFAULT_INPUT):
        sys.exit("ERROR: refusing to overwrite production financial_knowledge_graph.ttl.")
    if not os.path.exists(in_path):
        sys.exit(f"ERROR: input not found: {in_path}")

    print("=" * 70)
    print("PIT-SAFE KG POST-PROCESSOR (remove hasVolatilityProfile leak)")
    print("=" * 70)
    print(f"  Input : {in_path}")
    print(f"  Output: {out_path}")

    print("\n[1/4] Parsing Turtle (large file, may take a few minutes)...")
    t0 = time.time()
    g = Graph()
    g.parse(in_path, format="turtle")
    n_before = len(g)
    print(f"      Parsed {n_before:,} triples in {time.time() - t0:.1f}s")

    # --- Discover the volatility nodes that are the OBJECT of the leaked edge ---
    vol_nodes = set()
    edges = list(g.triples((None, HAS_VOLATILITY_PROFILE, None)))
    for _s, _p, o in edges:
        vol_nodes.add(o)

    print(f"\n[2/4] Found {len(edges):,} hasVolatilityProfile edges "
          f"referencing {len(vol_nodes)} distinct volatility node(s).")

    # --- Remove the leaked edges ---
    for triple in edges:
        g.remove(triple)
    edges_removed = len(edges)

    # --- Remove ALL triples touching those volatility nodes (subject OR object) ---
    # This drops the `?v rdf:type onto:VolatilityClass` statements and any other
    # stray triples referencing the now-orphaned nodes.
    vol_triples_removed = 0
    for v in vol_nodes:
        for triple in list(g.triples((v, None, None))):
            g.remove(triple)
            vol_triples_removed += 1
        for triple in list(g.triples((None, None, v))):
            g.remove(triple)
            vol_triples_removed += 1

    # Belt-and-suspenders: drop any remaining `?x a onto:VolatilityClass` nodes
    # (handles a clean graph or unexpected orphans not reachable via an edge).
    for v, _p, _o in list(g.triples((None, RDF.type, VOLATILITY_CLASS))):
        for triple in list(g.triples((v, None, None))):
            g.remove(triple)
            vol_triples_removed += 1
        for triple in list(g.triples((None, None, v))):
            g.remove(triple)
            vol_triples_removed += 1
        vol_nodes.add(v)

    print(f"\n[3/4] Removed:")
    print(f"      hasVolatilityProfile edges : {edges_removed:,}")
    print(f"      VolatilityClass node triples: {vol_triples_removed:,} "
          f"(across {len(vol_nodes)} node(s))")

    n_after = len(g)

    # --- Bind prefixes so the output stays readable like the original ---
    g.bind("ent", ENT)
    g.bind("onto", ONTO)

    print(f"\n[4/4] Serializing to {out_path} ...")
    t1 = time.time()
    g.serialize(destination=out_path, format="turtle")
    print(f"      Wrote in {time.time() - t1:.1f}s")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Triples BEFORE : {n_before:,}")
    print(f"  Triples AFTER  : {n_after:,}")
    print(f"  Net removed    : {n_before - n_after:,}")
    print(f"  Vol edges removed         : {edges_removed:,} (expected ~1900, 1/company)")
    print(f"  Vol node triples removed  : {vol_triples_removed:,} (expected 3-4 nodes)")
    print(f"  Output: {out_path}")
    if edges_removed == 0:
        print("  NOTE: graph was already PIT-safe (idempotent re-run).")
    print("=" * 70)


if __name__ == "__main__":
    main()
