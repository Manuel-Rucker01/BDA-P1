import os
from collections import defaultdict
from rdflib import Graph

# Configuración de rutas (apuntando a tu ExplotationZone)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLOITATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "ExploitationZone"))
FIN_KG_PATH = os.path.join(EXPLOITATION_DIR, "financial_knowledge_graph.ttl")

def inspect_ttl(kg_path):
    print(f"[INFO] Loading and parsing graph from: {os.path.basename(kg_path)}...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    print(f"[SUCCESS] Total triples found: {len(g)}\n")

    # Diccionario para agrupar ejemplos por cada relación
    relations_map = defaultdict(list)

    # Recorremos el grafo
    for h, r, t in g:
        # Simplificamos las URIs para que sean legibles en la consola
        h_short = str(h).split("#")[-1].split("/")[-1]
        r_short = str(r).split("#")[-1].split("/")[-1]
        t_short = str(t).split("#")[-1].split("/")[-1]
        
        # Guardamos un ejemplo vistoso: Sujeto -> Objeto
        if len(relations_map[r_short]) < 3:  # Guardamos solo 3 ejemplos por relación
            relations_map[r_short].append(f"    👉  {h_short}  ──({r_short})──>  {t_short}")

    print("=" * 70)
    print("      SCHEMA SUMMARY: UNIQUE RELATIONS & TRIPLE EXAMPLES")
    print("=" * 70)
    print(f"Found {len(relations_map)} unique relations in this graph:\n")

    for rel_name, examples in sorted(relations_map.items()):
        print(f"🔗 Relation: {rel_name}")
        for ex in examples:
            print(ex)
        print("-" * 70)

if __name__ == "__main__":
    if os.path.exists(FIN_KG_PATH):
        inspect_ttl(FIN_KG_PATH)
    else:
        print(f"[ERROR] Could not find file at: {FIN_KG_PATH}")