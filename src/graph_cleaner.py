# src/graph_cleaner.py
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

load_dotenv()

NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

GARBAGE_PATTERNS = [
    r'^\[[\d]+\]$',                    # citation numbers [1], [26]
    r'^[a-zA-Zθ∑αβγδεζηλμνξπρστφχψω]$',  # single characters
    r'^\d+$',                           # pure numbers
    r'^et al',                          # et al. shorthands
    r'this (paper|work)',               # self-references
    r'^(ours|we|none|unknown|authors?|unspecified|our work|our pipeline|our best model|evaluation|analysts|directness|pre-training|association for computational linguistics)$',
    r'arXiv:\d',                        # arxiv IDs
    r'^[A-Z]\d+-\d+$',                  # paper IDs like D19-1244
    r'^proc\.',                         # proc. ACL etc
    r'^\[',                             # starts with [
    r'^r\d[a-z0-9]+$',                  # random IDs
    r'^(TS|SS|C[0-9]|EP|PL|EQ|ENS|MLM|MLP|TQA|CO|NSF|IBM|CPU|GPT|LLM|NQ|WQ|CT|R3)$',  # known noisy abbreviations
    r'^\d{4}$',                         # years like 2020
    r'^and Grave',                      # partial author refs
    r'^\w+,\s+\d{4}$',                 # "Author, 2020" citation format
]
def is_garbage_entity(name: str) -> bool:
    name = name.strip()
    if len(name) < 2 or len(name) > 80:
        return True
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    if name.count(' ') > 6 and name[-1] in '.?,':
        return True
    return False


def connect():
    print("🔌 Connecting to Neo4j...")
    gs = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    print("✅ Connected\n")
    return gs


def delete_garbage_dynamically(gs):
    """
    Fetch all entity names and delete any that match garbage patterns.
    No hardcoded list — works for any future papers too.
    """
    print("🗑️  Scanning and deleting garbage entities...")

    # fetch all entity names
    result = gs.structured_query(
        "MATCH (e:__Entity__) RETURN e.name AS name"
    )

    garbage = [row["name"] for row in result if row["name"] and is_garbage_entity(row["name"])]

    print(f"   Found {len(garbage)} garbage entities out of {len(result)} total")

    # delete in batches
    deleted = 0
    for name in garbage:
        gs.structured_query(
            "MATCH (e:__Entity__ {name: $name}) DETACH DELETE e",
            param_map={"name": name}
        )
        print(f"   deleted: {name[:60]}")
        deleted += 1

    print(f"\n   Total deleted: {deleted}\n")


def delete_long_title_nodes(gs):
    """Delete nodes whose names are full paper titles (over 80 chars)."""
    print("📄 Deleting full-title nodes (length > 80 chars)...")
    result = gs.structured_query(
        """
        MATCH (e:__Entity__)
        WHERE size(e.name) > 80
        DETACH DELETE e
        RETURN count(*) AS deleted
        """
    )
    count = result[0]["deleted"] if result else 0
    print(f"   Deleted {count} long-title nodes\n")


def deduplicate_exact(gs):
    """Remove exact duplicate nodes."""
    print("🔁 Removing exact duplicates...")
    gs.structured_query(
        """
        MATCH (a:__Entity__)
        WITH a.name AS name, collect(a) AS nodes
        WHERE size(nodes) > 1
        FOREACH (n IN tail(nodes) | DETACH DELETE n)
        """
    )
    print("   Done\n")


def fix_authored_by(gs):
    """
    Delete all noisy AUTHORED_BY edges.
    Add back only verified ground-truth authors.
    Note: for future dynamic papers, this will be handled
    by the improved extraction prompt.
    """
    print("👤 Fixing AUTHORED_BY relations...")

    result = gs.structured_query(
        "MATCH ()-[r:AUTHORED_BY]->() DELETE r RETURN count(*) AS deleted"
    )
    deleted = result[0]["deleted"] if result else 0
    print(f"   Deleted {deleted} noisy AUTHORED_BY edges")

    # for current 5 papers — verified from paper headers
    correct_authors = [
        ("RAG",      "Patrick Lewis",      "Facebook AI Research"),
        ("RAG",      "Ethan Perez",        "New York University"),
        ("RAG",      "Aleksandra Piktus",  "Facebook AI Research"),
        ("RAG",      "Fabio Petroni",      "Facebook AI Research"),
        ("RAG",      "Vladimir Karpukhin", "Facebook AI Research"),
        ("RAG",      "Naman Goyal",        "Facebook AI Research"),
        ("RAG",      "Heinrich Küttler",   "Facebook AI Research"),
        ("RAG",      "Mike Lewis",         "Facebook AI Research"),
        ("RAG",      "Wen-tau Yih",        "Facebook AI Research"),
        ("RAG",      "Tim Rocktäschel",    "University College London"),
        ("RAG",      "Sebastian Riedel",   "Facebook AI Research"),
        ("RAG",      "Douwe Kiela",        "Facebook AI Research"),
        ("DPR",      "Vladimir Karpukhin", "Facebook AI Research"),
        ("DPR",      "Barlas Oguz",        "Facebook AI Research"),
        ("DPR",      "Sewon Min",          "University of Washington"),
        ("DPR",      "Patrick Lewis",      "Facebook AI Research"),
        ("DPR",      "Ledell Wu",          "Facebook AI Research"),
        ("DPR",      "Sergey Edunov",      "Facebook AI Research"),
        ("DPR",      "Danqi Chen",         "Princeton University"),
        ("DPR",      "Wen-tau Yih",        "Facebook AI Research"),
        ("REALM",    "Kelvin Guu",         "Google Research"),
        ("REALM",    "Kenton Lee",         "Google Research"),
        ("REALM",    "Zora Tung",          "Google Research"),
        ("REALM",    "Panupong Pasupat",   "Google Research"),
        ("REALM",    "Ming-Wei Chang",     "Google Research"),
        ("Fusion-in-Decoder",      "Gautier Izacard",    "Facebook AI Research"),
        ("Fusion-in-Decoder",      "Edouard Grave",      "Facebook AI Research"),
        ("GraphRAG", "Darren Edge",        "Microsoft Research"),
        ("GraphRAG", "Ha Trinh",           "Microsoft Research"),
        ("GraphRAG", "Newman Cheng",       "Microsoft Research"),
        ("GraphRAG", "Joshua Bradley",     "Microsoft Research"),
        ("GraphRAG", "Alex Chao",          "Microsoft Research"),
        ("GraphRAG", "Apurva Mody",        "Microsoft Research"),
        ("GraphRAG", "Steven Truitt",      "Microsoft Research"),
        ("GraphRAG", "Jonathan Larson",    "Microsoft Research"),
    ]

    for paper, author, institution in correct_authors:
        gs.structured_query(
            """
            MERGE (a:__Entity__ {name: $author})
            SET a.label = "AUTHOR"
            WITH a
            MATCH (p:__Entity__ {name: $paper})
            MERGE (p)-[:AUTHORED_BY]->(a)
            WITH a
            MERGE (i:__Entity__ {name: $institution})
            SET i.label = "INSTITUTION"
            MERGE (a)-[:AFFILIATED_WITH]->(i)
            """,
            param_map={"paper": paper, "author": author, "institution": institution}
        )
        print(f"   ✓ {paper} → {author}")

    print()


def inspect(gs):
    print("── Graph after cleaning ─────────────────────────────────")

    result = gs.structured_query(
        "MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC"
    )
    print("  Nodes by type:")
    for row in result:
        print(f"    {str(row['label']):<35} {row['count']}")

    result = gs.structured_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY count DESC"
    )
    print("\n  Relations by type:")
    for row in result:
        print(f"    {row['rel']:<30} {row['count']}")

    print("\n  Authors per paper:")
    for paper in ["RAG", "DPR", "REALM", "Fusion-in-Decoder", "GraphRAG"]:
        result = gs.structured_query(
            """
            MATCH (p:__Entity__ {name: $paper})-[:AUTHORED_BY]->(a)
            RETURN collect(a.name) AS authors
            """,
            param_map={"paper": paper}
        )
        authors = result[0]["authors"] if result else []
        print(f"    {paper:<12}: {', '.join(authors) if authors else 'none'}")


if __name__ == "__main__":
    gs = connect()
    deduplicate_exact(gs)
    delete_garbage_dynamically(gs)
    delete_long_title_nodes(gs)
    fix_authored_by(gs)
    inspect(gs)
    print("\n✅ Graph cleaning complete.")