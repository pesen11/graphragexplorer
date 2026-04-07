# src/entity_resolution.py
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

load_dotenv()

NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Canonical name → aliases to merge into it ─────────────────────────────────
ALIAS_MAP = {
    # DPR family
    "DPR": [
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "Dense Passage Retriever",
        "Dense Passage Retrieval",
        "DPR QA system",
        "DPR model",
        "Fixed DPR",
        "Multi DPR",
        "BM25+DPR",
        "DPR trained using 1,000 examples",
        "DPR trained using 10,000 examples",
        "DPR trained using 10k examples",
        "DPR trained using 1k examples",
        "DPR trained using 20,000 examples",
        "DPR trained using 20k examples",
        "DPR trained using 40,000 examples",
        "DPR trained using 40k examples",
        "DPR trained using all examples",
        "Dense Retr. +Transformer ICT +BERT",
        "Dense Retr. +Transformer REALM",
    ],

    # RAG family
    "RAG": [
        "RAG approaches",
        "RAG systems",
        "RAG-S",
        "RAG-Seq",
        "RAG-Seq.",
        "RAG-T",
        "RAG-Tok.",
        "RAG-generated answers",
        "canonical RAG approaches",
        "conventional RAG baseline",
        "adaptive benchmarking for RAG Evaluation",
        "hybrid RAG schemes",
        "na\u00efve RAG",
        "vector RAG",
        "vector RAG approaches",
        "vector RAG baseline",
        "vector RAG performance",
        "vector RAG systems",
        "vector RAG \"semantic search\" approach",
        "Vector RAG",
    ],

    # RAG-Sequence family
    "RAG-Sequence": [
        "RAG-Sequence Model",
        "RAG-Sequence-BM25",
        "RAG-Sequence-Frozen",
    ],

    # RAG-Token family
    "RAG-Token": [
        "RAG-Token Model",
        "RAG-Token-BM25",
        "RAG-Token-Frozen",
    ],

    # GraphRAG family
    "GraphRAG": [
        "Graph RAG",
        "Graph-based RAG approach",
        "GraphRAG Workflow",
        "GraphRAG approach and pipeline",
        "MultiHop-RAG",
    ],

    # BERT family
    "BERT": [
        "BERT (base, uncased",
        "BERT pre-training",
        "BERT-Baseline",
        "BERT-base",
        "BERT-base model",
        "BERT-based tagger",
        "BERT: Pre-training of deep bidirectional transformers for language understanding",
        "BERT's default optimizer",
        "BM25+BERT",
        "uncased BERT-base model",
        "BERTserini",
        "End-to-end open-domain question answering with BERTserini",
        "Multi-Passage BERT",
        "Multi-passage BERT",
        "Multi-passage BERT: A globally normalized BERT model for open-domain question answering",
        "Multi-passage BERT: A globally normalized bert model for open-domain question answering",
        "Passage re-ranking with BERT",
    ],

    # REALM family
    "REALM": [
        "REALM: Retrieval-augmented language model pre-training",
        "REALM retriever+Baseline encoder",
        "REALM with random span masks",
        "REALM with random uniform masks",
        "REALM News",
        "REALM Wiki",
        "REALMNews",
        "REALMWiki",
        "Baseline retriever+REALM encoder",
        "gradient of the REALM pre-training objective",
    ],

    # BART family
    "BART": [
        "BART-large",
    ],

    # Author canonical names → citation shorthands
    "Patrick Lewis": [
        "Lewis et al.",
        "Lewis et al",
    ],
    "Vladimir Karpukhin": [
        "Karpukhin et al.",
        "Karpukhin et al",
    ],
    "Kelvin Guu": [
        "Guu et al.",
        "Guu et al",
    ],
    "Gautier Izacard": [
        "Izacard et al.",
        "Izacard et al",
    ],
    "Darren Edge": [
        "Edge et al.",
        "Edge et al",
    ],
}

# ── Garbage nodes to delete outright ─────────────────────────────────────────
GARBAGE_NODES = [
    # malformed extractions
    "z |x))\n16. (REALM",
    "BERT (base, uncased",
    "BERTEND(s",
    "BERTMASK(j",
    "BERTMASK",
    "BERTSTART(s",
    "joinBERT(x)",
    "joinBERT(x, zbody",
    "\"z |x))\n16. (REALM\"",
    # wrong/noisy authors connected to RAG
    "unknown",
    "Kuratov et al.",
    "Liu et al.",
    "Baumel et al.",
    "Dang",
    "Laskar et al.",
    "Yao et al.",
]


def connect():
    print("🔌 Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    print("✅ Connected\n")
    return graph_store


def deduplicate_nodes(graph_store):
    """
    Merge exact duplicate nodes — same name, multiple nodes.
    Keeps one node and deletes the rest, redirecting all relationships.
    """
    print("🔁 Deduplicating exact duplicate nodes...")

    graph_store.structured_query(
        """
        MATCH (a:__Entity__)
        WITH a.name AS name, collect(a) AS nodes
        WHERE size(nodes) > 1
        FOREACH (n IN tail(nodes) | DETACH DELETE n)
        """
    )
    print("   Done.\n")


def merge_alias_into_canonical(canonical: str, alias: str, graph_store):
    """
    Merge alias node into canonical node using APOC:
    1. Redirect all incoming relationships to canonical
    2. Redirect all outgoing relationships to canonical
    3. Delete the alias node
    """
    graph_store.structured_query(
        """
        MATCH (canonical:__Entity__ {name: $canonical})
        MATCH (alias:__Entity__ {name: $alias})
        WHERE canonical <> alias

        WITH canonical, alias
        MATCH (other)-[r]->(alias)
        WHERE other <> canonical
        CALL apoc.create.relationship(other, type(r), {}, canonical) YIELD rel
        DELETE r

        WITH canonical, alias
        MATCH (alias)-[r]->(other)
        WHERE other <> canonical
        CALL apoc.create.relationship(canonical, type(r), {}, other) YIELD rel
        DELETE r

        WITH alias
        DETACH DELETE alias
        """,
        param_map={"canonical": canonical, "alias": alias}
    )


def delete_garbage(name: str, graph_store):
    """Delete a garbage node and all its relationships."""
    graph_store.structured_query(
        """
        MATCH (e:__Entity__)
        WHERE e.name = $name
        DETACH DELETE e
        """,
        param_map={"name": name}
    )


def add_real_authors(graph_store):
    """
    Manually add the correct authors for each paper
    and connect them with AUTHORED_BY and AFFILIATED_WITH edges.
    These are ground truth facts we know from the papers themselves.
    """
    print("👤 Adding correct author nodes...")

    authors = [
        # RAG paper authors
        {"name": "Patrick Lewis",          "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Ethan Perez",            "paper": "RAG",      "institution": "New York University"},
        {"name": "Aleksandra Piktus",      "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Fabio Petroni",          "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Vladimir Karpukhin",     "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Naman Goyal",            "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Heinrich Küttler",       "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Mike Lewis",             "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Wen-tau Yih",            "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Tim Rocktäschel",        "paper": "RAG",      "institution": "University College London"},
        {"name": "Sebastian Riedel",       "paper": "RAG",      "institution": "Facebook AI Research"},
        {"name": "Douwe Kiela",            "paper": "RAG",      "institution": "Facebook AI Research"},
        # DPR paper authors
        {"name": "Vladimir Karpukhin",     "paper": "DPR",      "institution": "Facebook AI Research"},
        {"name": "Barlas Oguz",            "paper": "DPR",      "institution": "Facebook AI Research"},
        {"name": "Sewon Min",              "paper": "DPR",      "institution": "University of Washington"},
        {"name": "Patrick Lewis",          "paper": "DPR",      "institution": "Facebook AI Research"},
        {"name": "Ledell Wu",              "paper": "DPR",      "institution": "Facebook AI Research"},
        {"name": "Sergey Edunov",          "paper": "DPR",      "institution": "Facebook AI Research"},
        {"name": "Danqi Chen",             "paper": "DPR",      "institution": "Princeton University"},
        {"name": "Wen-tau Yih",            "paper": "DPR",      "institution": "Facebook AI Research"},
        # REALM paper authors
        {"name": "Kelvin Guu",             "paper": "REALM",    "institution": "Google Research"},
        {"name": "Kenton Lee",             "paper": "REALM",    "institution": "Google Research"},
        {"name": "Zora Tung",              "paper": "REALM",    "institution": "Google Research"},
        {"name": "Panupong Pasupat",       "paper": "REALM",    "institution": "Google Research"},
        {"name": "Ming-Wei Chang",         "paper": "REALM",    "institution": "Google Research"},
        # FiD paper authors
        {"name": "Gautier Izacard",        "paper": "FiD",      "institution": "Facebook AI Research"},
        {"name": "Edouard Grave",          "paper": "FiD",      "institution": "Facebook AI Research"},
        # GraphRAG paper authors
        {"name": "Darren Edge",            "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Ha Trinh",               "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Newman Cheng",           "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Joshua Bradley",         "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Alex Chao",              "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Apurva Mody",            "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Steven Truitt",          "paper": "GraphRAG", "institution": "Microsoft Research"},
        {"name": "Jonathan Larson",        "paper": "GraphRAG", "institution": "Microsoft Research"},
    ]

    for entry in authors:
        graph_store.structured_query(
            """
            MERGE (author:__Entity__ {name: $author_name})
            SET author.label = "AUTHOR"
            WITH author
            MATCH (paper:__Entity__ {name: $paper_name})
            MERGE (paper)-[:AUTHORED_BY]->(author)
            WITH author
            MERGE (inst:__Entity__ {name: $institution})
            SET inst.label = "INSTITUTION"
            MERGE (author)-[:AFFILIATED_WITH]->(inst)
            """,
            param_map={
                "author_name":   entry["name"],
                "paper_name":    entry["paper"],
                "institution":   entry["institution"],
            }
        )
        print(f"   ✓ {entry['name']} → {entry['paper']} ({entry['institution']})")

    print()


def run_resolution(graph_store):
    """Run the full entity resolution pass."""

    # Step 1 — deduplicate exact duplicates first
    deduplicate_nodes(graph_store)

    # Step 2 — delete garbage nodes
    print("🗑️  Deleting garbage nodes...")
    for name in GARBAGE_NODES:
        delete_garbage(name, graph_store)
        print(f"   deleted: {name[:60]}")

    # Step 3 — merge aliases into canonical nodes
    print("\n🔗 Merging aliases into canonical nodes...")
    total_merged = 0
    for canonical, aliases in ALIAS_MAP.items():
        if not aliases:
            continue
        print(f"\n  [{canonical}]")
        for alias in aliases:
            try:
                merge_alias_into_canonical(canonical, alias, graph_store)
                print(f"    ✓ merged: {alias[:60]}")
                total_merged += 1
            except Exception as e:
                print(f"    ⚠ skipped {alias[:40]}: {e}")

    print(f"\n✅ Merged {total_merged} alias nodes.")

    # Step 4 — add correct authors manually
    add_real_authors(graph_store)


def inspect_after(graph_store):
    """Show graph stats after resolution."""
    print("\n── Graph after resolution ───────────────────────────────")

    result = graph_store.structured_query(
        "MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC"
    )
    print("  Nodes by type:")
    for row in result:
        print(f"    {str(row['label']):<35} {row['count']}")

    result = graph_store.structured_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY count DESC"
    )
    print("  Relations by type:")
    for row in result:
        print(f"    {row['rel']:<30} {row['count']}")

    # verify authors
    print("\n  Authors per paper:")
    for paper in ["RAG", "DPR", "REALM", "FiD", "GraphRAG"]:
        result = graph_store.structured_query(
            """
            MATCH (p:__Entity__ {name: $paper})-[:AUTHORED_BY]->(a)
            RETURN count(a) AS count
            """,
            param_map={"paper": paper}
        )
        count = result[0]["count"] if result else 0
        print(f"    {paper:<12} {count} authors")


if __name__ == "__main__":
    graph_store = connect()
    run_resolution(graph_store)
    inspect_after(graph_store)