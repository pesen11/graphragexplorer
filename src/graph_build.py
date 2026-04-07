# src/graph_build.py
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import PropertyGraphIndex, Settings
from extractor import SimpleGraphExtractor
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import asyncio

from schema import Entities, Relations, ValidationSchema
from ingest import fetch_papers_local, chunk_documents

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────
# SEARCH_QUERY = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# MAX_RESULTS  = 1
PAPERS_DIR   = Path("data/papers")

NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password_here")
# ──────────────────────────────────────────────────────────────────────────────


def build_llm_and_embeddings():
    """
    LLM  → Groq (free, via OpenAI-compatible endpoint)
    Embed → BAAI/bge-small-en-v1.5 (local HuggingFace, ~130MB, auto-downloads)
    """
    llm = OpenAILike(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        api_base="https://api.groq.com/openai/v1",
        temperature=0.0,
    )
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ LLM   : llama-3.1-70b-versatile via Groq")
    print("✅ Embed : BAAI/bge-small-en-v1.5 (local)\n")

    return llm, embed_model


def build_graph_store():
    """Connect to Neo4j and return the graph store."""
    print(f"🔌 Connecting to Neo4j at {NEO4J_URL}...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    print("✅ Neo4j connected.\n")
    return graph_store


def build_extractor(llm):
    return SimpleGraphExtractor(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
    )


async def extract_and_store(nodes, extractor, embed_model, graph_store):
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY

    # Step 1 — extract triples
    print(f"⚙️  Extracting triples from {len(nodes)} chunks...\n")
    nodes_with_triples = await extractor.acall(nodes, show_progress=False)

    # Step 2 — store entities and relations in Neo4j
    print("\n💾 Storing graph in Neo4j...")
    all_kg_nodes = []
    all_kg_relations = []

    for node in nodes_with_triples:
        kg_nodes = node.metadata.get(KG_NODES_KEY, [])
        kg_rels  = node.metadata.get(KG_RELATIONS_KEY, [])
        all_kg_nodes.extend(kg_nodes)
        all_kg_relations.extend(kg_rels)

    if all_kg_nodes:
        graph_store.upsert_nodes(all_kg_nodes)
    if all_kg_relations:
        graph_store.upsert_relations(all_kg_relations)

    print(f"  Stored {len(all_kg_nodes)} entity nodes")
    print(f"  Stored {len(all_kg_relations)} relations")

   # Step 3 — store chunk nodes with embeddings
    print("\n🔢 Generating and storing embeddings...")
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY

    for i, node in enumerate(nodes_with_triples):
        embedding = embed_model.get_text_embedding(node.text)
        
        # store only primitive fields — no nested objects
        graph_store.structured_query(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.embedding = $embedding
            """,
            param_map={
                "id": node.node_id,
                "text": node.text[:1000],   # trim to avoid hitting Neo4j property size limits
                "embedding": embedding,
            }
        )
        print(f"  [{i+1}/{len(nodes_with_triples)}] chunk stored", end="\r")

    print(f"\n  Stored {len(nodes_with_triples)} chunk nodes with embeddings")
    print("\n✅ Graph construction complete.")
    return nodes_with_triples


def create_mentions_edges(nodes_with_triples, graph_store):
    """
    Create MENTIONS edges between Chunk nodes and the entities
    extracted from them during the extraction step.
    """
    from llama_index.core.graph_stores.types import KG_NODES_KEY

    print("\n🔗 Creating MENTIONS edges between chunks and entities...")
    total_edges = 0

    for node in nodes_with_triples:
        kg_nodes = node.metadata.get(KG_NODES_KEY, [])
        if not kg_nodes:
            continue

        entity_names = [e.name for e in kg_nodes]

        graph_store.structured_query(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $entity_names AS name
            MATCH (e:__Entity__ {name: name})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            param_map={
                "chunk_id": node.node_id,
                "entity_names": entity_names,
            }
        )
        total_edges += len(entity_names)

    print(f"  Created {total_edges} MENTIONS edges")
    print("✅ Graph fully connected.\n")


def inspect_graph(graph_store):
    """Run Cypher queries to verify what landed in Neo4j."""
    print("\n── Graph inspection ─────────────────────────────────────")

    # Total nodes
    result = graph_store.structured_query(
        "MATCH (n) RETURN count(n) AS total_nodes"
    )
    print(f"  Total nodes      : {result[0]['total_nodes']}")

    # Nodes by type
    result = graph_store.structured_query(
        "MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC"
    )
    print("  Nodes by type:")
    for row in result:
        print(f"    {str(row['label']):<30} {row['count']}")

    # Relations by type
    result = graph_store.structured_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY count DESC"
    )
    print("  Relations by type:")
    for row in result:
        print(f"    {row['rel']:<30} {row['count']}")

    # Sample triples
    result = graph_store.structured_query(
        """
        MATCH (a)-[r]->(b)
        RETURN a.name AS from, type(r) AS rel, b.name AS to
        LIMIT 10
        """
    )
    print("\n  Sample triples (first 10):")
    for row in result:
        print(f"    {str(row['from']):<35} --[{row['rel']}]--> {row['to']}")

    print("\n  💡 Open Neo4j Browser at http://localhost:7474 and run:")
    print("     MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50\n")


if __name__ == "__main__":
    # 1. LLM + embeddings
    llm, embed_model = build_llm_and_embeddings()

    # 2. Fetch one paper
    full_docs = fetch_papers_local(PAPERS_DIR,exclude=["rag.pdf"])

    # 3. Chunk it
    nodes = chunk_documents(full_docs)
    print(f"  Processing {len(nodes)} chunks through the extractor...\n")

    # 4. Neo4j
    graph_store = build_graph_store()

    # 5. Extractor
    extractor = build_extractor(llm)

    # 6. extract and store
    nodes_with_triples=asyncio.run(extract_and_store(nodes, extractor, embed_model, graph_store))

    #7 Create MENTIONS edges
    create_mentions_edges(nodes_with_triples,graph_store)

    # 7. Inspect
    inspect_graph(graph_store)

    