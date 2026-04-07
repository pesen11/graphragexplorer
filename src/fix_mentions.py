# fix_mentions.py  (save in project root)
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

load_dotenv()

from ingest import fetch_papers_local, chunk_documents
from extractor import SimpleGraphExtractor

NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
PAPERS_DIR     = Path("data/papers")

async def main():
    # connect to Neo4j
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
    )
    print("✅ Neo4j connected\n")

    # re-run extractor on the same chunks to get entity metadata back
    # (we need kg_nodes metadata to know which entities came from which chunk)
    print("📂 Loading and chunking papers...")
    full_docs = fetch_papers_local(PAPERS_DIR)
    nodes = chunk_documents(full_docs)

    print(f"⚙️  Re-extracting to recover chunk→entity mappings ({len(nodes)} chunks)...")
    extractor = SimpleGraphExtractor(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
    )
    nodes_with_triples = await extractor.acall(nodes, show_progress=False)

    # create MENTIONS edges
    print("\n🔗 Creating MENTIONS edges...")
    total = 0
    from llama_index.core.graph_stores.types import KG_NODES_KEY

    for node in nodes_with_triples:
        kg_nodes = node.metadata.get(KG_NODES_KEY, [])
        if not kg_nodes:
            continue

        entity_names = [e.name for e in kg_nodes]

        graph_store.structured_query(
            """
            UNWIND $entity_names AS name
            MATCH (e:__Entity__ {name: name})
            MATCH (c:Chunk)
            WHERE c.text CONTAINS $text_snippet
            MERGE (c)-[:MENTIONS]->(e)
            """,
            param_map={
                "text_snippet": node.text[:100],  # first 100 chars as unique identifier
                "entity_names": entity_names,
            }
        )
        total += len(entity_names)
        print(f"  chunk {node.node_id[:8]}...  →  {len(entity_names)} entities linked")

    print(f"\n✅ Done — created {total} MENTIONS edges")

    # verify
    result = graph_store.structured_query(
        "MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN count(*) AS total"
    )
    print(f"   Verified: {result[0]['total']} MENTIONS edges in Neo4j")

asyncio.run(main())