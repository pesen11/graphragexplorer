# src/query_engine.py
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import streamlit as st

# load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

def get_secret(key):
    try:
        # Streamlit Cloud
        return st.secrets[key]
    except NameError:
        # Local
        return os.getenv(key)

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URL      = get_secret("NEO4J_URL")
NEO4J_USERNAME = get_secret("NEO4J_USERNAME")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD")
NEO4J_DATABASE=get_secret("NEO4J_DATABASE")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")
MODEL          = "llama-3.3-70b-versatile"
TOP_K          = 3
HOP_DEPTH      = 2

# ── Entity map — question keyword → graph node name ───────────────────────────
ENTITY_MAP = {
    "RAG":      "RAG",
    "DPR":      "DPR",
    "REALM":    "REALM",
    "BERT":     "BERT",
    "BART":     "BART",
    "GRAPHRAG": "GraphRAG",
    "FID":      "Fusion-in-Decoder",
    "FUSION":   "Fusion-in-Decoder",
}
# ──────────────────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """You are an expert research assistant analyzing academic papers about NLP and information retrieval.

You have access to a knowledge graph extracted from these papers:
- RAG (Retrieval-Augmented Generation) by Lewis et al.
- DPR (Dense Passage Retrieval) by Karpukhin et al.
- REALM (Retrieval-Augmented Language Model Pre-Training) by Guu et al.
- FiD (Fusion-in-Decoder) — "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering" by Izacard et al.
- GraphRAG — "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" by Edge et al.

Use the context below to answer the question accurately.
- Use the full paper names, not just abbreviations
- Only state what the context supports — do not guess
- Cite specific triples when relevant

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""


def connect():
    print("🔌 Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database=NEO4J_DATABASE
        
    )
    print("✅ Connected\n")

    print("🔢 Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("✅ Embedding model ready\n")

    client = Groq(api_key=GROQ_API_KEY)
    return graph_store, embed_model, client


def retrieve_chunks(query: str, embed_model, graph_store, top_k: int = TOP_K):
    """Vector search — find most similar chunks by embedding."""
    query_embedding = embed_model.get_text_embedding(query)

    results = graph_store.structured_query(
        """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        WITH c, vector.similarity.cosine(c.embedding, $embedding) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN c.id AS id, c.text AS text, score
        """,
        param_map={"embedding": query_embedding, "top_k": top_k}
    )
    return results


def retrieve_graph_context(chunk_ids: list, graph_store):
    """Graph traversal — follow edges from chunks to entities and beyond."""

    # general one-hop traversal from chunk-mentioned entities
    triples = graph_store.structured_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        WHERE c.id IN $chunk_ids
        WITH e
        MATCH (e)-[r]->(connected)
        RETURN e.name AS from, type(r) AS relation, connected.name AS to
        UNION
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        WHERE c.id IN $chunk_ids
        WITH e
        MATCH (other)-[r]->(e)
        RETURN other.name AS from, type(r) AS relation, e.name AS to
        """,
        param_map={"chunk_ids": chunk_ids}
    )

    # targeted structural relations for mentioned entities
    structural = graph_store.structured_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        WHERE c.id IN $chunk_ids
        WITH collect(DISTINCT e.name) AS entity_names
        MATCH (a)-[r:BUILDS_ON|CITES|AUTHORED_BY]->(b)
        WHERE a.name IN entity_names OR b.name IN entity_names
        RETURN a.name AS from, type(r) AS relation, b.name AS to
        """,
        param_map={"chunk_ids": chunk_ids}
    )

    # deduplicate
    all_rows = triples + structural
    seen = set()
    unique_triples = []
    for row in all_rows:
        key = (row["from"], row["relation"], row["to"])
        if key not in seen and row["from"] and row["to"]:
            seen.add(key)
            unique_triples.append(key)

    return unique_triples


def retrieve_entity_context(entity_name: str, graph_store) -> list:
    """
    Direct entity lookup — bypasses vector search entirely.
    Returns all relationships for a named entity.
    """
    results = graph_store.structured_query(
        """
        MATCH (e:__Entity__)
        WHERE toLower(e.name) = toLower($name)
        WITH e
        MATCH (e)-[r]->(connected)
        RETURN e.name AS from, type(r) AS relation, connected.name AS to
        UNION
        MATCH (e:__Entity__)
        WHERE toLower(e.name) = toLower($name)
        WITH e
        MATCH (other)-[r]->(e)
        RETURN other.name AS from, type(r) AS relation, e.name AS to
        """,
        param_map={"name": entity_name}
    )

    seen = set()
    triples = []
    for row in results:
        key = (row["from"], row["relation"], row["to"])
        if key not in seen and row["from"] and row["to"]:
            seen.add(key)
            triples.append(key)

    return triples


def extract_entities_from_question(question: str) -> list:
    """Find known entity names mentioned in the question."""
    question_upper = question.upper()
    found = []
    for keyword, graph_name in ENTITY_MAP.items():
        if keyword in question_upper:
            found.append(graph_name)
    return list(set(found))  # deduplicate

def find_common_entities(entity_names: list, graph_store) -> list:
    """
    Find entities connected to ALL of the given entity names via the same relation.
    """
    if len(entity_names) < 2:
        return []

    result = graph_store.structured_query(
        """
        MATCH (a:__Entity__ {name: $entity1})-[r1]->(common)
        WITH common, type(r1) AS rel
        MATCH (b:__Entity__ {name: $entity2})-[r2]->(common)
        WHERE type(r2) = rel
        RETURN
            $entity1   AS from1,
            rel        AS relation,
            common.name AS common_node,
            $entity2   AS from2
        """,
        param_map={
            "entity1": entity_names[0],
            "entity2": entity_names[1],
        }
    )

    triples = []
    seen = set()
    for row in result:
        t1 = (row["from1"], row["relation"], row["common_node"])
        t2 = (row["from2"], row["relation"], row["common_node"])
        if t1 not in seen:
            triples.append(t1)
            seen.add(t1)
        if t2 not in seen:
            triples.append(t2)
            seen.add(t2)

    return triples



def build_context(chunks: list, triples: list, question: str = "", priority_triples: list = None) -> str:
    priority_triples = priority_triples or []
    context_parts = []

    context_parts.append("--- Relevant text chunks ---")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        text  = chunk.get("text", "")[:200]
        context_parts.append(f"\n[Chunk {i} | similarity: {score:.3f}]\n{text}")

    if triples or priority_triples:
        question_keywords = set(question.lower().split())

        def relevance_score(triple):
            from_, rel, to = triple
            text = f"{from_} {rel} {to}".lower()
            keyword_hits = sum(1 for kw in question_keywords if kw in text)
            priority_bonus = 3 if rel in ["AUTHORED_BY", "BUILDS_ON", "CITES", "INTRODUCES", "AFFILIATED_WITH"] else 0
            return keyword_hits + priority_bonus

        # remove priority triples from main list to avoid duplicates
        priority_set = set(priority_triples)
        remaining = [t for t in triples if t not in priority_set]
        sorted_remaining = sorted(remaining, key=relevance_score, reverse=True)

        limit = 50 if any(w in question.lower() for w in ["author", "who wrote", "who worked", "common", "both"]) else 30

        # priority triples always go first, then fill remaining slots
        final_triples = priority_triples + sorted_remaining[:limit - len(priority_triples)]

        context_parts.append("\n--- Knowledge graph triples ---")
        for from_, rel, to in final_triples:
            context_parts.append(f"  ({from_}) --[{rel}]--> ({to})")

    return "\n".join(context_parts)


def answer(question: str, graph_store, embed_model, client) -> str:
    

    with st.status("Thinking...", expanded=True) as status:

        status.update(label="Retrieving relevant chunks...")
        chunks = retrieve_chunks(question, embed_model, graph_store)

        if not chunks:
            return "No relevant context found in the graph for this question."

        chunk_ids = [c["id"] for c in chunks if c.get("id")]

        status.update(label=f"Traversing graph from {len(chunk_ids)} chunks...")
        triples = retrieve_graph_context(chunk_ids, graph_store)

        entities = extract_entities_from_question(question)
        for entity in entities:
            direct = retrieve_entity_context(entity, graph_store)
            existing = set(triples)
            for t in direct:
                if t not in existing:
                    triples.append(t)
                    existing.add(t)

        cross_triples = []
        question_lower = question.lower()
        if len(entities) >= 2 and any(w in question_lower for w in ["common", "both", "shared", "same", "between"]):
            status.update(label=f"Finding connections between {', '.join(entities)}...")
            cross_triples = find_common_entities(entities, graph_store)

        status.update(label=f"Generating answer from {len(triples)} triples...")
        context = build_context(chunks, triples, question=question, priority_triples=cross_triples)

        prompt = ANSWER_PROMPT.format(context=context, question=question)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        status.update(label="Done", state="complete", expanded=False)

    return response.choices[0].message.content


# def run_interactive():
#     """Interactive query loop."""
#     graph_store, embed_model, client = connect()

#     print("=" * 60)
#     print("  GraphRAG Academic Explorer — Query Mode")
#     print("  Type 'quit' to exit")
#     print("=" * 60)

#     test_questions = [
#         "Who are the authors of DPR?",
#         "Which methods does RAG build on?",
#         "What datasets were used to evaluate REALM?",
#         "Which authors worked on both RAG and DPR?",
#     ]

#     print("\n💡 Suggested questions to try:")
#     for i, q in enumerate(test_questions, 1):
#         print(f"   {i}. {q}")
#     print()

#     while True:
#         question = input("Ask a question: ").strip()
#         if question.lower() in ("quit", "exit", "q"):
#             break
#         if not question:
#             continue

#         print()
#         result = answer(question, graph_store, embed_model, client)
#         print(f"📖 Answer:\n{result}")
#         print("\n" + "─" * 60 + "\n")


# if __name__ == "__main__":

#     run_interactive()
