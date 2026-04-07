# app.py  (project root)
import sys
sys.path.insert(0, "src")

import streamlit as st
from query_engine import connect, answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Academic Explorer",
    page_icon="🔬",
    layout="wide",
)

# ── Load resources once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Neo4j and loading models...")
def load_resources():
    graph_store, embed_model, client = connect()
    return graph_store, embed_model, client

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 GraphRAG Explorer")
    st.markdown("**Papers in the graph:**")
    papers = {
        "RAG": "Lewis et al. — Facebook AI",
        "DPR": "Karpukhin et al. — Facebook AI",
        "REALM": "Guu et al. — Google Research",
        "FiD": "Izacard et al. — Facebook AI",
        "GraphRAG": "Edge et al. — Microsoft Research",
    }
    for acronym, authors in papers.items():
        st.markdown(f"- **{acronym}** — {authors}")

    st.divider()
    st.markdown("**Example questions:**")
    example_qs = [
        "Who are the authors of DPR?",
        "Which methods does RAG build on?",
        "Which authors worked on both RAG and DPR?",
        "What datasets does REALM use?",
        "How does GraphRAG differ from RAG?",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("GraphRAG-Powered Academic Explorer")


# ── Project intro ─────────────────────────────────────────────────────────────
st.markdown("""
This explorer lets you ask complex relational questions across 5 papers spanning 
the evolution of retrieval-augmented generation — from DPR and REALM (2020) 
to GraphRAG (2024).

Instead of just finding similar text chunks, it builds a **knowledge graph** of papers, authors, 
methods, and datasets in Neo4j, then traverses real connections to answer your question.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**📄 5 Papers indexed**  \nRAG · DPR · REALM · FiD · GraphRAG")

with col2:
    st.success("**🕸️ Graph-powered retrieval**  \nFollows citation and authorship edges, not just keywords")

with col3:
    st.warning("**🤖 LLM synthesis**  \nGroq (Llama 3.3 70B) generates answers from structured context")

st.divider()
st.caption("Ask relational questions about NLP papers — powered by Neo4j + LlamaIndex")

# pre-fill from sidebar button click
prefill = st.session_state.pop("prefill", "")

question = st.text_input(
    "Your question",
    value=prefill,
    placeholder="e.g. Which papers cite RAG and use graph-based retrieval?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col2:
    st.write("")  # spacer

# ── Query + display ───────────────────────────────────────────────────────────
if ask and question.strip():
    graph_store, embed_model, client = load_resources()

    with st.spinner("Retrieving from graph and generating answer..."):
        # capture internal logs without cluttering the UI
        response = answer(question, graph_store, embed_model, client)

    st.subheader("Answer")
    st.markdown(response)

elif ask and not question.strip():
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Stack: LlamaIndex · Neo4j · Groq (Llama 3.3 70B) · HuggingFace Embeddings")