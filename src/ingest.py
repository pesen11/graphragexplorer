# src/ingest.py
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.readers.papers import ArxivReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
import time
from collections import defaultdict
from llama_index.core.schema import Document

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SEARCH_QUERY   = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
MAX_RESULTS    = 5
PAPERS_DIR     = Path("data/papers")
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
# ──────────────────────────────────────────────────────────────────────────────

def fetch_papers(query: str, max_results: int, papers_dir: Path):
    """Pull papers from ArXiv and return (full_docs, abstract_docs)."""
    papers_dir.mkdir(parents=True, exist_ok=True)
    reader = ArxivReader()

    print(f"\n📡 Fetching {max_results} papers for: '{query}'\n")
    time.sleep(3)
    full_docs, abstract_docs = reader.load_papers_and_abstracts(
        search_query=query,
        papers_dir=str(papers_dir),
        max_results=max_results,
    )
    return full_docs, abstract_docs

def fetch_papers_local(papers_dir: Path, exclude: list = None):
    """Read PDFs from disk, merging all pages per file into one Document."""
    exclude = exclude or []
    print(f"\n📂 Reading papers from {papers_dir}...\n")

    raw_docs = SimpleDirectoryReader(
        str(papers_dir),
        filename_as_id=True,
    ).load_data()

    # Group pages by filename
    grouped = defaultdict(list)
    meta_by_file = {}

    for doc in raw_docs:
        fname = doc.metadata["file_name"]
        if fname in exclude:
            continue
        grouped[fname].append(doc.text)
        meta_by_file[fname] = doc.metadata

    # Merge all pages of each file into one Document
    merged_docs = []
    for fname, pages in grouped.items():
        merged = Document(
            text="\n\n".join(pages),
            metadata={**meta_by_file[fname], "file_name": fname},
        )
        merged_docs.append(merged)

    print(f"  Found {len(merged_docs)} paper(s):\n")
    for doc in merged_docs:
        pages = len(grouped[doc.metadata["file_name"]])
        chars = len(doc.text)
        print(f"    {doc.metadata['file_name']:<35} {pages} pages   {chars:,} chars")
    print()

    return merged_docs



def inspect_documents(full_docs, abstract_docs=None):
    """Print a structured summary of every fetched document."""
    print(f"{'='*60}")
    print(f"  Loaded {len(full_docs)} documents from disk")
    print(f"{'='*60}\n")

    for i, doc in enumerate(full_docs, 1):
        meta = doc.metadata

        print(f"── Doc {i} ──────────────────────────────────────────────")
        print(f"  File      : {meta.get('file_name', 'N/A')}")
        print(f"  File path : {meta.get('file_path', 'N/A')}")
        print(f"  Size      : {len(doc.text):,} chars")
        print(f"  Preview   : {doc.text[:200].strip()}\n")

    # Show ALL metadata keys from the first doc
    if full_docs:
        print(f"\n── Available metadata keys (from doc 1) ──────────────")
        for k, v in full_docs[0].metadata.items():
            print(f"  {k:<20}: {str(v)[:80]}")


def chunk_documents(full_docs):
    """Split full-text docs into nodes and show stats."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(full_docs)

    print(f"\n── Chunking results ─────────────────────────────────────")
    print(f"  {len(full_docs)} docs  →  {len(nodes)} nodes")
    print(f"  Chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP}\n")

    # Sample 3 chunks spread across the set
    sample_indices = [0, len(nodes) // 2, len(nodes) - 1]
    for idx in sample_indices:
        node = nodes[idx]
        print(f"  [Node {idx}]  {len(node.text)} chars")
        print(f"  Source: {node.metadata.get('Title', 'unknown')}")
        print(f"  Text:   {node.text[:150].strip()}...\n")

    return nodes


