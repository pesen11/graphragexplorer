# src/extractor.py
import re
import asyncio
from typing import Any, Sequence
from groq import AsyncGroq
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.core.schema import BaseNode

EXTRACTION_PROMPT = """
You are an expert at extracting knowledge graph triples from academic NLP paper text.

From the text below, extract entities and relationships.

Entity types: PAPER, AUTHOR, METHOD, DATASET, CONCEPT, INSTITUTION
Relation types: CITES, INTRODUCES, BUILDS_ON, USES_DATASET, USES_METHOD, AUTHORED_BY, AFFILIATED_WITH, COMPARES_TO

STRICT RULES — violating these means the extraction is wrong:

Entity naming rules:
- Entity names must be 2-50 characters long
- Use short clean names only (e.g. "DPR" not "Dense Passage Retrieval for Open-Domain Question Answering")
- Never use single characters, math symbols, or LaTeX as entity names (e.g. NOT "θ", "∑", "x", "z")
- Never use citation numbers as entities (e.g. NOT "[1]", "[26]", "[53]")
- Never use "This paper", "This work", "Ours", "We", "authors", "None", "unknown" as entities
- For AUTHORS: use full names only (e.g. "Patrick Lewis" not "Lewis et al." or "Lewis")
- For AUTHORS: only extract if the text explicitly states who wrote something
- Never extract "et al." shorthand as an author name
- Never extract a full paper title as an entity — use its short name or acronym instead

Relation rules:
- AUTHORED_BY: only use when the text explicitly states authorship, never infer from citations
- CITES: use when paper A explicitly references paper B
- Only extract relations between valid entity types

Text:
{text}

Return ONLY a numbered list of triples. If nothing valid can be extracted, return "None".
1. (subject, RELATION, object)
2. (subject, RELATION, object)
"""

ALLOWED_RELATIONS = [
    "CITES", "INTRODUCES", "BUILDS_ON", "USES_DATASET",
    "USES_METHOD", "AUTHORED_BY", "AFFILIATED_WITH", "COMPARES_TO"
]




# patterns that indicate garbage entities
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
    """Return True if the entity name looks like noise."""
    name = name.strip()

    # too short or too long
    if len(name) < 2 or len(name) > 80:
        return True

    # check against garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True

    # full sentence (contains multiple spaces and ends with punctuation)
    if name.count(' ') > 6 and name[-1] in '.?,':
        return True

    return False

def parse_triples(response_text: str):
    """Parse (subject, RELATION, object) lines from LLM response."""
    triples = []
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, response_text)
    for match in matches:
        subj = match[0].strip()
        rel  = match[1].strip().upper()
        obj  = match[2].strip()
        if subj and rel and obj:
            triples.append((subj, rel, obj))
    return triples


class SimpleGraphExtractor:
    """
    Custom extractor calling Groq SDK directly —
    bypasses LlamaIndex LLM wrapper entirely.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model

    def __call__(self, nodes, show_progress=False, **kwargs):
        return asyncio.run(self.acall(nodes, show_progress=show_progress))

    async def acall(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        # one shared async client for all calls
        client = AsyncGroq(api_key=self.api_key)

        for i, node in enumerate(nodes):
            print(f"  [{i+1}/{len(nodes)}] ", end="", flush=True)

            if not node.text or len(node.text.strip()) < 50:
                node.metadata[KG_NODES_KEY] = []
                node.metadata[KG_RELATIONS_KEY] = []
                print("skipped (too short)")
                continue

            try:
                prompt = EXTRACTION_PROMPT.format(text=node.text[:2000])

                # direct Groq SDK call — guaranteed to hit /chat/completions
                chat_response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                response_text = chat_response.choices[0].message.content
                triples = parse_triples(response_text)

                kg_nodes = {}
                kg_relations = []

                for subj, rel, obj in triples:
                    if rel not in ALLOWED_RELATIONS:
                        continue
                    #filter garbage entities
                    if is_garbage_entity(subj) or is_garbage_entity(obj):
                        continue
                    if subj not in kg_nodes:
                        kg_nodes[subj] = EntityNode(name=subj, label="__Entity__")
                    if obj not in kg_nodes:
                        kg_nodes[obj] = EntityNode(name=obj, label="__Entity__")
                    kg_relations.append(
                        Relation(
                            source_id=kg_nodes[subj].id,
                            target_id=kg_nodes[obj].id,
                            label=rel,
                        )
                    )

                node.metadata[KG_NODES_KEY] = list(kg_nodes.values())
                node.metadata[KG_RELATIONS_KEY] = kg_relations
                print(f"✓  {len(kg_nodes):>2} entities  {len(kg_relations):>2} relations")

            except Exception as e:
                print(f"⚠  failed: {e}")
                node.metadata[KG_NODES_KEY] = []
                node.metadata[KG_RELATIONS_KEY] = []

        return nodes