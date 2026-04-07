# src/schema.py
from typing import Literal

# ── Entity types (always UPPER_CASE) ─────────────────────────────────────────
Entities = Literal[
    "PAPER",
    "AUTHOR",
    "METHOD",
    "DATASET",
    "CONCEPT",
    "INSTITUTION",
]

# ── Relation types ────────────────────────────────────────────────────────────
Relations = Literal[
    "CITES",           # PAPER → PAPER
    "INTRODUCES",      # PAPER → METHOD  or  PAPER → CONCEPT
    "BUILDS_ON",       # PAPER → PAPER   or  METHOD → METHOD
    "USES_DATASET",    # PAPER → DATASET
    "USES_METHOD",     # PAPER → METHOD
    "AUTHORED_BY",     # PAPER → AUTHOR
    "AFFILIATED_WITH", # AUTHOR → INSTITUTION
    "COMPARES_TO",     # PAPER → PAPER   or  METHOD → METHOD
]

# ── Validation schema ─────────────────────────────────────────────────────────
# Triples of (subject, relation, object) — the only combos the extractor will produce.
# Keeping this tight is what separates a clean academic graph from a noisy blob.
ValidationSchema = [
    ("PAPER",   "CITES",           "PAPER"),
    ("PAPER",   "INTRODUCES",      "METHOD"),
    ("PAPER",   "INTRODUCES",      "CONCEPT"),
    ("PAPER",   "BUILDS_ON",       "PAPER"),
    ("PAPER",   "USES_DATASET",    "DATASET"),
    ("PAPER",   "USES_METHOD",     "METHOD"),
    ("PAPER",   "AUTHORED_BY",     "AUTHOR"),
    ("PAPER",   "COMPARES_TO",     "PAPER"),
    ("METHOD",  "BUILDS_ON",       "METHOD"),
    ("METHOD",  "COMPARES_TO",     "METHOD"),
    ("AUTHOR",  "AFFILIATED_WITH", "INSTITUTION"),
]