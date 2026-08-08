"""Central configuration: filesystem paths and the (swappable) model choices.

Keeping every model name and tunable in one place is deliberate: the graph code
never hard-codes a model, so you can swap the generator or the embedder without
touching the logic. This is the "hardware-aware trade-off" surface the brief asks
you to reason about.
"""
from __future__ import annotations

from pathlib import Path

# --- Filesystem paths -------------------------------------------------------
# repo_root/src/orbitdesk/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
CASES_FILE = DATA_DIR / "resolved_cases.json"
SCHEMA_FILE = DATA_DIR / "output_schema.json"
SAMPLE_QUESTIONS_FILE = DATA_DIR / "sample_questions.json"

# --- Models (swappable) -----------------------------------------------------
# Retrieval: a local Hugging Face sentence-transformers embedding model.
# This satisfies the "at least one model loaded through a Hugging Face library"
# requirement, and runs on the Apple GPU (MPS) when available.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Generation + LLM triage: Gemma 3 4B, served locally by Ollama (no cloud API).
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"

# Device preference for the HF embedding model.
EMBEDDING_DEVICE = "mps"  # falls back to "cpu" automatically if MPS is absent

# --- Retrieval / routing tunables ------------------------------------------
TOP_K = 5                     # passages returned to the generator
RETRIEVAL_SCORE_FLOOR = 0.30  # below this, treat the corpus as "no real match"
MAX_REVISIONS = 1             # verification may trigger exactly one revision
GROUNDING_THRESHOLD = 0.45    # min fraction of answer content words found in evidence
