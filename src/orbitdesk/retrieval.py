"""Retrieval: embed the corpus with a local HF model and do in-memory search.

Design choices worth defending in the walkthrough:

  * No vector database. 53 passages fit in a tiny in-memory matrix, so we do
    *exact* cosine-similarity search (one matrix multiply). The brief explicitly
    says a managed vector DB is not required, and exact search is both simpler
    and more accurate than approximate ANN at this scale.

  * Embeddings are cached to `index/embeddings.npy`, keyed by a hash of
    (model name + every passage's text). On a later run, if nothing changed we
    load the vectors from disk instead of re-encoding. NOTE: the embedding model
    still loads at serve time because we must embed the *query* — so caching
    saves the corpus re-encode, not the model load. The model load is the
    dominant cost and is reported for the assignment.

  * Retrieval returns *pure similarity* only. The authority / superseded rules
    live downstream (triage + generation), keeping model reasoning and
    deterministic policy cleanly separated.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .data import Passage, build_corpus

# bge-v1.5 retrieves best when the *query* carries this instruction. Passages
# are embedded with no prefix. (Asymmetric query/document encoding.)
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


@dataclass
class Retrieved:
    """One search hit: the passage plus its similarity score (0..1)."""

    passage: Passage
    score: float


def _select_device(preferred: str) -> str:
    """Use the preferred device if the backend is actually available."""
    import torch

    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Retriever:
    def __init__(
        self,
        passages: list[Passage] | None = None,
        model_name: str | None = None,
        device: str | None = None,
        index_dir: Path | None = None,
    ) -> None:
        self.passages = passages if passages is not None else build_corpus()
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device or _select_device(config.EMBEDDING_DEVICE)
        self.index_dir = index_dir or (config.REPO_ROOT / "index")

        self._model = None
        self.embeddings: np.ndarray | None = None  # (N, D) L2-normalized float32

        # Timings recorded for the assignment's "load time / latency" requirement.
        self.load_time_s: float | None = None
        self.embed_time_s: float | None = None
        self.cache_hit: bool | None = None

    # -- internals ----------------------------------------------------------
    def _corpus_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        for p in self.passages:
            h.update(p.passage_id.encode())
            h.update(b"\x00")
            h.update(p.text.encode())
            h.update(b"\x01")
        return h.hexdigest()

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.load_time_s = time.perf_counter() - t0
        return self._model

    # -- public API ---------------------------------------------------------
    def build(self, force: bool = False) -> "Retriever":
        """Populate `self.embeddings`, loading from cache when possible."""
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "embeddings_meta.json"
        corpus_hash = self._corpus_hash()

        if not force and emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("hash") == corpus_hash:
                self.embeddings = np.load(emb_path)
                self.cache_hit = True
                self.load_time_s = meta.get("load_time_s")
                self.embed_time_s = meta.get("embed_time_s")
                return self

        # Cache miss (or forced): load model and re-encode the corpus.
        model = self._load_model()
        texts = [p.text for p in self.passages]
        t0 = time.perf_counter()
        emb = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False,
        )
        self.embed_time_s = time.perf_counter() - t0
        self.embeddings = emb.astype(np.float32)
        self.cache_hit = False

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, self.embeddings)
        meta_path.write_text(
            json.dumps(
                {
                    "hash": corpus_hash,
                    "model": self.model_name,
                    "device": self.device,
                    "count": len(self.passages),
                    "dim": int(self.embeddings.shape[1]),
                    "load_time_s": self.load_time_s,
                    "embed_time_s": self.embed_time_s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    def search(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        """Return the top_k passages by cosine similarity to `query`."""
        if self.embeddings is None:
            self.build()
        top_k = top_k or config.TOP_K

        model = self._load_model()  # required to embed the query
        q = model.encode(
            [QUERY_PROMPT + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Both sides are L2-normalized, so dot product == cosine similarity.
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return [Retrieved(self.passages[i], float(scores[i])) for i in order]


if __name__ == "__main__":
    # Smoke test: retrieve for each sample question and print the top hits.
    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]

    r = Retriever()
    t0 = time.perf_counter()
    r.build()
    build_wall = time.perf_counter() - t0

    print(f"device        : {r.device}")
    print(f"model         : {r.model_name}")
    print(f"cache_hit     : {r.cache_hit}")
    print(f"model load    : {r.load_time_s:.2f}s" if r.load_time_s else "model load    : (from cache)")
    print(f"embed corpus  : {r.embed_time_s:.2f}s" if r.embed_time_s else "embed corpus  : (from cache)")
    print(f"build wall    : {build_wall:.2f}s   (N={len(r.passages)}, dim={r.embeddings.shape[1]})")

    for q in questions:
        t0 = time.perf_counter()
        hits = r.search(q["question"], top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[{q['question_id']}]  ({latency_ms:.0f} ms)  {q['question'][:70]}...")
        for h in hits:
            flag = "  <SUPERSEDED>" if h.passage.is_superseded else ""
            print(f"    {h.score:.3f}  {h.passage.passage_id}{flag}")
