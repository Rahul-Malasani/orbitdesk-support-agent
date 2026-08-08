"""Deterministic test doubles.

These let the routing/graph tests run with NO models and NO network:
  * StubRetriever returns hand-picked real corpus passages as hits.
  * StubLLM returns a controlled classification label and scripted answers,
    so tests assert on routing/behaviour, never on model wording.
"""
from __future__ import annotations

from src.orbitdesk.data import build_corpus
from src.orbitdesk.retrieval import Retrieved

# Real passages, indexed by passage_id (parsing only — no model).
_CORPUS = {p.passage_id: p for p in build_corpus()}


def make_hits(spec: list[tuple[str, float]]) -> list[Retrieved]:
    """Build a hit list from (passage_id, score) pairs using real passages."""
    return [Retrieved(_CORPUS[pid], score) for pid, score in spec]


class StubRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def build(self) -> "StubRetriever":
        return self

    def search(self, question: str, top_k: int | None = None) -> list[Retrieved]:
        return self._hits[:top_k] if top_k else self._hits


class StubLLM:
    """Returns a fixed triage label; scripted answers for generation calls."""

    def __init__(self, label: str = "ANSWERABLE", gen_answers: list[str] | None = None) -> None:
        self.label = label
        self.gen_answers = list(gen_answers) if gen_answers else None
        self.calls: list[str] = []

    def health(self) -> bool:
        return True

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        self.calls.append(prompt)
        # Triage classification prompt -> return the controlled label.
        if "ANSWERABLE or NEEDS_CLARIFICATION" in prompt:
            return f"LABEL: {self.label}\nREASON: stubbed classification"
        # Generation prompt -> next scripted answer (keep last once exhausted).
        if self.gen_answers is not None:
            return self.gen_answers.pop(0) if len(self.gen_answers) > 1 else self.gen_answers[0]
        return "Stubbed answer.\nSOURCES:"
