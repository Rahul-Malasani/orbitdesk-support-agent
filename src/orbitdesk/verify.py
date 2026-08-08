"""Verification: the deterministic gatekeeper on Gemma's output.

Given an assembled response + the evidence it was built from, decide whether to
ACCEPT it, or reject it (which the graph turns into one revision, then a safe
failure). Every check here is deterministic code — no model reasoning — so the
accept/reject decision is auditable and testable.

Checks (per the brief):
  1. schema_valid       -- matches output_schema.json
  2. has_citations      -- answerable/escalation must cite at least one source
  3. citations_valid    -- every cited source_id exists in the corpus
  4. grounded           -- HEURISTIC: enough of the answer's content words appear
                           in the cited evidence (term overlap)
  5. no_forbidden       -- no superseded guidance, secret solicitation, or leaked tags

This module also owns response assembly (`build_response`) and the terminal
`safe_failure_response`, since verification is what governs the response object.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from .retrieval import Retrieved
from .triage import TriageResult
from .generate import Generation
from . import config

_SCHEMA = json.loads(config.SCHEMA_FILE.read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA)

# Routes whose answers are grounded claims we must police.
_GROUNDED_ROUTES = {"answerable", "requires_escalation"}

# Content-word grounding heuristic.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of",
    "in", "on", "at", "by", "with", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "you", "your", "we", "our", "they",
    "their", "can", "cannot", "not", "do", "does", "did", "have", "has", "had", "will",
    "would", "should", "could", "may", "might", "must", "from", "which", "who", "what",
    "when", "where", "how", "why", "there", "here", "into", "out", "up", "down", "so",
    "than", "also", "any", "all", "each", "some", "no", "yes", "please", "using", "use",
}
_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

# Forbidden-content patterns in the FINAL answer.
_FORBIDDEN_PATTERNS = [
    (re.compile(r"personal\s+token|profile\s*>\s*personal", re.I), "revives superseded personal-token guidance"),
    (re.compile(r"\b(paste|send|share|enter)\b.{0,30}(secret|token|password)", re.I), "solicits a secret value"),
    (re.compile(r"</?(passage|evidence|user_question)", re.I), "contains leaked prompt tags"),
]


@dataclass
class VerificationResult:
    passed: bool
    checks: dict[str, bool]
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Grounding heuristic
# --------------------------------------------------------------------------- #
def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def grounding_score(answer: str, evidence_text: str) -> float:
    """Fraction of the answer's content words that appear in the evidence."""
    ans = _content_words(answer)
    if not ans:
        return 0.0
    ev = _content_words(evidence_text)
    return len(ans & ev) / len(ans)


def _evidence_text(hits: list[Retrieved]) -> str:
    current = [h for h in hits if not h.passage.is_superseded][: config.TOP_K]
    return "\n".join(h.passage.text for h in (current or hits))


# --------------------------------------------------------------------------- #
# Response assembly
# --------------------------------------------------------------------------- #
def _confidence(route: str, hits: list[Retrieved]) -> float:
    top = hits[0].score if hits else 0.0
    if route == "answerable":
        return round(min(0.95, 0.55 + 0.4 * top), 2)
    return {"requires_escalation": 0.8, "requires_clarification": 0.5,
            "out_of_scope": 0.9, "safe_failure": 0.2}.get(route, 0.5)


def build_response(
    triage_result: TriageResult, generation: Generation, hits: list[Retrieved]
) -> dict:
    """Assemble a schema-shaped response object from triage + generation."""
    route = triage_result.classification
    response = {
        "classification": route,
        "answer": generation.answer or "(no answer produced)",
        "sources": generation.sources,
        "confidence": _confidence(route, hits),
        "requires_human": route in {"requires_escalation", "safe_failure"},
        "reason": triage_result.reason,
        "clarification_question": generation.answer if route == "requires_clarification" else None,
        "warnings": list(generation.warnings),
    }
    return response


def safe_failure_response(reason: str, hits: list[Retrieved] | None = None) -> dict:
    return {
        "classification": "safe_failure",
        "answer": (
            "I could not produce an answer that is fully supported by the available "
            "OrbitDesk documentation. Please rephrase or add detail, or this can be "
            "raised with a human support agent."
        ),
        "sources": [],
        "confidence": 0.2,
        "requires_human": True,
        "reason": f"Verification failed after revision: {reason}",
        "clarification_question": None,
        "warnings": [],
    }


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def verify(response: dict, hits: list[Retrieved]) -> VerificationResult:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    route = response.get("classification")

    # 1. Schema validity.
    schema_errors = sorted(_VALIDATOR.iter_errors(response), key=lambda e: e.path)
    checks["schema_valid"] = not schema_errors
    if schema_errors:
        failures.append(f"schema: {schema_errors[0].message}")

    # 2/3. Citations present + valid (only where an answer makes claims).
    sources = response.get("sources", [])
    if route in _GROUNDED_ROUTES:
        checks["has_citations"] = len(sources) > 0
        if not sources:
            failures.append("no citations for a claim-bearing answer")
    else:
        checks["has_citations"] = True

    from .data import build_corpus  # local import avoids a cycle at module load
    valid_ids = {p.source_id for p in build_corpus()}
    bad = [s["source_id"] for s in sources if s.get("source_id") not in valid_ids]
    checks["citations_valid"] = not bad
    if bad:
        failures.append(f"cited unknown source_id(s): {bad}")

    # 4. Grounding heuristic.
    if route in _GROUNDED_ROUTES:
        score = grounding_score(response.get("answer", ""), _evidence_text(hits))
        checks["grounded"] = score >= config.GROUNDING_THRESHOLD
        if score < config.GROUNDING_THRESHOLD:
            failures.append(f"low grounding ({score:.2f} < {config.GROUNDING_THRESHOLD})")
    else:
        checks["grounded"] = True

    # 5. No forbidden content.
    answer = response.get("answer", "")
    forbidden_hit = None
    for pat, desc in _FORBIDDEN_PATTERNS:
        if pat.search(answer):
            forbidden_hit = desc
            break
    checks["no_forbidden"] = forbidden_hit is None
    if forbidden_hit:
        failures.append(f"forbidden content: {forbidden_hit}")

    return VerificationResult(passed=not failures, checks=checks, failures=failures)


if __name__ == "__main__":
    from .retrieval import Retriever
    from .llm import OllamaLLM
    from .triage import triage
    from .generate import generate_response

    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]
    r = Retriever().build()
    llm = OllamaLLM()

    for q in questions:
        hits = r.search(q["question"], top_k=config.TOP_K)
        tr = triage(q["question"], hits, llm)
        gen = generate_response(tr, q["question"], hits, llm)
        resp = build_response(tr, gen, hits)
        v = verify(resp, hits)
        status = "PASS" if v.passed else "FAIL"
        checks = " ".join(f"{k}={'T' if val else 'F'}" for k, val in v.checks.items())
        print(f"[{q['question_id']}] {tr.classification:<24} verify={status}")
        print(f"    checks : {checks}")
        if v.failures:
            print(f"    fails  : {v.failures}")
