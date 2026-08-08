"""Triage: turn a question + retrieved evidence into exactly one of four routes.

Routing responsibility is split on purpose:

  * DETERMINISTIC (testable without the model):
      - safety_gate      -> out_of_scope   (injection / forbidden action / secrets)
      - score floor      -> out_of_scope   (nothing in the corpus is relevant)
      - escalation_rule  -> requires_escalation (documented trigger in Q + evidence)
  * MODEL (Gemma):
      - answerable vs requires_clarification  (a language judgment: is the
        question specific enough to follow a documented procedure?)

Every deterministic route records *why* it fired (`decided_by`) so the logs and
the routing test never depend on the model's wording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import OllamaLLM
from .retrieval import Retrieved
from . import config

# --------------------------------------------------------------------------- #
# Deterministic patterns
# --------------------------------------------------------------------------- #
# Prompt-injection: attempts to override the documentation/rules.
INJECTION_PATTERNS = [
    re.compile(r"\bignore\b.{0,25}(documentation|instructions|docs|rules|above|prior)", re.I),
    re.compile(r"\bdisregard\b.{0,25}(documentation|instructions|rules|above)", re.I),
    re.compile(r"\bforget\b.{0,20}(instructions|rules)", re.I),
    re.compile(r"\boverride\b.{0,20}(instructions|rules|system)", re.I),
]

# Forbidden actions the assistant must never perform (KB-010).
FORBIDDEN_PATTERNS = [
    (re.compile(r"\brefund\b", re.I), "refund"),
    (re.compile(r"\bcancel\b.{0,20}\bsubscription\b", re.I), "cancel_subscription"),
    (re.compile(r"\blegal advice\b|\bwrite.{0,15}legal\b|\bsue\b", re.I), "legal_advice"),
    (re.compile(r"\b(medical|financial) advice\b", re.I), "other_advice"),
    (re.compile(r"\breveal\b.{0,20}(secret|token|password)", re.I), "reveal_secret"),
]

# Attempts to extract a secret value.
SECRET_PATTERNS = [
    re.compile(r"\b(give|send|show|tell|paste)\b.{0,30}(api\s*)?(secret|token|password|key)", re.I),
    re.compile(r"\bwhat('?s| is)\b.{0,20}(my|the)\b.{0,15}(secret|token|password)", re.I),
]

# Escalation trigger: a repetition/failure signal in the QUESTION ...
_ESCALATION_REPEAT = re.compile(r"\b(two|second|twice|consecutive|in a row|repeated|again)\b", re.I)
_FAILURE_TERM = re.compile(r"\b(render_failed|connector_internal_error|failed|failure|error)\b", re.I)
# ... or a suspected-credential-exposure signal (escalates on its own).
_CRED_EXPOSURE = re.compile(
    r"\b(credential|secret|token|api key)\b.{0,40}\b(expos|leak|compromis|stolen|breach)", re.I
)
# Evidence that legitimately supports an escalation route.
_ESCALATION_SOURCES = {"KB-004", "KB-008"}


@dataclass
class TriageResult:
    classification: str          # schema value
    reason: str
    decided_by: str              # "gate" | "score_floor" | "rule" | "llm" | "default"
    llm_label: str | None = None
    gate_category: str | None = None


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #
def safety_gate(question: str) -> TriageResult | None:
    """Return an out_of_scope result if the request is unsafe/forbidden, else None."""
    for pat in INJECTION_PATTERNS:
        if pat.search(question):
            return TriageResult(
                "out_of_scope",
                "Request attempts to override the supplied documentation (prompt injection); "
                "handled per KB-010.",
                decided_by="gate",
                gate_category="injection",
            )
    for pat in SECRET_PATTERNS:
        if pat.search(question):
            return TriageResult(
                "out_of_scope",
                "Request asks the assistant to reveal or transmit a secret value; refused per KB-010.",
                decided_by="gate",
                gate_category="secret_request",
            )
    for pat, category in FORBIDDEN_PATTERNS:
        if pat.search(question):
            return TriageResult(
                "out_of_scope",
                f"Request asks for an action outside support scope ({category}); refused per KB-010.",
                decided_by="gate",
                gate_category=category,
            )
    return None


def _evidence_supports_escalation(hits: list[Retrieved]) -> bool:
    for h in hits:
        if h.passage.source_id in _ESCALATION_SOURCES:
            return True
        if h.passage.status == "escalated":
            return True
        if set(h.passage.metadata.get("source_documents", [])) & _ESCALATION_SOURCES:
            return True
    return False


def escalation_rule(question: str, hits: list[Retrieved]) -> TriageResult | None:
    """Escalate when a documented trigger appears in BOTH the question and the evidence."""
    if _CRED_EXPOSURE.search(question):
        return TriageResult(
            "requires_escalation",
            "Suspected credential exposure is a documented escalation condition (KB-008).",
            decided_by="rule",
        )
    repeated_failure = bool(_ESCALATION_REPEAT.search(question) and _FAILURE_TERM.search(question))
    if repeated_failure and _evidence_supports_escalation(hits):
        return TriageResult(
            "requires_escalation",
            "Repeated failure after documented checks matches an escalation condition "
            "(KB-004 / KB-008).",
            decided_by="rule",
        )
    return None


# --------------------------------------------------------------------------- #
# Model check: answerable vs. clarification
# --------------------------------------------------------------------------- #
_CLASSIFY_SYSTEM = (
    "You are a triage classifier for an OrbitDesk support assistant. "
    "You do NOT answer questions; you only classify. Use ONLY the provided passages."
)

_CLASSIFY_TEMPLATE = """Decide whether the retrieved passages let us directly answer the user's question.

Choose exactly one label:
- ANSWERABLE: the passages state the relevant policy, permission, or procedure that
  answers the question. A question about what a role can or cannot do, or how a
  documented feature works, is ANSWERABLE if the passages state that rule -- even if
  you do not know the user's specific account details.
- NEEDS_CLARIFICATION: choose this ONLY when you genuinely cannot proceed without a
  specific missing detail -- for example a troubleshooting request that needs an error
  code or an object ID to select the correct documented path, or a request so vague
  the passages indicate it cannot be diagnosed as stated.

Examples:
Question: "Can an Analyst delete a workspace?"
LABEL: ANSWERABLE
REASON: The roles passage states only an Owner can delete a workspace.

Question: "My export isn't working, please fix it."
LABEL: NEEDS_CLARIFICATION
REASON: No error code or schedule detail is given to select a documented troubleshooting path.

User question:
{question}

Retrieved passages:
{passages}

Respond in EXACTLY this format and nothing else:
LABEL: <ANSWERABLE or NEEDS_CLARIFICATION>
REASON: <one short sentence>"""

_LABEL_RE = re.compile(r"LABEL:\s*(ANSWERABLE|NEEDS_CLARIFICATION)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I)


def _format_passages(hits: list[Retrieved], limit: int = 5) -> str:
    lines = []
    for h in hits[:limit]:
        note = " [SUPERSEDED — not current guidance]" if h.passage.is_superseded else ""
        lines.append(f"[{h.passage.passage_id}]{note}\n{h.passage.text}")
    return "\n\n".join(lines)


def classify_answerable_or_clarify(
    question: str, hits: list[Retrieved], llm: OllamaLLM
) -> TriageResult:
    prompt = _CLASSIFY_TEMPLATE.format(question=question, passages=_format_passages(hits))
    raw = llm.generate(prompt, system=_CLASSIFY_SYSTEM, temperature=0.0, num_predict=80)

    m = _LABEL_RE.search(raw)
    reason_m = _REASON_RE.search(raw)
    reason = reason_m.group(1).strip() if reason_m else ""

    if not m:
        # Unparseable -> safe default: ask for clarification rather than guess.
        return TriageResult(
            "requires_clarification",
            "Classifier output was unparseable; defaulting to clarification for safety.",
            decided_by="default",
            llm_label=raw[:60],
        )

    label = m.group(1).upper()
    if label == "ANSWERABLE":
        return TriageResult(
            "answerable",
            reason or "Retrieved passages directly support an answer.",
            decided_by="llm",
            llm_label=label,
        )
    return TriageResult(
        "requires_clarification",
        reason or "Question lacks the specific details a documented procedure requires.",
        decided_by="llm",
        llm_label=label,
    )


# --------------------------------------------------------------------------- #
# Orchestrating triage function
# --------------------------------------------------------------------------- #
def triage(question: str, hits: list[Retrieved], llm: OllamaLLM) -> TriageResult:
    # 1. Safety gate (no model).
    gated = safety_gate(question)
    if gated is not None:
        return gated

    # 2. Nothing relevant in the corpus (no model).
    top_score = hits[0].score if hits else 0.0
    if top_score < config.RETRIEVAL_SCORE_FLOOR:
        return TriageResult(
            "out_of_scope",
            f"Top retrieval score {top_score:.2f} is below the relevance floor "
            f"{config.RETRIEVAL_SCORE_FLOOR}; question is outside the knowledge base.",
            decided_by="score_floor",
        )

    # 3. Documented escalation trigger (no model).
    escalated = escalation_rule(question, hits)
    if escalated is not None:
        return escalated

    # 4. Otherwise let Gemma decide answerable vs. clarification.
    return classify_answerable_or_clarify(question, hits, llm)


if __name__ == "__main__":
    import json

    from .retrieval import Retriever

    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]
    r = Retriever().build()
    llm = OllamaLLM()
    print(f"ollama healthy: {llm.health()}  model={llm.model}\n")

    for q in questions:
        hits = r.search(q["question"], top_k=config.TOP_K)
        result = triage(q["question"], hits, llm)
        print(f"[{q['question_id']}] -> {result.classification.upper():<24} "
              f"(by {result.decided_by})")
        print(f"    top evidence : {hits[0].passage.passage_id}  ({hits[0].score:.2f})")
        print(f"    reason       : {result.reason}\n")
