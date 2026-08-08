"""The typed shared state that flows through the graph.

Every node reads from and writes to this single object. `trace` uses an additive
reducer so each node *appends* its own log line rather than overwriting the log —
that list is the "which nodes executed" record the brief asks for.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .generate import Generation
from .retrieval import Retrieved
from .triage import TriageResult
from .verify import VerificationResult


class AgentState(TypedDict, total=False):
    # Input.
    question: str

    # Filled in as the graph runs.
    hits: list[Retrieved]
    triage_result: TriageResult
    generation: Generation
    response: dict[str, Any]          # the schema-shaped response object
    verification: VerificationResult

    # Retry control (the infinite-loop guard).
    revision_count: int
    revision_feedback: str | None

    # Execution log — additive across nodes.
    trace: Annotated[list[str], operator.add]
