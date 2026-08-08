"""Verification must reject bad responses (not just rubber-stamp good ones)."""
from src.orbitdesk.verify import verify
from tests.stubs import make_hits

_HITS = make_hits([("KB-002::Viewer", 0.70), ("KB-005::Creating a Credential", 0.70)])


def _resp(**over):
    base = {
        "classification": "answerable",
        "answer": "Only Owners and Admins can create workspace API credentials.",
        "sources": [{"source_id": "KB-002", "passage": "KB-002::Viewer"}],
        "confidence": 0.8,
        "requires_human": False,
        "reason": "test",
        "clarification_question": None,
        "warnings": [],
    }
    base.update(over)
    return base


def test_good_response_passes():
    assert verify(_resp(), _HITS).passed


def test_invented_citation_rejected():
    v = verify(_resp(sources=[{"source_id": "KB-999", "passage": "x"}]), _HITS)
    assert not v.passed and not v.checks["citations_valid"]


def test_ungrounded_answer_rejected():
    v = verify(_resp(answer="Bananas are yellow tropical fruit."), _HITS)
    assert not v.passed and not v.checks["grounded"]


def test_answerable_without_citations_rejected():
    v = verify(_resp(sources=[]), _HITS)
    assert not v.passed and not v.checks["has_citations"]


def test_superseded_guidance_rejected():
    v = verify(_resp(answer="Open Profile > Personal token to create a token."), _HITS)
    assert not v.passed and not v.checks["no_forbidden"]


def test_schema_violation_rejected():
    v = verify(_resp(confidence=2.0), _HITS)
    assert not v.passed and not v.checks["schema_valid"]
