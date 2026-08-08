"""The retry/fallback path — the 5th required test case.

A stubbed LLM returns a bad first answer (ungrounded, uncited) so verification
fails and the graph must revise. One variant recovers on the retry; the other
stays bad and must terminate in a safe failure (proving the loop guard).
"""
from src.orbitdesk.graph import build_app
from tests.stubs import StubLLM, StubRetriever, make_hits

_QUESTION = "Can a Viewer create an API credential?"
_HITS = make_hits([("KB-002::Viewer", 0.70), ("KB-005::Creating a Credential", 0.70)])

_BAD = "Bananas grow on trees.\nSOURCES:"  # ungrounded + no valid citation
_GOOD = (
    "Viewers cannot create API credentials; only Owners and Admins can create "
    "workspace credentials.\nSOURCES: KB-002, KB-005"
)


def _run(gen_answers):
    llm = StubLLM(label="ANSWERABLE", gen_answers=gen_answers)
    app = build_app(StubRetriever(_HITS), llm)
    return app.invoke(
        {"question": _QUESTION, "revision_count": 0, "revision_feedback": None, "trace": []},
        config={"recursion_limit": 25},
    )


def test_failed_verification_triggers_revision_then_recovers():
    final = _run([_BAD, _GOOD])
    trace = final["trace"]
    assert any("verify: FAIL" in t for t in trace)      # first attempt rejected
    assert any(t.startswith("revise:") for t in trace)  # revision fired
    assert final["response"]["classification"] == "answerable"
    assert final["verification"].passed                 # recovered


def test_persistent_failure_ends_in_safe_failure():
    final = _run([_BAD, _BAD])  # never improves
    assert final["response"]["classification"] == "safe_failure"
    assert any("safe_fail" in t for t in final["trace"])
    # Loop guard: exactly one revision, then stop (no infinite loop).
    assert final["revision_count"] == 1
