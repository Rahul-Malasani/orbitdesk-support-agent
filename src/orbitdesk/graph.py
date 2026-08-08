"""LangGraph wiring: the four responsibilities become an orchestrated graph.

Flow:

    START
      -> retrieve
      -> triage
      -> [conditional] out_of_scope? -> refuse        (deterministic, no model)
                        else          -> generate
      -> verify
      -> [conditional] passed?          -> END (accept)
                        revisable & <MAX -> revise -> generate -> verify ... (one loop)
                        else            -> safe_fail -> END

Requirements demonstrated here:
  * Shared typed state ......... AgentState (state.py)
  * Conditional routing ........ route_after_triage, route_after_verify
  * Retry / fallback path ...... verify -> revise -> generate, then safe_fail
  * Deterministic vs model ..... refuse/verify/routing are code; generate/triage use Gemma
  * Node-execution logs ........ every node appends to state["trace"]
  * Infinite-loop protection ... revision_count capped at MAX_REVISIONS
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import config
from .generate import generate_response, safe_refusal
from .llm import OllamaLLM
from .retrieval import Retriever
from .state import AgentState
from .triage import triage
from .verify import build_response, safe_failure_response, verify

# Routes whose answers can be revised if they fail verification.
_REVISABLE = {"answerable", "requires_escalation"}


def build_app(retriever: Retriever, llm: OllamaLLM):
    """Compile the graph, closing over a ready retriever + LLM client."""

    # -- nodes --------------------------------------------------------------
    def retrieve_node(state: AgentState) -> dict:
        hits = retriever.search(state["question"], top_k=config.TOP_K)
        top = hits[0]
        return {
            "hits": hits,
            "trace": [f"retrieve: top={top.passage.passage_id} score={top.score:.2f} (n={len(hits)})"],
        }

    def triage_node(state: AgentState) -> dict:
        tr = triage(state["question"], state["hits"], llm)
        return {"triage_result": tr, "trace": [f"triage: {tr.classification} (by {tr.decided_by})"]}

    def generate_node(state: AgentState) -> dict:
        tr = state["triage_result"]
        feedback = state.get("revision_feedback")
        gen = generate_response(tr, state["question"], state["hits"], llm, feedback=feedback)
        resp = build_response(tr, gen, state["hits"])
        tag = "revise-generate" if feedback else "generate"
        return {
            "generation": gen,
            "response": resp,
            "trace": [f"{tag}: {tr.classification} answer_len={len(gen.answer)} sources={len(gen.sources)}"],
        }

    def refuse_node(state: AgentState) -> dict:
        tr = state["triage_result"]
        gen = safe_refusal(tr)
        resp = build_response(tr, gen, state["hits"])
        return {"generation": gen, "response": resp, "trace": ["refuse: deterministic safe refusal"]}

    def verify_node(state: AgentState) -> dict:
        v = verify(state["response"], state["hits"])
        line = f"verify: {'PASS' if v.passed else 'FAIL'}"
        if not v.passed:
            line += f" -> {v.failures}"
        return {"verification": v, "trace": [line]}

    def revise_node(state: AgentState) -> dict:
        v = state["verification"]
        feedback = "; ".join(v.failures)
        attempt = state.get("revision_count", 0) + 1
        return {
            "revision_count": attempt,
            "revision_feedback": feedback,
            "trace": [f"revise: attempt {attempt} (feedback: {feedback})"],
        }

    def safe_fail_node(state: AgentState) -> dict:
        v = state["verification"]
        resp = safe_failure_response("; ".join(v.failures), state["hits"])
        return {"response": resp, "trace": ["safe_fail: returned safe_failure response"]}

    # -- conditional routers ------------------------------------------------
    def route_after_triage(state: AgentState) -> str:
        return "refuse" if state["triage_result"].classification == "out_of_scope" else "generate"

    def route_after_verify(state: AgentState) -> str:
        if state["verification"].passed:
            return "accept"
        route = state["response"]["classification"]
        if state.get("revision_count", 0) < config.MAX_REVISIONS and route in _REVISABLE:
            return "revise"
        return "safe_fail"

    # -- wiring -------------------------------------------------------------
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("triage", triage_node)
    g.add_node("generate", generate_node)
    g.add_node("refuse", refuse_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)
    g.add_node("safe_fail", safe_fail_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "triage")
    g.add_conditional_edges("triage", route_after_triage, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "verify")
    g.add_edge("refuse", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify,
        {"accept": END, "revise": "revise", "safe_fail": "safe_fail"},
    )
    g.add_edge("revise", "generate")
    g.add_edge("safe_fail", END)

    return g.compile()


class SupportAgent:
    """Convenience wrapper: load models once, answer many questions."""

    def __init__(self, retriever: Retriever | None = None, llm: OllamaLLM | None = None) -> None:
        self.retriever = (retriever or Retriever()).build()
        self.llm = llm or OllamaLLM()
        self.app = build_app(self.retriever, self.llm)

    def answer(self, question: str) -> dict:
        final = self.app.invoke(
            {"question": question, "revision_count": 0, "revision_feedback": None, "trace": []},
            config={"recursion_limit": 25},  # hard backstop beyond MAX_REVISIONS
        )
        return final

    def graph_object(self):
        return self.app.get_graph()


if __name__ == "__main__":
    import json

    agent = SupportAgent()
    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]

    for q in questions:
        final = agent.answer(q["question"])
        resp = final["response"]
        print(f"\n================= [{q['question_id']}] =================")
        print("TRACE:")
        for line in final["trace"]:
            print(f"   -> {line}")
        print(f"RESULT: classification={resp['classification']}  "
              f"confidence={resp['confidence']}  requires_human={resp['requires_human']}")
        print(f"answer: {resp['answer'][:200]}{'...' if len(resp['answer'])>200 else ''}")
        print(f"sources: {[s['source_id'] for s in resp['sources']]}")
