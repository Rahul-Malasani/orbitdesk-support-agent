"""Scripted demo for the video walkthrough.

Run:  python -m scripts.demo

Walks through, with a pause between sections (press Enter to advance):
  0. Models + device being loaded
  1. ANSWERABLE across TWO documents, showing retrieved evidence + sources
  2. CLARIFICATION route
  3. OUT_OF_SCOPE route (deterministic refuse, no model call)
  4. ESCALATION route (deterministic rule)
  5. Verification RETRY -> recover, and RETRY -> safe_failure

Each run prints the node-execution trace so the conditional path is visible.
"""
from __future__ import annotations

import httpx

from src.orbitdesk import config
from src.orbitdesk.graph import SupportAgent, build_app
from tests.stubs import StubLLM, StubRetriever, make_hits


def pause(msg: str = "\n[press Enter to continue] ") -> None:
    try:
        input(msg)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def show_run(agent: SupportAgent, question: str, show_evidence: bool = False) -> None:
    if show_evidence:
        print("RETRIEVED EVIDENCE (top 3 by cosine similarity):")
        for h in agent.retriever.search(question, top_k=3):
            print(f"   {h.score:.2f}  {h.passage.passage_id}")
        print()
    final = agent.answer(question)
    resp = final["response"]
    print("TRACE (nodes executed + conditional path):")
    for line in final["trace"]:
        print(f"   -> {line}")
    print(f"\nRESULT: {resp['classification']}  "
          f"confidence={resp['confidence']}  requires_human={resp['requires_human']}")
    print(f"ANSWER: {resp['answer']}")
    print(f"SOURCES: {[s['source_id'] for s in resp['sources']]}")


def main() -> None:
    banner("0. MODELS + DEVICE")
    agent = SupportAgent()  # loads the embedding model now
    print(f"Embedding model : {config.EMBEDDING_MODEL}  (Hugging Face / sentence-transformers)")
    print(f"Embedding device: {agent.retriever.device}")
    print(f"Model load      : {agent.retriever.load_time_s:.2f}s   "
          f"corpus embed: {agent.retriever.embed_time_s:.2f}s   "
          f"cache_hit={agent.retriever.cache_hit}")
    try:
        tags = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5).json()
        g = next(m for m in tags["models"] if m["name"] == config.OLLAMA_MODEL)
        print(f"Generation model: {config.OLLAMA_MODEL}  via Ollama (local)  "
              f"{g['details'].get('parameter_size')} {g['details'].get('quantization_level')}")
    except Exception:
        print(f"Generation model: {config.OLLAMA_MODEL}  via Ollama (local)")
    print(f"Ollama reachable: {agent.llm.health()}")

    # Run the EXACT questions provided in data/sample_questions.json — nothing
    # reworded — so it is verifiable these are the assignment's own questions.
    import json
    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]
    for i, q in enumerate(questions, start=1):
        pause()
        banner(f"{i}. [{q['question_id']}] live run  (from sample_questions.json)")
        print(f"QUESTION: {q['question']}\n")
        # Show the retrieved evidence for the first one (the two-document case).
        show_run(agent, q["question"], show_evidence=(q["question_id"] == "Q-001"))

    pause()
    banner("6. VERIFICATION RETRY  (stubbed bad first answer, forcing the loop)")
    hits = make_hits([("KB-002::Viewer", 0.70), ("KB-005::Creating a Credential", 0.70)])
    bad = "Bananas grow on trees.\nSOURCES:"
    good = ("Viewers cannot create API credentials; only Owners and Admins can create "
            "workspace credentials.\nSOURCES: KB-002, KB-005")
    q = "Can a Viewer create an API credential?"

    for label, answers in [("RETRY -> RECOVERS", [bad, good]), ("RETRY -> SAFE_FAILURE", [bad, bad])]:
        print(f"\n--- {label} ---")
        app = build_app(StubRetriever(hits), StubLLM(label="ANSWERABLE", gen_answers=answers))
        final = app.invoke(
            {"question": q, "revision_count": 0, "revision_feedback": None, "trace": []},
            config={"recursion_limit": 25},
        )
        for line in final["trace"]:
            print(f"   -> {line}")
        print(f"   FINAL classification: {final['response']['classification']}")

    print("\nDemo complete. (Automated tests: `.venv/bin/python -m pytest -v`)")


if __name__ == "__main__":
    main()
