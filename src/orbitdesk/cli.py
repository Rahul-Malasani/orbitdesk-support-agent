"""Command-line interface for the OrbitDesk support agent.

Examples:
  python -m src.orbitdesk.cli "Can a Viewer create an API credential?"
  python -m src.orbitdesk.cli --json "..."      # print the raw JSON response
  python -m src.orbitdesk.cli                    # interactive prompt loop
"""
from __future__ import annotations

import argparse
import json
import sys

from .graph import SupportAgent


def _print(final: dict, as_json: bool, show_trace: bool) -> None:
    resp = final["response"]
    if as_json:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return
    print(f"\n[{resp['classification']}]  confidence={resp['confidence']}  "
          f"requires_human={resp['requires_human']}")
    print(f"\n{resp['answer']}\n")
    if resp["sources"]:
        print("Sources: " + ", ".join(s["source_id"] for s in resp["sources"]))
    if show_trace:
        print("\nTrace:")
        for line in final["trace"]:
            print(f"  -> {line}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OrbitDesk local support agent")
    parser.add_argument("question", nargs="?", help="a support question")
    parser.add_argument("--json", action="store_true", help="print the JSON response only")
    parser.add_argument("--trace", action="store_true", help="also print the node-execution trace")
    args = parser.parse_args(argv)

    agent = SupportAgent()
    if not agent.llm.health():
        print(f"WARNING: Ollama model '{agent.llm.model}' not reachable at "
              f"{agent.llm.host}. Start Ollama and `ollama pull {agent.llm.model}`.",
              file=sys.stderr)

    if args.question:
        _print(agent.answer(args.question), args.json, args.trace)
        return

    print("OrbitDesk support agent — type a question (Ctrl-C to exit).")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                _print(agent.answer(q), args.json, args.trace)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
