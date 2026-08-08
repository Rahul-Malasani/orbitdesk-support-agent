"""Ingestion step: parse the corpus and write it to index/corpus.json.

Run:  python -m scripts.ingest

This is deterministic (no models). It exists so ingestion produces a tangible,
inspectable artifact rather than only living in memory.
"""
from __future__ import annotations

from pathlib import Path

from src.orbitdesk import config
from src.orbitdesk.data import build_corpus, save_corpus

OUT_PATH = config.REPO_ROOT / "index" / "corpus.json"


def main() -> None:
    passages = build_corpus()
    save_corpus(passages, OUT_PATH)

    kb = [p for p in passages if p.source_kind == "kb"]
    cases = [p for p in passages if p.source_kind == "case"]
    superseded = [p for p in passages if p.is_superseded]

    print("Ingestion complete.")
    print(f"  passages written : {len(passages)}")
    print(f"    KB sections    : {len(kb)}  (from {len({p.source_id for p in kb})} docs)")
    print(f"    resolved cases : {len(cases)}")
    print(f"    superseded     : {len(superseded)} -> {[p.source_id for p in superseded]}")
    print(f"  output file      : {OUT_PATH.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
