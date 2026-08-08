"""Data layer: load the knowledge base + resolved cases into searchable passages.

This module is 100% deterministic — no models. It is the single source of truth
that everything downstream grounds against. Two rules from the assignment are
enforced right here, at load time:

  1. Superseded resolved cases (e.g. CASE-0914, the obsolete "personal token"
     advice) are FLAGGED. They stay in the index so we can still retrieve them
     for testing, but the flag lets retrieval/verification refuse to present
     them as current guidance.
  2. Knowledge-base documents outrank resolved cases when they conflict. We tag
     every passage with `source_kind` and a numeric `authority` so later stages
     can prefer KB over cases deterministically.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import config


# Authority: higher wins when two passages conflict. KB > resolved case.
# Superseded cases are pushed to the bottom.
AUTHORITY_KB = 100
AUTHORITY_CASE = 50
AUTHORITY_SUPERSEDED = 0


@dataclass
class Passage:
    """One retrievable unit of evidence with a stable, citable identity."""

    source_id: str          # "KB-004" or "CASE-1103"  -> goes in the citation
    passage_id: str         # stable sub-id, e.g. "KB-004::Troubleshooting..."
    source_kind: str        # "kb" | "case"
    title: str              # human-readable doc/case title
    section: str            # KB heading, or the case's role in the record
    text: str               # the text we embed and show as evidence
    status: str             # "current" | "resolved" | "escalated" | "superseded"
    is_superseded: bool     # True only for superseded cases
    authority: int          # AUTHORITY_* — deterministic tie-breaker
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> dict[str, str]:
        """Shape used in the response `sources` array (matches output_schema)."""
        return {"source_id": self.source_id, "passage": self.passage_id}


# --------------------------------------------------------------------------- #
# Knowledge-base loading
# --------------------------------------------------------------------------- #
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_H2_SPLIT_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown file into (yaml frontmatter dict, body)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, match.group(2)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a KB body into (heading, content) chunks on `##` headers.

    Text before the first `##` (the `# Title` line + intro) becomes an
    "Overview" section so nothing is dropped.
    """
    parts = _H2_SPLIT_RE.split(body)
    sections: list[tuple[str, str]] = []

    # parts[0] is everything before the first `##` heading.
    intro = parts[0].strip()
    # Drop a leading `# Title` line from the intro; it repeats the frontmatter.
    intro = re.sub(r"^#\s+.*\n?", "", intro).strip()
    if intro:
        sections.append(("Overview", intro))

    # Remaining parts alternate: heading, content, heading, content, ...
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, content))
    return sections


def load_kb(kb_dir: Path | None = None) -> list[Passage]:
    """Load every KB Markdown file into section-level passages."""
    kb_dir = kb_dir or config.KB_DIR
    passages: list[Passage] = []

    for path in sorted(kb_dir.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("document_id", path.stem)
        title = meta.get("title", path.stem)
        status = meta.get("status", "current")

        for heading, content in _split_sections(body):
            # Embed with the doc title + heading as context — short passages
            # retrieve far better when they carry their own topic label.
            text = f"{title} — {heading}\n{content}"
            passages.append(
                Passage(
                    source_id=doc_id,
                    passage_id=f"{doc_id}::{heading}",
                    source_kind="kb",
                    title=title,
                    section=heading,
                    text=text,
                    status=status,
                    is_superseded=False,
                    authority=AUTHORITY_KB,
                    metadata={
                        "filename": path.name,
                        "updated": str(meta.get("updated", "")),
                        "tags": meta.get("tags", []),
                    },
                )
            )
    return passages


# --------------------------------------------------------------------------- #
# Resolved-case loading
# --------------------------------------------------------------------------- #
def _format_case(case: dict[str, Any]) -> str:
    """Render a resolved case into a single readable passage."""
    lines = [f"Case {case['case_id']}: {case['title']}"]
    if case.get("symptoms"):
        lines.append("Symptoms: " + "; ".join(case["symptoms"]))
    if case.get("resolution"):
        lines.append("Resolution: " + "; ".join(case["resolution"]))
    if case.get("important_limit"):
        lines.append("Important limit: " + case["important_limit"])
    if case.get("superseded_reason"):
        lines.append("Superseded: " + case["superseded_reason"])
    return "\n".join(lines)


def load_cases(cases_file: Path | None = None) -> list[Passage]:
    """Load resolved cases; flag superseded ones and lower their authority."""
    cases_file = cases_file or config.CASES_FILE
    data = json.loads(cases_file.read_text(encoding="utf-8"))
    passages: list[Passage] = []

    for case in data.get("cases", []):
        status = case.get("status", "resolved")
        is_superseded = status == "superseded"
        passages.append(
            Passage(
                source_id=case["case_id"],
                passage_id=case["case_id"],
                source_kind="case",
                title=case["title"],
                section=status,
                text=_format_case(case),
                status=status,
                is_superseded=is_superseded,
                authority=AUTHORITY_SUPERSEDED if is_superseded else AUTHORITY_CASE,
                metadata={
                    "product_version": case.get("product_version", ""),
                    "source_documents": case.get("source_documents", []),
                },
            )
        )
    return passages


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_corpus() -> list[Passage]:
    """The full searchable corpus: KB passages first, then resolved cases."""
    return load_kb() + load_cases()


def save_corpus(passages: list[Passage], path: Path) -> None:
    """Persist the ingested passages to a human-readable JSON file.

    This is a *derived* artifact (the md files remain the source of truth); it
    exists so you can open it and see exactly what ingestion produced, and so
    the embedding step has a stable input to point at.
    """
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passage_count": len(passages),
        "passages": [asdict(p) for p in passages],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    corpus = build_corpus()
    kb = [p for p in corpus if p.source_kind == "kb"]
    cases = [p for p in corpus if p.source_kind == "case"]
    superseded = [p for p in corpus if p.is_superseded]

    print(f"Total passages : {len(corpus)}")
    print(f"  KB sections  : {len(kb)}  (from {len({p.source_id for p in kb})} docs)")
    print(f"  Cases        : {len(cases)}")
    print(f"  Superseded   : {len(superseded)} -> {[p.source_id for p in superseded]}")
    print("\nSample KB passage ------------------------------------------")
    print(f"  source_id  : {kb[0].source_id}")
    print(f"  passage_id : {kb[0].passage_id}")
    print(f"  authority  : {kb[0].authority}")
    print(f"  text[:160] : {kb[0].text[:160]!r}")
    print("\nSuperseded case (must NOT be presented as current) ---------")
    sp = superseded[0]
    print(f"  source_id  : {sp.source_id}  status={sp.status}  authority={sp.authority}")
    print(f"  text       :\n{sp.text}")
