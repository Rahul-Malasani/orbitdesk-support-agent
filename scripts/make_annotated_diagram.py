"""Render an ANNOTATED graph diagram: every node's responsibility + every path.

Run:  python -m scripts.make_annotated_diagram

Writes docs/graph_annotated.mmd and docs/graph_annotated.png. Nodes are colored
green (deterministic code), blue (model), or amber (mixed) so the
deterministic-vs-model split is visible at a glance.
"""
from __future__ import annotations

from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from src.orbitdesk import config

DOCS = config.REPO_ROOT / "docs"

MERMAID = r"""flowchart TD
    START([START]) --> R

    R["<b>retrieve</b><br/>Embed the question, return the<br/>top-5 passages by cosine similarity"]:::model
    R --> T

    T["<b>triage</b> — pick ONE route<br/>1) safety gate  2) score floor<br/>3) escalation rule  = deterministic<br/>4) Gemma: answerable vs clarification"]:::mixed

    T -->|out_of_scope| RF["<b>refuse</b><br/>Fixed safe refusal<br/>(no model call)"]:::code
    T -->|"answerable / clarification / escalation"| G["<b>generate</b><br/>Gemma writes from evidence ONLY<br/>prompt isolation + code-validated citations<br/>(superseded passages filtered out)"]:::model

    RF --> V
    G --> V

    V["<b>verify</b> — deterministic checks<br/>schema · citations exist · grounded<br/>· no forbidden/superseded content"]:::code

    V -->|pass| ACC([END — accept])
    V -->|"fail &amp; revisions &lt; max"| RV["<b>revise</b><br/>Feed the failure reasons back,<br/>increment the revision counter"]:::code
    V -->|"fail &amp; at max revisions"| SF["<b>safe_fail</b><br/>Return the safe_failure response"]:::code

    RV --> G
    SF --> SFE([END — safe failure])

    subgraph Legend [ ]
      direction LR
      L1["deterministic code"]:::code
      L2["local model"]:::model
      L3["code + model"]:::mixed
    end

    classDef code fill:#e6f4ea,stroke:#137333,color:#0d652d
    classDef model fill:#e8f0fe,stroke:#1a73e8,color:#174ea6
    classDef mixed fill:#fef7e0,stroke:#f9ab00,color:#b06000
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "graph_annotated.mmd").write_text(MERMAID, encoding="utf-8")
    print(f"wrote {(DOCS / 'graph_annotated.mmd').relative_to(config.REPO_ROOT)}")
    try:
        png = draw_mermaid_png(MERMAID)
        (DOCS / "graph_annotated.png").write_bytes(png)
        print(f"wrote {(DOCS / 'graph_annotated.png').relative_to(config.REPO_ROOT)}")
    except Exception as exc:
        print(f"PNG render skipped ({exc}); the .mmd can be rendered manually.")


if __name__ == "__main__":
    main()
