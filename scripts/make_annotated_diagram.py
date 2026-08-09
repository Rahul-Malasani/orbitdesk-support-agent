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
    Q([User asks a question]) --> R[retrieve]:::model
    R --> T{triage}:::mixed

    T -->|out of scope / forbidden| RF[refuse]:::code
    T -->|answerable| G[generate]:::model
    T -->|needs clarification| G
    T -->|requires escalation| G

    RF --> V{verify}:::code
    G --> V

    V -->|pass| DONE([Return response]):::done
    V -->|fail · retry once| RV[revise]:::code
    V -->|still fails| SF[safe_fail]:::code
    RV --> G
    SF --> DONE

    subgraph Legend [ ]
      direction LR
      L1[deterministic code]:::code
      L2[local model]:::model
      L3[code + model]:::mixed
    end

    classDef code fill:#e6f4ea,stroke:#137333,color:#0d652d
    classDef model fill:#e8f0fe,stroke:#1a73e8,color:#174ea6
    classDef mixed fill:#fef7e0,stroke:#f9ab00,color:#b06000
    classDef done fill:#f3e8fd,stroke:#8430ce,color:#5b1a95
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
