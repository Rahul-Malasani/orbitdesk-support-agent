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
    R --> T{triage:<br/>classify request}:::mixed

    %% ---- the four classification routes ----
    T -->|out of scope / forbidden| RF[refuse]:::code
    T -->|answerable| G[generate]:::model
    T -->|needs clarification| G
    T -->|requires escalation| G

    RF --> V
    G --> V

    %% ---- verification / retry / fallback, grouped at the bottom ----
    subgraph BOTTOM [Verification and retry]
      direction TB
      V{verify}:::code
      V -->|fail · under retry limit| RV[revise and retry]:::code
      V -->|fail again| SF[safe_fail]:::code
    end

    V -->|pass| DONE([return response]):::done
    RV -->|regenerate once| G
    SF --> DONE

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
