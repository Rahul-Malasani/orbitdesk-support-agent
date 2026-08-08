"""Render the compiled graph to docs/graph.mmd and docs/graph.png.

Run:  python -m scripts.make_graph_diagram

The structure comes from the *actual* compiled graph, so the diagram can never
drift from the code. We pass dummy retriever/llm because building the graph only
closes over them; no model is loaded and nothing is called.
"""
from __future__ import annotations

from src.orbitdesk import config
from src.orbitdesk.graph import build_app

DOCS = config.REPO_ROOT / "docs"


def main() -> None:
    app = build_app(retriever=None, llm=None)  # structure only; nodes never run
    graph = app.get_graph()

    DOCS.mkdir(parents=True, exist_ok=True)
    mmd_path = DOCS / "graph.mmd"
    mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"wrote {mmd_path.relative_to(config.REPO_ROOT)}")

    png_path = DOCS / "graph.png"
    try:
        png_path.write_bytes(graph.draw_mermaid_png())
        print(f"wrote {png_path.relative_to(config.REPO_ROOT)}")
    except Exception as exc:  # offline / renderer unavailable
        print(f"PNG render skipped ({exc}); graph.mmd is available to render manually.")


if __name__ == "__main__":
    main()
