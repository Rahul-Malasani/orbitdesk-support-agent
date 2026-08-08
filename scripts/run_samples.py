"""Run all sample questions through the agent and save the outputs.

Run:  python -m scripts.run_samples

Writes sample_outputs/sample_outputs.json (structured) and
sample_outputs/sample_outputs.md (readable), including run metadata: exact model
names + revisions, hardware, and load/latency timings.
"""
from __future__ import annotations

import glob
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx

from src.orbitdesk import config
from src.orbitdesk.graph import SupportAgent

OUT_DIR = config.REPO_ROOT / "sample_outputs"


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def _embedding_revision() -> str:
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*"))
    return Path(hits[0]).name if hits else "unknown"


def _gemma_details() -> dict:
    try:
        tags = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5).json()
        for m in tags.get("models", []):
            if m["name"] == config.OLLAMA_MODEL:
                return {
                    "digest": m.get("digest", "")[:20],
                    "parameter_size": m.get("details", {}).get("parameter_size"),
                    "quantization": m.get("details", {}).get("quantization_level"),
                }
    except Exception:
        pass
    return {}


def _hardware() -> dict:
    try:
        ram = f"{int(_sysctl('hw.memsize')) / 1024 ** 3:.0f} GB"
    except Exception:
        ram = "unknown"
    import torch

    return {
        "platform": platform.platform(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "logical_cpus": _sysctl("hw.ncpu"),
        "ram": ram,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "embedding_device": "mps" if torch.backends.mps.is_available() else "cpu",
    }


def main() -> None:
    questions = json.loads(config.SAMPLE_QUESTIONS_FILE.read_text())["questions"]
    agent = SupportAgent()

    results = []
    for q in questions:
        t0 = time.perf_counter()
        final = agent.answer(q["question"])
        latency = time.perf_counter() - t0
        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "response": final["response"],
            "trace": final["trace"],
            "latency_s": round(latency, 2),
        })

    metadata = {
        "models": {
            "embedding": {
                "name": config.EMBEDDING_MODEL,
                "revision": _embedding_revision(),
                "library": "sentence-transformers (Hugging Face)",
                "device": _hardware()["embedding_device"],
            },
            "generation": {
                "name": config.OLLAMA_MODEL,
                "served_by": "Ollama (local, no cloud API)",
                **_gemma_details(),
            },
        },
        "hardware": _hardware(),
        "retriever": {
            "cache_hit": agent.retriever.cache_hit,
            "model_load_s": agent.retriever.load_time_s,
            "corpus_embed_s": agent.retriever.embed_time_s,
            "passages": len(agent.retriever.passages),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sample_outputs.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Readable Markdown.
    lines = ["# OrbitDesk Support Agent — Sample Outputs\n"]
    lines.append("## Run metadata\n")
    lines.append("```json")
    lines.append(json.dumps(metadata, indent=2, ensure_ascii=False))
    lines.append("```\n")
    for r in results:
        resp = r["response"]
        lines.append(f"## [{r['question_id']}] {resp['classification']}  ({r['latency_s']}s)\n")
        lines.append(f"**Question:** {r['question']}\n")
        lines.append(f"**Answer:** {resp['answer']}\n")
        lines.append(f"**Sources:** {[s['source_id'] for s in resp['sources']]}  ")
        lines.append(f"**confidence:** {resp['confidence']}  **requires_human:** {resp['requires_human']}\n")
        lines.append("**Trace:**\n")
        for t in r["trace"]:
            lines.append(f"- {t}")
        lines.append("")
    (OUT_DIR / "sample_outputs.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_DIR / 'sample_outputs.json'}")
    print(f"Wrote {OUT_DIR / 'sample_outputs.md'}")
    for r in results:
        print(f"  [{r['question_id']}] {r['response']['classification']:<22} {r['latency_s']}s")


if __name__ == "__main__":
    main()
