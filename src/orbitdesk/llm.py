"""Thin wrapper around a LOCAL Gemma 3 4B served by Ollama.

No cloud APIs. Everything goes to http://localhost:11434. temperature defaults to
0 so the model is as close to deterministic as a local LLM gets — important for
reproducible routing. The wrapper only *calls* the model and returns text; every
decision made from that text lives in deterministic code elsewhere.
"""
from __future__ import annotations

import time

import httpx

from . import config


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or config.OLLAMA_MODEL
        self.host = host or config.OLLAMA_HOST
        self.last_latency_s: float | None = None
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        num_predict: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        t0 = time.perf_counter()
        resp = httpx.post(f"{self.host}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        self.last_latency_s = time.perf_counter() - t0
        self.call_count += 1
        return resp.json()["response"].strip()

    def health(self) -> bool:
        """True if the local Ollama server is reachable and has the model."""
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=5).json()
        except Exception:
            return False
        return any(m["name"] == self.model for m in tags.get("models", []))
