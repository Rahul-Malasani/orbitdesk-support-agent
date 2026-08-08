# Project Cheatsheet — every file, every decision I own

Off-screen reference for the video and Q&A. The system is **one agent**, built as a
LangGraph pipeline: `retrieve → triage → generate → verify`, with a bounded
revise/safe-failure loop. The whole thing runs locally — bge-small embeddings via
Hugging Face on the Apple MPS GPU, and Gemma 3 4B via local Ollama. No cloud LLM.

---

## The 10 source files (`src/orbitdesk/`)

### `config.py` — the single control panel
- Holds **every knob** in one place: file paths, the two model names
  (`bge-small-en-v1.5`, `gemma3:4b`), device (`mps`), and tunables (`TOP_K=5`,
  `RETRIEVAL_SCORE_FLOOR=0.30`, `GROUNDING_THRESHOLD=0.45`, `MAX_REVISIONS=1`).
- **Nothing runs here** — it's just settings the other files import.
- **Why I did it:** so the logic never hardcodes a model or threshold — models are
  swappable and every tunable is auditable in one spot. That *is* the "hardware-aware
  trade-off" surface the brief asks for.

### `data.py` — raw files → clean, tagged passages (deterministic, no model)
- Parses the 10 KB markdown files (frontmatter + split **per `##` section**) and
  `resolved_cases.json` into a `Passage` dataclass → **53 passages**.
- Each passage carries `source_id`, `passage_id`, `text`, `authority` (KB=100,
  case=50, superseded=0) and `is_superseded`.
- **Why I did it:** section-level chunks make retrieval precise; the passages live in
  memory (the raw files stay the single source of truth). `corpus.json` is only a
  human-readable dump from `scripts/ingest.py` — nothing reads it back.
- **I own this caveat:** only the superseded rule (`is_superseded`) is *enforced* in
  code; the KB-vs-case authority *number* is reserved, not yet used as a tiebreak.

### `retrieval.py` — local embeddings + in-memory search
- Loads bge-small on MPS, encodes the 53 passages once, **L2-normalizes** them so
  cosine similarity becomes a plain dot product, and does **exact top-k search** —
  no vector database.
- Caches vectors to `index/embeddings.npy`, keyed by a hash of (model + passage
  text); recomputes only if the corpus changed. Records model-load and embed times.
- **Why I did it:** 53 passages is tiny — exact search is simpler and *more* accurate
  than an approximate vector DB, which the brief explicitly says isn't required.
  Returns pure similarity; all policy lives downstream.

### `llm.py` — the local Gemma wrapper
- Thin client over Ollama at `localhost:11434`; `generate()` returns text,
  `health()` checks the model is present. Temperature 0 by default.
- **Why I did it:** keep the model boundary thin and local — it only *produces text*;
  every decision made from that text is deterministic code elsewhere.

### `triage.py` — the routing brain (the 4-rung waterfall)
1. `safety_gate` — regex for injection / refunds / legal / secrets → `out_of_scope`.
2. score floor — top similarity `< 0.30` → `out_of_scope` (nothing relevant).
3. `escalation_rule` — a repeat-failure trigger in the **question** *and* supporting
   **evidence** → `requires_escalation`.
4. else → **Gemma** decides `answerable` vs `requires_clarification`.
- **Why I did it:** three routes are deterministic and testable; the model is trusted
  only with the genuine language judgment. Safety is an `if`, not a model call.

### `generate.py` — evidence-only answers
- Per route: `out_of_scope` = a **fixed deterministic refusal** (no model);
  answerable/escalation/clarification = Gemma with **prompt isolation** (rules in the
  system role; question and passages wrapped in `<user_question>`/`<evidence>` tags
  declared as *data, not instructions*).
- Superseded passages are **filtered out before the model sees them**; citations are
  model-proposed but **code-validated** (invented `source_id`s are dropped); stray
  tags are stripped.
- **Why I did it:** the model can't invent instructions or refuse in its own words,
  and it can't be injected via a retrieved document.

### `verify.py` — the independent gatekeeper
- Deterministic checks on the assembled response: `schema_valid` (vs
  `output_schema.json`), `has_citations`, `citations_valid`, `grounded` (content-word
  overlap ≥ 0.45, stop-words removed), `no_forbidden` (superseded phrases, secret
  solicitation, leaked tags). Also builds the response object and the `safe_failure`.
- **Why I did it:** a generator shouldn't grade its own homework — verification is a
  separate, testable checker. Grounding is a heuristic (a known limitation I'd swap
  for an NLI model).

### `state.py` — the shared typed state
- One `AgentState` TypedDict flows through every node: question, hits, triage result,
  response, verification, `revision_count`, and a `trace` list that **accumulates**
  (additive reducer) — that list is the "which nodes ran" log.
- **Why I did it:** typed shared state + an append-only trace is exactly what the
  orchestration requirements ask for.

### `graph.py` — the LangGraph wiring
- **7 nodes** (retrieve, triage, generate, refuse, verify, revise, safe_fail) + **2
  conditional routers** (`route_after_triage`, `route_after_verify`) — routers are
  *edge decisions*, not nodes, and no node is an "agent."
- Flow: `start → retrieve → triage → (refuse | generate) → verify → (accept |
  revise→generate | safe_fail)`. `MAX_REVISIONS=1` + a `recursion_limit` backstop
  guarantee termination. `SupportAgent` loads models once and answers many questions.
- **Why I did it:** the graph only *orchestrates* — the logic lives in the four
  modules, which is why each was testable on its own.

### `cli.py` — the interface
- `python -m src.orbitdesk.cli "question"` (or `--json` / `--trace` / interactive).
  Warns if Ollama is unreachable.
- **Why I did it:** the brief asks for a CLI/notebook/minimal API; this is the CLI.

---

## Decisions I own (one-liners for Q&A)

- **Local models:** Gemma 3 4B (Ollama) + bge-small (Hugging Face, MPS). No cloud, ever.
- **No vector DB:** exact cosine over 53 passages — simpler and more accurate at this scale.
- **Deterministic vs. model:** 3 of 4 routes + all verification + all safety are code;
  the model only embeds, classifies answerable-vs-clarify, and writes prose.
- **Prompt isolation** backs up the deterministic gate (defense-in-depth); the KB itself
  warns injections can arrive via retrieved docs, so I tag evidence too.
- **Superseded handling** is enforced in code (filtered before generation), not left to the model.
- **Validated citations:** any source the model invents is dropped before it ships.
- **Bounded retry:** one revision, then safe-failure — no infinite loop.
- **Python 3.11** (PyTorch wheels); generation temperature 0.1, triage classification 0.

## Likely questions — and my answers

- *How does it resist the refund/injection?* → deterministic gate (regex), before any
  model call; the injection literally can't reach the model's refusal.
- *How is escalation decided?* → a rule: repeat-failure trigger in the question **and**
  evidence that supports escalation (KB-004/KB-008 or an escalated case).
- *How do you know the answer is grounded?* → verify checks content-word overlap with
  the cited passages and validates every citation exists in the corpus.
- *What stops an infinite loop?* → `MAX_REVISIONS=1`; after one failed retry it
  returns `safe_failure`.
- *Why not a vector database?* → 53 passages; exact search is faster to build, more
  accurate, and the brief says a managed vector DB isn't required.
- *What's `confidence`?* → a heuristic by route (answerable scales with retrieval
  score), not a calibrated probability — a trust signal for a human, honestly rough.
- *Where's the model vs. code line?* → embeddings + answerable/clarify + generation are
  the model; gate, score-floor, escalation, verification, routing, refusal are code.
- *Biggest limitation?* → the regex gate and the term-overlap grounding heuristic;
  I'd harden the gate and move grounding to a local NLI model.
