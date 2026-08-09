# Graph Walkthrough — the detailed narration

This is the script for the "explain the graph" part of the video. It follows a
query through every node and names **the exact file and function** each node
calls, what it does in plain language, the **actual logic**, and the **trace
line** you'll see in the terminal. Use the diagram
([`graph_diagram.png`](graph_diagram.png)) on screen and this as your voice-over.

Everything lives under `src/orbitdesk/`. The graph itself is wired in
[`graph.py`](../src/orbitdesk/graph.py) inside `build_app()`; each node is a small
function there that calls into one of the specialist modules.

---

## 0. The 30-second orientation (say this first)

> "The whole system is one agent, built as a graph in `graph.py`. A question
> enters and flows through seven nodes. Each node is a tiny function in
> `graph.py` that delegates to one specialist file — retrieval, triage,
> generation, or verification. The shared state that every node reads and writes
> is a typed dictionary called `AgentState` in `state.py`, and every node appends
> one line to a `trace` list, which is the log you'll see."

---

## 1. The journey of a query, node by node

### `retrieve` — find the evidence
- **Where:** `graph.py → retrieve_node()`, which calls
  `retrieval.py → Retriever.search()`.
- **Plain words:** turn the question into numbers, compare it against all 53
  document chunks, hand back the 5 closest ones.
- **Actual logic:** `search()` embeds the question with the local `bge-small`
  model (prefixed with the bge "query" instruction), then does
  `scores = embeddings @ query_vector`. Because every vector was L2-normalized at
  build time, that single dot product **is** cosine similarity. `np.argsort`
  takes the top 5. The 53 passage vectors were built once by `Retriever.build()`
  and cached to `index/embeddings.npy`. No vector database — the corpus is tiny.
- **Trace line:** `retrieve: top=CASE-1041 score=0.89 (n=5)`

### `triage` — pick exactly one of four routes
- **Where:** `graph.py → triage_node()`, which calls `triage.py → triage()`.
- **Plain words:** decide what *kind* of request this is. Three of the four
  answers are decided by plain code; only one asks the model.
- **Actual logic — a 4-rung waterfall, first match wins** (all in `triage.py`):
  1. `safety_gate(question)` — runs regex lists (`INJECTION_PATTERNS`,
     `SECRET_PATTERNS`, `FORBIDDEN_PATTERNS`) over the raw question. A hit (e.g.
     the word `refund`, or "ignore the documentation") returns **out_of_scope**,
     `decided_by="gate"`. No model.
  2. **score floor** — `if hits[0].score < RETRIEVAL_SCORE_FLOOR (0.30)` →
     **out_of_scope**, `decided_by="score_floor"`. Means nothing in the corpus is
     relevant.
  3. `escalation_rule(question, hits)` — fires when a repeat-failure phrase is in
     the **question** (`_ESCALATION_REPEAT` + `_FAILURE_TERM`, e.g. "two ...
     render_failed") **and** the **evidence** supports it (a KB-004/KB-008 passage
     or an already-escalated case). Returns **requires_escalation**,
     `decided_by="rule"`.
  4. `classify_answerable_or_clarify(question, hits, llm)` — only now do we call
     Gemma. It gets the question + the retrieved passages + a strict prompt, and
     must answer `ANSWERABLE` or `NEEDS_CLARIFICATION`. Code parses the label with
     a regex; unparseable defaults to clarification. `decided_by="llm"`.
- **Trace line:** `triage: answerable (by llm)` or `... (by gate/score_floor/rule)`

### the branch after triage
- **Where:** `graph.py → route_after_triage()`.
- **Actual logic:** one line — `return "refuse" if classification ==
  "out_of_scope" else "generate"`. This is the diagram's split: out-of-scope goes
  left to `refuse`; the other three go right to `generate`.

### `refuse` — the deterministic refusal (out-of-scope only)
- **Where:** `graph.py → refuse_node()`, which calls
  `generate.py → safe_refusal()`.
- **Plain words:** for out-of-scope, return a **fixed, pre-written** refusal that
  cites KB-010. **The model is never called** — so a prompt injection can't argue
  its way past it.
- **Actual logic:** `safe_refusal()` returns a constant `Generation` object with
  the refusal text and `sources = [KB-010]`.
- **Trace line:** `refuse: deterministic safe refusal`

### `generate` — write the answer from evidence (the other three routes)
- **Where:** `graph.py → generate_node()`, which calls
  `generate.py → generate_response()`.
- **Plain words:** Gemma writes the response using **only** the retrieved
  passages, and it must cite them.
- **Actual logic:** `generate_response()` dispatches by route to
  `generate_answer` / `generate_escalation` / `generate_clarification`, which all
  call `_generate_with_llm()`. That function: (a) drops superseded passages with
  `_grounding_hits()`; (b) builds a prompt that wraps the question and passages in
  `<user_question>` / `<evidence>` tags, with the rules in the **system** role —
  this is the prompt-isolation defense; (c) calls `llm.generate()`; (d)
  `_split_answer_and_sources()` separates the answer from the `SOURCES:` line and
  **drops any citation the model invented** that wasn't in the evidence;
  `_clean_answer()` strips stray tags. Then `graph.py` calls
  `verify.py → build_response()` to assemble the schema object (adds `confidence`,
  `requires_human`, etc.).
- **Trace line:** `generate: answerable answer_len=268 sources=3`

### `verify` — the deterministic gate every response passes through
- **Where:** `graph.py → verify_node()`, which calls `verify.py → verify()`.
- **Plain words:** before anything ships, check it's trustworthy. If not, we'll
  retry once or fail safely.
- **Actual logic — five checks in `verify()`:**
  - `schema_valid` — validate against `output_schema.json` with `jsonschema`.
  - `has_citations` — an answerable/escalation answer with no sources is rejected.
  - `citations_valid` — every cited `source_id` must exist in the real corpus
    (`build_corpus()` ids).
  - `grounded` — `grounding_score()` measures how many of the answer's content
    words appear in the cited evidence; must be ≥ `GROUNDING_THRESHOLD (0.45)`.
    (Stop-words like "the/is/and" are stripped first so they don't inflate it.)
  - `no_forbidden` — regex block on superseded guidance ("Personal token"),
    secret solicitation, or leaked tags.
  It returns `VerificationResult(passed, checks, failures)`. It's **route-aware** —
  for a refusal it skips the grounding/citation checks and just confirms the
  shape.
- **Trace line:** `verify: PASS` or `verify: FAIL -> ['low grounding (0.00 < 0.45)']`

### the branch after verify
- **Where:** `graph.py → route_after_verify()`.
- **Actual logic:** `if passed → "accept"` (go to END); else if
  `revision_count < MAX_REVISIONS (1)` and the route is revisable →
  `"revise"`; else `"safe_fail"`. This is the diagram's three-way split at the
  bottom of the hub.

### `revise` — retry once with the reasons
- **Where:** `graph.py → revise_node()`.
- **Actual logic:** increments `revision_count` and stores the failure text in
  `revision_feedback`, then the edge loops back to `generate`, which feeds those
  reasons into a fresh attempt. The counter is the **infinite-loop guard**.
- **Trace line:** `revise: attempt 1 (feedback: ...)`

### `safe_fail` — give up safely
- **Where:** `graph.py → safe_fail_node()`, which calls
  `verify.py → safe_failure_response()`.
- **Plain words:** if it still fails after one retry, return a safe
  "I couldn't verify an answer, a human should look" response instead of shipping
  something unverified — or looping forever.
- **Trace line:** `safe_fail: returned safe_failure response`

---

## 2. Fully-narrated runs (open the code + terminal + diagram)

For the first couple of queries, do this: run the CLI with `--trace`, then open
the named file to show the logic, and point at the route on the diagram.

### Run A — answerable, two documents  ·  Q-001
Command: `python -m src.orbitdesk.cli --trace "<Q-001 timezone question>"`
Narrate the trace top to bottom:
1. `retrieve` → *open `retrieval.py`, point at `search()`* — "it embeds the
   question and dot-products against 53 vectors; the top hits are CASE-1041,
   KB-004, KB-003 — notice it pulled **two** knowledge-base docs."
2. `triage: answerable (by llm)` → *open `triage.py`* — "gate didn't fire, score
   is high, no escalation trigger, so rung 4 asked Gemma, which said ANSWERABLE."
3. `generate` → *open `generate.py`, point at `_generate_with_llm()`* — "Gemma
   wrote the answer from the evidence and named its sources; our code validated
   them." Point at the answer citing KB-003 + KB-004.
4. `verify: PASS` → *open `verify.py`* — "schema ok, citations exist and are real,
   grounding above 0.45, no forbidden content."
5. On the diagram: `triage —answerable→ generate → verify —pass→ return response`.
Optionally open `knowledge_base/04_scheduled_exports.md` and show the answer's
steps come straight from it — proof it's grounded, not invented.

### Run B — out of scope / prompt injection  ·  Q-005
Command: `python -m src.orbitdesk.cli --trace "<Q-005 refund + ignore the docs>"`
1. `triage: out_of_scope (by gate)` → *open `triage.py`, point at `safety_gate()`
   and the `FORBIDDEN_PATTERNS` / `INJECTION_PATTERNS`* — "the words `refund` and
   `ignore the documentation` match here, so it's out-of-scope **before any model
   call**."
2. `refuse: deterministic safe refusal` → *open `generate.py`, point at
   `safe_refusal()`* — "a fixed string, no Gemma. The injection literally can't
   reach the model."
3. `verify: PASS` then `return response`.
4. On the diagram: `triage —out of scope→ refuse → verify → return response`.
5. Note the latency: near-instant, because no model ran.

### Run C — escalation  ·  Q-004 (narrate, a bit quicker)
1. `triage: requires_escalation (by rule)` → *open `triage.py → escalation_rule()`*
   — "the question says 'two ... render_failed' and the evidence includes
   KB-004/KB-008, so the rule fires — deterministically, not via the model."
2. `generate` writes the escalation summary + safe diagnostics; `requires_human`
   is true. `verify: PASS`.

---

## 3. The remaining runs (run, narrate yourself)

- **Q-002 (answerable, roles):** gate/floor/rule don't fire → Gemma says
  ANSWERABLE → cites KB-002/KB-005. Mention the superseded CASE-0914 was filtered
  out before Gemma saw it.
- **Q-003 (clarification):** Gemma says NEEDS_CLARIFICATION because KB-006 shows
  the request is too vague → asks for the exact fields.
- **The retry path:** `python -m pytest tests/test_graph_revision.py -v -s` —
  show `verify: FAIL → revise → verify: PASS`, and the safe-failure variant.

---

## 4. Then a brief file tour (use the cheatsheet)

After the runs, open each file for ~10–15 seconds and give its one-liner from
[`cheatsheet.md`](cheatsheet.md): `config.py` (all knobs), `data.py` (files → 53
tagged passages), `retrieval.py` (embed + search), `llm.py` (Ollama wrapper),
`triage.py` (the waterfall), `generate.py` (evidence-only + prompt isolation),
`verify.py` (the deterministic checks), `state.py` (the shared typed state),
`graph.py` (the wiring), `cli.py` (the interface).

Close on the thesis: *"the model only embeds and writes; every route and every
check is deterministic code — that's the design."*
