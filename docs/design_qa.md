# Design Q&A — deterministic routes, robustness, and limitations

Honest answers for the "limitations / what I'd improve" part of the video. The
recurring theme to say out loud: **the architecture is production-shaped; the
individual components are PoC-grade and I can name exactly why.**

---

## 1. The three deterministic routes, their regex, and how robust they are

All in `triage.py`. Three of the four routes never touch the model.

### (a) out_of_scope via `safety_gate()` — injection / forbidden / secret
- **Injection:** `\bignore\b.{0,25}(documentation|instructions|docs|rules|above|prior)`
  (plus `disregard`, `forget`, `override`). Catches "ignore the supplied documentation".
- **Forbidden:** `\brefund\b`, `cancel .* subscription`, `legal advice|write .* legal|sue`,
  `(medical|financial) advice`, `reveal .* (secret|token|password)`.
- **Secret:** `(give|send|show|tell|paste) .* (secret|token|password|key)`, `what's my/the ... secret`.

### (b) out_of_scope via the score floor
`if hits[0].score < 0.30` → nothing relevant retrieved → out_of_scope.

### (c) requires_escalation via `escalation_rule()`
Fires when a **repeat word** (`two|second|twice|consecutive|in a row|repeated|again`)
**and** a **failure word** (`render_failed|connector_internal_error|failed|failure|error`)
are in the question **and** the evidence includes KB-004/KB-008 or an escalated case;
or a **credential-exposure** phrase (`credential|secret|token .* expos|leak|compromis|stolen|breach`).

### Are they robust / production-grade? **No — and here's exactly why.**
- **Brittle to paraphrase (false negatives).** `\brefund\b` misses "process my
  reimbursement", "give me my money back", "waive the charge". Injection misses
  "pretend the rules don't apply". Misspellings ("refnd"), other languages, or
  obfuscation (`r-e-f-u-n-d`, base64) all slip past.
- **Over-blocking (false positives).** "How do I read the *refund* policy doc?"
  contains `refund` → wrongly refused. "Should I *ignore the instructions* in that
  old email?" trips the injection rule. The escalation `_FAILURE_TERM` includes
  the very common words `error`/`failed`, so "I hit an error *again*" can over-fire
  if evidence happens to include KB-004.
- **Uncalibrated threshold.** The `0.30` score floor is a guess. `bge-small` gives
  unrelated English a moderate cosine — "weather in Mumbai" scored **0.47**, above
  the floor, so it wasn't caught. Raising the floor risks cutting real questions.
- **Corpus-hardcoded.** `_ESCALATION_SOURCES = {KB-004, KB-008}` is specific to
  this KB; a new doc means editing code.

### The production alternative (say this)
- Replace the regex gate with a **dedicated intent / safety classifier** — a
  small fine-tuned classifier or a local guard model (e.g. Llama-Guard-style), and
  an **embedding-based scope check** (compare the request embedding to labelled
  intent centroids) instead of a raw score floor; calibrate any threshold on
  held-out data or use a proper out-of-distribution detector.
- For injection specifically, a **prompt-injection detector model** + input
  normalization.
- Keep the deterministic layer as a **fast first filter**, but never as the only
  line of defence. (It already isn't — generation is evidence-only and
  verification rejects ungrounded output, so a miss is still contained.)

---

## 2. answerable vs. needs_clarification — how, and how robust

`triage.py → classify_answerable_or_clarify()` sends the question + retrieved
passages + a strict prompt (with two few-shot examples) to Gemma, which must reply
`ANSWERABLE` or `NEEDS_CLARIFICATION`; code parses the label and **defaults to
clarification** if unparseable.

**Robust? This is the least robust part** — it's the one genuinely model-dependent
decision. temperature 0 makes it stable per prompt, but a 4B model is
phrasing-sensitive (it misjudged Q-002 until I tightened the prompt). Importantly,
**the route itself is not verified** — verification checks the *answer's*
grounding, not whether "answerable vs clarify" was the right call. Mitigations: the
safe default is clarification, and an answerable-but-ungrounded answer is still
caught by verify. The residual risk: a question Gemma wrongly thinks is answerable,
that happens to be grounded in a plausible-but-wrong passage, ships confidently.

### Are KB-008 / KB-004 sufficient to base escalation on, for this corpus?
For **this** corpus, yes — escalation conditions are documented only in **KB-008**
(Escalation Conditions) and **KB-004** (escalate after two `render_failed`), plus
case CASE-1103, so keying on those is appropriate. **But we are missing parts of
KB-008**, and this is a great limitation to name:
- KB-008 also lists **"a reproducible error not in the knowledge base"** as an
  escalation condition — we **cannot** detect "not in the KB" deterministically, so
  we miss it.
- KB-008 lists **"billing / ownership / legal"** as *escalations*. But our gate
  currently classifies refund/legal as **out_of_scope** (refuse, `requires_human =
  false`). Per KB-008 those are arguably **escalations** (`requires_human = true`,
  hand to the billing/legal team). The refusal text does point to "a human team",
  but the *classification* is debatable. **I'd reclassify billing/legal disputes as
  escalation, not refusal.** (Naming this shows you actually read KB-008.)

---

## 3. What breaks at scale — and does this setup hold up?

**What breaks:**
- **Retrieval:** in-memory exact search is O(N)/query and holds all vectors in RAM
  — fine for 53, not for 10k–1M. Needs an ANN vector store (FAISS / pgvector /
  Qdrant) and incremental indexing instead of the full re-embed on any change.
- **Embedder/precision:** `bge-small` + no reranker is fine for 53 well-separated
  chunks; at scale you'd add a **cross-encoder reranker** and a stronger embedder.
- **Generation:** one local `gemma3:4b` instance won't serve concurrent users;
  you'd need batched GPU serving (vLLM) — but the assignment forbids cloud LLMs, so
  local stays the constraint.
- **Triage & thresholds:** the regex + `0.30` floor break on diverse real input
  (Section 1).
- **Grounding:** term-overlap breaks on paraphrase → needs an **NLI model**.
- **Missing production plumbing:** persistence, auth, rate-limiting, caching,
  observability, an eval harness, human-in-the-loop.

**Does a production system use a *similar* setup?** Yes — **the architecture is
genuinely production-shaped**: a graph orchestrating a boxed LLM, deterministic
control flow, `retrieve → triage → generate → verify`, with a bounded retry and a
safe-failure. What changes at scale is the **components, not the skeleton**: vector
DB, reranker, stronger/served model, NLI or LLM-judge grounding, an ML intent
classifier, calibrated thresholds, and observability. So the line is: **"the
skeleton is production-grade; the parts are PoC-grade, deliberately."**

---

## 4. The refuse path — which response, and it does NOT hit safe_fail

For out_of_scope the path is **`triage → refuse → verify → accept → return
response`**. It does **not** go through `safe_fail` (that's a *different* branch of
`verify`, taken only when a response *fails* verification).
- `refuse_node` calls `generate.py → safe_refusal()`, which returns a **hardcoded**
  refusal string citing KB-010 — **no model call**.
- `verify` runs on it but is route-aware: for out_of_scope it **skips** the
  grounding/citation checks and just confirms schema + no-forbidden, so it
  **passes** → `route_after_verify` returns `"accept"` → END.
- So the returned response is exactly the hardcoded refusal. `safe_fail` is only for
  a *failed* verification on a revisable route — a refusal never lands there.

---

## 5. No reranker / no separate classification model — is that OK?

Yes, and don't apologize — it's a scoped choice:
- **Reranker (cross-encoder):** improves top-k precision, but for 53 short,
  well-separated passages the bi-encoder already surfaces the right evidence (we saw
  0.7–0.9 scores on the correct passages). A reranker would add latency + another
  model load for negligible gain **at this scale**. Easy future add for a larger
  corpus.
- **Separate intent classifier:** we deliberately keep the model count minimal —
  deterministic rules for three routes, and the generator model itself for the one
  fuzzy call — so routing stays auditable and testable. A dedicated classifier is
  the production upgrade.
- **What to say:** *"Not needed for a 53-passage PoC — they'd add latency and a
  model load for little benefit here. For a larger corpus I'd add a cross-encoder
  reranker; for robust triage, a dedicated intent classifier."* The brief itself
  says a managed vector DB isn't required and model quality is secondary, so this is
  in-scope engineering judgment, not a gap.

---

## 6. Can an *answerable* query end in safe_fail?

**Yes — by design.** `answerable` is in `_REVISABLE`. If Gemma's answerable answer
fails verification (ungrounded / no citations / forbidden) on both the first attempt
**and** the one revision, `route_after_verify` sends it to `safe_fail`. It doesn't
happen on the five samples (all pass first try), but the path is **intended**: if an
answer can't be grounded even after a targeted retry, safe-failing (with
`requires_human = true`) is correct — far better than shipping a hallucination.

---

## 7. Is one retry enough? Does it break in production?

- **For this system: yes.** Verification failures are rare, and if a targeted
  revision (fed the exact failure reasons) doesn't fix it, repeating the same 4B
  call usually won't either. One retry balances recovery vs. latency vs.
  loop-safety, and it's already a **config knob** (`MAX_REVISIONS`), not a hard law.
- **Production nuance (say this):** more retries are **not** strictly better —
  diminishing returns, more latency/cost, and if the evidence is genuinely
  insufficient, no retry count helps (safe_fail is the right answer). The real
  production upgrade isn't *more* retries but a **smarter retry strategy**: on
  failure, **re-retrieve with an expanded/rewritten query** (not just re-prompt the
  same evidence), possibly escalate the model, and make the cap **configurable per
  route**, tuned from retry-success metrics.
