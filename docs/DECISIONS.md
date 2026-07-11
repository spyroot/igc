# Design decisions

Durable record of architecture decisions: what was decided, the alternatives considered, why,
the risks accepted, and the experiment that validates (or falsifies) each decision. Newest first.
Deeper design context lives in [ARCHITECTURE.md](ARCHITECTURE.md); training mechanics in
[TRAINING.md](TRAINING.md).

---

## D-003 — Backbone migration: retire GPT-2 as the state-encoder default (2026-07-11)

**Status: ACCEPTED** for the Phase 0 decoupling below; the *target default* and the
decoder-vs-encoder choice are the one open sub-decision, tracked in "The open fork".
**Method:** four-perspective design review (architect, adversarial skeptic, ML researcher,
pragmatic operator, run independently with full reasoning, then synthesized), grounded by a
usage census over the code.

### Problem

GPT-2 (2019) is the default `--model_type` backbone: the model that encodes each Redfish response
into the latent the M6 policy scores from. It is dated for this domain — a 1024-token window,
learned *absolute* position embeddings, and a BPE tokenizer that splits JSON braces and URLs into
many subtokens. The loader is *already* backbone-agnostic in its seams
(`AutoModelForCausalLM`/`AutoTokenizer` keyed on `--model_type`, Conv1D-vs-Linear LoRA handling in
`igc/modules/shared/llm_shared.py`, a Qwen fast-tokenizer path in `igc/ds/redfish_dataset.py`), but a
handful of GPT-2 assumptions are still hard-wired into the core encode path and would break — or
silently degrade — any modern backbone.

### Grounding census — where GPT-2 is welded in (grep-verified)

| Assumption (site) | What it does | Why it breaks a modern backbone | How it manifests |
|---|---|---|---|
| `wpe` positional indexing — `igc/modules/encoders/base_encoder.py`, `igc/envs/rest_encoder.py` (comment: "GPT-2 wpe positional table. Subtract 1…") | reaches into GPT-2's learned absolute-position table | RoPE models (Qwen, Llama, ModernBERT) have **no** `wpe` | `AttributeError` the moment a non-GPT-2 model loads |
| Hard-coded `1024` — `redfish_dataset.py` (`max_len` default and `"max_length": 1024`), `shared_arg_parser.py` (default 1024) | caps sequence length at GPT-2's window | modern models allow 8k–128k tokens | a long-context model is throttled to 1024; the chunking workaround stays alive for no reason |
| `GPT2Tokenizer.from_pretrained("gpt2")` — `redfish_dataset.py` (three fallbacks) | forces GPT-2's slow tokenizer | any other model needs its own tokenizer | **silent** quality collapse — the model receives token ids that are not its own (no crash) |
| Decoder-only load — `AutoModelForCausalLM` in `llm_shared.py` | assumes a causal LM | a true encoder (BERT-family) needs `AutoModel` + an MLM objective | blocks the encoder option without a wider change |
| `"gpt-2"` id — `igc/ds/ds_pairs.py` (untracked WIP) | invalid HF repo id (hyphen; the real id is `gpt2`) | — | load/download failure the first time that path runs |

(`igc/rl.py`, also an untracked WIP file, hard-codes `GPT2LMHeadModel`/`GPT2Tokenizer` and belongs in
the same sweep.)

### Verified state on `main` (2026-07-11) — most of Phase 0 was already done

A re-audit against the actual code (not the comments the census above was drawn from) found the
decoupling **largely complete**, and corrects two overstatements in the census:

- **`wpe` — already fixed.** `igc/modules/encoders/backbone_utils.py` (`backbone_module`,
  `max_positions`, `emb_shape`) derives module + shapes from `config`; both encoders
  (`base_encoder.py`, `rest_encoder.py`) use it and handle RoPE models with no positional table. The
  census matched a *comment* that referenced the legacy behavior, not live `.wpe` access.
- **The 1024 window and the tokenizer — already flags.** `igc_main.py` builds the dataset with
  `default_tokenize=--model_type` (→ `AutoTokenizer`) and `max_len=--seq_len`; `_load_tokenizer`
  rebuilds the cache on a backbone switch. So a modern long-context run is already
  `--model_type <model> --seq_len <N> --recreate_dataset` — no refactor required.
- **`AutoModelForCausalLM`** is the intended causal-LM track, already keyed on `--model_type`; not a
  defect.

**What genuinely remained (fixed in the follow-up PR):** the `JSONDataset.load_tokenizer` *classmethod*
hard-coded `GPT2Tokenizer` on the saved-tokenizer reload path (wrong class for a non-GPT-2 saved
tokenizer), one minor live fallback, and stale docstring examples — all moved to `AutoTokenizer`, with
an offline regression test. The remaining GPT-2 literals are the untracked WIP files above and a
standalone `chat_with_gpt2` demo helper. Net: Phase 0 is effectively closed; the open item is the
Phase-1/2 model choice below.

### The open fork (the one decision still to make)

| Track | Move | Cost |
|---|---|---|
| **Decoder-as-encoder (keep the objective)** | swap GPT-2 for a modern *decoder* — e.g. SmolLM2-135M (Apache-2.0, RoPE, 8k context, code-aware tokenizer) | low: same causal-LM training; near drop-in once Phase 0 is done |
| **True encoder** | move to ModernBERT-base (encoder, RoPE, 8192 context) | higher: changes the M1 objective from next-token (causal) to masked-LM |

For *structured, passive* JSON a bidirectional encoder is the better representation, but it is not a
drop-in — it recasts the M1 objective. **Recommendation: take the decoder track for the default
(Phase 1), and evaluate the encoder as the GPU backbone (Phase 2) where the gain justifies the
objective change.** This fork is recorded, not yet closed.

### Decision — phased

- **Phase 0 — decouple, do not dethrone (ACCEPTED; do first).** Remove the five GPT-2-isms above:
  load via `AutoModel` and read `last_hidden_state`; drop the `wpe` access (guard it to
  absolute-position models only); read the length cap from `config.max_position_embeddings`; replace
  the hard-coded `GPT2Tokenizer` fallbacks with `AutoTokenizer`; fix the `"gpt-2"` id. **GPT-2 stays
  the default**, so the offline CPU gate stays green. This is the prerequisite for any swap, and it
  ships with a `scripts/bench_hot_paths.py` number plus a perf-budget update per the hot-path rule
  (the encode path is a hot path).
- **Phase 1 — a modern small default, opt-in first.** Add SmolLM2-135M as a benchmarked option; flip
  the default only once it (a) beats GPT-2 on the zero-shot ranking harness
  (`igc/modules/eval/zero_shot_ranking.py`) and (b) keeps the offline gate download-free and fast.
  Keep `--model_type gpt2` as the pinned, reproducible baseline forever.
- **Phase 2 — the real win.** ModernBERT-base (8192 context) as the GPU fine-tune backbone → removes
  the chunking workaround; optionally distil a large on-cluster teacher into it (feature-level, on
  last hidden states). Accepts the causal→MLM objective change.

### Surfaced disagreement (kept, not averaged)

The adversarial-skeptic perspective argued **not to move the default at all**: GPT-2 is frozen,
permissively licensed, CPU-fast, and already cached; swapping it invalidates the tokenized dataset
cache and the `@odata` special-token setup, and risks the `wpe`/Conv1D alignment. This is reconciled
by phasing — Phase 0 *is* the skeptic's position (decouple, keep GPT-2 as the default); the default
only moves in Phase 1, and only behind a measured gate.

### Risks accepted

- **Cache invalidation.** A new tokenizer changes every token id, so cached tokenized datasets and
  the `@odata` special-token setup must be rebuilt. Keep GPT-2 pinned for regression.
- **Conv temporal view.** Changing sequence length alters the 1D-conv over `last_hidden_state`;
  re-validate the encode path (perf budget + an output-equivalence check on GPT-2 before/after
  decoupling).
- **Offline gate.** The default must stay download-free on CPU; a model that needs a download stays
  opt-in, not the default.
- **Objective change.** The encoder track (Phase 2) is *not* a drop-in — it recasts M1 training.

### Validation / go-no-go

- **Phase 0:** offline gate green with GPT-2 unchanged, and `bench_hot_paths` shows the decoupled
  encode path is output-equivalent to the current one and within budget.
- **Phase 1:** SmolLM2-135M ≥ GPT-2 top-5 on `zero_shot_ranking.py`, with the gate still fast and
  download-free, before the default flips.
- **Phase 2:** ModernBERT clears the D-001/D-002 held-out-vendor bar with the chunking workaround
  removed.

---

## D-002 — Action-candidate representation: text + graph features, v1 scoped (2026-07-11)

**Status: ACCEPTED** (v1 scope; extends D-001).
**Method:** owner proposal, refined through the same multi-perspective design review as D-001,
grounded by a feature-derivability census over the real capture corpora.

### Proposal under review

Represent each (endpoint, method) candidate for the D-001 pointer as both **text** and **graph
context** derived from the walked Redfish resource tree (nodes = resources, edges = containment /
links / action targets), rather than URL text alone. The original field list also included a
current-resource state summary and goal-relevance features.

### Grounding census (what the captured corpora actually support)

Measured over the real Supermicro (1,499 resources) and HPE iLO (167) fixture corpora:

| Feature source | Availability | Consequence |
|---|---|---|
| `@odata.type` (resource type + schema version) | 92–100% | `resource_type` is derivable nearly everywhere |
| Parent by URL-prefix containment | 90–98% | containment edges + `child_relation_name` nearly free |
| Explicit `Links` sections | **10–14%** | link edges are SPARSE — the graph builder must harvest `@odata.id` references from anywhere in the body, not just `Links` |
| `Actions` with a `target` | 5–17% | `has_action_target` is sparse and therefore discriminative |
| `Oem` sections / OEM-namespaced types | 16–26% | OEM markers exist but are vendor-specific |

### Decision — v1 candidate schema

```
candidate_v1 = {
  endpoint_path_tokens,   # URL path segments, ids normalized
  http_method,
  resource_type,          # from @odata.type (standard schema; namespace stays inside the string)
  child_relation_name,    # the link/containment name that makes this endpoint reachable
  has_action_target,      # boolean: body carries Actions[*].target
}
candidate_emb = concat(f_text(path, method), f_feats(type, relation, action_flag))  # NOT additive
score = state_latent^T · W · candidate_emb                                          # bilinear, v1
```

- **Fusion is concatenation, not addition.** Zero is not a neutral element in an additive
  scheme; under a partial crawl, missing graph features as zero vectors would systematically
  shift candidate embeddings. Concatenation with zero-padding lets the scorer learn to ignore
  missing dimensions. Gated fusion is deferred (added parameters that can overfit to the
  training vendors' crawl-completeness patterns).
- **Scorer is a bilinear dot product for v1**, not `MLP([s, c, s*c])`: smoother Q-surface
  (less overestimation with noisy TD targets), and one matrix multiply scores the whole
  candidate set — which matters because HER relabeling re-scores every candidate for each
  relabeled goal. Upgrade path if v1 underperforms: 2-layer MLP over `[s, c]` (the MLP can
  learn interactions; the explicit `s*c` term is not needed).

### Cut from the proposal (with reasons)

- **`current_resource_state_summary` — removed.** Duplicates what the state latent already
  encodes, and it is the one field that would make candidate embeddings state-dependent —
  breaking per-host precompute and creating staleness inconsistency under HER relabeling.
- **`goal_relevance_features` — removed.** A leakage vector: any precomputed relevance
  heuristic short-circuits the DQN's need to learn goal-action interaction, must be recomputed
  per relabeled goal, and collapses on unseen vendors/goal phrasings. Goal information reaches
  the score only through the goal-conditioned state latent.
- **`schema_or_oem_namespace` as a separate field — removed.** The namespace already lives
  inside `resource_type`; surfacing it separately invites the model to key on vendor tokens,
  directly hurting held-out-vendor transfer. Revisit later at most as a binary `is_oem` flag.
- **`depth_in_tree` — dropped** (duplicates path length; low information density).
  **`parent_resource_type`, `allowed_methods` — deferred to v2** (partial-crawl sensitive /
  largely redundant with the candidate's own method + `has_action_target`); add only if the
  v1 go/no-go fails, as controlled ablations.
- **Learned graph-neighborhood embeddings (GNN over the local tree) — deferred to v2.** The
  census shows explicit link coverage is too sparse for reliable neighborhoods on a partial
  crawl; v1 uses only the cheap, near-universal structural fields above.

### The payoff: caching

With the dynamic fields removed, **every candidate embedding is fully static per host** —
computed once from the walked tree, cached, and only *filtered* by the per-state legal catalog.
HER relabeling then re-scores cached embeddings against the new goal-conditioned state latent
(one matmul) instead of re-encoding candidates. This resolves D-001 binding requirement #2 by
construction.

**Measured throughput consequence (2026-07-11, `scripts/bench_hot_paths.py --section rl`).** The
pointer forward's ONLY expensive step is the candidate projection: projecting `[B, N, H]`
embeddings through the `ActionProjector` MLP (GELU + two Linears over 76.8M elements at B=256,
N=300, H=768) is 0.193s/step on CPU, while every other RL critical section — DQN target, HER
relabel loop, replay data feed, done-stacking, the scoring einsum — is under 6ms. Because the
projector weights are fixed within an optimizer step and a host's candidates are static, the
correct pattern is to **project the host's UNIQUE candidate set once per step and score with
`score_candidates` (einsum over cached keys)**, NOT call the full `Igc_PointerQNetwork.forward`
per state (which re-projects duplicated candidates). Measured: **0.193s → 0.0038s, ~51x**. The
key cache is not just a HER-relabel convenience — it is the per-step throughput lever for M6
training, guarded by a machine-independent ratio tripwire in `tests/perf/`.

### Risks accepted

- Text + shallow structural features may under-discriminate sibling endpoints that differ only
  deep in their bodies (no neighborhood embedding in v1) — measured by the same zero-shot
  ranking go/no-go as D-001 (≥ 80% top-5 on a held-out vendor); v2 features are the planned
  response, not a redesign.
- ID normalization in `endpoint_path_tokens` (collection members like `/Systems/1` vs
  `/Systems/Node0`) must not erase member identity where the goal targets a specific member —
  keep the raw id as a trailing token rather than deleting it.

### Experiment result (2026-07-11) — NO-GO for the untrained encoder; learned projection required

The go/no-go was run with the weakest instantiation first: a completely frozen character-trigram
encoder and NO learned projection (pure cosine between state body text and candidate text), over
the full walked trees, ground truth = each state's true graph neighbors
(`igc/modules/eval/zero_shot_ranking.py` + `igc/ds/sources/resource_graph.py`):

| Corpus | k=1 | k=5 | Bar (≥0.80 top-5) |
|---|---|---|---|
| Supermicro, in-domain, 1,499 nodes | 0.185 | **0.293** | NO-GO |
| HPE iLO, held-out vendor, 167 nodes | 0.323 | **0.754** | NO-GO (near) |

Interpretation: ~30× over the random baseline, but far under the bar on the large host — with
hundreds of near-identical sensor leaves, global text similarity fills the top-5 with lookalike
*siblings* rather than true transitions. Host size dominates difficulty (167-node HPE nearly
passes; 1,499-node Supermicro fails hard). **Consequence: the learned bilinear projection
`s^T W c` is load-bearing, not optional — representation similarity alone cannot rank
transitions.** Next step (before any M6 training spend): behavioral-cloning-train `W` on
in-domain graph transitions (Supermicro), then re-run this same harness zero-shot on the
held-out vendor (HPE) for the real go/no-go.

---

## D-001 — M6 action-selection objective: hybrid pointer + argument decoder (2026-07-11)

**Status: ACCEPTED** (pending the de-risk experiment below).
**Method:** four-perspective design review (architect, adversarial skeptic, ML researcher,
pragmatic operator viewpoints run independently with full reasoning, then synthesized).

### Problem

At each state the environment exposes a *dynamic* catalog of legal actions — an endpoint URL
from the walked Redfish resource tree, an HTTP method from that endpoint's `allowed_methods`
(both produced by the discovery crawl described in TRAINING.md §2), and optional argument
slots. Catalog size varies per state (tens to hundreds); the endpoint vocabulary is open —
new hosts and vendors introduce URLs never seen in training. The RL stack already has a replay
buffer with HER relabeling, DQN-style targets with terminal masking, and a per-slot enum
argument decoder. The M1 state encoder produces a latent per observation.

### Options considered

| | Option | Verdict |
|---|---|---|
| A | Pointer network scoring every legal action candidate | Good, but scores full argument combinations — candidate set balloons |
| B | Fixed-width Q-network, padded + `-inf`-masked (legacy) | **Dead end**: open endpoint vocabulary cannot be padded; a new vendor's URLs have no output head; capacity wasted on padding |
| C | LLM-native action decoding (backbone generates the action, constrained to the catalog) | **Breaks TD/HER**: the target max requires scoring every legal candidate — hundreds of LLM forward passes per target; forces a switch to policy-gradient, discarding the offline TD investment |
| D | **Pointer for (endpoint, method) + existing per-slot argument decoder** | **Adopted** — see below |

### Decision

Commit to **D**: a pointer network attends from the M1 state latent over *encoded
(endpoint, method) candidates* and is trained with DQN/TD targets; the existing argument
decoder fills slots. All four review perspectives independently converged on D.

Why it wins on the axes that matter:

- **Open vocabulary / unseen vendors** — candidates are scored by a *text* encoding of the
  endpoint + method, so structurally similar unseen URLs land near seen ones; nothing is keyed
  by a fixed action id.
- **Offline sample efficiency + HER/TD compatibility** — Q(s, a) over legal candidates keeps
  the standard target max (over the *next state's* legal set); HER relabeling works unchanged.
- **Catalog-size variance** — attention over a variable candidate set; no padding, no
  truncation, no dead output heads.
- **Inference cost** — the state is encoded once; candidates use a lightweight action encoder
  (not the full backbone); attention over ~200 candidates is sub-millisecond on the training
  GPUs.
- **Migration** — only the endpoint head of the legacy Q-network is replaced; the argument
  decoder (already tested) is untouched, allowing A/B against legacy during rollout.

### Binding implementation requirements (from the adversarial review)

1. **Compositional text action-encoder.** Encode URL path segments + method as text
   (seeded from the M1 backbone's token embeddings), never an id lookup — out-of-distribution
   vendor URL patterns are the #1 failure mode.
2. **HER relabeling changes the next state's legal catalog.** Action encodings for relabeled
   transitions must be recomputed or cached explicitly, or replay sampling cost silently
   explodes.
3. **Credit assignment through the argument decoder.** The pointer's Q reflects the return
   *after* the (initially frozen) argument decoder acts — train with a stop-gradient on the
   decoder first; joint training only once both are stable, else co-adaptation collapses when
   either is updated.
4. **Overestimation control.** With 100+ candidates the target max amplifies overestimation:
   use Double DQN and pretrain the pointer with behavioral cloning on the offline corpus
   before TD updates.
5. **URL granularity.** Endpoints differing only in query parameters must be split into base
   path + parameters in the encoding, or the pointer cannot generalize across them.

### Risks accepted

- **Embedding collapse on radically novel URL schemes** (an alien vendor hierarchy maps to an
  undifferentiated region of embedding space → near-random ranking). Mitigated by requirement 1
  and measured by the experiment below.
- **Decomposition bias** (independently maximizing endpoint then arguments can miss a jointly
  optimal action in the rare case where argument values determine which endpoint is best).
  Accepted as rare for Redfish semantics; monitored by comparing hybrid argmax against the
  legacy joint Q-network on logged data.

### De-risk experiment (go/no-go, inference-only)

Zero-shot endpoint ranking on a **held-out vendor**: run the prototype pointer with a frozen
text action-encoder over states from a vendor corpus excluded from training (the repo's
multi-vendor fixture corpora make this possible offline), rank each state's legal endpoints,
and score against the walked tree's true transitions.
**Go**: correct endpoint in top-5 for ≥ 80% of states. **No-go**: revisit the action encoder
(pretrained subword encoder) before any M6 training spend.

### Notes

One reviewer perspective, if unconstrained by the existing stack, would have preferred a
goal-conditioned planner over a learned world model instead of model-free RL for a
deterministic, schema-driven API — recorded here as a future alternative should model-free
training underperform. The adversarial review's final risk list was truncated by an output
limit in the raw transcript; the surviving content is reflected in the requirements above.
