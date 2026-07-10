# Design decisions

Durable record of architecture decisions: what was decided, the alternatives considered, why,
the risks accepted, and the experiment that validates (or falsifies) each decision. Newest first.
Deeper design context lives in [ARCHITECTURE.md](ARCHITECTURE.md); training mechanics in
[TRAINING.md](TRAINING.md).

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
