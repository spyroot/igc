# Critical sections & performance

> **⚠️ STATUS (2026-07-13, code audit).** The "RL training path" hot-path claims below (pointer
> forward, `score_candidates` cache 51×, `resource_graph.neighbors` O(1)) are **offline-only** — the
> live RL trainer builds `Igc_QNetwork` (the legacy DQN) and never touches the pointer or resource
> graph, so those optimizations are on the *data-gen/benchmark* path, **not** the running RL loop.
> Only the DQN/HER/TD/replay hot-path items are on the live path. Verify with
> `scripts/code_reality_check.py`.

Human-readable map of every performance-critical code path in igc: **where it is, what it
costs, what we optimized, and how it is guarded** so a slow path can never silently make
training take days. Every number here is reproducible with one command
([HOW_TO_PROFILE.md](HOW_TO_PROFILE.md)); every budget is a test
(`tests/perf/`, run with `pytest -m perf`).

The rule that produced this doc is binding: **hot-path code ships with numbers** — a PR that
touches a hot path without a benchmark table, a budget tripwire, and (for optimizations) a
before/after plus an output-equivalence check is rejected.

## Why this matters

The RL agent trains for many thousands of steps. A path that runs once per step at O(N²) instead
of O(N) does not look broken in a unit test — it just turns a 20-minute run into a 20-hour one.
So the hot paths are **measured**, not eyeballed, and the measurements live here and in CI.

## What is CPU-offline (and why it matters)

Every hot path below except the two model forwards runs on **CPU with no GPU, no model download,
and no network** — pure tensor/graph work on synthetic or fixture data. That means the whole
profiling suite can run **in parallel while the GPU is busy** with a training job, and in CI on a
plain runner. The model forwards (pointer / Q-network) are compute-bound matmuls that belong on
the GPU; they are benchmarked for visibility but are not CPU-offline and carry no CPU budget.

## The map

Measured on a laptop CPU (single-thread, `OMP_NUM_THREADS=1`); absolute times are machine-relative
— the **budgets and ratios** are what CI enforces. Reproduce with
`python scripts/bench_hot_paths.py --profile`.

### 1. Data-generation path — `igc/ds/sources/`

| Section | Where | Cost (real corpus) | Notes |
|---|---|---|---|
| Fixture load | `redfish_fixture_source.py` | 0.08 s / 1,499 records | JSON parse; linear |
| Graph build | `resource_graph.py::from_records` | 0.047 s / 1,499 nodes | one-pass; harvests `@odata.id` refs whole-body |
| **Neighbors, all nodes** | `resource_graph.py::neighbors` | **0.001 s** (was 0.710 s) | see optimization O-1 |
| Candidate cache | `candidate_features.py::build_candidate_cache` | 0.004 s | static per host (D-002) |
| Candidate embedding | `zero_shot_ranking.py::embed_candidates` | 0.12 s / 1,499 | trigram hash; once per host |
| Zero-shot rank+score | `zero_shot_ranking.py::top_k_hit_rate` | 4.0 s / 1,499 states | **known offline-eval cost** (O-3) |

### 2. RL training path — `igc/modules/rl/`, `igc/modules/policy/`, `igc/modules/igc_experience_buffer.py`

| Section | Where | Cost (B=256, N=300, H=768) | CPU-offline |
|---|---|---|---|
| DQN target | `rl/q_targets.py::q_learning_target` | 0.0003 s | ✅ |
| HER relabel (full episode) | `rl/q_targets.py::relabel_future` | 0.004 s (T=50 × k=8) | ✅ |
| Replay data feed | `igc_experience_buffer.py::sample_batch` | 0.0014 s | ✅ |
| `_stack_done` per-item loop | `igc_experience_buffer.py::_stack_done` | 0.0003 s | ✅ |
| Fixed Q-network forward | `igc_q_network.py` | 0.0005 s | ✅ (small) |
| **Pointer forward, naive** | `policy/pointer_policy.py::forward` | **0.193 s** | GPU-bound |
| **Pointer forward, cached** | project unique once + `score_candidates` | **0.0038 s** | see optimization O-2 |
| Candidate scoring einsum | `policy/pointer_policy.py::score_candidates` | 0.0026 s | ✅ |

**Read this table as:** everything the CPU can do in parallel with the GPU is sub-6 ms. The only
heavy step is the pointer's candidate projection, and it has a 51× fix (O-2).

## Optimizations we made

### O-1 — Resource-graph indexing (`resource_graph.py`, PR #25)

**Problem.** `children()` scanned every node per call, so building neighbors for all nodes was
O(V²); `child_relation` resolution re-walked each parent body once per child (a collection body
references every member). On the 2,024-node GB300 walk this was ~1.3 s per full pass — paid every
HER relabel and decision step.

**Fix.** Build one-pass `parent`/`children` indexes in `from_records`; share a single
`{referenced_url: key}` map per parent body across its children. `parent`/`children`/`neighbors`
are now O(1) dict lookups.

**Result.** All-node neighbors **0.710 s → 0.001 s (710×)**, build time unchanged, output verified
**bit-identical** to the previous implementation across every node (relations, neighbors, parents).

**Guard.** `tests/perf/test_hot_path_budgets.py::test_neighbors_all_nodes_budget` — 0.5 s per
1,000 nodes; the old O(V²) would fail it.

### O-2 — Candidate key cache (D-002 static-per-host, PR #37)

**Problem.** The pointer's only expensive step is the `ActionProjector` MLP (GELU + two Linears)
projecting `[B, N, H]` candidate embeddings — 76.8 M elements at B=256, N=300, H=768. cProfile
pinned it to `pointer_policy.py::ActionProjector.forward`. Calling the full
`Igc_PointerQNetwork.forward` per state re-projects the **same** candidates B times, because a
host's candidates are static and the projector weights are fixed within an optimizer step.

**Fix (design decision [D-002](DECISIONS.md)).** Project the host's **unique** candidate set once
per optimizer step, cache the keys, and score with `score_candidates` (an einsum over cached
keys) for every state in the batch. The primitive already exists — `score_candidates` takes
pre-projected keys — so the RLPolicy training loop must use it rather than the full per-state
forward.

**Result.** Pointer forward **0.193 s → 0.0038 s (~51×)** on CPU; on a 100k-step run this is the
difference between hours and minutes of pure projection.

**Guard.** `tests/perf/test_rl_hot_path_budgets.py::test_d002_key_cache_beats_reprojection` — a
**machine-independent ratio** tripwire: project-once must stay ≥ 10× cheaper than re-project-B×N
(measured 51×), so a training loop that reverts to the full forward trips CI regardless of runner
speed.

## Known costs we accepted (and why)

### O-3 — Zero-shot rank+score is 4.0 s / 1,499 states

The go/no-go harness embeds each state's full JSON body via a Python character-trigram loop.
It is **offline-eval only** (not on the RL decision path or the training loop), run occasionally
to score a representation, so it is deferred rather than optimized. If it ever moves onto a hot
path, the fix is vectorized/hashed embedding — recorded here so the decision is visible, not
forgotten.

## How the guards work

- `tests/perf/` holds every budget. They carry `@pytest.mark.perf` and are **excluded from the
  default gate** (`pytest.ini`) so machine speed never flakes a normal run; they run explicitly
  with `pytest -m perf` and in the CI `perf` job.
- Absolute budgets are set **50–100× looser than measured** — only an algorithmic regression
  (accidental O(V²), a per-item body walk, re-projecting duplicates) trips them.
- Where an absolute time would be too machine-dependent, the guard is a **ratio** between two
  implementations (O-2), which is invariant to runner speed.

## Adding a new hot path

1. Add a stage to `scripts/bench_hot_paths.py` (real corpus or realistic synthetic tensors).
2. Run it, paste the table into your PR; for an optimization, include before/after **and** an
   output-equivalence check against the old code.
3. Add a `tests/perf/` budget (absolute, 50–100× loose) or a ratio tripwire.
4. Name the dominant function from `--profile`. If you cannot, the work is not done.
5. Update this table.

See [HOW_TO_PROFILE.md](HOW_TO_PROFILE.md) for the exact commands and the CI job.
