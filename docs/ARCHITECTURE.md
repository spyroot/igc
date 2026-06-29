# IGC architecture & generalization plan

This document describes the target architecture for turning `igc` from a Redfish-specific
goal-conditioned RL project into a **generic, pluggable goal-conditioned tool-use agent
framework**, and the phased plan + training/MLOps roadmap to get there. Redfish becomes one
environment adapter among several (filesystem, SQL, GitHub), and the LLM backbone moves from GPT-2
to a config-driven large decoder model fine-tuned on the GB300 NVL72 cluster.

> Status: working design. Diagrams are theme-aware SVGs under [`docs/diagrams/`](diagrams/).

---

## 1. Where igc is today, and the five target layers

`igc` already has a working MDP shell (gym env + mock REST server + GPT-2 state encoder +
goal-conditioned DQN with HER), but every layer is welded to Redfish and GPT-2. The target is five
clean layers behind one interface. The map below marks each layer **refactor existing** (reuse what
is there) vs **build new** (greenfield), and calls out the keystone change.

![Five-layer gap map](diagrams/01-five-layer-gap-map.svg)

- **Goal** — `Goal(instruction, success_criteria, constraints, plan)`. Today only a `GoalTypeState`
  enum + a loose `self.goals` dict. *Build new.*
- **Environment** — a `GoalEnvironment` interface (`reset / available_actions / step / verify`) with
  Redfish as one adapter. The `MockServer` simulator + gym shell are reusable. *Refactor.*
- **Trajectory** — typed `Observation / ToolAction / Transition` + a recorder. Only `MockResponse`
  exists; transitions are currently discarded. *Build new.*
- **Learning & eval** — evaluators, preference data, reward model. SFT/RL trainers exist; evaluators
  and preference data are greenfield. *Build new.*
- **Runtime guardrail** — dry-run → approval → executor. Today a single `_is_live` boolean. *Refactor.*

**Keystone:** replace the one-hot `Box(num_urls + 6)` action with
`ToolAction(tool_name, op, arguments, target, risk_level)`. The env action space, the `JSONDataset`
adler32 codec, the Q-network head, and the goal extractor all depend on it — so it is migrated
**coexistence-then-cutover** behind an `action_repr ∈ {onehot, tool}` flag and a golden parity test
that pins slice order before any default flips.

## 2. How a new simulator plugs into the agent

The framework uses **typed Protocols + a decorator registry + per-env plugin packages**. The core
never changes when you add an environment; the existing `MockServer.request → MockResponse` seam is
kept intact behind a `RedfishSimulator` adapter.

- Core contracts in `igc/core/`: `ToolAction`, `Observation`, `Goal`, `Transition`, `SimResult`,
  `ToolSpec`, `RiskLevel`, plus `Simulator` and `GoalEnvironment` as `@runtime_checkable` Protocols.
- One registry `igc/envs/registry.py`: `@register("name")` + `make(name, cfg, model=, tokenizer=)` /
  `make_vec(...)`. Trainers obtain envs only via `make` — never by naming a concrete class.
- Each env is a plugin package `igc/envs/<name>/` with `simulator.py`, `catalog.py`, `evaluator.py`,
  `recorder.py`, and a manifest.

**Recipe to add a new simulator (touches no core file):**

1. Copy `igc/envs/_template/` → `igc/envs/<name>/`.
2. Implement the `Simulator` (`execute(target, op, args)→SimResult`, plus `snapshot`/`restore`).
3. Declare the `ToolCatalog` — `ToolSpec`s with per-op arg schema + `RiskLevel` (sizes the head, feeds
   the guardrail).
4. Implement `Evaluator.verify(goal, obs)→(reached, dense_reward)` with domain-correct success.
5. Add a `Recorder` (cassette) for record/replay, or `NullRecorder` for self-simulating envs.
6. `@register("<name>")` a manifest in `__init__.py`.
7. Add an offline pytest (build, codec round-trip, scripted episode reaches goal, destructive action
   blocked without an approval token) on a `StubEncoder` — no GPU.

### The four environments

| Env | Why it is in the mix | What it proves |
| --- | --- | --- |
| Redfish | the real target domain | live infrastructure control (the actual goal) |
| Filesystem | cheapest offline env with real destructive actions | `ToolAction.arguments`, dynamic `available_actions`, destructive risk for the gate |
| SQL (SQLite) | self-simulating, exact verification | exact `verify()` + `BEGIN/ROLLBACK` as the dry-run guardrail primitive |
| GitHub | a real third-party REST API via record/replay | the capture→simulate→live story generalizes; multi-step planning |

![GitHub environment](diagrams/02-github-env.svg)

![SQL environment](diagrams/03-sql-env.svg)

## 3. Workloads are hierarchical: planning, preconditions, discovery

A workload like *"tune BIOS, then boot Ubuntu"* is not one action — it decomposes into ordered
sub-goals with hard preconditions (on iDRAC, a BIOS change is staged as a pending `@Redfish.Settings`
object, applied only after a config **job** and a **reboot**). The terminal reward is clear but
sparse, so the agent must *plan*, and it must *discover* both the API (action space) and the state
machine (Current→Pending→Applied, job Scheduled→Running→Done).

![Hierarchical workload plan](diagrams/04-hierarchical-workload-plan.svg)

Design consequences: `Goal` carries a `plan` (sub-goal DAG); `ToolAction` carries `preconditions`
and `effects`; the simulator models deferred/async state; reward is decomposed (terminal +
per-sub-goal + progress shaping) with HER over reachable sub-goals; the planner ("casting" layer)
generalizes `GoalExtractorTrainer` from one action to an ordered plan; and API/state discovery become
explicit pipeline stages.

## 4. The six models and the training curriculum

`igc` is not one model — it is six, with real dependencies. `M1` (the backbone) is the single root
and the riskiest node: if hidden size `H` changes, every downstream magic dim breaks, so dims are
de-hardcoded before the first backbone fit.

![Training curriculum](diagrams/05-training-curriculum.svg)

| Stage | Model | Objective | Key metric | Compute |
| --- | --- | --- | --- | --- |
| 0 | — | make it runnable + de-hardcode dims + backbone config-driven | `pytest -q` green, ruff clean | CPU |
| 1 | M1 backbone / state encoder | causal-LM SFT over Redfish JSON → checkpoint A | held-out token acc ↑ vs measured GPT-2 baseline | 1-GPU |
| 2 | M2 autoencoder · M3 planner · M4 reward | pool→latent 64 · NL→sub-goal DAG · decomposed reward | recon MSE · DAG topo-validity · reward AUC | 1-GPU ×3 |
| 3 | M5 world/transition | next-latent + status/job-phase classification | 1-step error, phase accuracy, rollout drift | 1-GPU |
| 4 | M6 RL agent | goal-cond. DQN + HER on `ToolAction`, learned reward | success_rate per workload, episode_length | 1-GPU (longest) |

## 5. Backbone modernization (GPT-2 → flash-class LLM)

The central loader already uses `AutoTokenizer` + a class arg and `ValueHead` already reads
`config.hidden_size`, so the model is *nearly* backbone-agnostic. The welds to remove:

- **Kill `.transformer`/`.wpe` reads** → a `backbone_utils.py` helper (`base_model_prefix`,
  `config.hidden_size`, `max_position_embeddings`). RoPE models have no positional table, so
  `emb_shape` must be `(seq_len, hidden_size)`, never a weight tensor.
- **Replace the conv1d `AutoStateEncoder`** (`nn.Linear(1026,…)` + a `seq_len*H` decoder head — a
  ~4M-wide memory bomb at H=4096) with a masked-mean/last-token `StatePooler` + MLP that reconstructs
  the *pooled* vector.
- **Config-driven `--model_type`** (drop gpt2-only choices), `--pooler`, `--use_peft`, `--precision`.
- **LoRA → PEFT.** The existing `lora1d.py` wrapper is GPT-2-only and crashes on construction today;
  adopt HF PEFT targeting `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`.
- **Policy:** LoRA + bf16 + gradient checkpointing on a single GB300 first; full FT only for a tiny
  validation model; fp8/NVFP4 opt-in after bf16 is green.
- **Flash as teacher only** (offline synthetic plan/trajectory generation, sequence-level
  distillation, eval judge) — never fine-tuned, never in the training loop, never near a BMC.
- **Small-model-first:** validate the whole refactor against a tiny CausalLM on CPU before the cluster.

## 6. Phased roadmap

| Phase | Goal | Gate |
| --- | --- | --- |
| 0 · Stabilize | fix launch blockers + de-hardcode dims + backbone config-driven | offline `pytest -q` + ruff |
| 1 · Core types | `igc/core/` contracts + Protocols; `Goal.plan`; Buffer → 5-tuple (done flag) | round-trip/type tests |
| 2 · Registry + Redfish adapter | `register/make`; Redfish plugin; `action_repr` flag + parity test; route `igc_rl_module` through `make_vec` | parity test green (one-hot ≡ tool for GET/HEAD) |
| 3 · Offline prover envs | filesystem + sqlite plugins (real args, dynamic actions, exact verify, dry-run, destructive blocked) | per-env offline tests |
| 4 · Trajectory + github | recorder + `TrajectoryDataset`; `CassetteSimulator`; github plugin (offline replay, live-gated) | record→load round-trip; github offline via cassette |
| 5 · Backbone + M1/M2 | backbone-agnostic refactor; `StatePooler`; train M1 + M2 on NVL72 | small-model CPU green → 1-GPU overfit-a-batch smoke |
| 6 · Learning layer | planner (M3) + reward (M4) + world model (M5) + evaluators + preference data → RL agent (M6) | offline eval harness scores traces; GPU-marked training |
| 7 · Guardrail + deploy | dry-run/approval/executor over `_is_live`; serve backbone+planner via vLLM (TP=1); live Redfish canary behind the gate | guardrail blocks destructive-without-approval; offline gate stays green |

## 7. Infra · monitoring · deploy (NVL72)

- **Slurm runbook:** 1-GPU first, NGC `pytorch:26.03-py3` via pyxis, **always**
  `--exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16`, `NCCL_NVLS_ENABLE=0`, stage data
  + HF cache to node-local NVMe and pin with `-w slotN` (no shared FS). 4-GPU FSDP only after a 2-GPU
  smoke passes (NVLink fabric is flaky — TP=4 crashes).
- **Monitoring** reuses the existing `MetricLogger` (`--metric_report` tb/wandb/mlflow): per-model
  curves (loss/perplexity/accuracy/grad-norm/tokens-per-sec; reward/success-rate/episode-length for
  M6).
- **Training-loop sanity checks:** overfit-a-batch, grad-norm clip + `isfinite` guard, LR
  warmup/cosine, deterministic seed, resume-from-checkpoint, early stop, a startup dim-contract
  assertion (`observation_space == latent_dim + goal_dim`), and a Buffer arity check.
- **Deploy, two paths:** (A) backbone + planner served OpenAI-compatible via vLLM, **TP=1 mandatory**
  (blocked until the backbone refactor lands); (B) the RL policy runs in the mock-env eval harness,
  never wired to a live BMC by default. Any real-Redfish move is read-only GET/HEAD first, paced,
  approved non-prod host, behind the guardrail.

## 8. Verified defects to clear in Phase 0

Found in-tree during design review: `spec.raw_data_dir` AttributeError (RL path unrunnable today) ·
`lora1d` wrapper crashes on construction · `load_checkpoint` bare `raise` breaks resume ·
`rl_batch_size` typed float · stale `.sh` scripts point at a nonexistent `trainer.py` ·
`train_rl_agent.py` tuple-unpack bug · Buffer 4-tuple has no `done` · magic dims `1026/1025/seq*768`.

## 9. Open decisions

1. Trainable base model id (the `--model_type` value for the flash-class backbone).
2. Planner build — learned (`GoalExtractor`→planner), LLM-via-Flash, or both.
3. `--raw_data_dir` — new flag or alias of the existing `--json_data_dir`.
4. Live canary — is an approved non-production Redfish host available (with pacing)?
5. Reward decomposition — how per-sub-goal rewards sum-consistently with the terminal reward.
