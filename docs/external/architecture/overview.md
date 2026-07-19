# IGC architecture & generalization plan

> **⚠️ STATUS (2026-07-13, code audit).** This document is design **intent**, not a description of the
> running system. Today the live agent is a legacy fixed-width one-hot DQN over the state encoder's
> embedding of the **raw Redfish JSON**. The structured state (`RedfishStateV0`), resource graph,
> pointer/action-selector, and candidate-ranking system described below are **coded-but-unwired,
> offline-only, or absent** — audit found 13 of 41 designed objects actually live. Treat any
> "landed / implemented / current RL path" phrasing here as **unverified until you check the code**
> with `scripts/code_reality_check.py`.

This document describes the target architecture for turning `igc` from a Redfish-specific
goal-conditioned RL project into a **generic, pluggable goal-conditioned tool-use agent
framework**, and the phased plan + training/MLOps roadmap to get there. Redfish becomes one
environment adapter among several (filesystem, SQL, GitHub), and the LLM backbone moves from GPT-2
to a config-driven large decoder model fine-tuned on the GB300 NVL72 cluster.

The end goal is not one Redfish policy but a **transferable, multi-step meta-learner**: an agent that
extracts a high-level goal, decomposes it into sub-goals, *discovers* the action space of whatever
REST API (or other tool backend) it is plugged into, finds an optimal strategy to reach the goal, and
carries that capability to new, unseen environments with little or no retraining.

> Status: working design. Diagrams are theme-aware SVGs under [`docs/diagrams/`](../diagrams/).

---

## 0. Current reality, target interaction, and model map

The diagrams in this section are a design map. They intentionally separate the current Phase 0 code
path from the target architecture so planning work does not mistake scaffolding for an integrated
training path.

### Current Phase 0 code path

Today, the default Redfish path is still a captured-data, one-hot-action RL shell. The
`MockServer` class, defined in `igc/envs/rest_mock_server.py`, replays captured Redfish responses;
the Gym environments encode responses with the legacy GPT-2-shaped encoder path; and the RL trainer
still writes four-field replay tuples. Treat any DQN/HER metric as a smoke/debug signal until the
transition, terminal-mask, and evaluator contracts are fixed.

```mermaid
flowchart LR
    Capture["Captured Redfish data\n~/.json_responses + rest_api_map.npy"]
    Dataset["JSONDataset\nigc/ds/redfish_dataset.py\nURL/method maps + loose action targets"]
    Mock["MockServer\nigc/envs/rest_mock_server.py\nreplay + callback mutation"]
    Env["RestApiEnv / VectorizedRestApiEnv\nigc/envs/*\none-hot URL + HTTP method"]
    Encoder["Legacy response encoder\nraw JSON -> GPT-2-shaped tensor"]
    Policy["Igc_QNetwork\nfixed output width = URL count + methods"]
    Buffer["Buffer\nigc/modules/igc_experience_buffer.py\n(state, action, reward, next_state)"]
    Trainer["IgcAgentTrainer\nigc/modules/igc_train_agent.py\nlegacy DQN/HER loop"]

    Capture --> Dataset --> Mock --> Env --> Encoder --> Policy --> Env
    Env --> Buffer --> Trainer --> Policy
```

### Target interaction model

The target path keeps Redfish as one adapter, but moves the decision boundary to structured
observations, legal candidate actions, domain evaluators, and guarded execution. `RedfishStateV0` is
the proposed first compact-state schema: a JSON-serializable structured payload under
`Observation.structured`, not a learned graph model yet.

```mermaid
flowchart LR
    Goal["GoalEnvelope\noperator text -> GoalRefs + explicit dependencies"]
    Extractor["GoalExtractor / GoalEncoder\ntext -> atomic z_sub_goal(s)"]
    State["RedfishStateV0\nstructured resources, links,\nstatus, tasks, settings, telemetry"]
    Catalog["ToolCatalog.available_actions(obs)\nplanned Redfish adapter\nlegal ToolAction candidates"]
    Codec["ActionCodec\nigc/modules/policy/action_codec.py\nrender + cache candidate templates"]
    Pointer["Pointer Q head\nigc/modules/policy/pointer_policy.py\nscore legal candidates only"]
    Args["Argument stage\nigc/modules/policy/argument_decoder.py\nfills per-slot enum values"]
    Guard["Guardrail\nplanned dry-run -> approval -> executor"]
    Sim["Simulator\nMockServer adapter first\nlive Redfish gated"]
    Eval["Evaluator.verify(goal, obs)\nplanned domain reward + success"]
    Step["StepResult / Transition\nigc/core/types.py\nthin step + rich replay record"]
    Replay["Replay / HER\nterminal masks + achieved_goal"]

    Goal --> Extractor --> Pointer
    State --> Catalog --> Codec --> Pointer --> Args --> Guard --> Sim
    Sim --> State
    State --> Eval --> Step --> Replay --> Pointer
    Extractor --> Eval
```

### Model dependency map

The model stack is a dependency graph, not six independent training jobs. The backbone has to be
made config-driven before downstream heads can be trusted; the RL policy should not report real
learning metrics until structured state, legal actions, evaluator rewards, and replay masks are in
place. The math gate for these claims lives in [math checks](../research/math-checks.md).

```mermaid
flowchart TD
    RB["RedfishBackbone / model_x\nconfig-driven causal/decoder model"]
    S0["RedfishStateV0 extractor\nstructured state contract + fixtures"]
    SE["StateEncoder / StatePooler\npooled structured/text state"]
    GE["GoalExtractor / GoalEncoder\ninstruction -> ordered REST goals + z_goal"]
    RV["RewardVerifier\nstructured verify + dense reward"]
    WM["WorldModel\nnext-state + status/task phase"]
    RP["RLPolicy\ncandidate scoring + replay/HER"]
    Gate["Offline eval harness\nno GPU/network/live host by default"]

    S0 --> SE
    RB --> SE
    RB --> GE
    S0 --> RV
    SE --> WM
    RV --> WM
    SE --> RP
    GE --> RP
    RV --> RP
    WM --> RP
    RP --> Gate
```

### RedfishStateV0 field budget

`RedfishStateV0` is the smallest useful structured state target to validate before any learned graph
pooling. It should be deterministic, JSON-serializable, and testable from tiny synthetic Redfish
fixtures.

| Field group | Minimum content | Why it stays |
| --- | --- | --- |
| Resource identity | canonical URI, `@odata.type`, schema version, collection membership | preserves stable entities |
| Topology | selected `@odata.id` links, parent/subordinate edges | keeps Redfish's hypermedia graph visible |
| Health/control | `Status.State`, `Status.Health`, power/boot/firmware/storage summaries | carries control-relevant state |
| Action surface | allowed methods, `Actions`, targets, argument schemas, allowable values, risk level | builds legal `ToolAction` candidates |
| Deferred state | current vs pending settings, task/job phase, reboot/apply-time hints | prevents false Markov collapse |
| Observation metadata | HTTP status, error class, freshness/staleness, ETag when present | separates unknown, stale, and failed reads |
| Goal context | goal spec fragment, sub-goal index, achieved-goal fields, previous action/result | supports evaluator rewards and HER |

Validation comes before pooling: extractor golden tests, action-catalog parity, pending-settings
counterexamples, component-order invariance, and replay/HER shape tests.

## 1. Where igc is today, and the five target layers

`igc` already has a Redfish-specific MDP shell (Gym env + mock REST server + GPT-2-shaped state
encoder + legacy goal-conditioned DQN/HER trainer), but every layer is welded to Redfish and GPT-2.
The legacy trainer is useful for smoke/debug work, not for trusted RL metrics yet. The target is five
clean layers behind one interface. The map below marks each layer **refactor existing** (reuse what is
there) vs **build new** (greenfield), and calls out the keystone change.

![Five-layer gap map](../diagrams/01-five-layer-gap-map.svg)

- **Goal** — `Goal(instruction, spec, constraints, plan)`. Today only a `GoalTypeState`
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

### Scalable action selection — the pointer / candidate-scoring policy

The one-hot action space explodes because the policy's output width equals the number of discovered
URLs (`num_actions`), and adding methods/arguments multiplies it. The target fix is a **pointer /
candidate-scoring policy**: instead of a fixed `Linear(hidden, num_actions)` head, the policy scores
the *legal* candidate actions that a Redfish `ToolCatalog.available_actions(obs)` adapter will return
for the current state. The pointer head and action codec are scaffolded; the Redfish adapter and env
cutover are still planned.

- The policy encodes state + goal into a query `q`; each candidate `ToolAction` is rendered to a
  canonical, value-independent string (`igc/core/action_render.py`, landed) and embedded into a key
  `k_i` by the shared backbone (cached by `action_template_key`). The score is `Q(s, a_i) = q · k_i`,
  so the output width = number of *currently legal* candidates (local fan-out, tens) — **never the
  global catalog**.
- Adding tools/URLs/methods grows (cached) encoding compute, not policy weights. A brand-new tool is
  intended to be scorable in the same embedding space — no output-layer resize, no adler32 re-index,
  no head retrain — but transfer remains an evaluation target until the offline harness proves it.
- Arguments are filled in a second stage from `ToolSpec.arg_schema` (small categorical heads for
  enumerated/bounded slots; constrained backbone decoding for freeform path/SQL/body), then
  `ToolCatalog.validate` gates execution.
- The large LLM is the shared encoder for state, candidate-action text, and freeform-argument
  generation (LoRA + bf16); only tiny heads (state-query, action-projector, argument-decoder) train.
- Migration: `--action_repr {onehot, pointer}` (default `onehot`); the legacy `Igc_QNetwork` + adler32
  codec stay byte-for-byte intact until a parity test (pointer ≡ one-hot on the enumerated Redfish
  case) is green, then the default flips.

Target state encoding uses the same backbone (pooled last-hidden of `Observation.text` plus
`Goal.instruction`) with dims derived from `config.hidden_size`, not hardcoded `1025` / `768`. The
current environment path still encodes raw JSON responses into legacy tensors, so `RedfishStateV0`
and state-pooler work must land before this becomes the training path.

## 2. How a new simulator plugs into the agent

The target framework uses **typed Protocols + a decorator registry + per-env plugin packages**. The
core should not change when a new environment is added; the existing `MockServer.request → MockResponse`
seam is kept intact behind a planned `RedfishSimulator` adapter.

Implemented today in Phase 0: `igc/core/types.py` defines the core dataclasses and enums, and
`igc/core/protocols.py` defines the runtime-checkable Protocols. Still planned: `igc/envs/registry.py`,
`make(...)` / `make_vec(...)`, per-env plugin packages under `igc/envs/<name>/`, and the Redfish adapter
that preserves the mock-server seam.

**Target recipe to add a new simulator (touches no core file once the registry lands):**

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

![GitHub environment](../diagrams/02-github-env.svg)

![SQL environment](../diagrams/03-sql-env.svg)

## 3. Workloads are hierarchical: planning, preconditions, discovery

A workload like *"tune BIOS, then boot Ubuntu"* is not one action — it decomposes into ordered
sub-goals with hard preconditions (on iDRAC, a BIOS change is staged as a pending `@Redfish.Settings`
object, applied only after a config **job** and a **reboot**). The terminal reward is clear but
sparse, so the agent must *plan*, and it must *discover* both the API (action space) and the state
machine (Current→Pending→Applied, job Scheduled→Running→Done).

![Hierarchical workload plan](../diagrams/04-hierarchical-workload-plan.svg)

Design consequences: `GoalEnvelope` carries atomic `GoalRef` targets and only the dependency hints
explicitly stated in text; `ToolAction` carries `preconditions` and `effects`; the simulator models
deferred/async state; reward is decomposed (terminal + per-sub-goal + progress shaping) with HER over
reachable sub-goals; and API/state discovery become explicit pipeline stages. The RL policy learns
the concrete action ordering; the language model does not emit a strategy.

### Meta-learning: a transferable multi-step agent

The end goal is not a single Redfish policy but an agent that **transfers** — plug it into a new REST
API (or filesystem/SQL/GitHub) and it should operate that backend with little or no retraining. This
is the target hypothesis; it becomes a claim only after offline eval traces and metrics support it.
The design makes the hypothesis concrete, layer by layer:

1. **Extract the goal.** `GoalExtractor` maps a high-level instruction to atomic `GoalRef` targets
   and explicit dependency hints; `GoalEncoder` aligns the text with concrete Redfish goal surfaces.
2. **Discover the action space.** A new backend exposes its tools/ops through `ToolCatalog` +
   `available_actions(obs)` (for Redfish, derived from the `redfish_ctl` crawl + the `.npy` map). The
   agent never needs a fixed, pre-enumerated action set — it reads what is legal *now*.
3. **Act zero/few-shot.** Because the policy is the **pointer / candidate-scoring** head (§1), a tool
   it has never seen is still scorable: its `tool_name/op/schema` text is embedded by the shared
   backbone, so there is no output-layer resize, no re-index, no head retrain. This is what turns "a
   new API" from a retraining event into an inference-time lookup.
4. **Find an optimal strategy.** Goal-conditioned RL (DQN + HER) over the active `z_sub_goal` learns
   the ordering and the fewest-steps path; the world model supplies preconditions/dynamics so the
   agent plans rather than flails.
5. **Adapt on the new environment.** Trained across *multiple* environments (filesystem, SQL, GitHub,
   Redfish) — the meta-training distribution — the shared backbone, pointer policy, and state encoder
   transfer; a short interaction (plus an optional LoRA / world-model update) adapts to the new
   backend's quirks. This is the meta-learning ("learn to learn") payoff: capability acquired on known
   APIs carries to unknown ones.

In short: **extract → discover → execute → transfer.** Every layer is built so the unit of
generalization is *the skill of operating a tool API*, not *a specific API*.

## 4. Trainable Components And Training Curriculum

`igc` is not one model. It is a dependency graph of separately named components with separate
weights. The Redfish-aware backbone checkpoint is `model_x`; downstream state, goal, argument,
reward, world-model, and policy components must name their own weight roles and output directories
instead of reusing historical `M*` aliases. If hidden size `H` changes, every downstream magic dim
breaks, so dims are de-hardcoded before the first backbone fit.

![Training curriculum](../diagrams/05-training-curriculum.svg)

| Stage | Component | Objective | Key metric | Compute |
| --- | --- | --- | --- | --- |
| 0 | — | make it runnable + de-hardcode dims + backbone config-driven | `pytest -q` green, ruff clean | CPU |
| 1 | RedfishBackbone / `model_x` | causal-LM SFT over Redfish JSON → Phase 1 checkpoint | held-out token acc ↑ vs measured GPT-2 baseline | 1-GPU → multi-GPU |
| 2 | StateEncoder · GoalExtractor/GoalEncoder · RewardVerifier | pool→latent 64 · text→ordered REST goals · decomposed reward | recon MSE · extraction exact match · reward AUC | 1-GPU ×3 |
| 3 | WorldModel | next-latent + status/job-phase classification | 1-step error, phase accuracy, rollout drift | 1-GPU |
| 4 | RLPolicy | goal-cond. DQN + HER on `ToolAction`, learned reward | success_rate per workload, episode_length | 1-GPU (longest) |

## 5. Backbone modernization (GPT-2 → flash-class LLM)

The target loader should use `AutoTokenizer` + a class arg, and `ValueHead` should read
`config.hidden_size`, so the model becomes backbone-agnostic. The welds to remove:

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
| 5 · Backbone + state representation | backbone-agnostic refactor; `StatePooler`; train `model_x` and StateEncoder on NVL72 | small-model CPU green → 1-GPU overfit-a-batch smoke |
| 6 · Learning layer | GoalExtractor/GoalEncoder + RewardVerifier + WorldModel + evaluators + preference data → RLPolicy | offline eval harness scores traces; GPU-marked training |
| 7 · Guardrail + deploy | dry-run/approval/executor over `_is_live`; serve backbone + goal extractor via vLLM (TP=1); live Redfish canary behind the gate | guardrail blocks destructive-without-approval; offline gate stays green |

## 7. Infra · monitoring · deploy (NVL72)

The canonical runtime setup lives in private `docs/internal/environment.md` when the operator context is present. Architecturally, training stays
separate from local development: Phase 0 gates on CPU, while training/fine-tuning runs on the GB300
NVL72 through a one-GPU-first Slurm/pyxis workflow.

- **Monitoring** reuses the existing `MetricLogger` (`--metric_report` tb/wandb/mlflow): per-model
  curves (loss/perplexity/accuracy/grad-norm/tokens-per-sec; reward/success-rate/episode-length for
  RLPolicy).
- **Training-loop sanity checks:** overfit-a-batch, grad-norm clip + `isfinite` guard, LR
  warmup/cosine, deterministic seed, resume-from-checkpoint, early stop, a startup dim-contract
  assertion (`observation_space == latent_dim + goal_dim`), and a Buffer arity check.
- **Deploy, two paths:** (A) backbone + GoalExtractor served OpenAI-compatible via vLLM, **TP=1 mandatory**
  (blocked until the backbone refactor lands); (B) the RL policy runs in the mock-env eval harness,
  never wired to a live BMC by default. Any real-Redfish move is read-only GET/HEAD first, paced,
  approved non-prod host, behind the guardrail.

## 8. Verified defects to clear in Phase 0

Found in-tree during design review:

- `igc/shared/shared_arg_parser.py` currently defines `--json_data_dir`, while
  `igc/modules/igc_rl_module.py` references missing `spec.raw_data_dir`; Phase 0 must alias or rename
  one side.
- `lora1d` wrapper crashes on construction.
- `load_checkpoint` uses a bare `raise`, which breaks resume handling.
- `rl_batch_size` is typed as a float.
- Stale shell scripts still point at a nonexistent `trainer.py`.
- `train_rl_agent.py` has a tuple-unpack bug.
- ~~Buffer entries are 4-tuples with no `done` flag.~~ **Fixed:** `igc/modules/igc_experience_buffer.py`
  now stores/returns a per-transition `done`; the trainer's target uses
  `igc.modules.rl.q_targets.q_learning_target` with a `(1 - done)` mask.
- Magic dimensions such as `1026`, `1025`, and `seq*768` are still hard-coupled to GPT-2 shapes.

## 9. Open decisions

1. Trainable base model id (the `--model_type` value for the flash-class backbone).
2. Goal extraction build — learned `GoalExtractor`/`GoalEncoder`, teacher-generated text drafts, or both.
3. `--raw_data_dir` — new flag or alias of the existing `--json_data_dir`.
4. Reward decomposition — how per-sub-goal rewards sum-consistently with the terminal reward.

### Resolved design decisions (review "hard flags")

1. **`Transition` vs `StepResult`** — both, distinct roles (`igc/core/types.py`).
   `GoalEnvironment.step()` returns the thin Gymnasium `StepResult`
   (`observation, reward, terminated, truncated, info`); `Transition` is the rich
   replay-buffer record adding `action, next_observation, desired_goal,
   achieved_goal` plus a `relabel()` helper. The env stays goal-agnostic; the agent
   assembles a `Transition` before pushing to replay.
2. **terminal vs truncated semantics** — Gymnasium semantics. `terminated` is a true
   MDP terminal (goal reached / unrecoverable) → bootstrapping stops (target =
   reward). `truncated` is a budget/time cut → bootstrapping continues. The DQN mask
   is `(1 - terminated)` only; a transition is terminal iff the env goal was reached.
3. **legal-action source of truth** — merged catalog. The `.npy`
   `allowed_methods_mapping` is authoritative for *verb legality* per URL (the binding legacy
   `redfish_ctl` discovery contract); the JSON `Actions` / `@Redfish.ActionInfo` blocks supply the
   *parameter space* (enums) consumed by the stage-2 argument decoder. CSDL schema is a
   later enrichment.
4. **credential logging** — redacted before any live canary
   (`MockServer._redact_headers`; no username/password/token in request logs).
5. **two-stage action with values** — `action_to_prompt` intentionally drops argument
   values, so mutating actions are completed by `igc/modules/policy/argument_decoder.py`
   (per-slot enum scoring, no cross-product explosion).

## 10. Meta-RL direction (agreed 2026-06-30)

**Final goal (Tier 4):** a generalist RL agent that operates ANY REST API — given a goal it
*discovers* the API's action space, plans the call sequence to reach it, and **generalizes to an
unseen API**, adapting from a few examples (meta-learning). Redfish is the first proving ground because
we can simulate it; the recipe (build a simulator → train) repeats per API. Tiers: 1 = Redfish loop
works; 2 = generalization + ablations + safety (research result); 4 = several simulators → generalize to
an unseen REST API. Tier 3 (filesystem/SQL) folds in as diverse sims that force generalization.

**It is a sparse-reward problem.** Reward arrives only at goal completion of a multi-step API
interaction, so three complementary attacks: **HER** (relabel achieved states — `q_targets.relabel_future`
+ `Transition.achieved_goal`), **LLM priors + meta-learning** (smart exploration), and **action-space
discovery + the pointer policy** (score only discovered legal candidates → shrink the search).

**Architecture — RL² cast as a Transformer.** The LLM backbone carries cross-episode history in its
context = the in-context meta-learner (the Transformer analog of RL²'s RNN; cf. Decision Transformer /
Algorithm Distillation). StateEncoder/StatePooler compress observations so a long history fits the
context budget. **Dense first; MoE is a later upgrade** (upcycle/distill once there are many APIs + data + expert-parallel infra)
— MoE adds capacity/specialization, not the meta-learning (which lives in attention-over-history).

**Backbone sizing — decoupled.** Small dense bf16 backbone for the hot-loop encoder/RL
(RedfishBackbone/StateEncoder/RLPolicy, runs every step); a larger model may be used for the GoalExtractor if extraction metrics justify it
(runs once per episode). Trainable =
small/mid dense bf16 + LoRA + ZeRO-3 (`--sharding zero3`). DeepSeek-V4-Flash (284B FP8/FP4 MoE) is the
**inference/teacher** only (deployed NVFP4), never the trainable backbone.

**Goal extraction — standalone distillation.** Own backbone + tokenizer + Redfish-corpus-backed
`(operator text -> atomic GoalRef set + explicit dependency hints)` data. A teacher model may draft
operator text, but deterministic code owns `true_y`; verify generated text against the current
GoalExtractor before using it as supervised data.

**Training curriculum.**
1. **Represent** (supervised, separate): RedfishBackbone SFT + StateEncoder/StatePooler → compact
   states. Pretrained first; RL LoRA-adapts on top (do not co-train the whole backbone with RL).
2. **Extract** (distill) + **Reward** (build, parallel): GoalExtractor/GoalEncoder from the
   generated goal dataset; RewardVerifier schema-driven verifier.
3. **Meta-RL — ONE optimization path:** the in-context RL²-Transformer policy over the API-sim
   distribution, with HER, a **BC/SFT warm-start** (imitate teacher demos before RL — the main lever
   against the sparse-reward cold start), and a **narrow→broad task curriculum**. Meta is not a phase
   after single-task RL; it *is* RL over a task distribution with cross-episode memory.
4. **Scale** (later): MoE via upcycle/distillation.

**Eval (the contribution):** on a held-out API, *success vs. number of adaptation examples* — the
few-shot adaptation curve.

## 11. Architecture diagrams (Tier-4 meta-RL)

Four views of the agreed design (§10). Maintainable mermaid; render on GitHub.

### 11.1 Tier-4 system — generalist API meta-RL agent

```mermaid
flowchart TB
    NL["Natural-language goal<br/>(set boot=PXE, power-cycle)"]
    NL --> GE["GoalExtractor / GoalEncoder<br/>-> atomic GoalRefs + z_sub_goal"]
    GE -->|goal-conditions| AGENT

    subgraph AGENT["Goal-conditioned RL2 agent (one shared policy)"]
        ENC["StateEncoder / StatePooler<br/>API response -> compact state vector"]
        HIST["Transformer over history<br/>state, action, reward = in-context task memory"]
        POL["Pointer policy<br/>score discovered legal candidates"]
        ARG["Argument decoder<br/>fill action values"]
        ENC --> HIST --> POL --> ARG
    end

    subgraph ENVS["Environment = any simulated REST API"]
        DISC["Action-space DISCOVERY<br/>.npy / OpenAPI / HATEOAS links"]
        SIM["Simulator<br/>mock REST from captured data or spec"]
    end

    ARG -->|REST call| SIM
    SIM -->|response| ENC
    DISC -->|legal candidates| POL
    SIM --> RV["RewardVerifier<br/>verify Goal.spec vs state -> sparse reward"]
    RV --> HER["HER replay<br/>relabel achieved states"]
    HER --> POL

    META["Meta-train over MANY API simulators<br/>-> generalize few-shot to a HELD-OUT API"]
    ENVS -.-> META
```

### 11.2 RL2 cast as a Transformer (the in-context meta-learner)

```mermaid
flowchart LR
    subgraph TRIAL["One trial = a task (API) sampled from the distribution"]
        E1["episode 1<br/>s,a,r ..."]
        E2["episode 2<br/>s,a,r ..."]
        Ek["episode k<br/>s,a,r ..."]
    end
    E1 --> CTX
    E2 --> CTX
    Ek --> CTX
    CTX["Transformer context window<br/>compressed states + prev action + prev reward"]
    CTX --> POL["policy head (pointer scoring)"]
    POL --> ADAPT["in-context adaptation:<br/>behaviour improves across episodes<br/>WITHOUT weight updates"]
    NOTE["StateEncoder/StatePooler compress each observation<br/>so a long history fits the context budget"]
    CTX -.- NOTE
```

### 11.3 Training curriculum (separate prerequisites -> one meta-RL path)

```mermaid
flowchart TB
    subgraph SEP["Separate prerequisite phases"]
        REP["Represent<br/>RedfishBackbone SFT + StateEncoder<br/>small dense, pretrained first"]
        PLAN["Extract<br/>GoalExtractor / GoalEncoder"]
        REW["Reward<br/>RewardVerifier schema-driven verifier"]
    end
    REP --> META
    PLAN --> META
    REW --> META
    subgraph ONE["Meta-RL = ONE optimization path"]
        META["in-context RL2-Transformer policy<br/>over the API-simulator distribution<br/>+ HER + BC warm-start + task curriculum (narrow -> broad)"]
    end
    META --> SCALE["Scale (later)<br/>MoE via upcycle / distillation"]
```

### 11.4 Backbone topology + distillation (decoupled sizes)

```mermaid
flowchart LR
    DS["Teacher LLM<br/>inference-only bootstrap"]
    DS -->|draft operator text X only| DISTILL["goal text set<br/>deterministic true_y labels"]
    DISTILL -->|SFT / contrastive| GEB["GoalExtractor / GoalEncoder<br/>(runs once per episode)"]
    SMALL["small dense bf16 backbone 1-7B<br/>+ LoRA + ZeRO-3 (hot loop)"]
    SMALL --> RB["RedfishBackbone / model_x"]
    SMALL --> SE["StateEncoder / StatePooler"]
    SMALL --> RP["RLPolicy heads"]
    DS -.->|direct call for bootstrap| GEB
```

## 12. Tool-teaching (LLM-taught few-shot tool acquisition)

Today the pointer policy (§1, "pointer / candidate-scoring policy") is *passively*
tool-aware: `action_to_prompt` (`igc/core/action_render.py`, landed) renders an unseen
op's `tool=… op=… target=… args=[slot:type] schema=…` text, and `ActionCodec`
(`igc/modules/policy/action_codec.py`) embeds it through the shared backbone, so a
brand-new tool is *scorable* with no head resize — `Igc_PointerQNetwork`
(`igc/modules/policy/pointer_policy.py`) is pure `q · k_i` over the legal-candidate
fan-out (§3.3, step 3). But that signal is only the tool's *declared surface*; the agent
still learns *how to actually drive* the tool by sparse-reward trial-and-error (reward
arrives only at goal completion — §10). **Tool-teaching adds an active loop**: the
same gated LLM-teacher path used for goal-dataset bootstrapping reads
the agent's own `k` real `(call → result/error)` interactions with an unknown op and
induces a grounded, versioned **ToolCard**, which flows through the *exact existing*
render→cache→pointer→argument pipeline to accelerate few-shot acquisition. Passive
scorability makes an unseen tool *scorable*; tool-teaching makes it *quickly learnable*.

```mermaid
flowchart TB
    FACTS["Discovery facts (passive, today)<br/>ToolSpec(tool, op, arg_schema, risk_level)<br/>Redfish: .npy allowed_methods_mapping +<br/>@Redfish.ActionInfo enums (§9.3 resolved)"]
    OBS["k observed Transitions THIS trial<br/>igc/core/types.py: .action,<br/>.next_observation.status/.error/.text"]
    FACTS --> TEACH
    OBS --> TEACH
    TEACH["ToolTeacher.induce()<br/>reuse DeepSeek-V4-Flash (§11.4)<br/>evidence-bound JSON prompt · StubTeacher offline"]
    TEACH --> CARD["ToolCard (igc/core/tool_card.py)<br/>keyed (env_name, tool_name, op), per-trial<br/>effective_signature · expected_response ·<br/>error_taxonomy · preconditions · grounding · version"]
    CARD --> GATES
    subgraph GATES["Grounding gates (run BEFORE any injection)"]
        G1["1 evidence-bound prompt<br/>uncited claims dropped"]
        G2["2 schema gate + .npy/ActionInfo override (§9.3)"]
        G3["3 RewardVerifier replay · Evaluator.verify + MockServer"]
        G4["4 online falsification<br/>confirm vs contradict => falsified"]
        G1 --> G2 --> G3 --> G4
    end
    GATES --> STATUS{"grounding.status"}
    STATUS -->|grounded: may up-rank| INFER
    STATUS -->|provisional: widen only| INFER
    subgraph INFER["Injection (inference-time, no weight update)"]
        A["A candidate text<br/>action_to_prompt(card) re-keys<br/>ActionCodec cache · NO head resize"]
        B["B RL2 context block (§11.2)"]
        C["C argument_decoder enum tightening<br/>bounded by ActionInfo"]
        D["D READ_ONLY safe-probe gate"]
    end
    GATES --> TRAIN
    subgraph TRAIN["Gradient-trained (offline meta-train only)"]
        BC["BC warm-start (§10 main lever)"]
        AD["Algorithm Distillation<br/>teacher-free at test"]
        SHAPE["potential shaping F=γΦ'−Φ over q_targets<br/>source reward stays RewardVerifier.verify()"]
    end
```

### 12.1 Why — passive scorability vs. quick learnability

Passive scorability (§1, §3.3 step 3) embeds an unseen op's declared text so the pointer
head can *score* it; it does nothing to teach the op's effective signature, response
shape, or error semantics, so cold-start exploration on a never-seen tool is blind
trial-and-error under RewardVerifier's sparse reward (§10). Tool-teaching layers an active induction
loop on top, leaving the passive path byte-identical when no card is present (the
`card=None` parity guarantee, pinned by `tests/core/test_action_render_card.py`).

### 12.2 The ToolCard artifact

A `ToolCard` (`igc/core/tool_card.py` — pure-stdlib dataclass, `to_dict`/`from_dict`
round-trip, mirroring `igc/core/types.py`) is an evidence-checked spec of one op, keyed
`(env_name, tool_name, op)` and **per-trial namespaced** (`ToolCardStore`) so a
held-out-API trial's cards never leak across trials. Fields: `effective_signature` (a
*refinement* of `ToolSpec.arg_schema[op]`, never a replacement; enums ⊆
`@Redfish.ActionInfo` allowable values), `expected_response` (field → type, distilled
from observed 2xx bodies), `error_taxonomy` (`retriable | fatal | precondition_unmet`,
each citing ≥1 `Transition`), `preconditions`/`usage_tips`, `provenance`
(`llm | stub`, evidence ids, `k_observed`), `grounding` (`status ∈
{provisional, grounded, contradicted}` + confirm/contradict + RewardVerifier counters), `version`,
`content_hash` (blake2b of the canonical learned content — excludes the volatile
grounding tallies so a counter tick does not churn the embedding cache), and
`spec_fingerprint` (blake2b of `ToolSpec.arg_schema[op]` — a moved op schema
auto-invalidates a stale card).

### 12.3 Induction — ToolTeacher

`ToolTeacher.induce()` (`igc/modules/teach/tool_teacher.py`, slice 6b) issues a
JSON-schema'd, **evidence-bound** prompt against the same gated LLM-teacher path used for
goal-dataset bootstrapping — a reuse, not a new LLM. Inputs are
the discovery facts (`ToolSpec`, and for Redfish the `.npy` `allowed_methods_mapping` +
`@Redfish.ActionInfo` enums) plus the `k` observed `Transition` records
(`.action`, `.next_observation.status/.error`, truncated `.text/.structured`,
`.terminated`). The teacher redacts host/credentials before prompting (mirroring
`MockServer._redact_headers`). A `StubTeacher` replays recorded cards for the offline
CPU subset (no network, no DeepSeek). The teacher is **inference-only, never trained**
(§10, §11.4).

### 12.4 Four injection seams

Only **grounded** claims may up-rank; **provisional** claims may only widen exploration.
All four seams are inference-time (no weight update during held-out adaptation):

- **A · candidate text** — `action_to_prompt(action, spec, card=None)` appends a bounded,
  value-independent `card=[…]` clause that mixes into `action_template_key`, **re-keying
  exactly that candidate's blake2b `ActionCodec` cache entry** so the backbone re-embeds
  it (every other candidate stays byte-identical; pinned by
  `tests/modules/test_action_codec_card.py`). `card=None` reproduces today's string
  byte-for-byte. **No head resize** — the card rides the candidate *text*, so no `prior`
  argument is added to the cosine-normalized `score_candidates` (avoids scale-mixing a
  logit bias).
- **B · RL² context block** — the same card text as a compact token block prepended once
  per trial to the cross-episode history (§11.2); StateEncoder/StatePooler keeps it within the
  context budget.
- **C · argument-decoder enum tightening** — `card.effective_signature` tightens
  `ArgumentSlot.choices`/`required` in `arg_slots_for`
  (`igc/modules/policy/argument_decoder.py`), but **only within `@Redfish.ActionInfo`
  enums**, never beyond them.
- **D · READ_ONLY safe-probe gate** — `safe_probe_actions(card, catalog, obs)`
  (`igc/modules/teach/safe_probe.py`) = the card's probes ∩
  `ToolCatalog.available_actions(obs)`, filtered by `ToolCatalog.validate` and
  `RiskLevel ≤ READ_ONLY`. This drives cold-start probing on GET/HEAD only, before any
  mutating op is even legal.

### 12.5 Grounding and the safety invariant

The teacher *will* fabricate; nothing is trusted on assertion. `ToolCardGrounder`
(`igc/modules/teach/grounding.py`) runs four gates before any injection: (1)
**evidence** — every `error_taxonomy` entry must cite a real `Transition`; uncited
entries are dropped. (2) **schema + enum** — every arg slot must exist in
`ToolSpec.arg_schema[op]`, and enum claims are clipped to `@Redfish.ActionInfo`
allowable values; **the `.npy`/ActionInfo overrides the teacher on any enum conflict**
(§9.3, resolved — binding contract). (3) **RewardVerifier replay** — each claim becomes an assertion
checked via `Evaluator.verify` (RewardVerifier, `igc/core/protocols.py`) + a no-op replay against
`MockServer`/cassette; only passing claims flip `grounding.status` to `grounded` (lands
in slice 6b/6c — the offline slice carries gates 1, 2, 4). (4) **online falsification** —
every real `StepResult` updates `n_confirmations`/`n_contradictions`; the status flows
`provisional → grounded`, or `→ contradicted` once contradictions dominate (prior
zeroed), and a `spec_fingerprint` mismatch auto-invalidates the card.

**Safety invariant (binding):** a card may only **raise** caution, never lower
`RiskLevel` or unlock a DESTRUCTIVE/MUTATING op. `ToolCatalog.validate` plus the Phase-7
dry-run/approval guardrail remain the sole authority; cards are advisory to scoring
only. *Honest caveat:* the Phase-7 guardrail is itself **queued** (see ROADMAP) — only
credential redaction has landed — so this is a design-time guarantee, not yet enforced,
and must not be reported as already-enforced.

### 12.6 Training

Tool-teaching is **not a new phase**; it rides §10's single meta-RL optimization path
(§11.3) inside Phase 6, and depends on StateEncoder/StatePooler (compress so `k` interactions +
card fit the RL² budget), the gated teacher path, and RewardVerifier (the grounder). The interplay:

- **BC/SFT warm-start** (§10's main lever against the sparse-reward cold start): teacher
  demos enter BC *only after* RewardVerifier confirms they reached a sub-goal, carrying their ToolCard
  in context, so the policy learns the card→action mapping before RL.
- **Algorithm Distillation** (the learned core): next-action cross-entropy over
  `[ToolCard, ep1(s,a,r), ep2…]` cross-episode histories, internalizing the across-episode
  improvement operator (§10/§11.2's "improves across episodes WITHOUT weight updates") so
  the **test-time agent needs no teacher in the loop**.
- **HER unchanged**: `relabel_future` + `q_learning_target` with the `(1-done)` mask
  (`igc/modules/rl/q_targets.py`) run over the card-conditioned episodes — the card is part
  of the observation, not a separate loss.
- **Potential-based shaping**: a grounded card supplies a potential Φ over (precondition-met,
  expected-progress); shaping is `F = γΦ′ − Φ` over `q_targets`, which *provably cannot
  change the optimal policy*, and decays as real-Q confidence grows. **The source reward
  stays RewardVerifier's `verify()` — this is not a new reward channel** (it deliberately avoids a
  second, hallucination-amplifying reward signal adjacent to the still-open §9.4
  reward-decomposition decision).

The only new trainable parameters are the LoRA adapter already in budget (small/mid dense
bf16 + LoRA + ZeRO-3). The honest core: **few-shot adaptation on a held-out API happens at
inference time** (seams A–D + AD-internalized in-context improvement); the *ability to use
cards* is what gets gradient-trained offline over the meta-train API distribution.

### 12.7 Evaluation

The primary metric extends §10's contribution (success_rate vs. number of adaptation
interactions on a **held-out API**) into a with-vs-without A/B: `curve_A` = passive
pointer text only (§3.3); `curve_B` = + ToolCard teaching. The claim is that `curve_B`
reaches a target success_rate in fewer interactions and at lower `episode_length` (the two
RLPolicy metrics, §4). Crucially, the **teacher-free distilled agent at test time must match the
teacher-in-loop agent** — the honest proof the across-episode operator was internalized,
not a present-teacher crutch.

Honesty guardrails: a **negative control** (converged success_rate *and* episode_length
must match the no-teaching baseline — proves teaching left no permanent crutch and shaping
did not hack the final metric); **StubTeacher + per-`env_name` keying** (held-out cards
come only from that API's own few shots, no cross-API leakage); calibration (Brier score
of teacher confidence; falsification latency). Ablations: (1) no-grounding (quantifies the
hallucination tax); (2) k-sweep (0,1,3,5); (3) text-in-candidate vs RL²-context vs both;
(4) frozen vs refreshed card; (5) provisional-allowed vs grounded-only; (6) AD on/off; (7)
BC warm-start on/off; (8) shuffled error-taxonomy control. All offline: held-out API via
`CassetteSimulator`/`MockServer`, CPU + StubEncoder for plumbing, `@pytest.mark.gpu` for
backbone-in-the-loop curves.

### 12.8 Novelty and honest risks

The learned core *is* Algorithm Distillation (§10's RL²-as-Transformer); the novelty is
not the objective but (i) the acquired unit is a REST-op acquisition skill over a
*discovered* action surface, and (ii) the in-context history is seeded by an explicit,
schema-bound, evidence-grounded, verifier-gated ToolCard — where vanilla
AD/Decision-Transformer/in-context-RL start from raw `(s,a,r)`. Versus Voyager, a ToolCard
is a grounded declarative spec adversarially checked against RewardVerifier/observed evidence, not
self-verified executable code; versus Toolformer, adaptation to a *new* tool is
inference-time with zero weight update and the knowledge is an inspectable, RewardVerifier-verified
data structure, not latent weights; versus RAP / ReAct / Reflexion, the card is the
structured, schema-typed, verifier-gated counterpart that flows into a learned pointer
Q-score plus argument-enum pruning *and* is distilled into weights.

Honest risks: (1) hallucinated knowledge mis-ranking — mitigated by the four gates +
grounded-only up-ranking + the `.npy`/ActionInfo override + potential-based shaping. (2)
**AD-transfer-to-held-out** and **"the enriched candidate key measurably lifts an unseen
op's rank"** are **hypotheses**, load-bearing and proven only by the curve eval. (3)
**Enum-tightening value is narrow**: where `@Redfish.ActionInfo` was captured, enums
already live in `spec.arg_schema[op][slot]['enum']` and already feed `arg_slots_for`, so
the card's enum term adds value mainly where capture is *missing* — ablation (3) must
isolate this honestly. (4) Card-as-context could become a test-time crutch — measured by
the teacher-free vs teacher-in-loop ablation. (5) Safety rides on the **still-queued
Phase-7 guardrail**. (6) The **WorldModel precondition seed is deferred** until WorldModel exists (§4
stage 3); it is not load-bearing for the contribution. Cards are dataset artifacts
(gitignored like `~/.json_responses`), never committed.
