# Training Optimization Plan

The Phase 1 profile/adapter/precision plan for Redfish JSON pretraining of `model_x`
(the Phase 1 backbone named in [P0_PHASE_WORKFLOW.md](P0_PHASE_WORKFLOW.md)). It keeps
large-model runs from inheriting GPT-2 / small-GPU defaults: every run resolves to a
named, fully-logged profile, and adapter/precision choices are promoted only on
held-out evidence.

Scope: Phase 1 only. Phase 2 (unordered `rest_api_list` set extraction) and Phase 3
(unordered `calls: list[Call]` method/argument binding) are specified in
[phase_2.md](phase_2.md) and [phase_3.md](phase_3.md); their canonical shapes are
frozen by the machine-readable schema under `configs/contracts/` (the contract
authority of [gates.md](gates.md)), and call order is separate RL-oracle evidence
(`expert_call_order`), never a Phase 2/3 output. The `z_rest` and `z_method` encoders
of [GOAL_LATENT_DESIGN.md](GOAL_LATENT_DESIGN.md) are separate later stages, out of
scope here. This doc is public-safe: no secrets, live endpoints, raw captures, or
private node details.

## Progression

1. Prove the training and data path with a tiny model on CPU or one GPU.
2. Run a GPT-2 smoke job to catch launch and logging failures cheaply.
3. Compare 3B LoRA and 7B LoRA on the same data manifest and eval split.
4. Try full fine-tuning only when LoRA underfits trusted held-out metrics.
5. Scale out for controlled ablations, not one large fragile run.

## Target Training Profiles

The profile registry is implemented in `igc/modules/train/profiles.py`
(`PROFILES` / `resolve_profile`); `scripts/run_profile.sh` launches by name via
`IGC_PROFILE=<name>` and prints the resolved config before training. Per-run
hyperparameter values (model id, precision, batch, accumulation, LR, scheduler,
warmup, sharding, sequence length, adapter spec) live in that registry and the YAML
specs under `configs/training/` — this table is the intent map, the registry is
authoritative.

| Profile | Purpose | Backbone | Adaptation | First-use gate |
| --- | --- | --- | --- | --- |
| `phase1_gpt2_smoke` | Validate launch, labels, logging, checkpoint write, data loading cheaply. | GPT-2 class | Full fine-tune acceptable. | One short run with finite loss and checkpoint output. |
| `phase1_3b_lora` | Fast serious baseline for Redfish JSON reconstruction. | 3B dense decoder | LoRA r=16, bf16 | Held-out loss and structural metrics better than smoke baseline. |
| `phase1_7b_lora` | Default large-backbone Phase 1 target. | 7B dense decoder | LoRA r=16, bf16 | Beats 3B on held-out metrics within a reasonable cost budget. |
| `phase1_7b_rslora_r32` | First serious 7B adapter candidate. | 7B dense decoder | rsLoRA r=32, bf16 | Runs after the plain LoRA control is green. |
| `phase1_local` | Same LoRA recipe over a node-local weights dir (`$IGC_MODEL_DIR`, resolved from the environment — no path in committed code). | local | LoRA r=16, bf16 | Same gates as the matching hub-model profile. |
| `phase1_3b_full` | Controlled full fine-tune comparison. | 3B dense decoder | Full FT, ZeRO-3 | Runs only after LoRA metrics and data splits are stable. |
| `phase1_7b_full_zero3` | Higher-ceiling experiment when LoRA underfits. | 7B dense decoder | Full FT, ZeRO-3 | Requires measured LoRA underfit and a stable smaller full-FT control. |

Guardrails baked into the profiles and trainer (not optional per run): bf16
precision and `bfloat16` load dtype for large backbones (fp8/NVFP4 stays opt-in
after bf16 is green), true gradient accumulation enforced around backward/step,
gradient clipping via `max_grad_norm`, scheduler stepped only when the optimizer
steps, no per-step CUDA cache clearing, and warmup computed from actual dataloader
sizing.

## Adapter Strategy

Plain LoRA is the baseline control, not the end of the search. The adapter ablation
axis is `--adapter_method {lora,rslora,dora}` with `--lora_init
{default,pissa,eva,loftq}` (flags defined in `igc/shared/shared_arg_parser.py`;
PEFT construction in `igc/modules/llm/peft_lora.py`). Recommended order:

```text
LoRA rank 16 -> rsLoRA rank 32/64 -> DoRA -> PiSSA/EVA/LoftQ initializers
```

| Method | Role | Gate before adoption |
| --- | --- | --- |
| LoRA | Baseline adapter for 3B/7B. | Must beat GPT-2 smoke with stable held-out metrics. |
| rsLoRA | Default candidate for higher-rank sweeps. | Compare rank 16/32/64 against plain LoRA at equal data and steps. |
| DoRA | Higher-quality candidate at extra memory/runtime cost. | After the LoRA/rsLoRA baseline; requires memory and tokens/sec report. |
| PiSSA / EVA / LoftQ | Initializer ablations only, not new objectives. | Same rank/targets as the control; change only the initializer. |

The first serious 7B adapter candidate is rsLoRA over all major decoder linear
projections — this exact config is the `phase1_7b_rslora_r32` profile and is pinned
by `tests/modules/train/test_profiles.py`:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_rslora=True,
    init_lora_weights=True,
)
```

For a PiSSA arm, keep the same rank/targets and change only the initializer (e.g.
`init_lora_weights="pissa_niter_16"` where the installed PEFT version supports it).
QLoRA/LoftQ arms are for memory-constrained runs only; do not make them the default
when bf16 LoRA fits.

## Fair Comparison and Report Contract

Comparisons are apples-to-apples only when these match across arms — enforced as the
fairness keys of `igc/modules/train/report.py`:

```text
model  tokenizer  data_manifest  eval_split  max_steps  seq_len
```

Every adapter arm emits the same result bundle (the `RunManifest` of `report.py`):
resolved profile, model id, tokenizer, adapter method + rank + init, data manifest,
eval split, max optimizer steps, sequence length, tokens/sec, peak memory, and the
metric table — plus checkpoint path, five best/worst examples, and known blockers.
Reports render from offline artifacts only; no live Redfish endpoint, private host,
or raw captured secrets. Do not promote an adapter from training loss alone.

Promotion rule: an adapter replaces plain LoRA only if it improves at least one
held-out metric without regressing generalization, throughput, or memory beyond the
target profile budget.

## Evaluation Gates

A Phase 1 profile is promoted on held-out, source-separated evidence, not training
loss:

| Gate | Required signal |
| --- | --- |
| Held-out language loss | Lower validation loss/perplexity on unseen Redfish snapshots. |
| JSON validity | Reconstructed structures remain parseable and schema-like. |
| Resource reconstruction | Identity, health, status, link, and action fields preserved. |
| Performance budget | Tokens/sec, memory, and wall-clock acceptable for the target profile. |

Downstream extraction quality is judged in the later phases, against their own
contracts: Phase 2 by unordered `rest_api_list` set match and Phase 3 by method
accuracy, argument JSON validity, and unsafe/unsupported-argument rejection (see
[TRAINING.md](TRAINING.md) §3). Ordering is evaluated only against the separate
RL-oracle `expert_call_order` evidence, never as a Phase 1/2/3 language metric.

## GPU Performance and Profiling

Optimize measured bottlenecks, not intuition. The measurement flow (remote
profiling commands, CI budgets) lives in [HOW_TO_PROFILE.md](HOW_TO_PROFILE.md);
the measured hot-path map and budgets live in
[CRITICAL_SECTIONS.md](CRITICAL_SECTIONS.md). Standard order of fixes before any
kernel work:

1. cache tokenization by manifest/tokenizer/max-length/masking;
2. reduce padding with packing/chunking;
3. tune dataloader workers, pinned memory, local storage — after measuring tokens/sec;
4. verify optimized attention backends (SDPA / FlashAttention) where supported;
5. keep logging/checkpointing syncs out of the hot step loop.

Custom CUDA/Triton kernels are justified only when the profiler shows a stable hot
op not already covered by fused PyTorch, SDPA/FlashAttention, `torch.compile`, or a
PEFT/optimizer configuration change. Any custom kernel ships with:

```text
correctness tests against a PyTorch reference
dtype coverage for bf16/fp16/fp32 as needed
shape coverage for expected sequence and batch sizes
benchmark against the baseline implementation
fallback path when the kernel is unavailable
```

## Data Provenance

Training and evaluation must separate data sources by provenance tag:

| Source | Use | Trust boundary |
| --- | --- | --- |
| Real captured Redfish snapshots | Grounding and final held-out evaluation. | Approval-gated collection; scrubbed before any shared use. |
| Replay traces | Deterministic offline parity tests. | Default safe mode after capture. |
| DMTF mockups / emulators | Schema variety, legal actions, error cases. | Schema-valid but not vendor truth. |
| Public unit-test fixtures | Edge behavior and negative cases. | Never train on random source text. |

Live recording remains an approval-gated operation; replay, scrub, diff, and export
are the offline defaults. Redfish-shaped data here is the current test environment,
not a permanent ontology.

## Non-Goals

- No public claims that 7B, full fine-tune, fp8, or multi-node scaling is validated
  without metric evidence.
- No committed raw captures, checkpoints, tokenizers, tarballs, private endpoints,
  or secrets.
- No live Redfish collection as a training or test default.
- No simulator-only metrics as proof of real infrastructure behavior.
- No planner, scheduler, or curriculum inside Phase 1/2/3 training; ordering,
  prerequisites, retries, waiting, and recovery belong to the separate RL policy
  stage ([RL_SCALING_PLAN.md](RL_SCALING_PLAN.md)).
