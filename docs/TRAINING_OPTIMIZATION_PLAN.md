# Training Optimization Plan

This plan keeps the large-model training path from inheriting old GPT-2 and
small-GPU assumptions. It is a code-improvement roadmap for Phase 1 Redfish JSON
pretraining of the M1 backbone/state encoder, GPU efficiency, Redfish data variation,
simulator-backed data generation, and the evaluation gates needed before claiming that a 3B or 7B
backbone is better.

The plan is public-safe by design. Operational secrets, live endpoints, raw
captures, credentials, private node details, and local runbooks stay outside the
repository.

## Purpose

The M1 state encoder is the first large-model representation stage in `igc`. Phase 1 is the
concrete Redfish JSON pretraining/fine-tune that starts that stage: it teaches `model_x` to
reconstruct Redfish resource JSON from `x = {rest_api, allowed_methods, json}` before Phase 2/3
specialize goal and argument extraction. That goal needs a modern large-model path, not defaults
carried over from a GPT-2 smoke test.

Profile names use the `phase1_*` prefix because they are launchable Phase 1 training profiles.
`M1` and `M2` remain architecture-stage names in `docs/ARCHITECTURE.md`.

The intended progression is:

1. prove the training and data path with a tiny model on CPU or one GPU;
2. run a GPT-2 smoke job to catch launch and logging failures cheaply;
3. compare 3B LoRA and 7B LoRA on the same fixture/eval split;
4. try full fine-tuning only when LoRA underfits trusted metrics;
5. scale out across NVL72 for controlled ablations, not one large fragile run.

## Current Limitations

These are the current improvement targets. Some are already configurable, but
the defaults and launchers do not yet make a safe large-model profile obvious.

| Area | Limitation | Risk |
| --- | --- | --- |
| Accelerator path | `--use_accelerator` is opt-in while large-model sharding and mixed precision need it. | A 7B run may silently use the plain one-device path. |
| Launch scripts | Older scripts still encode stale trainer names, fixed epoch counts, or local assumptions. | Operators may run an obsolete path and misread the result. |
| Scheduler | The parser exposes warmup-like knobs, but selected PyTorch scheduler constructors only consume matching argument names. | A run can appear to use warmup or max-step control while the scheduler ignores it. |
| Precision | fp16 is still the legacy default in parts of the parser surface. | GB300 large-model training should start with bf16 unless a measured reason says otherwise. |
| Gradient accumulation | Accumulation is configured, but the trainer loop should enforce it explicitly around backward/step. | Effective batch size and learning-rate schedule can differ from the run spec. |
| Gradient clipping | The current logging path can compute grad norm without applying the parsed clipping limit. | Large-model runs lose a basic stability guardrail. |
| Cache handling | Clearing the CUDA cache inside the hot loop hurts throughput. | Step time is inflated and memory behavior becomes harder to interpret. |
| Profiles | There is no single profile table for smoke, 3B LoRA, 7B LoRA, and full fine-tune controls. | Old small-model defaults become the practical ceiling. |

## Target Training Profiles

The following profile names are planned training-profile concepts. They should
become explicit config or launcher presets before routine 3B/7B training.

| Profile | Purpose | Backbone | Adaptation | First-use gate |
| --- | --- | --- | --- | --- |
| `phase1_gpt2_smoke` | Validate launch, labels, logging, checkpoint write, and data loading cheaply. | GPT-2 class | Full fine-tune is acceptable. | One short run with finite loss and checkpoint output. |
| `phase1_3b_lora` | Fast serious baseline for Redfish JSON reconstruction. | 3B dense decoder | LoRA, bf16 | Held-out loss and structural metrics better than smoke baseline. |
| `phase1_7b_lora` | Default large-backbone Phase 1 target. | 7B/8B dense decoder | LoRA, bf16 | Beats 3B on held-out structure/action/state metrics within a reasonable cost budget. |
| `phase1_3b_full` | Controlled full fine-tune comparison. | 3B dense decoder | Full fine-tune, bf16 | Runs only after LoRA metrics and data splits are stable. |
| `phase1_7b_full_zero3` | Higher-ceiling experiment when LoRA underfits. | 7B/8B dense decoder | Full fine-tune with ZeRO-3/FSDP | Requires measured LoRA underfit and a successful smaller full-FT control. |

Planned profile knobs:

| Knob | Meaning |
| --- | --- |
| `IGC_PROFILE` | Named training profile selected by the launcher. |
| `IGC_MODEL` | Hugging Face model id or local model path. |
| `IGC_USE_PEFT` | Enables PEFT/LoRA for large-backbone adaptation. |
| `IGC_PRECISION` | Compute precision such as `bf16`; fp8 remains opt-in after bf16 is green. |
| `IGC_TORCH_DTYPE` | Model load dtype, expected to be `bfloat16` for large models. |
| `BATCH_SIZE` | Per-device train batch size. |
| `GRAD_ACCUM` | Gradient accumulation steps. |
| `LR` | Optimizer learning rate. |
| `SCHEDULER` | Scheduler family, preferably one with explicit warmup semantics. |
| `WARMUP_RATIO` | Warmup fraction derived into warmup steps after dataloader sizing. |
| `MAX_STEPS` | Hard step cap for smoke and comparison runs. |
| `SHARDING` | `none`, `zero3`, or `fsdp` style model/optimizer sharding. |
| `LORA_R` | LoRA rank. |
| `LORA_ALPHA` | LoRA scaling. |
| `ADAPTER_METHOD` | Planned adapter family, for example `lora`, `rslora`, `dora`, `pissa`, `eva`, or `corda`. |
| `LORA_INIT` | Planned LoRA-family initialization, for example default, PiSSA, EVA, or CorDA. |

## Adapter Strategy

Plain LoRA is the baseline control, not the end of the search. The first 7B
run should use the simplest stable adapter so the data path, labels, metrics,
and launcher are debuggable. After that, adapter variants are a high-value
NVL72 ablation axis.

| Method | Planned role | Why it matters | Gate before adoption |
| --- | --- | --- | --- |
| LoRA | Baseline adapter for 3B/7B. | Simple, widely supported, cheap, and easy to compare. | Must beat GPT-2 smoke and produce stable held-out metrics. |
| rsLoRA | Default candidate for higher-rank sweeps. | Rank-stabilized scaling can make larger ranks less fragile. | Compare rank 16/32/64 against plain LoRA at equal data and steps. |
| DoRA | Higher-quality adapter candidate. | Decomposes magnitude and direction, often improving adaptation at extra memory/runtime cost. | Use after LoRA/rsLoRA baseline; require memory and tokens/sec report. |
| PiSSA | Weight-SVD initialization experiment. | Can start adapters closer to useful subspaces than random/default LoRA init. | Run as an initializer ablation, not a new training objective. |
| EVA | Activation/data-driven initialization experiment. | Uses representative activations, so it may fit Redfish structure sooner. | Requires a fixed calibration dataloader and manifest. |
| CorDA | Context-oriented initialization experiment. | Useful when preserving base knowledge while adapting to a target context matters. | Requires clear mode/config choice and a held-out generalization check. |

The recommended ablation order is:

```text
LoRA rank 16 -> rsLoRA rank 32/64 -> DoRA -> PiSSA/EVA/CorDA initializers
```

Keep the comparison fair: same backbone, same tokenizer, same train/eval split,
same max optimizer steps, same sequence length, and the same metrics. Do not
promote an adapter from training loss alone.

The first serious 7B adapter candidate should be rsLoRA over all major decoder
linear projections:

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

Use this as the `phase1_7b_rslora_r32` candidate after the plain LoRA control is
green. For a PiSSA run, keep the same rank/targets and change only the
initializer, for example `init_lora_weights="pissa_niter_16"` if the installed
PEFT version supports that initializer.

The concrete state-encoder ablation should be:

| Arm | Configuration | Purpose |
| --- | --- | --- |
| A | Frozen base + projection/pooling head only | Measures how much structure the pretrained backbone already exposes without adapting transformer weights. |
| B | LoRA, `r=16` and `r=32`, all linear decoder layers | Baseline PEFT adaptation across attention and MLP projections. |
| C | rsLoRA, `r=32` and `r=64` | Tests whether rank-stabilized scaling improves higher-rank adapters. |
| D | DoRA, `r=16` and `r=32` | Tests quality gain from magnitude/direction decomposition against extra memory/runtime cost. |
| E | PiSSA-LoRA or EVA-LoRA, `r=16` and `r=32` | Tests whether weight-SVD or activation-driven initialization improves sample efficiency. |
| F | QLoRA + PiSSA/LoftQ | Only for memory-constrained runs; do not make this the default when bf16 LoRA fits. |

Each arm must produce the same result bundle: resolved profile, model id, adapter
method, adapter rank, initialization method, data manifest, eval split, max
optimizer steps, sequence length, tokens/sec, peak memory, and metric table.

## Code Improvement Roadmap

1. **Make profiles explicit.**
   Add one source of truth for training profiles and make launchers print the
   resolved profile before training starts. Profile resolution should record the
   final model id, dtype, batch size, accumulation, scheduler, max steps, LoRA
   settings, and sharding mode.

2. **Clean up parser and config semantics.**
   Make large-model defaults conservative: bf16 first, accelerator explicit,
   LoRA opt-in for large backbones, and full fine-tune treated as an experiment.
   Rename or document any scheduler knob that is not consumed by the selected
   scheduler.

3. **Fix trainer-loop mechanics.**
   Enforce gradient accumulation, clip with `max_grad_norm`, step the scheduler
   only when the optimizer steps, honor `max_train_steps`, and avoid per-step
   CUDA cache clearing. Keep the corrected causal-LM label path as the only
   state-encoder training path.

4. **Modernize scheduler and optimizer setup.**
   Prefer explicit warmup plus cosine or constant-with-warmup schedules for 3B
   and 7B runs. Compute total optimizer steps from dataloader size, epochs,
   accumulation, and max-step caps, then pass exact scheduler arguments.

5. **Make precision and sharding visible.**
   Load large backbones with bf16 dtype, report accelerator mixed precision,
   and require ZeRO/FSDP only for full fine-tune or memory pressure. LoRA 7B
   should first prove itself on one GPU.

6. **Unify launchers.**
   Make the main training launcher the blessed path and move stale shell scripts
   into a clearly marked legacy area or remove them after review. Every launch
   should emit the resolved profile and the exact command-equivalent settings.

7. **Improve checkpoint and resume hygiene.**
   Store the resolved profile, source data manifest, git commit, dependency
   versions, and evaluation split id beside each checkpoint. Resume should
   restore model, optimizer, scheduler, scaler/accelerator state, and step count.

## GPU Performance Plan

Performance work should optimize measured bottlenecks, not just occupy hardware.

| Area | Plan |
| --- | --- |
| Precision | Start with bf16. Treat fp8/NVFP4 as a later opt-in after bf16 correctness and metrics are stable. |
| Memory | Use LoRA, gradient checkpointing, smaller per-device batches, and accumulation before sharding. |
| Sequence efficiency | Add packed/chunked sequence training so padding does not dominate Redfish JSON batches. |
| Tokenization | Cache tokenized fixtures by dataset manifest, tokenizer hash, max length, and masking strategy. |
| Data loading | Tune workers, pinned memory, persistent workers, and local storage only after measuring tokens/sec. |
| Logging | Report tokens/sec, samples/sec, step time, optimizer steps, allocated/reserved memory, LR, loss, and grad norm. |
| Multi-GPU | Use NVL72 first for parallel ablations. Use ZeRO-3/FSDP for full fine-tune or larger batches only after one-GPU profiles are green. |

The default 7B sequence should be:

```text
GPT-2 smoke -> 3B LoRA -> 7B LoRA -> multi-GPU smoke -> full fine-tune experiments
```

## Profiling and Custom Kernel Policy

Optimization starts with measurement. Do not write custom CUDA or Triton kernels
until profiling proves that a stable, repeated hot operation remains after the
standard PyTorch/Transformers/PEFT path is configured correctly.

The profiling pass should start on one GPU, then scale to multiple GPUs only
after single-GPU bottlenecks are understood. It must use offline data and must
not require a live Redfish endpoint.

| Profiling area | Required signal |
| --- | --- |
| Step-time breakdown | Dataloader, tokenization/collation, host-to-device copy, forward, backward, optimizer, scheduler, logging, checkpointing. |
| Throughput | Tokens/sec, samples/sec, optimizer steps/sec, and p50/p95 step time. |
| GPU utilization | SM activity, memory allocated/reserved/peak, HBM bandwidth where available, and kernel-vs-Python time. |
| Data path | Tokenization cache hit rate, padding waste, sequence-length distribution, dataloader worker utilization. |
| Distributed path | Per-rank step time, communication time, all-reduce/sharding overhead, and straggler rank. |

Recommended tools:

```text
torch.profiler with NVTX ranges
Nsight Systems for timeline and CPU/GPU overlap
Nsight Compute only after a specific hot CUDA kernel is identified
nvidia-smi dmon or DCGM-style telemetry for coarse utilization
W&B or TensorBoard plots for tokens/sec, memory, and step time
```

Fix high-level bottlenecks before kernel work:

1. remove step-loop cache clearing;
2. cache tokenization;
3. reduce padding with packing/chunking;
4. tune dataloader workers, pinned memory, and local storage;
5. enforce true gradient accumulation;
6. use bf16 model load and compute for large backbones;
7. verify optimized attention backends such as SDPA or FlashAttention where supported;
8. avoid logging/checkpointing syncs in the hot path.

Custom kernels are justified only when the profiler identifies a stable hot op
that is not already covered by fused PyTorch, SDPA/FlashAttention, `torch.compile`,
or a PEFT/optimizer configuration change. Any custom kernel must include:

```text
correctness tests against a PyTorch reference
dtype coverage for bf16/fp16/fp32 as needed
shape coverage for expected sequence and batch sizes
benchmark against the baseline implementation
fallback path when the kernel is unavailable
```

## Data and Simulator Plan

Better state representation depends more on diverse transition data than raw
token volume. Data must be provenance-tagged so training and evaluation can
separate real, replayed, simulated, and public fixture sources.

| Source | Use | Trust boundary |
| --- | --- | --- |
| Real captured Redfish snapshots | Grounding and final held-out evaluation. | Approval-gated collection; scrubbed before any shared use. |
| Replay/cassette traces | Deterministic offline parity tests and transition examples. | Default safe mode after capture. |
| DMTF mockups and emulators | Schema variety, legal actions, error cases, and synthetic transitions. | Schema-valid but not vendor truth. |
| Vendor-flavored simulators | OEM behavior and schema extensions. | Validate against real/replay traces before treating as ground truth. |
| Synthetic resource graphs | Topology/action/state variation and rare edge cases. | Training augmentation, not final truth. |
| Public unit-test fixtures | Expected edge behavior and negative cases. | Convert fixtures to structured transitions; do not train on random source text. |

Normalized training examples should eventually look like:

```text
source_domain
trust_level
schema_version
resource_graph_before
request_or_action
response
resource_graph_after
allowed_methods
expected_semantics
```

The Redfish recorder should capture ordered traces:

```text
timestamp
request method/path/body
response status/body/headers
state_before_hash
state_after_hash
resource graph delta
job/task ids
goal or actor label
provenance tag
```

Live recording remains an approval-gated operation. Replay, scrub, diff, and
export should be offline defaults.

## Evaluation Gates

Training loss is not enough. A profile can only be promoted when it improves
held-out, source-separated metrics.

| Gate | Required signal |
| --- | --- |
| Held-out language loss | Lower validation loss/perplexity on unseen Redfish snapshots. |
| JSON validity | Decoded or reconstructed structures remain parseable and schema-like. |
| Resource reconstruction | Important identity, health, status, link, and action fields are preserved. |
| Action availability | The representation predicts legal actions and allowed methods. |
| Transition prediction | Before/action/after pairs preserve causality and job/task semantics. |
| Embedding retrieval | Similar semantic states cluster while different operational states separate. |
| Downstream policy value | Action ranking or RL evaluation improves on offline traces. |
| Performance budget | Tokens/sec, memory, and wall-clock cost are acceptable for the target profile. |

The adapter experiment should report these metrics, not just training loss:

| Metric | What it answers |
| --- | --- |
| State retrieval accuracy | Does the embedding retrieve the same semantic state across renamed ids, topology variation, and source domains? |
| Next-state prediction / transition consistency | Does the representation preserve action-conditioned state changes, including async job/task phases? |
| Action prediction accuracy | Can a small head recover legal actions, allowed methods, and action targets from the state representation? |
| Linear probe labels | Are useful labels linearly accessible, such as health, power state, pending/applied config, task phase, and resource class? |
| Downstream RL return / planning success | Does the adapter improve offline planning or policy success on held-out traces? |
| Embedding collapse / nearest-neighbor sanity | Do embeddings avoid collapsing to one cluster, and do nearest neighbors make operational sense? |

Promotion rule: an adapter can replace plain LoRA only if it improves at least
one semantic metric without regressing public safety, held-out generalization,
throughput, or memory beyond the target profile budget.

## Report and Plot Suite

Every adapter arm must produce the same tight report. A run without these plots
is a smoke result, not a promotion candidate.

| Case | Required metrics | Required plots | Status |
| --- | --- | --- | --- |
| State retrieval accuracy | recall@1, recall@5, MRR, source-domain split. | retrieval@k bars by source, nearest-neighbor table, embedding UMAP/t-SNE colored by resource class and source. | Planned harness. |
| Next-state prediction / transition consistency | delta-field F1, task/job phase accuracy, one-step error, rollout drift. | transition confusion matrix, error-by-field bars, rollout drift over step horizon. | Planned harness. |
| Action prediction accuracy | top-1/top-3 action accuracy, method accuracy, illegal-action false-positive rate. | top-k accuracy bars, action-class confusion matrix, precision/recall by action family. | Planned harness. |
| Linear probe labels | accuracy/F1 per label, macro-F1, AUROC for binary labels. | per-label F1 bars, probe learning curves, label-support table. | Planned harness. |
| Downstream RL return / planning success | success rate, return, episode length, action invalid-rate. | success-vs-step curves, return curves, episode-length distribution, invalid-action trend. | Blocked on trustworthy offline RL eval. |
| Embedding collapse / nearest-neighbor sanity | cosine distribution, effective rank, cluster purity, duplicate-collapse rate. | pairwise cosine histogram, singular-value spectrum, nearest-neighbor audit table. | Planned harness. |

Each report should include:

```text
run id
profile and adapter arm
model and tokenizer
data manifest
eval split
metric table
plots directory
five best examples
five worst examples
known blockers
```

Plots should be generated from offline artifacts only. Do not require a live
Redfish endpoint, a private host, or raw captured secrets to render a report.

Results must report:

```text
profile
model
data manifest
eval split
command or launcher settings
checkpoint path
metric table
known blockers
```

## Execution Order

1. Add docs and profile definitions.
2. Add offline tests for profile resolution and scheduler argument construction.
3. Fix trainer-loop semantics under tiny CPU/GPT-2 tests.
4. Add throughput and memory telemetry.
5. Add a one-GPU profiling report for the GPT-2 smoke and one adapter smoke.
6. Modernize the blessed launcher around explicit profiles.
7. Run the CPU/offline gate.
8. Run one-GPU GPT-2 smoke.
9. Run 3B LoRA and 7B LoRA on the same data split.
10. Use NVL72 for parallel ablations across LoRA rank, sequence length, learning rate, and data mix.
11. Try full fine-tune only if LoRA underfits trusted metrics and the full-FT control is stable.
12. Consider custom kernels only after profiler evidence survives all higher-level fixes.

## Non-Goals

- Do not make public claims that 7B, full fine-tune, fp8, or NVL72 scaling is
  validated without metric evidence.
- Do not commit raw captures, checkpoints, tokenizers, tarballs, private
  endpoints, or secrets.
- Do not run live Redfish collection as part of a training or test default.
- Do not use simulator-only metrics as proof of real infrastructure behavior.
