# Training & reproducibility

How to reproduce an igc training run end to end: environment, data, launch, experiment
tracking (Weights & Biases), checkpoints, and weight sharing. Cluster topology and hard
cautions live in the private cluster runbook; environment setup is in
[ENVIRONMENT.md](ENVIRONMENT.md).

## 0. Secrets (read first)

Never put a token in a script, a committed file, or chat. Auth comes from the environment:

```bash
export WANDB_API_KEY=...        # from wandb.ai → User → API keys; rotate if ever exposed
export HUGGINGFACE_TOKEN=...    # only if the Hugging Face tooling needs gated-model auth
```

Alternatively keep them in the gitignored `.internal/` (never committed) and source it:

```bash
# .internal/wandb.env   (chmod 600) — project: https://wandb.ai/spyroot/igc
export WANDB_API_KEY=...           # training runs
export WANDB_WEAVE_API_KEY=...     # W&B Weave (LLM tracing / eval)
export WANDB_ENTITY=spyroot
export WANDB_PROJECT=igc
```

`scripts/train_m1.sbatch` auto-sources `$IGC_DIR/.internal/wandb.env` when present (copy it to the
cluster login node alongside the checkout — it is not in git). If a key has been shared anywhere
(chat, a paste, a screenshot), **rotate it** before using it.

For Hugging Face gated models, prefer a staged local model path in `IGC_MODEL`/`--model_type` or a
pre-authenticated Hugging Face cache/session on the training node. The igc loader does not write token
values into repo files.

## 1. Environment

- **Local (CPU) — dev + offline gate:** `conda env create -f environment-dev.yaml && conda activate igc-dev`.
- **GPU (NVL72) — training:** the NGC container `nvcr.io/nvidia/pytorch:26.03-py3` + `docker/requirements-train.txt`
  (installed at job start by the sbatch). Do not rebuild the CUDA `conda-recipe.yaml` on a mac.

## 2. Data

Training reads captured Redfish JSON from a materialized `redfish_ctl` dataset artifact. Use the
corpus workflow in the `redfish_ctl` checkout to pull, strictly verify, and materialize the selected
dataset into a stable vendor/model/capture layout. Then point IGC at the manifest and materialized
root with `--corpus_manifest`, `--corpus_root`, and `--corpus_kind`, which are parser flags defined in
`igc/shared/shared_arg_parser.py`:

```bash
IGC_PROFILE=phase1_gpt2_smoke \
IGC_METRIC_REPORT=tensorboard \
bash scripts/run_profile.sh \
  --corpus_manifest /path/to/redfish_ctl/corpora/manifest.v1.json \
  --corpus_root /path/to/materialized/corpus \
  --corpus_kind dataset \
  --corpus_objective phase1_pretrain
```

The legacy `~/.json_responses` capture layout remains a compatibility input; its `.npy` map is the
binding legacy contract. The cluster has a shared filesystem mounted at `/models` on every node —
stage large shared artifacts (staged weights, checkpoints, built corpora) there, in packed form
(tarballs, JSONL) rather than many small files. Keep per-run scratch (the igc checkout, a run's
working dataset copy) on the target node's local NVMe, and pin the job to that node with the
scheduler's node-selection flag.

## 3. Experiment tracking (Weights & Biases)

The metric backend is selected by `--metric_report` (the run wires `igc.modules` `MetricLogger`).
The Phase 1 profile launcher defaults to `wandb`:

```bash
export WANDB_API_KEY=...                 # required for wandb
export WANDB_PROJECT=igc                 # optional; defaults to "igc"
# WANDB_NAME is auto-derived from the model + Slurm job id
```

Current Phase 1 corpus runs log under the `phase1_finetune/*` namespace when
`--corpus_objective phase1_pretrain` is active. Current RL-agent runs log `epoch_mean_loss`,
`epoch_cumulative_reward`, and `epoch_goal_reached_count`. Treat any metric not listed in
`igc/modules/base/metric_keys.py` as planned instrumentation unless a later change adds that exact
producer.

To use TensorBoard instead, choose the variable for the launcher you are using:
`IGC_METRIC_REPORT=tensorboard bash scripts/run_profile.sh` for the profile wrapper, or
`IGC_REPORT=tensorboard sbatch scripts/train_m1.sbatch` for the sbatch smoke path.

## 4. Launch (Phase 1 JSON pretraining)

There are two launch routes today:

- `scripts/train_m1.sbatch`, the Slurm smoke launcher, is the normal scheduled GPU path.
- `scripts/run_profile.sh`, the profile-backed wrapper, is for direct named-profile execution.

The profile registry, defined in `igc/modules/train/profiles.py`, is the committed source of truth
for named Phase 1 Redfish JSON pretraining runs. In the architecture docs, `M1` still names the
backbone/state-encoder stage; the concrete launch profiles are named `phase1_*` so their data
objective is explicit. Inspect a profile locally before spending GPU time:

```bash
python -m igc.modules.train.launch --profile phase1_gpt2_smoke
python -m igc.modules.train.launch --profile phase1_7b_rslora_r32 --print-argv
```

The smoke argv should include `--profile phase1_gpt2_smoke`, `--corpus_objective phase1_pretrain`,
`--model_type gpt2`, and `--max_train_steps 50`; the rsLoRA argv should include
`--profile phase1_7b_rslora_r32`, `--adapter_method rslora`, `--lora_r 32`, and
`--lora_alpha 64`.

The `scripts/run_profile.sh` wrapper, committed as the profile-name launcher, resolves `IGC_PROFILE`
through `igc.modules.train.launch`, then supplies `--json_data_dir`, `--output_dir`, and
`--metric_report` from `IGC_DATA_DIR`, `IGC_OUTPUT_DIR`, and `IGC_METRIC_REPORT`:

```bash
IGC_PROFILE=phase1_gpt2_smoke \
IGC_DATA_DIR=$HOME/.json_responses \
IGC_OUTPUT_DIR=experiments/phase1_gpt2_smoke \
bash scripts/run_profile.sh
```

Use `IGC_SET`, read by `scripts/run_profile.sh` as profile-field overrides, for small controlled
changes such as a shorter smoke or batch-size check:

```bash
IGC_PROFILE=phase1_3b_lora IGC_SET="batch_size=16 lr=2e-4" bash scripts/run_profile.sh
```

`scripts/run_profile.sh` is the profile-backed local launcher. `scripts/train_m1.sbatch`, the older
Slurm smoke launcher, does not read `IGC_PROFILE`; it still uses `IGC_MODEL`, `EPOCHS`, and
`IGC_USE_PEFT`.

```bash
# on a GB300 login node, with $HOME/igc and $HOME/.json_responses staged on the node
export WANDB_API_KEY=...
sbatch scripts/train_m1.sbatch                                  # gpt2 smoke — validates the path
IGC_MODEL=<large-hf-decoder> EPOCHS=3 sbatch scripts/train_m1.sbatch   # then scale up
squeue --me ; tail -f igc-m1-*.out                              # watch
```

[scripts/train_m1.sbatch](../scripts/train_m1.sbatch) runs `igc_main.py --train llm --llm latent`
on 1 GPU with conservative fabric settings and unavailable nodes excluded by the launcher. **Start
with the `gpt2` smoke** before spending a large model's time — it surfaces any remaining launch
issues cheaply. A *large* model on one GPU needs LoRA — set `IGC_USE_PEFT=1` (LoRA via HF PEFT);
full fine-tune is for the small validation model only.

The code-improvement and GPU-efficiency roadmap for 3B/7B state-encoder training lives in
[TRAINING_OPTIMIZATION_PLAN.md](TRAINING_OPTIMIZATION_PLAN.md). Keep this page as the runnable launch
guide; put deeper optimization decisions in that roadmap.

Knobs (env): `IGC_MODEL`, `EPOCHS`, `SEED`, `IGC_USE_PEFT` (+`LORA_R`/`LORA_ALPHA`), `IGC_DIR`,
`DATA_DIR`, `NGC_IMAGE`, `IGC_REPORT`, `WANDB_PROJECT`, `WANDB_NAME`.

### Backbone, precision, and sharding flags

The first GPU loop above is intentionally a one-GPU smoke. The training CLI defines broader knobs in
`igc/shared/shared_arg_parser.py`; the sbatch wrappers expose only the common path through environment
variables.

Wrapper mappings:

- `IGC_MODEL` maps to `--model_type`, a HuggingFace repo id or local weights path; it defaults to
  `gpt2` for the cheap smoke.
- `IGC_USE_PEFT=1` maps to `--use_peft`, enabling LoRA via HuggingFace PEFT for adapter tuning.
- `LORA_R` and `LORA_ALPHA` map to `--lora_r` and `--lora_alpha`; the remaining LoRA knobs are CLI-only.

CLI-only flags today:

- `--trust_remote_code` allows model/tokenizer Python from a trusted weights repo or local path.
- `--llm_torch_dtype` selects the load dtype; use `bfloat16` for a large GPU run when supported.
- `--lora_dropout` and `--lora_target_modules` tune adapter dropout and target module names.
- `--use_accelerator` builds the Accelerate runtime instead of the plain single-process path.
- `--sharding` accepts `none`, `ddp`, `zero2`, `zero3`, `zero3_offload`, or `fsdp`.
- `--mixed_precision` accepts `fp16`, `bf16`, or `fp8`; sharded runs default to `bf16` when unset.
- `--gradient_accumulation_steps` accumulates micro-batches before an optimizer step.

For a staged local backbone, pass its path through `IGC_MODEL` or `--model_type`, add
`--llm_torch_dtype bfloat16` when appropriate, and enable `--trust_remote_code` only when the model
path is trusted.

Sharding is not just a flag flip in the current sbatch wrappers. `--sharding zero3` or `--sharding fsdp`
needs `--use_accelerator` plus a matching multi-process `accelerate launch` command inside the
container; the checked-in wrappers still start with one process so the smoke path is easy to debug.
Keep the conservative fabric settings from the environment runbook, start with one GPU, and record
the exact launch command and metrics before calling any sharded run successful.

### Profile launcher for tuning

`scripts/run_profile.sh` (documented in §4 above) also passes trailing arguments through to the
trainer — use it when you need knobs that `scripts/train_m1.sbatch` does not expose directly:

```bash
IGC_PROFILE=phase1_gpt2_smoke \
IGC_DATA_DIR=$HOME/.json_responses \
IGC_METRIC_REPORT=tensorboard \
bash scripts/run_profile.sh --gradient_accumulation_steps 4 --num_workers 2
```

For the first large-backbone run, keep `per_device_train_batch_size` small, raise
`--gradient_accumulation_steps` before raising the micro-batch size, and watch the reported GPU memory
and tokens/sec in the selected metric backend. Tune `--num_workers` only if loader throughput becomes
the bottleneck.

### Multi-stage launcher

[scripts/train_igc.sbatch](../scripts/train_igc.sbatch) is the staged curriculum launcher; it writes
stage outputs under `experiments/${IGC_RUN}`. Keep the same `IGC_RUN` for every stage so later stages
can find earlier checkpoints; see the script header for stage names and epoch knobs.

## 5. Checkpoints & weight sharing

Checkpoints are node-local and must not be committed. With `scripts/train_m1.sbatch`, the default
`output_dir` comes from `igc/shared/shared_main.py` and resolves to
`experiments/<model_batch_optimizer_scheduler_lr>/<module>`. With `scripts/train_igc.sbatch`, the
launcher sets `--output_dir experiments/${IGC_RUN}`, so stages land under
`experiments/${IGC_RUN}/<module>`.

Validate the path before publishing:

```bash
find experiments -maxdepth 2 -type d -name state_encoder -print
```

Share weights through a separate store with [scripts/publish_checkpoint.sh](../scripts/publish_checkpoint.sh):

```bash
# a) separate git-lfs weights repo or a synced path:
IGC_WEIGHTS_DEST=/data/igc-weights scripts/publish_checkpoint.sh experiments/<run>/state_encoder
# b) Google Drive (needs `rclone config` once):
IGC_WEIGHTS_DEST=gdrive:igc/weights scripts/publish_checkpoint.sh experiments/<run>/state_encoder
```

It copies the checkpoint plus a `SHA256SUMS.txt` manifest, and (for a git-lfs repo) prints the
`git lfs track` + commit/push commands. The repo's own `datasets/models.json` mirror mechanism is the
other supported distribution path for published artifacts.

## 6. Reproducibility

- The exact parameters of every run are dumped to `parameters.json` (via `save_spec`) alongside the
  checkpoints — keep it with the weights.
- `RunManifest`, `ResultBundle`, and `compare`, defined in `igc/modules/train/report.py`, are the
  library contract for adapter comparisons, not automatic launcher output. Until a reporting command
  exists, the eval/reporting caller creates each `ResultBundle` after metrics are computed, writes
  `report.json` next to the checkpoint artifacts, and compares bundles across LoRA, rsLoRA, and
  full-fine-tune arms.
- `--seed` (default 42) is recorded; set it explicitly for a reproducible run.
- Report results with evidence: the exact command, the logged metric (wandb run URL), and where the
  checkpoint landed — never paste secrets or raw auth responses.
