# Training & reproducibility

How to reproduce an igc training run end to end: environment, data, launch, experiment
tracking (Weights & Biases), checkpoints, and weight sharing. Cluster topology and hard
cautions live in the private cluster runbook; environment setup is in
[ENVIRONMENT.md](ENVIRONMENT.md).

## 0. Secrets (read first)

Never put a token in a script, a committed file, or chat. Auth comes from the environment:

```bash
export WANDB_API_KEY=...        # from wandb.ai → User → API keys; rotate if ever exposed
export HUGGINGFACE_TOKEN=...    # only if pulling a gated model
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

## 1. Environment

- **Local (CPU) — dev + offline gate:** `conda env create -f environment-dev.yaml && conda activate igc-dev`.
- **GPU (NVL72) — training:** the NGC container `nvcr.io/nvidia/pytorch:26.03-py3` + `docker/requirements-train.txt`
  (installed at job start by the sbatch). Do not rebuild the CUDA `conda-recipe.yaml` on a mac.

## 2. Data

Training reads captured Redfish JSON from `~/.json_responses` (collected by `idrac_ctl`; the
`.npy` map is the binding contract). There is **no shared filesystem** on the cluster, so stage the
data (and the igc checkout) onto the target node's local NVMe, and pin the job to that node with
the scheduler's node-selection flag.

## 3. Experiment tracking (Weights & Biases)

The metric backend is selected by `--metric_report` (the run wires `igc.modules` `MetricLogger`).
The M1 launcher defaults to `wandb`:

```bash
export WANDB_API_KEY=...                 # required for wandb
export WANDB_PROJECT=igc                 # optional; defaults to "igc"
# WANDB_NAME is auto-derived from the model + Slurm job id
```

You will see the usual training-loop curves in the wandb run: training/val **loss**, **perplexity**,
token **accuracy**, **grad-norm**, **learning rate**, tokens/sec, and GPU memory. For the RL agent
(later phases) the same logger reports reward, success-rate, and episode length.

To use TensorBoard instead, choose the variable for the launcher you are using:
`IGC_METRIC_REPORT=tensorboard bash scripts/run_profile.sh` for the profile wrapper, or
`IGC_REPORT=tensorboard sbatch scripts/train_m1.sbatch` for the sbatch smoke path.

## 4. Launch (first GPU loop — M1 state encoder)

There are two launch routes today:

- `scripts/train_m1.sbatch`, the Slurm smoke launcher, is the normal scheduled GPU path.
- `scripts/run_profile.sh`, the profile-backed wrapper, is for direct named-profile execution.

The profile registry, defined in `igc/modules/train/profiles.py`, is the committed source of truth
for named M1 state-encoder runs. Inspect a profile locally before spending GPU time:

```bash
python -m igc.modules.train.launch --profile m1_gpt2_smoke
python -m igc.modules.train.launch --profile m1_7b_rslora_r32 --print-argv
```

The smoke argv should include `--model_type gpt2` and `--max_train_steps 50`; the rsLoRA argv should
include `--adapter_method rslora`, `--lora_r 32`, and `--lora_alpha 64`.

The `scripts/run_profile.sh` wrapper, committed as the profile-name launcher, resolves `IGC_PROFILE`
through `igc.modules.train.launch`, then supplies `--json_data_dir`, `--output_dir`, and
`--metric_report` from `IGC_DATA_DIR`, `IGC_OUTPUT_DIR`, and `IGC_METRIC_REPORT`:

```bash
IGC_PROFILE=m1_gpt2_smoke \
IGC_DATA_DIR=$HOME/.json_responses \
IGC_OUTPUT_DIR=experiments/m1_gpt2_smoke \
bash scripts/run_profile.sh
```

Use `IGC_SET`, read by `scripts/run_profile.sh` as profile-field overrides, for small controlled
changes such as a shorter smoke or batch-size check:

```bash
IGC_PROFILE=m1_3b_lora IGC_SET="batch_size=16 lr=2e-4" bash scripts/run_profile.sh
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

## 5. Checkpoints & weight sharing

Checkpoints land in `output_dir/run_name/<module>` (node-local). **Weights never go into the igc
repo** (datasets/checkpoints/tokenizers/tarballs are gitignored). Share them through a separate store
with [scripts/publish_checkpoint.sh](../scripts/publish_checkpoint.sh):

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
