# Training & reproducibility

How to reproduce an igc training run end to end: environment, data, launch, experiment
tracking (Weights & Biases), checkpoints, and weight sharing. Cluster topology and the
hard cautions live in `GPU_ACCESS.md`; environment setup in [ENVIRONMENT.md](ENVIRONMENT.md).

## 0. Secrets (read first)

Never put a token in a script, a committed file, or chat. Auth comes from the environment:

```bash
export WANDB_API_KEY=...        # from wandb.ai → User → API keys; rotate if ever exposed
export HUGGINGFACE_TOKEN=...    # only if pulling a gated model
```

If a key has been shared anywhere (chat, a paste, a screenshot), **rotate it** before using it.

## 1. Environment

- **Local (CPU) — dev + offline gate:** `conda env create -f environment-dev.yaml && conda activate igc-dev`.
- **GPU (NVL72) — training:** the NGC container `nvcr.io/nvidia/pytorch:26.03-py3` + `docker/requirements-train.txt`
  (installed at job start by the sbatch). Do not rebuild the CUDA `conda-recipe.yaml` on a mac.

## 2. Data

Training reads captured Redfish JSON from `~/.json_responses` (collected by `idrac_ctl`; the
`.npy` map is the binding contract). There is **no shared filesystem** on the cluster, so stage the
data (and the igc checkout) onto the target node's local NVMe, and pin the job to that node with
`-w gb300-poc1-slotN`.

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

To use TensorBoard instead: `IGC_REPORT=tensorboard sbatch scripts/train_m1.sbatch`.

## 4. Launch (first GPU loop — M1 state encoder)

```bash
# on a GB300 login node, with $HOME/igc and $HOME/.json_responses staged on the node
export WANDB_API_KEY=...
sbatch scripts/train_m1.sbatch                                  # gpt2 smoke — validates the path
IGC_MODEL=<large-hf-decoder> EPOCHS=3 sbatch scripts/train_m1.sbatch   # then scale up
squeue --me ; tail -f igc-m1-*.out                              # watch
```

[scripts/train_m1.sbatch](../scripts/train_m1.sbatch) runs `igc_main.py --train llm --llm latent`
on 1 GPU with `NCCL_NVLS_ENABLE=0` and slots 2/15/16 excluded. **Start with the `gpt2` smoke** before
spending a large model's time — it surfaces any remaining launch issues cheaply. A *large* model on
one GPU needs LoRA/PEFT (the PEFT integration is a tracked next step); full fine-tune is for the small
validation model only.

Knobs (env): `IGC_MODEL`, `EPOCHS`, `SEED`, `IGC_DIR`, `DATA_DIR`, `NGC_IMAGE`, `IGC_REPORT`,
`WANDB_PROJECT`, `WANDB_NAME`.

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
- `--seed` (default 42) is recorded; set it explicitly for a reproducible run.
- Report results with evidence: the exact command, the logged metric (wandb run URL), and where the
  checkpoint landed — never paste secrets or raw auth responses.
