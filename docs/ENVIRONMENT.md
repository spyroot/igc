# Environments: where igc runs

Three surfaces matter for `igc`: local CPU development, the reproducible Docker test image, and the
GB300 NVL72 training cluster. This doc covers environment setup only; the training runbook (data,
launch, W&B, checkpoints) is [TRAINING.md](TRAINING.md).

| Surface | Purpose | How |
| --- | --- | --- |
| Local CPU (mac/linux) | development and offline smoke tests | conda env `igc-dev`, created by `environment-dev.yaml` (repo root) |
| Docker test image | reproducible CPU checks for mac/linux/CI | `docker/Dockerfile.test` |
| GB300 NVL72 cluster | GPU training and fine-tuning | NGC `pytorch:26.03-py3` container plus `docker/requirements-train.txt` via Slurm/pyxis |

## 1. Local CPU dev/test env

The repo's `conda-recipe.yaml` is the CUDA/GPU recipe. It is not the local macOS happy path. For
local CPU work, use `environment-dev.yaml`, which creates `igc-dev` with CPU PyTorch, pytest, ruff,
and mock-REST dependencies:

```bash
conda env create -f environment-dev.yaml
conda activate igc-dev
```

For local smoke runs, set the guard variables in the same shell before invoking pytest.
`KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS`, local shell variables for macOS OpenMP behavior,
avoid duplicate OpenMP loader failures and keep CPU checks single-threaded. For unattended
offline gates, also set `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` — local shell variables
honored by HuggingFace Transformers and Datasets — so the smoke gate does not attempt model or
dataset downloads:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The explicit smoke checks:

```bash
python -m pytest -q tests/core
ruff check igc/core tests/core
```

A green local run currently ends with `29 passed` for `tests/core` and `All checks passed!` from
`ruff`.

The integration gate is `pytest -q` plus `ruff check <changed files>` once GPU/download/live tests
are marked and the shared fixtures are in place. Until then, new tests should name their safe node
ids or files directly and must not require a GPU, network, HuggingFace download, live Redfish host,
or real `redfish_ctl` crawl.

## 2. Reproducible test image

`docker/Dockerfile.test`, the repo-local CPU test image, installs Python 3.10, CPU PyTorch wheels,
pytest, ruff, and mock-REST dependencies.

```bash
docker build -f docker/Dockerfile.test -t igc-test:cpu .
docker run --rm igc-test:cpu python -m pytest -q tests/core
docker run --rm igc-test:cpu ruff check igc/core tests/core
```

The Docker build context is trimmed by `.dockerignore` (repo root), which excludes datasets,
tarballs, `.npy` files, checkpoints, and local-only coordination files.

## 3. Training on the GB300 NVL72

Training runs on the cluster, not on the local CPU env. The training base is the NGC image already
pulled on the nodes; igc's extra training dependencies live in `docker/requirements-train.txt`.

```bash
srun \
  --partition=debug \
  --gres=gpu:1 \
  --exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16 \
  --container-image=nvcr.io/nvidia/pytorch:26.03-py3 \
  --container-mounts=$HOME/igc:/workspace/igc,$HOME/data:/data \
  --time=02:00:00 \
  --pty \
  bash
cd /workspace/igc && pip install -r docker/requirements-train.txt
NCCL_NVLS_ENABLE=0 accelerate launch --num_processes 1 igc_main.py --train ...
```

Start with one GPU. Always exclude slots 2, 15, and 16, set `NCCL_NVLS_ENABLE=0`, and stage large
artifacts on the shared BeeGFS filesystem mounted at `/models` on every node. Use node-local NVMe
only for short-lived scratch data. Do not report a training run as successful without the exact
command, logged metric, and artifact location.

## Verification split

- **Pure-Python checks:** run locally in `igc-dev` or base Python for dataclasses, protocols, codecs,
  and other stdlib-only logic.
- **Torch-importing offline checks:** run in `igc-dev` or `docker/Dockerfile.test`; keep them CPU-only.
- **GPU training/eval:** run on the NVL72 and mark any corresponding tests as opt-in.
