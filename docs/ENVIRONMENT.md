# Environments: where igc runs

Three surfaces, by purpose. Local is for development + the offline gate; the GB300 NVL72 is for
training/fine-tuning; Flash is the code-generation helper (not a training target).

| Surface | Purpose | How |
| --- | --- | --- |
| Local CPU (mac/linux) | dev + offline gate (`pytest -q`, `ruff`) | conda env `igc-dev` (`environment-dev.yaml`) or `docker/Dockerfile.test` |
| GB300 NVL72 cluster | training / fine-tuning (GPU) | NGC container `nvcr.io/nvidia/pytorch:26.03-py3` + `docker/requirements-train.txt`, via Slurm/pyxis (`GPU_ACCESS.md`) |
| Flash endpoint | offline code/test/doc generation | `deepseek-v4-flash` over VPN; draft-only (`FLASH_BRAIN.md`) |

## 1. Local CPU dev/test env (the offline gate)

The repo's `conda-recipe.yaml` is CUDA-pinned (pytorch cuda mutex, `cuda-toolkit`, `cuda-nvcc`) and is
the **cluster/GPU** recipe — it does **not** build on a mac. For local development and the offline
gate, use the CPU env instead:

```bash
conda env create -f environment-dev.yaml      # creates env `igc-dev` (CPU torch, pytest, ruff, mock-REST deps)
conda activate igc-dev
pytest -q                                      # offline subset: no GPU/network/HF-download/live host
ruff check igc tests
```

Notes:
- `numpy` is pinned `<2`: the legacy `igc` conda env on this machine has numpy 2.x, which breaks
  `accelerate` / `scikit-learn` / `nltk` / `evaluate` imports. `igc-dev` is clean.
- Pure-stdlib modules (e.g. `igc/core/**`) can also be gated with the base `python3` directly
  (`python3 -m pytest tests/core -q`) — no heavy env needed.
- Heavy/opt-in tests (GPU, HF download, live Redfish) are marked `@pytest.mark.gpu/slow/download/live`
  and skip by default.
- **macOS only:** importing `torch` together with `scikit-learn`/`nltk` aborts with `OMP: Error #15`
  (duplicate libomp on Apple Silicon). Export `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` before such
  runs; the offline gate sets this in `tests/conftest.py`. Not needed in the Linux `docker/Dockerfile.test`.

## 2. Reproducible test image (mac/linux CI)

```bash
docker build -f docker/Dockerfile.test -t igc-test:cpu .
docker run --rm igc-test:cpu                    # runs `pytest -q`
docker run --rm igc-test:cpu ruff check igc tests
```

CPU-only (`python:3.10-slim` + CPU torch wheels). Build context is trimmed by `.dockerignore`
(datasets, tarballs, `.npy`, checkpoints, the `idrac_ctl` submodule, and agent docs are excluded).

## 3. Training on the GB300 NVL72

Training runs on the cluster, not locally. The base is the NGC image already pulled on the nodes; igc's
extra deps are in `docker/requirements-train.txt` (installed on top, torch comes from the base).

```bash
# Interactive 1-GPU shell (see GPU_ACCESS.md for the full runbook)
srun --partition=debug --gres=gpu:1 \
     --exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16 \
     --container-image=nvcr.io/nvidia/pytorch:26.03-py3 \
     --container-mounts=$HOME/igc:/workspace/igc,$HOME/data:/data \
     --time=02:00:00 --pty bash
cd /workspace/igc && pip install -r docker/requirements-train.txt
NCCL_NVLS_ENABLE=0 accelerate launch --num_processes 1 igc_main.py --train ...
```

Hard cautions (from `GPU_ACCESS.md`): always `--exclude` slots 2/15/16; start 1-GPU and set
`NCCL_NVLS_ENABLE=0` (the NVLink fabric is flaky under multi-GPU); stage data + the HF cache to
node-local NVMe (no shared filesystem) and pin the job with `-w slotN`. To bake a reusable image
instead of `pip install` at job start, build `docker/Dockerfile.train` and convert it to an enroot
`.sqsh`.

## Verification split (be explicit about where a gate ran)

- **Pure-Python** (`igc/core/**`, codecs, dataclass logic): gate locally on base `python3` or `igc-dev`.
- **Torch-importing offline subset**: gate in `igc-dev` or `docker/Dockerfile.test` (CPU) — not the
  CUDA `conda-recipe.yaml`.
- **Training / eval (GPU)**: runs on the NVL72; never reported as trained/converged without the exact
  command, the logged metric, and the artifact location.
