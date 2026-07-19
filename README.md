# igc

Infrastructure Goal-Condition Reinforce Learner (`igc`) is a Python research project for training a
goal-conditioned reinforcement-learning agent to operate Redfish-exposed infrastructure as a Markov
Decision Process. The agent observes Redfish GET/HEAD responses, chooses REST actions, and learns
from captured Redfish data produced by `redfish_ctl`.

## Start here: local CPU smoke gate

Use the CPU development environment first. `environment-dev.yaml`, the repo-local conda file,
creates `igc-dev` with CPU PyTorch, pytest, ruff, and the mock-REST dependencies needed for the
offline gate.
The local smoke gate is the first check before broader tests.

```bash
conda env create -f environment-dev.yaml
conda activate igc-dev
```

On macOS, set the OpenMP guard before tests that import Torch together with scientific Python
packages:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
```

Run the current local smoke gate:

```bash
python -m pytest -q tests/core
ruff check igc/core tests/core
```

Success means `pytest` exits 0 for `tests/core` and `ruff` exits 0 for `igc/core` and `tests/core`.
Until the broader harness and markers are fully in place, keep new tests explicit, offline, and
CPU-only.
Docker, GPU, and cluster details live in the private operator runtime guide under
`docs/internal/environment.md` when the private context is present.

## What is in this repository

- `igc/ds/` contains the Redfish dataset pipeline, including JSON response loading, tokenizer input
  construction, masked datasets, and the `rest_api_map.npy` loader for the map written by
  `redfish_ctl` discovery.
- `igc/envs/` contains the mock REST environment and Gym-style wrappers used for offline simulation.
- `igc/interfaces/` contains the REST mapping interface that binds URLs to captured response files.
- `igc/modules/` contains model and training code for the language model, state encoder, value head,
  and RL agent.
- `igc/shared/` contains shared argument parsing and utility code.
- `igc/core/` contains typed contracts for the generic tool-use agent architecture.
- `tests/` contains pytest coverage. Default tests must stay offline: no GPU, network, HuggingFace
  download, live Redfish host, or real `redfish_ctl` crawl.
- `docs/` contains the deeper design and environment material. Start with
  [docs/README.md](docs/README.md).

## Data flow

The preferred input is a materialized `redfish_ctl` dataset artifact. Use the corpus workflow in the
`redfish_ctl` checkout to pull, verify, and materialize selected archives into a stable
vendor/model/capture layout. IGC consumes that output through `--corpus_manifest`,
`--corpus_root`, and `--corpus_kind`, which are defined in `igc/shared/shared_arg_parser.py`:

```bash
IGC_PROFILE=phase1_gpt2_smoke \
IGC_METRIC_REPORT=tensorboard \
bash scripts/run_profile.sh \
  --corpus_manifest /path/to/redfish_ctl/corpora/manifest.v1.json \
  --corpus_root /path/to/materialized/corpus \
  --corpus_kind dataset \
  --corpus_objective phase1_pretrain
```

The legacy compatibility input is still supported: `redfish_ctl discovery` writes captured JSON to
`~/.json_responses/<host>/...` and writes `rest_api_map.npy`, a NumPy map with `url_file_mapping` and
`allowed_methods_mapping`. `igc/ds/ds_rest_trajectories.py` loads that map with
`np.load(..., allow_pickle=True).item()`; those keys are the binding legacy contract between the two
repositories. New dataset consumers should prefer `rest_api_map.v1.json`, whose file paths are
relative to the materialized capture root.

The default development path does not run discovery. Use captured data or the mock REST server in
`igc/envs/` for offline tests. A live Redfish crawl can affect real management controllers, so it
requires explicit current-task approval, an approved non-production host, credentials passed through
environment variables or flags, and pacing. Never hardcode or print `REDFISH_IP`,
`REDFISH_USERNAME`, or `REDFISH_PASSWORD` values.

`HUGGINGFACE_TOKEN`, when set by the developer for opt-in model downloads, is not needed for the
local smoke gate and must not be logged or committed.

## How the agent learns

The current Redfish-specific stack is being generalized into a pluggable goal-conditioned tool-use
agent framework. The target architecture keeps Redfish as one environment adapter while adding typed
`Goal`, `Observation`, `ToolAction`, and `Transition` contracts, environment registries, trajectory
recording, evaluators, and runtime guardrails.

Training is intentionally staged. First the backbone language model learns Redfish JSON structure;
then pooling/autoencoder, planner, reward, world-model, and RL policy pieces build on that
representation. The full plan and model curriculum live in
[docs/external/architecture/overview.md](docs/external/architecture/overview.md).

## Working rules

- Keep README as the quickstart; put detailed setup, architecture, and training material in `docs/`.
- Keep default validation local, offline, and CPU-only.
- Mark GPU, HuggingFace download, slow, or live Redfish tests so they skip by default.
- Do not commit credentials, captured responses, datasets, checkpoints, tokenizers, tarballs, or
  generated raw model output.
- Changes to the capture tool belong in the `spyroot/redfish_ctl` repository and should be consumed
  here through a released artifact or an explicit dataset materialization step.

## Next reading

- [docs/external/architecture/overview.md](docs/external/architecture/overview.md) — target architecture, simulator plugin model,
  training curriculum, and current implementation status.
- [docs/README.md](docs/README.md) — index for the docs directory and diagrams.
- Private operator context: `docs/internal/environment.md` for runtime, Docker, GPU, and cluster
  details when available.
