# igc

Infrastructure Goal-Condition Reinforce Learner (`igc`) is a Python research project for training a
goal-conditioned reinforcement-learning agent to operate Redfish-exposed infrastructure as a Markov
Decision Process. The agent observes Redfish GET/HEAD responses, chooses REST actions, and learns
from captured Redfish data collected by the `idrac_ctl/` git submodule.

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
For Docker, GPU, and cluster details, see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), the runtime
and verification guide.

## What is in this repository

- `igc/ds/` contains the Redfish dataset pipeline, including JSON response loading, tokenizer input
  construction, masked datasets, and the `rest_api_map.npy` loader for the map written by
  `idrac_ctl/` discovery.
- `igc/envs/` contains the mock REST environment and Gym-style wrappers used for offline simulation.
- `igc/interfaces/` contains the REST mapping interface that binds URLs to captured response files.
- `igc/modules/` contains model and training code for the language model, state encoder, value head,
  and RL agent.
- `igc/shared/` contains shared argument parsing and utility code.
- `igc/core/` contains typed contracts for the generic tool-use agent architecture.
- `tests/` contains pytest coverage. Default tests must stay offline: no GPU, network, HuggingFace
  download, live Redfish host, or real `idrac_ctl` crawl.
- `docs/` contains the deeper design and environment material. Start with
  [docs/README.md](docs/README.md).
- `idrac_ctl/` is a git submodule for Redfish data collection. Do not edit it from this repository.

## Data flow

`idrac_ctl/`, the pinned data-collection submodule, discovers Redfish resources and writes captured
JSON to `~/.json_responses/<host>/...`. The same discovery step writes `rest_api_map.npy`, a numpy map
with `url_file_mapping` and `allowed_methods_mapping`. `igc/ds/ds_rest_trajectories.py` loads that map
with `np.load(..., allow_pickle=True).item()`; those keys are the binding contract between the two
repositories.

The default development path does not run discovery. Use captured data or the mock REST server in
`igc/envs/` for offline tests. A live Redfish crawl can affect real management controllers, so it
requires explicit current-task approval, an approved non-production host, credentials passed through
environment variables or flags, and pacing. Never hardcode or print `IDRAC_IP`, `IDRAC_USERNAME`, or
`IDRAC_PASSWORD` values.

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
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Working rules

- Keep README as the quickstart; put detailed setup, architecture, and training material in `docs/`.
- Keep default validation local, offline, and CPU-only.
- Mark GPU, HuggingFace download, slow, or live Redfish tests so they skip by default.
- Do not commit credentials, captured responses, datasets, checkpoints, tokenizers, tarballs, or
  generated raw model output.
- Do not edit `idrac_ctl/` here. Changes to that tool belong in the `spyroot/idrac_ctl` repository,
  followed by a deliberate submodule bump in `igc`.

## Next reading

- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — local CPU env, Docker test image, and GB300/NVL72
  training surfaces.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — target architecture, simulator plugin model,
  training curriculum, and current implementation status.
- [docs/README.md](docs/README.md) — index for the docs directory and diagrams.
