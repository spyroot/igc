# Script Layout

Scripts are grouped by operator intent. Unit tests live under `tests/`; scripts here are launchers,
gates, profilers, corpus producers, or operational sanity checkers.

| Directory | Purpose |
| --- | --- |
| `bmc_corpus/` | Approved Redfish corpus capture and current labelled-request producers. |
| `nv72/` | NV72/GB300 deployment, image, staging, and fleet preflight helpers. |
| `training/` | Training and run-orchestration launchers. |
| `artifacts/` | Checkpoint publishing and Git LFS artifact movement helpers. |
| `gates/` | CI or remote-validation gate entrypoints and policy checks. |
| `profilers/` | CPU/GPU profiling and metrics snapshot helpers. |
| `sanity_checkers/current/` | Current operational sanity checkers; not unit tests. |
| `sanity_checkers/need_refactor/` | Parked legacy helpers kept for refactor reference only. |
| `research/` | Standalone research experiments that are not runtime entrypoints. |

Do not run IGC validation on the operator laptop. Gate and sanity scripts are executed only by
Internal GitLab/Kubernetes or an approved remote GB300/NV72 container.
