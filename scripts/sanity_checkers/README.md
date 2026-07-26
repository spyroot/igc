# Sanity Checkers

These scripts are operational probes, not unit tests. They are for approved remote execution surfaces
such as GB300/NV72 containers or Internal GitLab/Kubernetes jobs.

- `current/` contains checkers that still match the current Phase 1/2/3 or runtime contracts.
- `need_refactor/` contains legacy helpers that are intentionally retained but not authoritative.
