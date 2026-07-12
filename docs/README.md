# igc docs

Architecture, environment, and training-plan notes for `igc` (Infrastructure Goal-Condition
Reinforce Learner).

## Start here

- [ENVIRONMENT.md](ENVIRONMENT.md) — the local CPU setup, current offline smoke gate, Docker test
  image, and GB300 NVL72 training surface.
- [TRAINING.md](TRAINING.md) — the reproducible GPU training runbook: secret handling, data staging,
  launch, experiment tracking, checkpoints, and reporting evidence.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the generic goal-conditioned tool-use framework design,
  simulator plugin plan, hierarchical workload model, model curriculum, backbone modernization, and
  Phase 0 stabilization notes.
- [DECISIONS.md](DECISIONS.md) — the design-decision log: each entry records what was decided,
  the alternatives weighed, binding implementation requirements, accepted risks, and the
  experiment that validates it.
- [MATH_CHECKS.md](MATH_CHECKS.md) — the math and optimization gate for Bellman targets, HER,
  replay masks, tensor shapes, finite-gradient checks, and trustworthy training claims.
- [STATE_GRAPH_PLAN.md](STATE_GRAPH_PLAN.md) — the compact state-graph plan
  for `RedfishStateV0`, action-candidate graph features, HER compatibility,
  and validation gates.
- [CRITICAL_SECTIONS.md](CRITICAL_SECTIONS.md) — the performance map: every hot path, its cost,
  what we optimized (with numbers), and the budget tripwire that guards it.
- [HOW_TO_PROFILE.md](HOW_TO_PROFILE.md) — the one-command way to measure hot paths, find the
  critical section, and prove a change is faster; how CI runs it.
- [NODE_ARTIFACTS.md](NODE_ARTIFACTS.md) — pushing large LFS artifacts from a training node over
  its own uplink (never the VPN), git-lfs without OS changes, and the Docker Hub image flow.
- [RUN_ORCHESTRATION_PLAN.md](RUN_ORCHESTRATION_PLAN.md) — the spec-driven Docker and Slurm
  launcher automation roadmap: image reuse, storage, checkpoints, dry-runs, sanity checks, and CI.
- [TRAINING_OPTIMIZATION_PLAN.md](TRAINING_OPTIMIZATION_PLAN.md) — the large-model training,
  optimization, and GPU-efficiency roadmap for 3B/7B state-encoder work.

The repository [README](../README.md) stays short and tutorial-oriented. Put deeper design and runtime
material in this directory so setup instructions and architecture details do not drift apart.

## Diagrams

Theme-aware standalone SVGs live under [`diagrams/`](diagrams/):

1. [Five-layer gap map](diagrams/01-five-layer-gap-map.svg) — target layers versus what exists today.
2. [GitHub environment](diagrams/02-github-env.svg) — a non-Redfish `GoalEnvironment` adapter via record/replay.
3. [SQL environment](diagrams/03-sql-env.svg) — a self-simulating environment with transactions as the dry-run guardrail.
4. [Hierarchical workload plan](diagrams/04-hierarchical-workload-plan.svg) — "tune BIOS, boot Ubuntu" as ordered sub-goals plus discovery.
5. [Training curriculum](diagrams/05-training-curriculum.svg) — the six models and their staged training path on the NVL72.
