# igc docs

Architecture, environment, and training-plan notes for `igc` (Infrastructure Goal-Condition
Reinforce Learner).

## Start here

- [ENVIRONMENT.md](ENVIRONMENT.md) — the local CPU setup, current offline smoke gate, Docker test
  image, and GB300 NVL72 training surface.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the generic goal-conditioned tool-use framework design,
  simulator plugin plan, hierarchical workload model, model curriculum, backbone modernization, and
  Phase 0 stabilization notes.

The repository [README](../README.md) stays short and tutorial-oriented. Put deeper design and runtime
material in this directory so setup instructions and architecture details do not drift apart.

## Diagrams

Theme-aware standalone SVGs live under [`diagrams/`](diagrams/):

1. [Five-layer gap map](diagrams/01-five-layer-gap-map.svg) — target layers versus what exists today.
2. [GitHub environment](diagrams/02-github-env.svg) — a non-Redfish `GoalEnvironment` adapter via record/replay.
3. [SQL environment](diagrams/03-sql-env.svg) — a self-simulating environment with transactions as the dry-run guardrail.
4. [Hierarchical workload plan](diagrams/04-hierarchical-workload-plan.svg) — "tune BIOS, boot Ubuntu" as ordered sub-goals plus discovery.
5. [Training curriculum](diagrams/05-training-curriculum.svg) — the six models and their staged training path on the NVL72.
