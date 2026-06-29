# igc docs

Architecture and the generalization/training plan for `igc` (Infrastructure Goal-Condition
Reinforce Learner).

## Contents

- [ARCHITECTURE.md](ARCHITECTURE.md) — the generic goal-conditioned tool-use framework design, the
  pluggable-environment mechanism, the hierarchical workload/planning model, the six-model training
  curriculum, the backbone modernization, the phased roadmap, and the NVL72 infra/monitoring/deploy
  runbook.

## Diagrams

Theme-aware standalone SVGs (light + dark), under [`diagrams/`](diagrams/):

1. [Five-layer gap map](diagrams/01-five-layer-gap-map.svg) — the target layers vs what exists today.
2. [GitHub environment](diagrams/02-github-env.svg) — a non-Redfish `GoalEnvironment` adapter via record/replay.
3. [SQL environment](diagrams/03-sql-env.svg) — a self-simulating env; transactions as the dry-run guardrail.
4. [Hierarchical workload plan](diagrams/04-hierarchical-workload-plan.svg) — "tune BIOS, boot Ubuntu" as ordered sub-goals + discovery.
5. [Training curriculum](diagrams/05-training-curriculum.svg) — the six models and their training stages on the NVL72.
