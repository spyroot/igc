# IGC docs

Start with the root [../README.md](../README.md) for setup and the offline CPU gate. This directory
is the design + runtime authority. In the current design, **Phase 2 and Phase 3 outputs are
unordered**, and **order is separate RL-oracle evidence (`expert_call_order`)**, not part of the
language contract. Redfish is the first proof environment, not a permanent ontology.

The machine-readable schema under `configs/contracts/*.yaml` is authoritative for the Phase 2/3
output shapes; every example in these docs is illustrative only.

## Architecture & contract

- [ARCHITECTURE.md](ARCHITECTURE.md) — the current pipeline: D0 -> Phase 1 -> `model_x` -> D1
  (judge-verified inverse labels) -> Phase 2 `rest_api_list` (unordered unique set) -> Phase 3
  `calls: list[Call]` (unordered, explicit `http_method` + `operation_name` + `arguments`) -> two SEPARATE encoders
  `z_rest` + `z_method` -> a separate RL policy -> JSON simulator.
- [GOAL_LATENT_DESIGN.md](GOAL_LATENT_DESIGN.md) — the `z_rest` / `z_method` separate-latent
  design; exact argument VALUES stay raw/outside both latents.
- [mdp_formulation.md](mdp_formulation.md) — SSP/MDP formulation and convergence proof underpinning
  the separate RL recovery policy.

## Phase runbooks

- [phase_1.md](phase_1.md) — Phase 1 Redfish JSON-reconstruction pretraining for `model_x`.
- [phase_2.md](phase_2.md) — Phase 2 `labelled_requests` builder (unordered `rest_api_list` set
  extraction; order is separate RL-oracle evidence).
- [phase_3.md](phase_3.md) — Phase 3 method/argument binding (unordered `calls: list[Call]`; order
  is separate RL-oracle evidence).

## Training, environment, artifacts

- [ENVIRONMENT.md](ENVIRONMENT.md) — local CPU dev/test, Docker CPU image, GB300 surface.
- [TRAINING.md](TRAINING.md) — training runbook: secrets, corpus staging, launch, W&B, checkpoints.
- [TRAINING_OPTIMIZATION_PLAN.md](TRAINING_OPTIMIZATION_PLAN.md) — Phase 1 profile/adapter/precision
  plan.
- [NODE_ARTIFACTS.md](NODE_ARTIFACTS.md) — node-uplink LFS artifact + Docker image handling.
- [RUN_ORCHESTRATION_PLAN.md](RUN_ORCHESTRATION_PLAN.md) — spec-driven single launch contract.
- [DISTRIBUTED_PLAN.md](DISTRIBUTED_PLAN.md) — DDP vs FSDP2 distributed-training plan.
- [SMOKE_LADDER.md](SMOKE_LADDER.md) — R0..R5 gated promotion ladder.

## Performance, decisions, math

- [HOW_TO_PROFILE.md](HOW_TO_PROFILE.md) — profiling + CI budget one-command flow.
- [CRITICAL_SECTIONS.md](CRITICAL_SECTIONS.md) — measured hot-path map + budgets.
- [DECISIONS.md](DECISIONS.md) — accepted decision log.
- [MATH_CHECKS.md](MATH_CHECKS.md) — numerical-checkout discipline for trainable
  objectives/metrics.
- [REDFISH_ENUM_SPACES.md](REDFISH_ENUM_SPACES.md) — offline argument value-space extraction
  (values stay raw).
- [RL_SCALING_PLAN.md](RL_SCALING_PLAN.md) — separate RL policy scaling + DQN/HER contracts.

## Use cases

- [use_cases/README.md](use_cases/README.md) — illustrative end-to-end Redfish episodes.

## Diagram

- [diagrams/01-current-pipeline.svg](diagrams/01-current-pipeline.svg) — the current-pipeline
  architecture diagram (theme-aware).

The repository [README](../README.md) stays short and tutorial-oriented; this directory is the
entry point for deeper design and runtime material, so setup instructions and architecture details
do not drift apart.
