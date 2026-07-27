# Design Decisions

This file keeps active architecture decisions only. The current phase and checkpoint architecture is
defined by the [architecture overview](../architecture/overview.md) and machine-readable contracts;
superseded experiments remain available in Git history.

## D-004: Stateful Mutation Simulator (2026-07-11)

**Status:** Accepted architecture; implementation is phased.

### Problem

The RL agent must learn multi-step, order-dependent mutation strategies. Each write must evolve the
resource tree so the next observation reflects it. A read-only or stateless simulator cannot teach
prerequisite actions, waits, retries, verification, or recovery.

### Decision

1. The stateful multi-step mutation simulator lives in `igc`. RL-specific reset, determinism,
   batching, transition journals, and reward evidence remain owned by this project.
2. Vendor mutation semantics are data supplied by `redfish_ctl`: order-independent rules matched on
   method, path pattern, and preconditions over current state.
3. `igc` consumes the Redfish corpus, `rest_api_map.npy`, and mutation rules. It does not duplicate
   live Redfish transport or vendor-specific mutation code.
4. Simulator actions emit Redfish-shaped status, JSON, task, and error evidence through the same
   observation contract used by the RL `StateEncoder`.

### Required Behavior

- Declarative rules, not hardcoded vendor callbacks.
- Pending versus current state, including apply-on-reset behavior.
- Async task lifecycle advanced by polling.
- Boot-once reversion, collection create/delete, and power-state transitions.
- Full resource-tree snapshots before and after every step for HER and offline RL relabeling.
- Seed-controlled failure injection for realistic `4xx`/`5xx`, task failure, transient transport
  failure, and successful-status no-op behavior.
- Failures consume a step but do not automatically end the episode; the evaluator verifies final
  state against the goal.
- Deterministic replay for a fixed seed and parity checks across mock, simulator, and approved real
  validation surfaces.

### Risks And Gates

- Semantics duplication is prevented by treating exported mutation rules as the vendor authority.
- Async oversimplification is checked by poll-to-advance task tests.
- Shallow journals are rejected by before/after snapshot gates.
- Simulator drift is measured with fixture parity and separately approved live canaries.

### Build Order

1. Export order-independent mutation rules from `redfish_ctl`.
2. Implement the `igc` rule engine, state store, deterministic faults, and transition journal.
3. Bind simulator transitions to the Gym environment and RL observation/action contracts.
4. Gate fixture parity, recovery behavior, and approved live canaries before RL promotion.
