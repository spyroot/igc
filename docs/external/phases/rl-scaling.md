# RL Scaling Plan

This plan keeps the RL phase honest before it grows from the current offline
DQN/HER shell into a high-throughput rollout and learner system. It is
public-safe: no private hosts, credentials, internal endpoints, captured
payloads, or operator-only workflow details belong here.

The current code path is still value-based DQN/HER over the vectorized REST
mock environment. Future LLM-rollout or policy-gradient modes are planned
architecture, not implemented training evidence.

## State And Goal Graph Anchor

The compact state and goal graph plan already lives in
[architecture overview](../architecture/overview.md). Its `RedfishStateV0` contract describes a
structured state with resource identity, topology, health/control fields, legal
actions, deferred state, observation metadata, and goal context.

The action-candidate graph feature decision lives in
[decisions](../roadmap/decisions.md). The v1 scope uses text plus graph features for
candidate actions; learned graph-neighborhood embeddings are deferred until the
structured state and offline evaluation contracts are stable.

This RL scaling plan assumes those state and goal graph contracts are the input
surface. It does not replace them.

## Current Safety Finding

The first blocker is a current DQN/HER correctness issue, independent of any
future distributed design.

`IgcAgentTrainer.train_goal`, defined in `igc/modules/igc_train_agent.py`,
receives `terminated` and `truncated` from `VectorizedRestApiEnv.step`, defined
in `igc/envs/rest_gym_batch_env.py`, but does not store those flags in
`episode_experience`. Later, `update_replay_buffer` reconstructs the terminal
mask as:

```text
done = rewards >= 1.0
```

That is correct only for success terminals. A negative terminal, such as a
dead-end server error with reward `-0.5`, is sampled back as `done=0` and
therefore bootstraps through `next_q`. The correct one-step target for a true
terminal is:

```text
target = reward
```

not:

```text
target = reward + gamma * max_a Q_target(next_state, a)
```

This is an engine-lane fix because it changes the trainer/replay tuple contract.
The docs/tests lane pins it with an offline strict-xfail contract until the
engine owner updates the rollout record shape.

## DQN/HER Safety Contracts

Before scaling RL, the current value-learning path needs these contracts green:

- Negative terminals preserve `done=True`.
  Dead ends must not bootstrap just because their reward is below success.
  Current status: strict xfail.
- Truncation is separate from termination.
  Time limits may bootstrap; true terminals may not.
  Current status: partially covered.
- Zero legal actions are terminal dead ends.
  Pointer/candidate mode can produce no legal candidates.
  Current status: strict xfail.
- Non-finite Q-values fail fast.
  NaN/+inf should not be silently treated as a terminal.
  Current status: strict xfail.
- Reward/done shapes are rank-1 `[B]`.
  This avoids accidental broadcast to `[B, B]` targets.
  Current status: strict xfail.
- `done` is binary.
  Values outside `{0, 1}` invert or scale the bootstrap term.
  Current status: strict xfail.
- Partial-done vector env rows preserve batch axes.
  Batched rollout accounting must keep one row per sub-env.
  Current status: strict xfail.

Promotion rule: do not treat DQN/HER curves as trustworthy until these contracts
are passing or explicitly waived in a design review.

## Scaling Principles

The RL phase should scale by producing more useful trajectories, not by making
one enormous policy replica.

1. Use many rollout replicas with modest tensor parallelism, not one 72-GPU
   rollout engine.
2. Keep learner and rollout engines separable when throughput is the priority.
3. Bound queues by items, tokens, bytes, age, and policy-version lag.
4. Synchronize fresh weights through a fast broadcast/refit path, not checkpoint
   files in the hot loop.
5. Batch variable-length records by token budget, not trajectory count.
6. Keep every grouped policy update homogeneous by policy version and tokenizer
   identity.
7. Use the same tokenizer, chat template, EOS handling, padding, generation
   config, and attention masks on rollout and learner paths.
8. Recompute and compare rollout logprobs against the learner before trusting a
   policy-gradient loss.
9. Scale dispatchers per node or per shard; do not let one Python or HTTP
   front end feed every rollout replica.
10. Profile one full learner/rollout/reward/update cycle before scaling to 4,
    8, or 72 GPUs.

## Target Architecture

The future architecture has four independent loops with explicit contracts:

- Rollout workers generate trajectories from the current policy against
  replayable simulators or approved gated environments.
  Required contract: stamp every record with policy version, tokenizer
  fingerprint, generation config, env id, seed, token count, and created learner
  step.
- The bounded queue buffers fresh trajectory records between rollout and
  learner.
  Required contract: enforce max items/tokens/bytes, max age, max
  policy-version lag, and deterministic drop/backpressure policy.
- The learner consumes fresh records, recomputes logprobs or Q-targets, updates
  weights, and publishes a new policy version.
  Required contract: reject records whose version, tokenizer identity, masks, or
  prompt group membership are incompatible with the update.
- The weight publisher moves updated weights to rollout workers.
  Required contract: use a measured fast path such as NCCL broadcast or runtime
  refit when available; checkpoint files are for persistence/resume, not
  hot-loop sync.

For the current DQN/HER path, the same shape applies with Q-learning records:
`policy_version` and `created_step` still matter for freshness, even if replay
is intentionally off-policy.

For a future grouped policy-gradient path, prompt groups must be homogeneous:

```text
same policy_version
same tokenizer_fingerprint
same chat_template_hash
same generation_config_hash
same prompt_group_id
```

Mixed groups are rejected before loss computation.

## Data Model Concepts

These are planned contracts. They are not claims that the implementation exists
today.

```text
RolloutRecord
  env_id
  trajectory_id
  step_index
  policy_version
  created_learner_step
  tokenizer_fingerprint
  chat_template_hash
  generation_config_hash
  prompt_group_id
  input_ids
  attention_mask
  eos_token_id
  action
  reward
  terminated
  truncated
  logprob
  token_count
```

For DQN/HER, `logprob` may be absent and the required fields are state/action/
reward/next-state plus terminal metadata. For policy-gradient modes, `logprob`
and token/mask fields are mandatory.

## Offline Test Plan

The first implementation slice should stay local and deterministic.

- Negative terminal replay mask:
  a reward `-0.5`, `terminated=True`, `truncated=False` record samples back as
  `done=1` and targets `reward`.
- Zero-candidate target:
  `next_q` with width zero returns `reward`, finite, with no bootstrap.
- Non-finite target guard:
  NaN/+inf next-Q values raise a clear error; only all `-inf` masked rows become
  terminal.
- Shape guard:
  `reward`, `done`, and `next_q` batch dimensions are validated before
  arithmetic.
- Partial-done vector env:
  done sub-env rows keep observation/reward/flag axes aligned with `num_envs`.
- Freshness queue:
  old records are dropped or excluded by max age and max policy-version lag.
- Token batch planner:
  variable-length records pack by total tokens and reject or isolate oversized
  records deterministically.
- Prompt-group integrity:
  mixed policy versions or tokenizer fingerprints are rejected before learner
  update.
- Logprob parity:
  a tiny deterministic model produces matching rollout and learner logprobs
  over identical tokens and masks.

The first five tests pin current DQN/HER safety. The remaining tests define the
future rollout/learner boundary and should land with the modules they exercise.

## Profiling Gates

Scaling starts only after one complete cycle is measured:

```text
reset environment
roll out actions
encode observations
compute reward/evaluator signal
enqueue records
sample learner batch
compute target or logprob loss
backward + optimizer step
publish new policy version
```

The report must include:

```text
transitions/sec
tokens/sec where tokens exist
env step p50/p95
encoder p50/p95
mock or simulator dispatch p50/p95
queue depth p50/p95/max
record age p50/p95/max
policy-version lag p50/p95/max
learner samples/sec
H2D/D2H transfer time
peak memory
per-rank skew
```

Only after the one-cycle profile is understood should runs expand to 4 GPUs,
then 8 GPUs, and then the full fleet.

## Execution Order

1. Pin current DQN/HER safety with strict-xfail offline tests.
2. Fix the trainer/replay terminal-mask contract in the engine lane.
3. Turn negative-terminal, zero-candidate, shape, non-finite, and partial-done
   xfails into passing tests.
4. Add rollout profiler JSON summaries to the existing RL sanity/profiling
   scripts.
5. Add replay freshness metadata and bounded-queue tests.
6. Add token-count batch planner tests and implementation.
7. Add tokenizer/config identity and logprob parity contracts if an LLM rollout
   learner is introduced.
8. Run one complete 1-GPU cycle profile.
9. Run 4-GPU and 8-GPU profiles only if the 1-GPU cycle is balanced.
10. Scale farther only with measured queue freshness, per-rank skew, and
    learner/rollout utilization.

## Non-Goals

- Do not refactor the engine in this documentation slice.
- Do not run a live Redfish crawl, GPU job, or distributed training job as part
  of the default tests.
- Do not claim policy convergence or scaling efficiency without a logged command
  and metric artifact.
- Do not make simulator-only success a substitute for held-out replay or live
  approval-gated validation.
