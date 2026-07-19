# IGC Architecture

This document is the current architecture anchor for `igc`. It is intentionally shorter than older
design notes: historical experiments and rejected names belong in [decisions](../roadmap/decisions.md), while
this file names the pieces agents should build now.

## Current Truth

`igc` trains a goal-conditioned RL system for Redfish infrastructure control. The current unlock path
is:

```text
Phase 1: model_x
  Redfish JSON/API/method pretraining.

Phase 2: goal_extractor
  accepted human text -> rest_api_list.

Phase 3: argument_extractor
  same accepted text + rest_api_list + JSON/method evidence -> target_calls.

RL refactor:
  target_calls or latent z_goal + current Redfish observations -> execution strategy.
```

The legacy code was a complete earlier implementation for GPT-2-scale constraints. It is not the
source of truth for the new `D1`/Pro-judge path because that path did not exist when the legacy code
was written. Trace the legacy code to salvage reusable mechanics: dataset loading, encoder heads,
replay/HER, DQN targets, simulator seams, launch/profile infrastructure, and offline tests. Do not
search it for already-finished `D1`, Pro judging, `goal_extractor`, or `argument_extractor`
implementations.

Avoid historical `M1`/`M2`/`M3`/`M4`/`M5`/`M6` aliases in new docs, profiles, metrics, or code. Use
the component names in this document.

## Data And Label Flow

The Redfish corpus starts with machine evidence only:

```text
rest_api
allowed_methods
json response body
```

It does not contain human operator text such as "mount an ISO and boot the server." Phase 2 creates
that missing label at scale:

```text
sample 1-3 REST API records
  -> model_x drafts plausible operator text
  -> Pro judges whether the text maps to exactly that API set
  -> accepted row enters D1
```

Phase 2 then trains in the reverse direction:

```text
x = accepted human text plus optional same-row context
y_true = rest_api_list
```

Cross entropy trains against one canonical serialized JSON target, but Phase 2 correctness is API-set
coverage unless the text explicitly states ordering. `[] == []` is a valid hard-negative/no-action
match.

Phase 3 reuses the same accepted `D1.x.text`. It does not create new text and does not choose new
APIs. It fills method and argument labels for the Phase 2 APIs:

```text
x = text + rest_api_list + allowed_methods + JSON evidence
y_true = calls[{rest_api, allowed_methods, method, arguments}]
```

`arguments` means HTTP request-body fields or Redfish action parameters for that exact selected API
and method. For read-only `GET`/`HEAD`, `arguments` is `{}`.

## Runtime Contract

The normal inference contract after Phase 3 is:

```json
{
  "text": "mount ISO X and boot from it",
  "target_calls": [
    {
      "rest_api": "/redfish/v1/Managers/1/VirtualMedia/CD/Actions/VirtualMedia.InsertMedia",
      "allowed_methods": ["POST"],
      "method": "POST",
      "arguments": {"Image": "X"}
    }
  ]
}
```

`target_calls` are a visible goal specification, not an execution script. The RL policy may need to
read state, wait for tasks, retry transient failures, verify state, or call prerequisite/recovery APIs
that are absent from `target_calls`.

Example: if ISO `Y` is already mounted, a correct strategy for "mount ISO X" may be:

```text
GET VirtualMedia
EjectMedia {}
wait/poll until empty
InsertMedia {"Image": "X"}
GET/verify mounted image is X
```

The simulator and real BMC path must expose the same kind of evidence to the RL system:

```text
HTTP status + Redfish JSON + task/error metadata
  -> ObservationEncoder
  -> compact RL observation
  -> legal action catalog / candidate-action scorer
```

## Post-Phase-3 Goal Representation

After Phase 3, the open research question is how RL should consume the goal. Keep this swappable:

```text
concrete baseline:
  condition RL directly on target_calls

latent experiment:
  encode target_calls into z_goal / z_sub_goal

hybrid:
  use target_calls plus z_goal
```

Concrete `target_calls` are enough for same-vendor/same-surface experiments. For transfer, the same
human intent may map to different Dell, HPE/iLO, or Supermicro REST surfaces. In that case the latent
goal should represent the desired state facts and argument values, while the policy grounds that goal
against the current BMC's legal action catalog.

The default goal representation should be order-invariant. Phase 3 does not know execution order, and
it does not include hidden prerequisite actions such as ejecting an existing ISO before inserting a
new one. RL learns execution order from simulator/real transitions. Optional dependency edges are
allowed only when the text or evidence explicitly carries ordering.

HER depends on this goal layer. Before "fixing HER", the project must define the goal representation,
transition record, and verifier/reward recomputation contract. HER relabels achieved semantic goals
verified from future observations; it must not copy future rewards backward or compare only
vendor-specific REST API strings except in a deliberate same-surface baseline.

## Observation Encoder

The architecture name is `ObservationEncoder` or `RedfishObservationEncoder`, not `StatePooler`.
Pooling is only one possible compression strategy inside the encoder.

The intended shape is:

```text
Redfish JSON/status/method/path evidence
  -> model_x hidden states and/or typed Redfish fact extraction
  -> StateCompressor
  -> graph/history/goal features
  -> compact RL state
```

`StateCompressor` is a pluggable head over model hidden states and structured facts. Valid
implementations include 1D conv, MLP, masked mean, last-token pooling, attention pooling, graph
pooling, or a hybrid. The old mistake to remove is not "1D conv existed"; the mistakes are hardcoded
GPT-2 dimensions, giant flattened `seq_len * H` assumptions, and unclear coupling between encoder
training and the RL loop.

The compact state should carry enough information for decision-making without feeding raw JSON into
the policy at every step:

```text
current resource identity and @odata type
selected graph edges / @odata.id links
allowed methods and action targets
HTTP status and Redfish error class
pending task/job/apply-time state
goal satisfaction and blockers
previous action / previous REST API
retry count and recent failure facts
```

The same encoder must be used for simulator responses and real BMC responses to preserve sim/real
parity.

### StateEncoder v1 — binding contract

This locks the v1 contract for the state encoder. The paper (`docs/external/research/paper/IGC.tex`) calls this component
`StateEncoder`; in this document it is the `ObservationEncoder`/`RedfishObservationEncoder` above. It
emits `z_state` — the paper's `z_t`, the compact RL state that `StateCompressor` produces from a
Redfish observation. Other components may depend only on what this subsection lists; everything else is
an internal implementation detail that may change without notice.

- **Backbone.** v1 is built on the **immutable Phase 1 `model_x` backbone** (the Redfish-aware base
  model produced by Phase 1; see `docs/external/phases/phase-1.md`). v1 does not fine-tune, replace, or reshape it.
- **Output.** v1 emits a **fixed-width `z_state`**. Downstream code may rely on the width being
  constant within an experiment; no per-dimension meaning is part of the contract.
- **Internal pooler is private.** The pooler (`StateCompressor`, the head over backbone hidden states
  and typed Redfish facts introduced above) may be Conv1D/MLP, resource-set pooling, or graph pooling.
  That choice is a v1 experiment, not a public interface.
- **Graph structure is private.** Any graph-pooling variant's **JSON graph structure is an
  implementation experiment, not a public dependency.** Consistent with the paper, `z_state`
  *semantically* represents the currently observable JSON graph, but the specific graph encoding is
  private and no component may couple to it.
- **What `z_state` retains.** **Exact observed state values and API/Redfish error facts MUST influence
  `z_state`; they are not discarded.** This is the state-side mirror of the action side: exact argument
  bindings `b_t` stay *outside* `z_rest`/`z_method` and only parameterize execution and transition
  prediction (see `docs/external/research/paper/IGC.tex` and `docs/external/architecture/goal-latent.md`), whereas observed values and error
  class stay *inside* `z_state`. Concretely, HTTP status, Redfish error class, pending
  task/job/apply-time state, and the concrete field values that distinguish transitions must remain
  recoverable influences on `z_state`.
- **Frozen and version-pinned.** During each RL experiment the encoder is **frozen and
  version-pinned**: its weights and the `model_x` checkpoint do not change mid-experiment, and the
  exact encoder version is recorded with the run so `z_state` is reproducible. The same frozen encoder
  serves simulator and real BMC observations (sim/real parity, above).
- **Selection criterion.** v1 **compactness is chosen by state-sufficiency and RL-performance gates,
  not by reconstruction loss alone.** A smaller `z_state` is accepted only if it still supports the
  policy's decisions and RL return; reconstruction quality is supporting evidence, not the acceptance
  gate.

## RL Refactor Surface

Phase 1/2/3 do not make existing RL curves trustworthy by themselves. After Phase 3, the RL stack must
be refactored around the concrete/latent goal choice and observation-encoder contract:

- **Simulator / mutation model:** Redfish-like JSON output, stateful transitions, hidden
  prerequisites, idempotence, async tasks, stale reads, transient transport failure, and
  vendor-shaped errors.
- **ObservationEncoder:** shared sim/real compression from Redfish JSON/status into compact
  state, graph/history/goal facts, and legal-action features.
- **HER:** relabel from achieved state facts in before/after journals, including partial successes
  such as "old ISO ejected" or "target field already satisfied."
- **DQN / candidate-action scorer:** score all legal actions in the current state, not just
  `target_calls`.
- **TD target and replay:** preserve `terminated`, `truncated`, zero-candidate, non-finite, and
  partial-done metadata correctly.
- **Evaluator / reward:** verify final Redfish state against the intended goal, not a `2xx` or a
  replayed target call.
- **Rollout records:** store target calls, selected action, legal-action catalog identity,
  before/after observations, reward, termination flags, and achieved subgoal facts.

## Component Map

| Component | Role | Current status |
| --- | --- | --- |
| `model_x` | Phase 1 Redfish JSON/API/method model | current training focus |
| `D1` builder | synthetic-but-judged text-label dataset construction | planned after Phase 1 acceptance |
| `goal_extractor` | Phase 2 text to `rest_api_list` model | planned |
| `argument_extractor` | Phase 3 method/argument model | planned |
| `ObservationEncoder` | Redfish JSON/status to compact RL state | legacy mechanics exist; needs refactor |
| `StateCompressor` | compression head inside `ObservationEncoder` | implementation choice, not architecture name |
| `RLPolicy` | execution strategy over legal action catalog | needs SIM/HER/DQN/TD/evaluator refactor |
| `redfish_ctl` corpus | authoritative Redfish discovery/corpus provider | external data contract |

## Build Order

1. Finish Phase 1 acceptance: full-corpus `model_x`, W&B/reconstruction/test evidence, and reviewed
   artifact metadata.
2. Build Phase 2 `D1`: sample 1-3 APIs, generate text with `model_x`, judge with Pro, train
   `goal_extractor`, and evaluate API-set coverage.
3. Build Phase 3: reuse accepted `D1.text`, label method/arguments from API/method/JSON evidence,
   train `argument_extractor`, and evaluate call/method/argument accuracy.
4. Refactor RL around the concrete/latent goal choice: simulator, observation encoder, HER,
   DQN/candidate scoring, TD/replay, evaluator/reward, and rollout records.
5. Only after the refactor, scale rollout/training.

## Safety And Evidence

Default work is offline and CPU-safe. GPU training, W&B readback, live fleet checks, and Redfish
canaries require current-task authorization and must follow the project runbooks. Do not claim a
dataset shape, model quality, RL curve, or cluster state without real evidence.
