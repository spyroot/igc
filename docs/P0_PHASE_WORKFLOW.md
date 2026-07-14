# P0 Phase Workflow

This document defines the public Phase 1/2/3 workflow for the Redfish language-model path.
The executable profile, metric, optimizer, PEFT, and distributed-training contract lives in
`configs/phase_training/profiles.yaml`; the docs explain what each phase is allowed to train.

## Shared Contract

All three phases use the same naming convention:

- `x`: the model input rendered from the dataset row.
- `y_true`: the exact target JSON used for supervised learning.
- `y_pred`: the parsed model output during evaluation or inference.
- `rest_api`: one concrete Redfish URI from a captured corpus.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `json`: the full Redfish resource body.

The shared profile file is the gate. A new trainer or launcher must consume that profile shape
instead of introducing a phase-specific profile dialect.

## Phase 1

Phase 1 pretrains `model_x` on Redfish JSON reconstruction. It teaches the model Redfish URI
grammar, resource shape, method context, and JSON completion. It does not train operator-goal
extraction, method/argument extraction, rewards, or RL behavior.

See `docs/phase_1.md`.

## Phase 2

Phase 2 builds and trains `D1`: operator text plus Redfish context to ordered `rest_api_list`.
It preserves explicit operator order when the sentence contains sequence words, and reports ordered
exact match separately from set match.

See `docs/phase_2.md`.

## Phase 3

Phase 3 starts from accepted Phase 2 rows. Given operator text and an ordered `rest_api_list`, it
produces one ordered call per REST API: `rest_api`, `allowed_methods`, selected `method`, and
`arguments`.

See `docs/phase_3.md`.

## Inference Contract

After Phase 2 and Phase 3, a user sentence should produce testable JSON:

```json
{
  "text": "check the task queue, then list the available computer systems",
  "ordered_goals": [
    {
      "rest_api": "/redfish/v1/TaskService/Tasks",
      "allowed_methods": ["GET", "HEAD"],
      "method": "GET",
      "arguments": {}
    },
    {
      "rest_api": "/redfish/v1/Systems",
      "allowed_methods": ["GET", "HEAD"],
      "method": "GET",
      "arguments": {}
    }
  ]
}
```

Order matters in this JSON when the operator text states an order. RL can still learn execution
strategy, retries, recovery, and consequences from state transitions.

Author:
Mus mbayramo@stanford.edu
