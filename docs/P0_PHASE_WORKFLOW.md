# P0 Phase Workflow

This is the current P0 split for Redfish language-model work. It keeps the immediate pretraining
goal separate from later goal/argument extraction, so Phase 1 can finish without being blocked by
Phase 2 and Phase 3 dataset generation.

Naming rule: current docs use phase names for launch surfaces and component names for weights. Do
not introduce historical model-number aliases for profiles; those aliases were retired because they
hide the dataset objective. Launchable profiles use `phase1_*`, `phase2_*`, and `phase3_*` names,
while model artifacts use explicit roles such as `model_x`, `goal_extractor`, and
`argument_extractor`.

| Phase | Profile prefix | Weight role | W&B namespace | Dataset objective | Checkpoint rule |
| --- | --- | --- | --- | --- | --- |
| 1 | `phase1_*` | `model_x` | `phase1_finetune/*` | `phase1_pretrain` / Redfish JSON reconstruction | writes a separate Phase 1 checkpoint |
| 2 | `phase2_goal_extractor_*` (planned) | `goal_extractor` | `phase2_goal_extraction/*` | `text_to_rest_api_list` | initializes from `model_x` only when requested; never overwrites it |
| 3 | `phase3_argument_extractor_*` (planned) | `argument_extractor` | `phase3_argument_extraction/*` | `text_and_rest_api_list_to_calls` | initializes from `model_x` or `goal_extractor` only when requested; never overwrites either |

Phase 2 and Phase 3 profile names are planned until their trainers land. Do not add live profiles
unless `igc.modules.train.launch.profile_to_argv` can emit a runnable command with a distinct
checkpoint/output directory and W&B namespace.

## Phase 1: Pretrain `model_x`

Purpose: train `model_x` on Redfish JSON reconstruction using full Redfish corpora and same-run
method maps. This creates the Redfish-aware base model used by later phases.

Deliverables:

- Phase 1 dataset renderer: `x = {rest_api, allowed_methods, json}` and `y_true = {json}`.
- Causal-LM labels masked over `x` and active only on the `y_true` JSON completion.
- GPU launch ladder: local/offline unit tests, cheap smoke, short large-model smoke, then full
  Phase 1 run on the approved multi-node GPU training surface.
- W&B metrics under `phase1_finetune/*`, with train/eval/throughput/data/calibration/test
  subgroups.
- Checkpoint and evaluation report showing JSON parse rate, exact-match rate, and `@odata.id`
  match rate.

Acceptance gate:

- Full Redfish JSON pretraining has completed over the approved full corpora, not just fixtures.
- W&B contains readable Phase 1 plots for loss, perplexity, token accuracy, throughput, JSON
  validity, exact reconstruction, calibration, and test-time evaluation.
- The checkpoint and evaluation report are stored in the approved shared model store.
- The repo receives only reviewed artifact metadata through Git LFS pointers; bulky weights are not
  copied back into the source tree.

The first serious model profile should use a 7B instruct backbone with rsLoRA unless a smoke gate
proves that profile cannot run. GPT-2 remains a path smoke only; it is not evidence that the Redfish
representation is useful.

## Phase 2: Build And Train `phase2_labelled_requests`

Purpose: after Phase 1, use `model_x` plus a review/judging pass to build the
canonical `phase2_labelled_requests` dataset, then fine-tune a specialized
model that maps operator text and current Redfish context to a canonical
`rest_api_list`. Older notes may call this artifact `D1`; new configs, code,
metrics, and docs use `phase2_labelled_requests`.

Current P0 scope: mock-dataset plumbing and offline tests only.

Real dataset build waits for the Phase 1 checkpoint. The accepted row shape is:

```json
{
  "phase": 2,
  "dataset": "phase2_labelled_requests",
  "task": "text_to_rest_api_list",
  "x": {
    "text": "check the task queue, then list the available computer systems",
    "json": [],
    "allowed_methods": {}
  },
  "y_true": {
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ]
  }
}
```

Evaluation treats REST API set match as primary; ordered exact match is reported
separately only for rows with explicit order evidence. W&B metrics live under
`phase2_goal_extraction/*`, with train/eval/order/throughput/data/calibration/test
subgroups, while dataset-builder metrics use `phase2_labelled_requests/*`.

## Phase 3: Method And Argument Extraction

Purpose: after Phase 2, fine-tune a specialized model that maps operator text plus the canonical
`rest_api_list` to calls with `rest_api`, `allowed_methods`, `method`, and `arguments`.

Current P0 scope: mock-dataset plumbing and offline tests only.
W&B metrics live under `phase3_argument_extraction/*`, with train/eval/order/throughput/data/calibration/test
subgroups.

The accepted row shape is:

```json
{
  "phase": 3,
  "task": "text_and_rest_api_list_to_calls",
  "x": {
    "text": "check the task queue, then list the available computer systems",
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ],
    "json": [],
    "allowed_methods": {}
  },
  "y_true": {
    "calls": [
      {
        "rest_api": "/redfish/v1/TaskService/Tasks",
        "allowed_methods": [
          "GET",
          "HEAD"
        ],
        "method": "GET",
        "arguments": {}
      },
      {
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": [
          "GET",
          "HEAD"
        ],
        "method": "GET",
        "arguments": {}
      }
    ]
  }
}
```

## Inference Contract

After Phase 2 and Phase 3, a sentence should produce testable JSON:

```json
{
  "text": "check the task queue, then list the available computer systems",
  "target_calls": [
    {
      "rest_api": "/redfish/v1/TaskService/Tasks",
      "allowed_methods": [
        "GET",
        "HEAD"
      ],
      "method": "GET",
      "arguments": {}
    },
    {
      "rest_api": "/redfish/v1/Systems",
      "allowed_methods": [
        "GET",
        "HEAD"
      ],
      "method": "GET",
      "arguments": {}
    }
  ]
}
```

Order is explicit only when the operator text carries order evidence. RL can
still learn execution strategy, retries, and recovery from the environment.

Author:
Mus mbayramo@stanford.edu
