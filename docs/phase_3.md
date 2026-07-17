# Phase 3: Ordered Method And Argument Extraction

Phase 3 fine-tunes argument extraction. It starts from accepted
`phase2_labelled_requests` rows. Legacy notes may call that Phase 2 artifact
`D1`; new code and docs use the canonical name. Given operator text and the
ordered `rest_api_list` from Phase 2, the model must produce the method and
arguments for each `rest_api` in the same order.

Names are fixed as follows:

- `phase2_labelled_requests`: Phase 2 text-to-REST-API-set data.
- `model_x`: the Phase 1 Redfish-tuned LLM.
- `goal_extractor`: the separate Phase 2 weight role, used upstream to produce `rest_api_list`.
- `argument_extractor`: the separate Phase 3 fine-tuned weight role.
- `profile`: a planned `phase3_argument_extractor_*` training profile.
- `base_weights_role`: `model_x` or `goal_extractor`, whichever the run explicitly initializes from.
- `weights_role`: `argument_extractor`; Phase 3 must not overwrite Phase 1 or Phase 2 checkpoints.
- `x`: the input context shown to the model.
- `y_true`: the exact target label stored in the dataset.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete Redfish URI.
- `rest_api_list`: canonical list of concrete Redfish URIs from `phase2_labelled_requests`.
- `method`: the selected HTTP method for a `rest_api`.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `arguments`: request body or action arguments; `{}` means no arguments.
- `json`: full Redfish JSON resource body.

Phase 3 does not choose new REST APIs. It fills `method` and `arguments` for the ordered
`rest_api_list` already produced by Phase 2.

Checkpoint rule: Phase 3 writes only `argument_extractor` artifacts. A run config must record the
input checkpoint path, the output path for `argument_extractor`, the `phase3_argument_extraction/*`
W&B namespace, and the exact dataset manifest used for training.

## Phase 3 Row From Phase 2

This example continues the accepted `phase2_labelled_requests` row from Phase 2.
Both resources are read-only checks with `GET` and `HEAD` allowed, so the correct
method is `GET` and `arguments` is empty for each one.

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
    "json": [
      {
        "@odata.context": "/redfish/v1/$metadata#TaskCollection.TaskCollection",
        "@odata.id": "/redfish/v1/TaskService/Tasks",
        "@odata.type": "#TaskCollection.TaskCollection",
        "Description": "Collection of Tasks",
        "Members": [],
        "Members@odata.count": 0,
        "Name": "Task Collection"
      },
      {
        "@odata.context": "/redfish/v1/$metadata#ComputerSystemCollection.ComputerSystemCollection",
        "@odata.id": "/redfish/v1/Systems",
        "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
        "Description": "Collection of Computer Systems",
        "Members": [
          {
            "@odata.id": "/redfish/v1/Systems/System.Embedded.1"
          }
        ],
        "Members@odata.count": 1,
        "Name": "Computer System Collection"
      }
    ],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": [
        "GET",
        "HEAD"
      ],
      "/redfish/v1/Systems": [
        "GET",
        "HEAD"
      ]
    }
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

`y_true.calls` preserves the order from `x.rest_api_list`. Evaluation parses `y_pred.calls` and
checks ordered exact match plus per-call `method` and canonical `arguments`.

## Non-Empty Arguments

For a write/action resource, Phase 3 uses the same shape. The only difference is that `arguments`
contains the request body or action parameters. A non-empty argument row is accepted only when the
sampled JSON and method evidence support it: for example an action target with
`@Redfish.ActionInfo`, inline `@Redfish.AllowableValues`, a settings resource, or a recorded
successful write.

The target shape remains:

```json
{
  "y_true": {
    "calls": [
      {
        "rest_api": "/redfish/v1/example/action/target",
        "allowed_methods": [
          "POST"
        ],
        "method": "POST",
        "arguments": {
          "ExampleArgument": "ExampleAllowedValue"
        }
      }
    ]
  }
}
```

Do not create mutation arguments from arbitrary scalar values in `json`. A value observed in a GET
body is not proof that the resource accepts that value in a PATCH or POST body.

## Phase 3 W&B Metrics

- `phase3_argument_extraction/train/loss`
- `phase3_argument_extraction/train/perplexity`
- `phase3_argument_extraction/train/optimizer_step`
- `phase3_argument_extraction/eval/loss`
- `phase3_argument_extraction/eval/perplexity`
- `phase3_argument_extraction/eval/token_accuracy`
- `phase3_argument_extraction/eval/call_ordered_exact_match_rate`
- `phase3_argument_extraction/eval/call_order_correct_rate`
- `phase3_argument_extraction/eval/step_exact_match_rate`
- `phase3_argument_extraction/eval/rest_api_exact_match_rate`
- `phase3_argument_extraction/eval/allowed_methods_exact_match_rate`
- `phase3_argument_extraction/eval/method_exact_match_rate`
- `phase3_argument_extraction/eval/arguments_json_parse_rate`
- `phase3_argument_extraction/eval/arguments_exact_match_rate`
- `phase3_argument_extraction/eval/invalid_method_rate`
- `phase3_argument_extraction/eval/readonly_empty_arguments_rate`
- `phase3_argument_extraction/order/kendall_tau`
- `phase3_argument_extraction/order/edit_distance`
- `phase3_argument_extraction/throughput/train_tokens_per_sec`
- `phase3_argument_extraction/throughput/train_samples_per_sec`
- `phase3_argument_extraction/throughput/eval_tokens_per_sec`
- `phase3_argument_extraction/throughput/eval_samples_per_sec`
- `phase3_argument_extraction/data/avg_num_calls`
- `phase3_argument_extraction/data/avg_arguments_length`
- `phase3_argument_extraction/data/mean_sequence_length`
- `phase3_argument_extraction/data/padding_ratio`
- `phase3_argument_extraction/calibration/log_prob_per_call`
- `phase3_argument_extraction/calibration/ece_method`
- `phase3_argument_extraction/test/latency_sec_p50`
- `phase3_argument_extraction/test/latency_sec_p95`
- `phase3_argument_extraction/test/memory_peak_mb`

When Phase 3 moves beyond mock plumbing, its acceptance gate must mirror Phase 1: approved full
corpora, readable W&B plots, checkpoint/report storage, and reviewed Git LFS artifact metadata.

## Inference Handoff

At normal inference time the chain is:

1. Phase 2 receives operator text plus current Redfish JSON/method context and emits an ordered
   `rest_api_list`.
2. Phase 3 receives the same text plus ordered `rest_api_list` and emits one ordered call per
   `rest_api`.
3. RL receives the ordered REST goals and can still learn environment strategy, retries, recovery,
   and consequences from state transitions.

The combined inference JSON should be easy to test:

```json
{
  "text": "check the task queue, then list the available computer systems",
  "ordered_goals": [
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

## Offline Smoke Profile

`configs/inference/phase3_argument_extractor_smoke.yaml` defines the first
runnable `argument_extractor` smoke profile. It is a tiny offline profile for
the existing ordered-call contract, not a training or GPU launch profile. The
profile locks `weights_role: argument_extractor`, the
`text_and_rest_api_list_to_calls` task, the `render_ordered_call_example`
renderer, and acceptance thresholds for ordered calls, method validity,
argument JSON, and read-only empty arguments.

Run the smoke through `scripts/build_phase3_argument_smoke.py`. In default mock
mode it renders two built-in fixtures: a read-only ordered GET/HEAD sequence and
a PATCH row with explicit arguments. File mode accepts one local fake
`argument_extractor` JSON prediction per fixture via `--predictions-jsonl`.
Both modes write JSONL row evidence plus metrics JSON; neither mode loads model
weights, opens W&B, downloads corpora, calls Redfish, or uses a GPU.

Example:

```bash
python scripts/build_phase3_argument_smoke.py \
  --output-jsonl /tmp/phase3_argument_smoke.jsonl \
  --metrics-out /tmp/phase3_argument_smoke_metrics.json
```

Do not launch the bounded GPU smoke until this profile, runner, and offline
gates pass in the approved remote CPU-only validation environment.

Author:
Mus mbayramo@stanford.edu
