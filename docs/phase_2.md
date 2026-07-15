# Phase 2: Ordered REST Goal Extraction

Phase 2 builds and trains `D1`, the dataset for extracting Redfish REST goals from an operator
sentence. `model_x` must already have completed Phase 1 Redfish JSON pretraining.

Names are fixed as follows:

- `D0`: Phase 1 JSON reconstruction data.
- `D1`: Phase 2 text-to-ordered-REST-goal data.
- `model_x`: the Phase 1 Redfish-tuned LLM.
- `x`: the input context shown to the model.
- `y_true`: the exact target label stored in the dataset.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete Redfish URI.
- `rest_api_list`: ordered list of concrete Redfish URIs.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `json`: full Redfish JSON resource body.

Phase 2 output is ordered. If the operator sentence says "check tasks, then check systems", the
target order is tasks first and systems second. If the sentence has no explicit order, the dataset
builder still stores a deterministic order and marks that order as weak evidence so evaluation can
separate exact-order accuracy from set-recall.

## D1 Build Input

To build one `D1` row, sample one, two, or three Redfish records from `D0` or the same captured
corpus. Each sampled record must carry its `rest_api`, full `json`, and `allowed_methods`.
The following block is an intermediate synthetic-text generation request, not the final stored row:
`x` is the sampled Redfish context, and `y_pred.text` is the draft operator sentence.

```json
{
  "x": {
    "task": "produce_human_text_for_ordered_rest_api_list",
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
    },
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ]
  },
  "y_pred": {
    "text": "check the task queue, then list the available computer systems"
  }
}
```

`y_pred.text` is only a draft. If `model_x` emits junk text, that text must not enter `D1`. It must
be cleaned or rejected.

## Review Cleanup And Judge

The generated text should be passed through a review/judge step with the same `json`,
`allowed_methods`, and ordered `rest_api_list`. The reviewer has two jobs:

1. rewrite the text into a natural operator sentence if needed;
2. judge whether the sentence asks for all and only the sampled `rest_api_list`, in the intended
   order when the sentence contains ordering language.

Accepted output becomes the final `D1` row. Rejected output is not used for fine-tuning.

## Final D1 Row

This is the row used to fine-tune text-to-REST-goal extraction:

```json
{
  "phase": 2,
  "dataset": "D1",
  "task": "text_to_ordered_rest_api_list",
  "x": {
    "text": "check the task queue, then list the available computer systems",
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
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ],
    "order_evidence": "explicit_then"
  },
  "validation": {
    "text_source": "model_x_then_review",
    "review_judged": true,
    "all_rest_api_present": true,
    "extra_rest_api_present": false,
    "order_preserved": true
  }
}
```

## Fine-Tuning Target

The rendered training target is canonical JSON:

```json
{
  "rest_api_list": [
    "/redfish/v1/TaskService/Tasks",
    "/redfish/v1/Systems"
  ]
}
```

Cross-entropy is computed on the target JSON tokens. Evaluation parses `y_pred`, reads
`y_pred.rest_api_list`, and reports both ordered exact match and set match.

## Phase 2 W&B Metrics

- `phase2_goal_extract/train/loss`
- `phase2_goal_extract/train/perplexity`
- `phase2_goal_extract/train/optimizer_step`
- `phase2_goal_extract/eval/loss`
- `phase2_goal_extract/eval/perplexity`
- `phase2_goal_extract/eval/token_accuracy`
- `phase2_goal_extract/eval/ordered_exact_match_rate`
- `phase2_goal_extract/eval/set_match_rate`
- `phase2_goal_extract/eval/precision`
- `phase2_goal_extract/eval/recall`
- `phase2_goal_extract/eval/f1`
- `phase2_goal_extract/eval/top_k_api_accuracy`
- `phase2_goal_extract/eval/invalid_api_rate`
- `phase2_goal_extract/eval/missing_required_api_rate`
- `phase2_goal_extract/eval/missing_allowed_methods_rate`
- `phase2_goal_extract/eval/order_violation_rate`
- `phase2_goal_extract/order/kendall_tau`
- `phase2_goal_extract/order/edit_distance`
- `phase2_goal_extract/throughput/train_tokens_per_sec`
- `phase2_goal_extract/throughput/train_samples_per_sec`
- `phase2_goal_extract/throughput/eval_tokens_per_sec`
- `phase2_goal_extract/throughput/eval_samples_per_sec`
- `phase2_goal_extract/data/avg_num_apis`
- `phase2_goal_extract/data/max_num_apis`
- `phase2_goal_extract/data/mean_sequence_length`
- `phase2_goal_extract/data/padding_ratio`
- `phase2_goal_extract/calibration/log_prob_per_sequence`
- `phase2_goal_extract/calibration/ece`
- `phase2_goal_extract/test/latency_sec_p50`
- `phase2_goal_extract/test/latency_sec_p95`
- `phase2_goal_extract/test/memory_peak_mb`

When Phase 2 moves beyond mock plumbing, its acceptance gate must mirror Phase 1: approved full
corpora, readable W&B plots, checkpoint/report storage, and reviewed Git LFS artifact metadata.

Author:
Mus mbayramo@stanford.edu
