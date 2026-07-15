# Phase 2: Labelled REST Request Extraction

Legacy notes and old branches may call the Phase 2 text-labelled artifact `D1`.
New code, configs, metrics, and docs use the canonical dataset name
`phase2_labelled_requests`. `model_x` must already have completed Phase 1
Redfish JSON pretraining before real generation runs; offline plumbing may use
tiny fixtures and injected providers only.

Names are fixed as follows:

- `D0`: Phase 1 JSON reconstruction data.
- `phase2_labelled_requests`: Phase 2 text-to-REST-API-set data.
- `model_x`: the Phase 1 Redfish-tuned LLM.
- `x`: the input context shown to the model.
- `y_true`: the exact target label stored in the dataset.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete Redfish URI.
- `rest_api_list`: canonical list of concrete Redfish URIs; correctness is the
  unordered API set unless `order_evidence` is explicit.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `json`: full Redfish JSON resource body.

Phase 2 output is a REST API set extraction target. If the operator sentence
says "check tasks, then check systems", the row records explicit order evidence
for auxiliary order metrics. If the sentence has no explicit order, evaluation
treats `[A, B]` and `[B, A]` as the same target. Empty set equals empty set for
hard-negative or no-action rows.

## Builder Spec

The offline builder spec lives in `configs/phase2_labelled_requests.yaml`.
That spec owns prompt text, `model_x` identifiers, judge route/profile fields,
generation settings, sample widths, W&B namespace/key lists, and acceptance
thresholds. Runtime Python must load those values rather than hardcoding prompt
or model literals.

## Build Input

To build one `phase2_labelled_requests` row, sample one, two, or three Redfish
records from Phase 1 rows or the same captured corpus. Each sampled record must
carry its `rest_api`, full `json`, and `allowed_methods`.
The following block is an intermediate synthetic-text generation request, not the final stored row:
`x` is the sampled Redfish context, and `y_pred.text` is the draft operator sentence.

```json
{
  "x": {
    "task": "produce_human_text_for_rest_api_set",
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

`y_pred.text` is only a draft. If `model_x` emits junk text, that text must not
enter `phase2_labelled_requests`. It must be cleaned or rejected.

## Review Cleanup And Judge

The generated text should be passed through a review/judge step with the same `json`,
`allowed_methods`, and canonical `rest_api_list`. The reviewer has two jobs:

1. rewrite the text into a natural operator sentence if needed;
2. judge whether the sentence asks for all and only the sampled REST API set,
   considering order only when the sentence contains ordering language.

Accepted output becomes the final `phase2_labelled_requests` row. Rejected
output is not used for fine-tuning.

## Final Row

This is the row used to fine-tune text-to-REST-goal extraction:

```json
{
  "phase": 2,
  "dataset": "phase2_labelled_requests",
  "task": "text_to_rest_api_set",
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

Cross-entropy is computed on the target JSON tokens. Evaluation parses `y_pred`,
reads `y_pred.rest_api_list`, and treats unordered set match as the primary
correctness metric. Ordered exact match is auxiliary and applies only when
`order_evidence` is explicit.

## Labelled-Request W&B Metrics

The labelled-request generation builder records these keys under the
`phase2_labelled_requests/*` namespace:

- `phase2_labelled_requests/draft_total`
- `phase2_labelled_requests/accepted_total`
- `phase2_labelled_requests/rejected_total`
- `phase2_labelled_requests/nonsense_rate`
- `phase2_labelled_requests/invalid_json_rate`
- `phase2_labelled_requests/pro_accept_rate`
- `phase2_labelled_requests/rest_api_set_match_rate`
- `phase2_labelled_requests/empty_set_match_rate`
- `phase2_labelled_requests/sample_width/k`
- `phase2_labelled_requests/vendor/source_corpus`
- `phase2_labelled_requests/prompt_spec_version`
- `phase2_labelled_requests/model_x/artifact_sha`
- `phase2_labelled_requests/judge/model`
- `phase2_labelled_requests/judge/profile`

## Goal-Extractor Training W&B Metrics

After accepted labelled-request rows exist, the later `goal_extractor` training
path may emit these legacy trainer metrics. They are separate from
`phase2_labelled_requests/*` generation metrics above.

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
