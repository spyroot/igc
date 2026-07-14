# Phase 2: Ordered REST Goal Extraction

Phase 2 builds and trains `D1`, the dataset for extracting Redfish REST goals from an operator
sentence. `model_x` should already have completed Phase 1 Redfish JSON pretraining.

The executable training contract is `configs/phase_training/profiles.yaml`. Use the
`phase2_goal_extract` phase and one of its profiles, such as `phase2_gpt2_smoke` for a path smoke or
`phase2_qwen2_5_7b_rslora` for the 7B adapter run.

## Dataset Build Input

To build one `D1` row, sample one, two, or three Redfish records. Each sampled record must carry its
`rest_api`, full `json`, and same-run `allowed_methods`. A generation/review step can turn that
context into operator text, but only reviewed text enters the final dataset.

The intermediate generation input is not the final row:

```json
{
  "x": {
    "task": "produce_human_text_for_ordered_rest_api_list",
    "json": [
      {
        "@odata.id": "/redfish/v1/TaskService/Tasks",
        "@odata.type": "#TaskCollection.TaskCollection",
        "Name": "Task Collection"
      },
      {
        "@odata.id": "/redfish/v1/Systems",
        "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
        "Name": "Computer System Collection"
      }
    ],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
      "/redfish/v1/Systems": ["GET", "HEAD"]
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

## Final D1 Row

The fine-tuning row maps operator text plus context to an ordered list of REST APIs:

```json
{
  "phase": 2,
  "dataset": "D1",
  "task": "text_to_ordered_rest_api_list",
  "x": {
    "text": "check the task queue, then list the available computer systems",
    "json": [
      {"@odata.id": "/redfish/v1/TaskService/Tasks", "Name": "Task Collection"},
      {"@odata.id": "/redfish/v1/Systems", "Name": "Computer System Collection"}
    ],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
      "/redfish/v1/Systems": ["GET", "HEAD"]
    }
  },
  "y_true": {
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ],
    "order_evidence": "explicit_then"
  }
}
```

If the text has no explicit order, the dataset builder still stores a deterministic order and marks
that order as weak evidence so exact-order accuracy and set-match metrics can be separated.

## Metrics

The canonical metric names are defined in `configs/phase_training/profiles.yaml` under
`metric_sets.phase2_goal_extract`. Required plot families include:

- `phase2_dataset/*` for record counts, list length, method distribution, and token shape.
- `phase2_goal_extract/*` for loss, perplexity, optimizer step, throughput, padding, and memory.
- `phase2_eval/*` for JSON validity, ordered exact match, set F1, endpoint recall, order relation,
  hallucinated endpoint rate, text-interface success, and inference latency.

Author:
Mus mbayramo@stanford.edu
