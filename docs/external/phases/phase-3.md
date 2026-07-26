# Phase 3: Method And Argument Extraction

Phase 3 initializes `argument_extractor` from the promoted Phase 2 `goal_extractor`. It receives the
same accepted `D1` text, the Phase 2 REST API set, and current API evidence. It emits one unordered
call for every input API. It does not choose new APIs or determine execution order.

The machine-readable authority is `configs/contracts/rest_goal.yaml`.

## Concrete Bindings

| Role | Binding |
| --- | --- |
| Row builder/parser | `igc.ds.rest_goal_contract.build_call_row` / `parse_calls_y_pred` |
| Master/view renderer | `build_d1_master_record` / `render_d1_master_views` in the same module |
| Grounded release tool | `scripts/build_phase3_grounded_dataset.py` |
| Stored training dataset | `igc.ds.sft_dataset.PromptCompletionJSONLDataset` |
| Renderer | `igc.ds.rest_goal_contract.render_phase3_sft` |
| Trainer | `igc.modules.train.sft.SFTTrainer` |
| Parent checkpoint | promoted Phase 2 `goal_extractor` |
| Output checkpoint | `argument_extractor` |
| Training profile | `phase3_7b_rslora_r32` in `configs/training/profiles.yaml` |
| Task/prompt spec | `text_and_rest_api_list_to_calls` in `configs/training/sft_tasks.yaml` |
| Machine contract | `configs/contracts/rest_goal.yaml` |
| Promotion gate | `configs/inference/phase3_argument_extractor_promotion.yaml` |

## Row Contract

```json
{
  "phase": 3,
  "source_dataset": "D1",
  "task": "text_and_rest_api_list_to_calls",
  "target_semantics": "unordered_unique_call_set",
  "x": {
    "text": "set x to 1 and report status",
    "rest_api_list": [
      "/api/configuration",
      "/api/status"
    ],
    "api_context": [
      {
        "rest_api": "/api/configuration",
        "allowed_methods": ["GET", "PATCH"],
        "operation_names": [],
        "argument_schema": {"x": {"type": "integer"}},
        "json": {"x": 0}
      },
      {
        "rest_api": "/api/status",
        "allowed_methods": ["GET"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"state": "ready"}
      }
    ]
  },
  "y_true": {
    "calls": [
      {
        "rest_api": "/api/configuration",
        "http_method": "PATCH",
        "operation_name": null,
        "arguments": {"x": 1}
      },
      {
        "rest_api": "/api/status",
        "http_method": "GET",
        "operation_name": null,
        "arguments": {}
      }
    ]
  }
}
```

`allowed_methods` is input evidence and never an output field. `http_method` is explicit and must be
legal for that API. `operation_name` is an explicit string or `null`. `arguments` is always an
object; `GET` and `HEAD` require `{}`, while mutation arguments must be explicitly labelled and
grounded in the operator text plus API schema/action evidence. A current JSON value alone is not
proof that it is a valid mutation argument.

An argument is the concrete value binding applied by a call, analogous to `variable = value`:

| Operator intent | Method | Arguments |
| --- | --- | --- |
| set the management address to `192.168.1.1` | `PATCH` | `{"Address": "192.168.1.1"}` |
| set the admin password to `admin` | `PATCH` | `{"Password": "admin"}` |
| set NTP to `ntp.pool.org` | `PATCH` | `{"NTPServers": ["ntp.pool.org"]}` |

These values must come from the operator text and the selected API's schema/action evidence. The
resource JSON may describe the current value, but it cannot supply the requested new value.

The call set is keyed by `rest_api`. List position has no semantic meaning. There must be exactly one
call per Phase 2 API, including when the input contains only one API. Missing, extra, duplicate, or
unknown APIs are contract failures.

## Grounded Dataset Views

The judged Phase 2 row supplies the accepted text and selected API set, but it does not authorize a
method or mutation value. A separate explicit call-label record binds that stable D1 `row_id` to a
method, operation name, arguments object, and argument-value grounding evidence for every selected
API. Non-empty arguments must cite `operator_text` plus `argument_schema` or
`operation_definition`; arbitrary labels and `current_json` are forbidden.

`scripts/build_phase3_grounded_dataset.py` joins complete immutable D1 and call-label inputs, builds
one private `d1_master.v1` record, and renders strict Phase 2 and Phase 3 views. The selected API
identity exists once in the master call keys. Publication succeeds only when:

```text
set(phase2.y_true.rest_api_list)
== set(call.rest_api for call in phase3.y_true.calls)
```

The release directory is published atomically and contains master, Phase 2, and Phase 3 JSONL plus
per-view manifests. Both SFT phases verify exact artifact SHA, row count, `immutable=true`, and
`complete=true` before loading data.

## Promotion

The promotion gate reads `configs/inference/phase3_argument_extractor_promotion.yaml`. Required
metrics are:

- JSON parse rate
- call-set exact match
- REST API coverage exact match
- method exact match and invalid-method rate
- operation-name exact match
- argument exact match, schema-valid rate, and value-grounding rate
- missing, extra, and duplicate call rates
- read-only empty-arguments rate
- breakdowns by `k=1`, `k=2`, `k=3`, and argument class
- held-out coverage of `GET`, `HEAD`, `PATCH`, `POST`, and `DELETE`, including the required
  scalar/nested/no-argument/one-argument/multiple-argument cases

Promotion also requires a real promoted Phase 2 parent, immutable real heldout evidence, checkpoint
reload, an inference smoke, finite metrics, and exact artifact and code lineage.

## Inference Handoff

The runtime handoff preserves the same contract:

```json
{
  "text": "check the task queue and list the available systems",
  "rest_api_list": [
    "/redfish/v1/TaskService/Tasks",
    "/redfish/v1/Systems"
  ],
  "calls": [
    {
      "rest_api": "/redfish/v1/Systems",
      "http_method": "GET",
      "operation_name": null,
      "arguments": {}
    },
    {
      "rest_api": "/redfish/v1/TaskService/Tasks",
      "http_method": "GET",
      "operation_name": null,
      "arguments": {}
    }
  ]
}
```

The differing list order is intentional and valid. The RL policy receives the goal call set and
learns execution order, prerequisite actions, retries, waiting, and verification from environment
transitions.

Author:
Mus mbayramo@stanford.edu
