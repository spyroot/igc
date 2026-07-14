# Phase 2/3 Mock Dataset Contract

`D0`, the tiny Phase 2 mock row tuple defined in
`igc/ds/mock_inference_contract.py`, covers only inference plumbing. It is not a
trained dataset. Each row keeps the shared Phase 1 split: `x` is the rendered input
context, and `y_true` is the JSON completion that a tokenizer label mask should train
on.

`D1`, the tiny Phase 3 mock row tuple defined in
`igc/ds/mock_inference_contract.py`, consumes the ordered `rest_api_list` from Phase 2
and emits ordered call records. Real Pro-generated `D1` rows wait for a Phase 1
`model_x` checkpoint and a reviewed dataset build.

## Phase 2 Row

```python
{
    "model_x": "model_x",
    "x": {
        "text": "Read /redfish/v1/Managers/1 first, then /redfish/v1/Systems/1.",
        "rest_api": "/redfish/v1",
        "method": "GET",
        "json": [{"@odata.id": "/redfish/v1", "Links": [...]}],
        "allowed_methods": {
            "/redfish/v1": ["GET", "HEAD"],
            "/redfish/v1/Managers/1": ["GET", "HEAD"],
            "/redfish/v1/Systems/1": ["GET", "HEAD"],
        },
    },
    "y_true": {
        "rest_api_list": [
            "/redfish/v1/Managers/1",
            "/redfish/v1/Systems/1",
        ],
    },
    "y_pred": {"rest_api_list": []},
}
```

Contract: `x.text` is the operator sentence, and `x.json` plus `x.method` and
`x.allowed_methods` provide current Redfish context. The target is the ordered
`y_true.rest_api_list`. The mock helper preserves the order expressed by exact URI
mentions in `x.text`; it does not call a model or train an extractor.

## Phase 3 Row

```python
{
    "model_x": "model_x",
    "x": {
        "text": "Read /redfish/v1/Systems/1, then reset with GracefulRestart ...",
        "rest_api": "/redfish/v1/Systems/1",
        "method": "GET",
        "json": [{"@odata.id": "/redfish/v1/Systems/1", "Actions": {...}}],
        "rest_api_list": [
            "/redfish/v1/Systems/1",
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        ],
        "allowed_methods": {
            "/redfish/v1/Systems/1": ["GET", "HEAD"],
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": ["POST"],
        },
    },
    "y_true": {
        "calls": [
            {
                "rest_api": "/redfish/v1/Systems/1",
                "allowed_methods": ["GET", "HEAD"],
                "method": "GET",
                "arguments": {},
            },
            {
                "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                "allowed_methods": ["POST"],
                "method": "POST",
                "arguments": {"ResetType": "GracefulRestart"},
            },
        ],
    },
    "y_pred": {"calls": []},
}
```

Contract: Phase 3 does not choose new REST APIs. It receives `x.rest_api_list` and
predicts ordered `y_true.calls` objects. Each call carries `rest_api`,
`allowed_methods`, selected `method`, and `arguments`. Read-only `GET` calls always use
`arguments={}`.

## Renderer And Labels

`render_model_x`, the prompt renderer defined in `igc/ds/mock_inference_contract.py`,
renders only the `x` context sections: REST API, method, allowed methods, Redfish JSON
input, optional REST API list, and operator sentence.

`render_y_true`, the target renderer defined in `igc/ds/mock_inference_contract.py`,
renders the canonical JSON completion. A tokenizer label mask should mask the
`render_model_x(...)` prompt tokens and compute loss only on `render_y_true(...)`, matching
the Phase 1 training shape.

## Metric Keys

`PHASE23_WANDB_METRIC_KEYS`, the tuple defined in
`igc/modules/base/metric_keys.py`, documents the shared W&B keys planned for Phase 2/3
training and evaluation:

```text
phase2_goal_extract/eval/ordered_exact_match_rate
phase2_goal_extract/eval/set_match_rate
phase2_goal_extract/eval/missing_allowed_methods_rate
phase3_argument_extract/eval/call_ordered_exact_match_rate
phase3_argument_extract/eval/method_exact_match_rate
phase3_argument_extract/eval/readonly_empty_arguments_rate
```

These are string constants only. The mock contract does not import `wandb`, initialize a
run, or log live metrics.
