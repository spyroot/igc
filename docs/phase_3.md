# Phase 3: Ordered Method And Argument Extraction

Phase 3 starts from accepted Phase 2 rows. Given operator text and the ordered `rest_api_list`, the
model produces the method and arguments for each `rest_api` in the same order.

The executable training contract is `configs/phase_training/profiles.yaml`. Use the
`phase3_argument_extract` phase and one of its profiles, such as `phase3_gpt2_smoke` for a path
smoke or `phase3_qwen2_5_7b_rslora` for the 7B adapter run.

## Dataset Row

Phase 3 does not choose new REST APIs. It fills `method` and `arguments` for the ordered
`rest_api_list` already produced by Phase 2:

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
      {"@odata.id": "/redfish/v1/TaskService/Tasks", "Name": "Task Collection"},
      {"@odata.id": "/redfish/v1/Systems", "Name": "Computer System Collection"}
    ],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
      "/redfish/v1/Systems": ["GET", "HEAD"]
    }
  },
  "y_true": {
    "calls": [
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
}
```

`y_true.calls[i].rest_api` must equal `x.rest_api_list[i]`. Tests should reject reversed or shuffled
calls.

## Non-Empty Arguments

For a write/action resource, `arguments` contains the request body or action parameters. A non-empty
argument row is accepted only when the sampled JSON and method evidence support it: for example an
action target with `@Redfish.ActionInfo`, inline `@Redfish.AllowableValues`, a settings resource, or
recorded successful write evidence.

Do not create mutation arguments from arbitrary scalar values in a GET response. A value observed in
a response body is not proof that the resource accepts that value in a PATCH or POST body.

## Metrics

The canonical metric names are defined in `configs/phase_training/profiles.yaml` under
`metric_sets.phase3_argument_extract`. Required plot families include:

- `phase3_dataset/*` for call counts, method distribution, argument-required rate, and token shape.
- `phase3_argument_extract/*` for loss, perplexity, optimizer step, throughput, padding, and memory.
- `phase3_eval/*` for JSON validity, ordered call match, REST order match, method match, argument
  match, enum validity, hallucinated endpoint/action rate, executable-against-catalog rate, and
  inference latency.

Author:
Mus mbayramo@stanford.edu
