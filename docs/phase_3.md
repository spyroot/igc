# Phase 3: Method And Argument Extraction

Phase 3 fine-tunes method and argument binding. It starts from accepted `D1` rows (the Phase 2
text-to-REST-goal dataset). Given operator text and the Phase 2 `rest_api_list`, the model binds an
HTTP method and an arguments object to each selected API and emits `calls: list[Call]`. The set of
calls is **UNORDERED**: Phase 3 says *what* to call and *with what*, never *in which order*.

The machine-readable schema under `configs/contracts/*.yaml` (checked into the repo) is the
authoritative definition of the Phase 3 contract. Every example in this document is ILLUSTRATIVE
only; when a doc example and the schema disagree, the schema wins.

Names are fixed as follows:

- `D1`: Phase 2 text-to-REST-goal data (built by the Phase 2 dataset pipeline).
- `model_x`: the Phase 1 Redfish-tuned LLM.
- `goal_extractor`: the separate Phase 2 weight role, used upstream to produce `rest_api_list`.
- `argument_extractor`: the separate Phase 3 fine-tuned weight role.
- `base_weights_role`: `model_x` or `goal_extractor`, whichever the run explicitly initializes from.
- `weights_role`: `argument_extractor`; Phase 3 must not overwrite Phase 1 or Phase 2 checkpoints.
- `x`: the input context shown to the model.
- `y_true`: the exact target label stored in the dataset.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete REST URI.
- `rest_api_list`: the UNORDERED `list[str]` of REST URIs emitted by Phase 2. One selected API is
  still a list of length one, never a scalar.
- `calls`: the UNORDERED `list[Call]` Phase 3 emits. One call is still a list of length one, never a
  scalar or a bare object.
- `Call`: one object `{rest_api, http_method, operation_name, arguments}` — `operation_name` names the action/function when one exists (never inferred) and is `null` for plain REST verbs.
- `http_method`: the explicitly bound HTTP method for a `rest_api`. Always present.
- `arguments`: the request-body or action-parameter object. Always present; `{}` means no arguments.
- `allowed_methods`: per-URL legal methods from the same discovery run's `rest_api_map.npy` (written
  by `redfish_ctl` discovery). Input evidence only.
- `json`: full JSON resource bodies given as input evidence.
- `expert_call_order`: separate RL-oracle ordering evidence, recorded outside this contract (see
  "Ordering Is Not Part Of This Contract").

Phase 3 does not choose new REST APIs. It binds `http_method` and `arguments` for exactly the
`rest_api_list` already produced by Phase 2. There is no planner, scheduler, or curriculum engine
inside Phase 3 inference; ordering, prerequisites, retries, waiting, recovery, hidden state, and
error handling belong to the separate RL policy stage.

Checkpoint rule: Phase 3 writes only `argument_extractor` artifacts. A run config must record the
input checkpoint path, the output path for `argument_extractor`, the `phase3_argument_extraction/*`
W&B namespace, and the exact dataset manifest used for training.

## Contract v1 Hard Requirements

- Every API in the input `rest_api_list` has **exactly one** matching call in `calls`. No extra
  call, no duplicate, no dropped API.
- `http_method` is explicit on every call. A missing method must NOT silently become `GET`.
- `http_method` must be legal for that API per the input evidence.
- `arguments` is always present as an object.
- `GET`/`HEAD` calls use `arguments: {}`.
- A zero-argument function/action also uses `arguments: {}` — explicitly emitted, not omitted.
- Mutation arguments are explicit. Missing mutation arguments must NOT silently become `{}`.
- Arguments are never inferred from scalar values seen in response JSON bodies.
- `allowed_methods` is input evidence only; it is never copied into output calls.
- `calls` is unordered; no ordering claim may be read from or written into it.

## Generic Examples (k = 1, 2, 3)

Generic shapes for contract explanation; APIs and fields are placeholders, not any real device.

k = 1 — "set x to 1". One call is still a length-1 list:

```json
{
  "x": {"text": "set x to 1", "rest_api_list": ["/api/x"]},
  "y_true": {
    "calls": [
      {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}}
    ]
  }
}
```

k = 2 — "set x to 1 and read z". The read uses `{}`:

```json
{
  "x": {"text": "set x to 1 and read z", "rest_api_list": ["/api/x", "/api/z"]},
  "y_true": {
    "calls": [
      {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}},
      {"rest_api": "/api/z", "http_method": "GET", "operation_name": null, "arguments": {}}
    ]
  }
}
```

k = 3 — "set x to 1, set y to 2, and read z":

```json
{
  "x": {"text": "set x to 1, set y to 2, and read z",
        "rest_api_list": ["/api/x", "/api/y", "/api/z"]},
  "y_true": {
    "calls": [
      {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}},
      {"rest_api": "/api/y", "http_method": "PATCH", "operation_name": null, "arguments": {"y": 2}},
      {"rest_api": "/api/z", "http_method": "GET", "operation_name": null, "arguments": {}}
    ]
  }
}
```

In every example `calls` is a set serialized as a list; any permutation of the same calls is the
same label.

## Redfish-Shaped Example (Current Test Environment)

Redfish is the current test environment, not part of the contract definition. This example
continues an accepted `D1` row. Both resources are read-only checks, so both calls bind `GET` with
`arguments: {}`. Note that `allowed_methods` appears only in `x` as evidence, never in the output.

```json
{
  "phase": 3,
  "task": "text_and_rest_api_list_to_calls",
  "x": {
    "text": "check the task queue and list the available computer systems",
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ],
    "json": [
      {
        "@odata.id": "/redfish/v1/TaskService/Tasks",
        "@odata.type": "#TaskCollection.TaskCollection",
        "Members": [],
        "Members@odata.count": 0,
        "Name": "Task Collection"
      },
      {
        "@odata.id": "/redfish/v1/Systems",
        "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
        "Members": [{"@odata.id": "/redfish/v1/Systems/System.Embedded.1"}],
        "Members@odata.count": 1,
        "Name": "Computer System Collection"
      }
    ],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
      "/redfish/v1/Systems": ["GET", "HEAD"]
    }
  },
  "y_true": {
    "calls": [
      {"rest_api": "/redfish/v1/TaskService/Tasks", "http_method": "GET", "operation_name": null, "arguments": {}},
      {"rest_api": "/redfish/v1/Systems", "http_method": "GET", "operation_name": null, "arguments": {}}
    ]
  }
}
```

Evaluation parses `y_pred.calls` and checks unordered set match: each input API is matched to
exactly one predicted call, then per-call `http_method` and canonical `arguments` are compared.

## Non-Empty Arguments

For a write/action resource, Phase 3 uses the same shape; the only difference is that `arguments`
carries the request body or action parameters. A non-empty argument row is accepted only when the
sampled JSON and method evidence support it: for example an action target with
`@Redfish.ActionInfo`, inline `@Redfish.AllowableValues`, a settings resource, or a recorded
successful write.

```json
{
  "y_true": {
    "calls": [
      {
        "rest_api": "/redfish/v1/example/action/target",
        "http_method": "POST", "operation_name": null,
        "arguments": {"ExampleArgument": "ExampleAllowedValue"}
      }
    ]
  }
}
```

Do not create mutation arguments from arbitrary scalar values in `json`. A value observed in a GET
body is not proof that the resource accepts that value in a PATCH or POST body.

Exact argument VALUES stay raw text/JSON end to end. `z_rest` (the REST-goal encoder from Phase 2)
and `z_method` (the Phase 3 method encoder) are SEPARATE encoders; neither encodes concrete
argument values, and v1 makes no shared-latent, unified-encoder, or zero-shot-universal claim.

## Ordering Is Not Part Of This Contract

Phase 2 and Phase 3 are both unordered. When a legal execution order matters (prerequisites,
mutate-before-verify, wait-for-task), that order is recorded as separate RL-oracle evidence in
`expert_call_order`, produced by the RL data pipeline alongside the row — never inside
`rest_api_list` or `calls`. The RL policy is the stage that learns ordering, prerequisites,
retries, waiting, recovery, hidden-state effects, and error handling; it may also execute legal
reads, waits, and recovery calls outside the target set when the environment requires them.

## Combination-Coverage Curriculum

Phase 3 trains on one-, two-, and three-API combinations (k = 1, 2, 3), not mostly singletons. The
first curriculum must include every category below, so that method binding and argument shapes are
each exercised in combination, not only alone:

Combination categories:

- all GET/observe
- observe + mutate
- observe + invoke (action)
- mutate + invoke
- delete + observe

Argument-shape categories:

- zero-argument function (explicit `arguments: {}`)
- one scalar argument
- nested object argument
- multiple arguments

Schema-origin categories:

- standard + standard
- standard + OEM
- OEM + OEM, where the corpus provides them

## Phase 3 W&B Metrics

Metric names below are the intended namespace; the schema and metric-key definitions in the repo
are authoritative.

- `phase3_argument_extraction/train/loss`
- `phase3_argument_extraction/train/perplexity`
- `phase3_argument_extraction/train/optimizer_step`
- `phase3_argument_extraction/eval/loss`
- `phase3_argument_extraction/eval/perplexity`
- `phase3_argument_extraction/eval/token_accuracy`
- `phase3_argument_extraction/eval/call_set_exact_match_rate`
- `phase3_argument_extraction/eval/rest_api_exact_match_rate`
- `phase3_argument_extraction/eval/method_exact_match_rate`
- `phase3_argument_extraction/eval/arguments_json_parse_rate`
- `phase3_argument_extraction/eval/arguments_exact_match_rate`
- `phase3_argument_extraction/eval/invalid_method_rate`
- `phase3_argument_extraction/eval/readonly_empty_arguments_rate`
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

Ordering metrics (Kendall tau, edit distance, ordered exact match) belong to RL-oracle
`expert_call_order` evaluation, not to this namespace.

When Phase 3 moves beyond mock plumbing, its acceptance gate must mirror Phase 1: approved full
corpora, readable W&B plots, checkpoint/report storage, and reviewed Git LFS artifact metadata.

## Inference Handoff

At normal inference time the chain is:

1. Phase 2 receives operator text plus current JSON/method context and emits an unordered
   `rest_api_list`.
2. Phase 3 receives the same text plus that `rest_api_list` and emits one call per `rest_api` —
   an unordered `calls: list[Call]`.
3. The RL policy receives the unordered target calls and learns environment strategy: ordering,
   prerequisites, retries, waiting, recovery, and consequences from state transitions.

The combined inference JSON should be easy to test:

```json
{
  "text": "check the task queue and list the available computer systems",
  "calls": [
    {"rest_api": "/redfish/v1/TaskService/Tasks", "http_method": "GET", "operation_name": null, "arguments": {}},
    {"rest_api": "/redfish/v1/Systems", "http_method": "GET", "operation_name": null, "arguments": {}}
  ]
}
```

Author:
Mus mbayramo@stanford.edu
