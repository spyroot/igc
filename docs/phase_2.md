# Phase 2: Labelled Request Dataset

Phase 2 builds `phase2_labelled_requests`, the dataset name defined by
`configs/phase2_labelled_requests.yaml` in this repository. The dataset adds the
missing human request text label for one, two, or three Redfish REST API records
that already carry paths, allowed HTTP methods, and JSON bodies.

`model_x`, produced by Phase 1 Redfish JSON pretraining, drafts the missing
operator text. The private Pro judge, selected by the `judge` section in
`configs/phase2_labelled_requests.yaml`, decides whether that text maps back to
the same unordered REST API set. Empty set equals empty set. Order is recorded
only as secondary evidence when the text explicitly says things like "then",
"before", "after", or uses numbered steps.

## Config Contract

`configs/phase2_labelled_requests.yaml`, added by the Phase 2 labelled-request
plumbing step, is the single runtime source for:

- prompt templates and prompt spec versions under `prompts`;
- `model_x` identifiers and the model artifact SHA under `model_x`;
- private judge route metadata, judge model, and judge profile under `judge`;
- generation settings under `generation`;
- sample widths `1`, `2`, and `3` under `sample_widths`;
- W&B namespace metadata under `wandb`;
- acceptance thresholds under `acceptance_thresholds`.

Runtime code must load this YAML spec. Prompt text, model IDs, judge route names,
generation temperatures, token limits, W&B namespaces, and acceptance thresholds
must not be embedded directly in dataset-builder code.

## Build Input

One build item samples `k in {1, 2, 3}` records from the Redfish corpus. Each
record has this minimum shape:

```json
{
  "rest_api": "/redfish/v1/Systems",
  "allowed_methods": ["GET", "HEAD"],
  "json": {
    "@odata.id": "/redfish/v1/Systems",
    "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
    "Members": [],
    "Members@odata.count": 0,
    "Name": "Computer System Collection"
  },
  "vendor": "fixture",
  "source_corpus": "unit_fixture"
}
```

The sampled records are rendered into the YAML `draft_human_request` prompt. A
later approved run can send that prompt to `model_x`; offline unit tests use only
fixtures and do not call a model.

## Draft And Judge

`model_x` should return JSON containing the drafted text:

```json
{
  "text": "check the available computer systems"
}
```

The draft text, sampled records, allowed methods, and expected REST API set are
then rendered into the YAML `pro_judge_set_match` prompt. The private Pro judge
returns JSON with the judged REST API set:

```json
{
  "accepted": true,
  "rest_api_list": ["/redfish/v1/Systems"],
  "nonsense": false,
  "reason": "The request asks for the sampled Systems collection only."
}
```

The offline parser accepts a row only when the judge JSON parses, the judge did
not mark the draft as nonsense, and the judged REST API set equals the expected
set after ignoring order. If the expected and judged sets are both empty, the
row counts as an empty-set match.

## Final Row

An accepted row stores the text label with the sampled REST API evidence:

```json
{
  "phase": 2,
  "dataset": "phase2_labelled_requests",
  "task": "text_to_rest_api_set",
  "x": {
    "text": "check the available computer systems",
    "records": [
      {
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["GET", "HEAD"],
        "json": {
          "@odata.id": "/redfish/v1/Systems",
          "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
          "Members": [],
          "Members@odata.count": 0,
          "Name": "Computer System Collection"
        }
      }
    ]
  },
  "y_true": {
    "rest_api_set": ["/redfish/v1/Systems"],
    "order_evidence": "none"
  },
  "validation": {
    "text_source": "model_x_then_private_pro_judge",
    "pro_judged": true,
    "rest_api_set_match": true,
    "empty_set_match": false,
    "nonsense": false,
    "invalid_json": false
  }
}
```

The row may preserve a deterministic list order for storage, but Phase 2
acceptance is set-based unless the text itself carries explicit ordering
language. Phase 3 argument extraction remains a separate phase and consumes
accepted request labels only after a downstream contract chooses how to preserve
or infer call order.

## Metrics

`PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS`, defined in
`igc/modules/base/metric_keys.py`, records the metric keys used by the offline
builder seam. The canonical namespace comes from `wandb.namespace` in
`configs/phase2_labelled_requests.yaml` and is `phase2_labelled_requests`.

Required keys:

- `phase2_labelled_requests/build/draft_total`
- `phase2_labelled_requests/build/accepted_total`
- `phase2_labelled_requests/build/rejected_total`
- `phase2_labelled_requests/eval/nonsense_rate`
- `phase2_labelled_requests/eval/invalid_json_rate`
- `phase2_labelled_requests/eval/pro_accept_rate`
- `phase2_labelled_requests/eval/rest_api_set_match_rate`
- `phase2_labelled_requests/eval/empty_set_match_rate`
- `phase2_labelled_requests/sample_width/k`
- `phase2_labelled_requests/vendor/source_corpus`
- `phase2_labelled_requests/spec/prompt_spec_version`
- `phase2_labelled_requests/model/model_x_artifact_sha`
- `phase2_labelled_requests/judge/model`
- `phase2_labelled_requests/judge/profile`

These keys can be emitted by offline counters without opening a live W&B run.

## Acceptance Gate

The offline plumbing is accepted when focused pytest coverage proves:

- sample widths `k=1`, `k=2`, and `k=3` are supported;
- prompt specs load from YAML and render sampled record payloads;
- REST API set comparison is unordered;
- empty set equals empty set;
- Pro judge JSON parsing handles accepted, rejected, nonsense, and invalid JSON
  outcomes;
- nonsense, invalid JSON, Pro accept, REST API set match, and empty-set match
  counters emit the required keys;
- Phase 3 argument extraction remains outside this builder.

A later approved run can point the YAML route metadata at the fine-tuned
`model_x` artifact and private Pro judge. That run is outside the local CPU
plumbing gate and must not be simulated with live GPU inference, live W&B, live
Redfish crawls, model downloads, or cluster jobs.

Author:
Mus mbayramo@stanford.edu
