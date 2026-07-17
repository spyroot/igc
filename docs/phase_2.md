# Phase 2: Labelled REST Request Extraction

Legacy notes and old branches may call the Phase 2 text-labelled artifact `D1`.
New code, configs, metrics, and docs use the canonical dataset name
`phase2_labelled_requests`. `model_x` must already have completed Phase 1
Redfish JSON pretraining before real generation runs; offline plumbing may use
tiny fixtures and injected providers only.

`model_x`, produced by Phase 1 Redfish JSON pretraining, drafts the missing
operator text. The private Pro judge, selected by the `judge` section in
`configs/phase2_labelled_requests.yaml`, decides whether that text maps back to
the same unordered REST API set. Empty set equals empty set. Order is recorded
only as secondary evidence when the text explicitly says things like "then",
"before", "after", or uses numbered steps.

- `D0`: Phase 1 JSON reconstruction data.
- `phase2_labelled_requests`: Phase 2 text-to-REST-API-set data.
- `model_x`: the Phase 1 Redfish-tuned LLM.
- `goal_extractor`: the separate Phase 2 fine-tuned weight role.
- `profile`: a planned `phase2_goal_extractor_*` training profile.
- `base_weights_role`: `model_x`, when Phase 2 initializes from the Phase 1 checkpoint.
- `weights_role`: `goal_extractor`; Phase 2 must not overwrite the Phase 1 checkpoint.
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

Checkpoint rule: Phase 2 writes only `goal_extractor` artifacts. A run config must record the input
checkpoint path for `model_x`, the output path for `goal_extractor`, the `phase2_goal_extraction/*`
W&B namespace, and the exact `phase2_labelled_requests` dataset manifest used for training. Older
coordination notes may call that manifest `D1`; new runtime code, configs, metrics, and docs use the
canonical `phase2_labelled_requests` name.

The offline builder spec lives in `configs/phase2_labelled_requests.yaml`.
That spec owns prompt text, `model_x` identifiers, provider adapter metadata,
judge route/profile fields, generation settings, sample widths, W&B
namespace/key lists, safety caps, and acceptance thresholds. Runtime Python
must load those values rather than hardcoding prompt or model literals.

The offline fixture CLI is `scripts/build_phase2_labelled_requests.py`. It
loads the YAML spec, reads tiny JSONL records with `rest_api`,
`allowed_methods`, `json`, `vendor`, and `source_corpus`, then writes accepted
`phase2_labelled_requests` JSONL plus an aggregate metrics JSON. Its provider
modes are YAML-selected config, local-only mock, local file fixtures, and an
OpenAI-compatible live adapter. The checked-in config keeps both draft and
judge adapters on `mock`; operators must explicitly select the live adapter by
YAML or CLI and provide the environment variables named by the spec before any
live `model_x` or private Pro judge call is possible.

The live adapter is only a provider surface. It resolves the model and route
placeholders from environment variables, reads base URLs from the spec's
`base_url_env` fields, extracts text from the configured response JSON path,
and sends no W&B, Redfish, GPU, or dataset-scale work on its own. A live run
whose `--count` exceeds `safety.live_without_gate_max_candidates` must pass
`--live-provider-gate-passed`; otherwise the CLI exits before opening a live
provider connection.

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
  "vendor": "fixture",
  "source_corpus": "unit_fixture"
}
```

`y_pred.text` is only a draft. If `model_x` emits junk text, that text must not
enter `phase2_labelled_requests`. It must be cleaned or rejected.

## Draft And Judge

The generated text should be passed through a review/judge step with the same `json`,
`allowed_methods`, and canonical `rest_api_list`. The reviewer has two jobs:

1. rewrite the text into a natural operator sentence if needed;
2. judge whether the sentence asks for all and only the sampled REST API set,
   considering order only when the sentence contains ordering language.

Accepted output becomes the final `phase2_labelled_requests` row. Rejected
output is not used for fine-tuning.

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
  "task": "text_to_rest_api_list",
  "x": {
    "text": "check the available computer systems",
    "json": [
      {
        "@odata.id": "/redfish/v1/Systems",
        "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
        "Members": [],
        "Members@odata.count": 0,
        "Name": "Computer System Collection"
      }
    ],
    "allowed_methods": {
      "/redfish/v1/Systems": ["GET", "HEAD"]
    },
    "rest_api_list": ["/redfish/v1/Systems"]
  },
  "y_true": {
    "rest_api_list": ["/redfish/v1/Systems"],
    "order_evidence": "none"
  },
  "validation": {
    "text_source": "model_x_then_private_judge",
    "review_judged": true,
    "set_coverage_preserved": true,
    "nonsense": false
  },
  "metadata": {
    "prompt_spec_version": "phase2-labelled-requests-v1",
    "vendor": ["fixture_vendor"],
    "source_corpus": ["fixture_corpus"]
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
- `phase2_labelled_requests/empty_set_expected_total`
- `phase2_labelled_requests/sample_width/k`
- `phase2_labelled_requests/vendor/source_corpus`
- `phase2_labelled_requests/prompt_spec_version`
- `phase2_labelled_requests/model_x/artifact_sha`
- `phase2_labelled_requests/judge/model`
- `phase2_labelled_requests/judge/profile`

`accepted_total` counts rows that pass JSON parsing, Pro acceptance, nonsense
rejection, and unordered REST API set matching. `pro_accept_rate` is narrower:
it tracks valid judge responses whose own `accepted` flag is true, even if the
row later fails the set-match gate.

## Goal-Extractor Training W&B Metrics

After accepted labelled-request rows exist, the later `goal_extractor` training
path may emit these legacy trainer metrics. They are separate from
`phase2_labelled_requests/*` generation metrics above.

- `phase2_goal_extraction/train/loss`
- `phase2_goal_extraction/train/perplexity`
- `phase2_goal_extraction/train/optimizer_step`
- `phase2_goal_extraction/eval/loss`
- `phase2_goal_extraction/eval/perplexity`
- `phase2_goal_extraction/eval/token_accuracy`
- `phase2_goal_extraction/eval/ordered_exact_match_rate`
- `phase2_goal_extraction/eval/set_match_rate`
- `phase2_goal_extraction/eval/precision`
- `phase2_goal_extraction/eval/recall`
- `phase2_goal_extraction/eval/f1`
- `phase2_goal_extraction/eval/top_k_api_accuracy`
- `phase2_goal_extraction/eval/invalid_api_rate`
- `phase2_goal_extraction/eval/missing_required_api_rate`
- `phase2_goal_extraction/eval/missing_allowed_methods_rate`
- `phase2_goal_extraction/eval/order_violation_rate`
- `phase2_goal_extraction/order/kendall_tau`
- `phase2_goal_extraction/order/edit_distance`
- `phase2_goal_extraction/throughput/train_tokens_per_sec`
- `phase2_goal_extraction/throughput/train_samples_per_sec`
- `phase2_goal_extraction/throughput/eval_tokens_per_sec`
- `phase2_goal_extraction/throughput/eval_samples_per_sec`
- `phase2_goal_extraction/data/avg_num_apis`
- `phase2_goal_extraction/data/max_num_apis`
- `phase2_goal_extraction/data/mean_sequence_length`
- `phase2_goal_extraction/data/padding_ratio`
- `phase2_goal_extraction/calibration/log_prob_per_sequence`
- `phase2_goal_extraction/calibration/ece`
- `phase2_goal_extraction/test/latency_sec_p50`
- `phase2_goal_extraction/test/latency_sec_p95`
- `phase2_goal_extraction/test/memory_peak_mb`

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

A later approved run can point the YAML provider metadata at the restored
`model_x` artifact and private Pro judge. That run is outside the local CPU
plumbing gate and must not be simulated with live GPU inference, live W&B, live
Redfish crawls, model downloads, cluster jobs, or dataset-scale generation
until the provider and launch gates have passed.

Author:
Mus mbayramo@stanford.edu
