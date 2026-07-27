# Phase 2: Labelled REST Request Extraction

Phase 2 creates and learns from `D1`, the dataset that adds the human text absent from `D0`.
`model_x`, the promoted Phase 1 checkpoint, drafts a natural operator request from one, two, or
three selected Redfish APIs. The private Pro judge accepts the draft only when it expresses all and
only that unordered API set. Phase 2 then trains `goal_extractor` in the reverse direction:

```text
x = accepted text + API context containing targets and realistic distractors
y_true = {"rest_api_list": [...]}
```

`D1` is the dataset identity. `phase2_labelled_requests` is the builder/W&B metric namespace, not a
second dataset name. API order has no meaning. Separately bounded `k=0` hard negatives use natural
unsupported requests over realistic distractor contexts and preserve `[] == []`; they are judged
and released in D1 but excluded from the positive `k=1/2/3` balance calculation.

## Concrete Bindings

| Role | Binding |
| --- | --- |
| Dataset builder | `igc.ds.phase2_labelled_requests.Phase2LabelledRequestBuilder` |
| Dataset release | `igc.ds.d1_release.release_d1_jsonl` |
| Stable row identity | `igc.ds.rest_goal_contract.d1_row_id` |
| Stored training dataset | `igc.ds.sft_dataset.PromptCompletionJSONLDataset` |
| Renderer | `igc.ds.rest_goal_contract.render_phase2_sft` |
| Trainer | `igc.modules.train.sft.SFTTrainer` |
| Parent checkpoint | promoted Phase 1 `model_x` |
| Output checkpoint | `goal_extractor` |
| Training profile | `phase2_7b_rslora_r32` in `configs/training/profiles.yaml` |
| Task/prompt spec | `text_to_rest_api_list` in `configs/training/sft_tasks.yaml` |
| D1 builder spec | `configs/phase2_labelled_requests.yaml` |
| Machine contract | `configs/contracts/d1_contract.yaml` and `configs/contracts/rest_goal.yaml` |
| Promotion gate | `configs/inference/phase2_goal_extractor_promotion.yaml` |

`model_x`, its artifact SHA, the judge route/model/profile, generation settings, prompts, sampling
widths, distractor count, metrics, and thresholds come from the YAML specs or their named runtime
environment values. A live run resolves those identities before its first request and records the
resolved values in the release manifest.

The YAML also owns finite build ceilings: total candidates, total accepted rows, accepted variants
per API combination, attempts per API combination, and accepted rows per API. The builder checks
these limits before a model or judge request. A D1 build is therefore bounded; it never enumerates
all combinations or permutations.

## Starting Evidence

`D0` contains only:

```text
rest_api + allowed_methods + Redfish JSON
```

It does not contain labels such as "mount an ISO and boot the server." Hand-writing roughly 100,000
requests is not practical. Instead, the builder samples `k=1`, `k=2`, and `k=3` API combinations,
uses `model_x` to draft the missing text, and uses Pro as the independent quality opinion. This is
label creation, not Phase 2 model evaluation: the selected API set is already known because it was
the input to generation.

The draft and judge see only the selected records. After acceptance, the stored Phase 2 input adds
at least four unselected contexts. It never exposes `rest_api_list`, `selected`, target indices, or
any hidden membership field in `x`.

## Accepted Row

```json
{
  "phase": 2,
  "dataset": "D1",
  "source_dataset": "D0",
  "task": "text_to_rest_api_list",
  "contract_version": "phase2-rest-api-set/v1",
  "x": {
    "text": "check the available computer systems",
    "api_context": [
      {
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": "/redfish/v1/Systems"}
      },
      {
        "rest_api": "/redfish/v1/Managers",
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": "/redfish/v1/Managers"}
      },
      {
        "rest_api": "/redfish/v1/Chassis",
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": "/redfish/v1/Chassis"}
      },
      {
        "rest_api": "/redfish/v1/TaskService/Tasks",
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": "/redfish/v1/TaskService/Tasks"}
      },
      {
        "rest_api": "/redfish/v1/UpdateService",
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": "/redfish/v1/UpdateService"}
      }
    ]
  },
  "y_true": {
    "rest_api_list": ["/redfish/v1/Systems"]
  },
  "validation": {
    "valid_json": true,
    "accepted": true,
    "natural": true,
    "nonsense": false,
    "ambiguous": false,
    "duplicate_intent": false,
    "extra_intents": false,
    "method_semantics_valid": true,
    "covered_api_set": ["/redfish/v1/Systems"]
  },
  "metadata": {
    "row_id": "sha256:28689285115f451588c30d92729415f6d3d8f9936fd49537ab5203259fbcc735",
    "prompt_spec_version": "phase2-labelled-requests-v1",
    "sample_width_k": 1,
    "vendor": ["example", "example", "example", "example", "example"],
    "source_corpus": ["example", "example", "example", "example", "example"],
    "heldout_vendor_or_model": ["example"]
  }
}
```

## Judge And Release Gates

A draft is accepted only when the verdict is valid JSON, accepted, natural, non-nonsense,
unambiguous, non-duplicated, free of extra intent, method-semantics-valid, and its
`covered_api_set` equals the sampled API set. Judge calibration requires both human-accepted and
human-rejected examples and gates precision, recall, and false-accept rate.

The release path builds a `.pending` directory, validates every JSONL row, judge evidence, and
balanced positive `k=1/2/3` counts, validates the separately bounded `k=0` negatives, computes the
artifact SHA, and writes `data.jsonl` plus `manifest.json`. One directory rename publishes the
complete immutable release, so a failed batch cannot expose a canonical D1 path. Mock or
file-provider output may exercise the contract but cannot pass real D1 promotion.
Promotion requires resolved live model/judge identities, the promoted `model_x` parent SHA, real
held-out evidence, checkpoint reload, and inference smoke evidence.

Every released row has a content-derived `metadata.row_id`. Phase 2/3 training requires explicit
immutable train and held-out manifests whose artifact SHA and row count match the exact JSONL bytes.
One deterministic `d1_phase23_split_release.v1` manifest proves that Phase 2 and Phase 3 views use
aligned source row IDs and that train and held-out source row IDs are disjoint. The run report records
the split release, data, and manifest identities; a path alone is not lineage evidence.

## Goal-Extractor Promotion

The model output is exactly:

```json
{"rest_api_list": ["/redfish/v1/Systems", "/redfish/v1/TaskService/Tasks"]}
```

The strict parser rejects extra keys, scalar output, duplicates, and APIs absent from
`x.api_context`. Evaluation reports JSON parse rate, unordered set exact match, precision, recall,
F1, cardinality accuracy, invalid/duplicate rates, empty-set exact match, and `k=1/2/3` plus
vendor/model slices. The same semantic case must remain correct when context order and JSON key
order change, target serialization is reversed, or irrelevant distractors are added.

Phase 2 writes only `goal_extractor`; it never overwrites `model_x`. Its parent artifact SHA,
foundation model SHA, tokenizer SHA, task-spec SHA, data/eval identities, optimizer steps, and best
checkpoint are recorded in the run report.

Author:
Mus mbayramo@stanford.edu
