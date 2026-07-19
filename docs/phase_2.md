# Phase 2: D1 — Labelled REST Request Extraction

`D1` is the Phase 2 dataset: operator text paired with the **unordered set** of
REST APIs that text asks for. In runtime code, configs, and W&B keys the D1
builder uses the identifier `phase2_labelled_requests` (the module
`igc/ds/phase2_labelled_requests.py`, the spec
`configs/phase2_labelled_requests.yaml`, and the
`phase2_labelled_requests/*` metric namespace).

The machine-readable schema under `configs/contracts/*.yaml` is the
authoritative definition of every row shape in this document. All JSON examples
below are ILLUSTRATIVE only; when a doc example and the contract YAML disagree,
the contract YAML wins.

## Names

- `D0`: Phase 1 JSON reconstruction data.
- `D1`: the Phase 2 text-to-REST-API-set dataset (code identifier
  `phase2_labelled_requests`).
- `model_x`: the Phase 1 Redfish-tuned LLM; it must have completed Phase 1
  pretraining before real D1 generation runs. Offline plumbing may use tiny
  fixtures and injected providers only.
- `goal_extractor`: the separate Phase 2 fine-tuned weight role.
- `base_weights_role`: `model_x`, when Phase 2 initializes from the Phase 1
  checkpoint.
- `weights_role`: `goal_extractor`; Phase 2 writes only `goal_extractor`
  artifacts and must never overwrite the `model_x` checkpoint.
- `x`: the input context shown to the model.
- `y_true`: the exact target label stored in the dataset.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete REST URI string.
- `rest_api_list`: the target — always `list[str]`, never a scalar. One target
  API is still a list of length one.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`
  (written by `redfish_ctl` discovery).

## The Phase 2 Contract

Phase 2 maps operator text to an **unordered set of unique REST APIs**, emitted
as `rest_api_list: list[str]`:

- `[A, B]` equals `[B, A]`. Correctness is set equality.
- `[]` equals `[]` for hard-negative / no-action rows.
- One requested operation is still a length-1 list, never a bare string.
- The list contains unique entries; duplicates are invalid output.

Generic contract examples for sample widths `k=1`, `k=2`, `k=3` (illustrative;
the stored shape is owned by `configs/contracts/*.yaml`):

```json
{"text": "set x to 1",
 "y_true": {"rest_api_list": ["/api/v1/x"]}}
```

```json
{"text": "set x to 1 and read z",
 "y_true": {"rest_api_list": ["/api/v1/x", "/api/v1/z"]}}
```

```json
{"text": "set x to 1, set y to 2, and read z",
 "y_true": {"rest_api_list": ["/api/v1/x", "/api/v1/y", "/api/v1/z"]}}
```

### Order is not part of Phase 2

Phase 2 targets are unordered, always. Execution order — including
prerequisites, retries, waiting, and recovery — is learned by the separate RL
policy stage. When ground-truth ordering exists it is stored as separate
RL-oracle evidence (`expert_call_order`, defined by its own contract in
`configs/contracts/*.yaml`), outside the Phase 2 row and outside Phase 2
evaluation. There is no planner, scheduler, or curriculum inside Phase 2.

### Mention-order variants

Different phrasings that mention the same operations in different order are
**different text examples mapping to the SAME unordered target set**. All three
of these map to the same two-operation target:

- "set x, then read z"
- "read z after setting x"
- "update x and report z"

```json
{"y_true": {"rest_api_list": ["/api/v1/x", "/api/v1/z"]}}
```

The builder may emit several such surface variants per API combination; they
enlarge text coverage without changing the label.

## D1 Generation

D1 exists because the captured corpus has REST API paths, allowed methods, and
JSON bodies, but no human request text. Generation is inverse labelling:

1. **Combination generation.** The builder enumerates API *combinations* of the
   sampled corpus — combinations, not permutations, because the target is
   unordered. Sample widths come from `sample_widths` in the builder spec and
   are `[1, 2, 3]`.
2. **Draft.** `model_x` drafts an operator sentence for the sampled combination.
3. **Judge.** An independent judge (the private Pro judge, selected by the
   `judge` section in `configs/phase2_labelled_requests.yaml`) verifies the
   sentence asks for **all and only** the sampled set. Acceptance is
   deterministic: the judged set must equal the expected set after ignoring
   order; both-empty counts as an empty-set match.
4. **Store.** Accepted rows become D1; rejected rows are excluded from
   fine-tuning entirely.

### Judge rejection semantics

The judge rejects a draft for any of:

- **missing intent** — the text omits a sampled API's operation;
- **extra intent** — the text asks for an operation outside the sampled set;
- **duplicate intent** — the text asks for the same operation twice as if
  distinct;
- **ambiguous** — the text cannot be mapped to a definite set;
- **nonsense** — the text is not a plausible operator request;
- **method-semantic mismatch** — the text implies a write against a read-only
  API (or vice versa) per `allowed_methods`;
- **invalid structured output** — the judge response itself fails to parse
  against the required JSON shape.

### Mandatory finite bounds

Every D1 build is bounded by spec-owned caps; no D1 process is exhaustive or
unbounded, and no doc or run report may describe one as such:

- `sample_widths` — fixed at `[1, 2, 3]`;
- `max_accepted_rows` — total accepted-row budget for the build;
- `max_candidates` — total draft attempts for the build;
- `max_accepted_per_combination` — cap on accepted surface variants per API
  combination;
- `max_attempts_per_combination` — cap on retries before a combination is
  abandoned;
- `max_accepted_per_api` — cap on how often any single API appears across
  accepted rows, keeping the dataset balanced.

### Distractor context — no label leakage

The context `x` shown to the model contains the target records **plus
distractor records** that are not in the target set. The context must leak
nothing about which entries are targets: no "selected" marker, no target-only
catalog, no ordering trick that distinguishes targets from distractors. The
model must recover the set from the text alone.

## Builder Spec

The offline builder spec lives in `configs/phase2_labelled_requests.yaml`. That
spec owns prompt text, `model_x` identifiers, provider adapter metadata, judge
route/profile fields, generation settings, sample widths, the finite bounds
above, W&B namespace/key lists, safety caps, and acceptance thresholds. Runtime
Python must load those values rather than hardcoding prompt or model literals.

The builder CLI is `scripts/build_phase2_labelled_requests.py`. It loads the
YAML spec, reads JSONL records with `rest_api`, `allowed_methods`, `json`,
`vendor`, and `source_corpus`, then writes accepted D1 JSONL plus an aggregate
metrics JSON. Its provider modes are YAML-selected config, local-only mock,
local file fixtures, and an OpenAI-compatible live adapter. The checked-in
config keeps both draft and judge adapters on `mock`; operators must explicitly
select the live adapter by YAML or CLI and provide the environment variables
named by the spec before any live `model_x` or private Pro judge call is
possible.

The live adapter is only a provider surface. It resolves the model and route
placeholders from environment variables, reads base URLs from the spec's
`base_url_env` fields, extracts text from the configured response JSON path,
and sends no W&B, Redfish, GPU, or dataset-scale work on its own. A live run
whose `--count` exceeds `safety.live_without_gate_max_candidates` must pass
`--live-provider-gate-passed`; otherwise the CLI exits before opening a live
provider connection.

Live launches use the `providers:` block in
`configs/phase2_labelled_requests.yaml`; `igc/ds/phase2_labelled_requests.py`
resolves each provider's `base_url_env` and `api_key_env` value before it
builds the OpenAI-compatible provider clients. The concrete environment
variables are:

- `PHASE2_MODEL_X_BASE_URL`: the draft provider base URL named by
  `providers.draft.base_url_env` in `configs/phase2_labelled_requests.yaml`.
- `PHASE2_MODEL_X_API_KEY`: the draft provider API key named by
  `providers.draft.api_key_env` in `configs/phase2_labelled_requests.yaml`.
- `PHASE2_JUDGE_BASE_URL`: the judge provider base URL named by
  `providers.judge.base_url_env` in `configs/phase2_labelled_requests.yaml`.
- `PHASE2_JUDGE_API_KEY`: the judge provider API key named by
  `providers.judge.api_key_env` in `configs/phase2_labelled_requests.yaml`.

If any required live-provider variable is unset, the builder fails closed before
generation instead of silently falling back to mock providers or hardcoded
endpoints; `scripts/build_phase2_labelled_requests.py` enforces that check when
it constructs the live OpenAI-compatible provider.

### Checkpoint rule

Phase 2 writes only `goal_extractor` artifacts; it must never overwrite the
`model_x` checkpoint. A run config must record the input checkpoint path for
`model_x`, the output path for `goal_extractor`, the `phase2_goal_extraction/*`
W&B namespace, and the exact D1 dataset manifest used for training.

## Example Row (Redfish test environment)

Redfish is the **current test environment**, so stored rows carry
Redfish-shaped URIs and bodies; the contract itself is environment-generic.
This example is illustrative; `configs/contracts/*.yaml` owns the row shape.

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
    }
  },
  "y_true": {
    "rest_api_list": ["/redfish/v1/Systems"]
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

A stored row may keep a deterministic list order for reproducible serialization,
but the label is the unordered set. Phase 3 argument extraction is a separate
phase with its own `calls` contract; it consumes accepted D1 text and sets, not
this document.

## D1 Builder W&B Metrics

`PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS`, defined in
`igc/modules/base/metric_keys.py`, records the metric keys used by the offline
builder seam. The canonical namespace comes from `wandb.namespace` in
`configs/phase2_labelled_requests.yaml` and is `phase2_labelled_requests`:

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

After accepted D1 rows exist, the `goal_extractor` training path emits these
trainer metrics under `phase2_goal_extraction/*`. They are separate from the
builder metrics above. Cross-entropy is computed on the canonical serialized
target tokens; evaluation parses `y_pred.rest_api_list` and scores unordered
set match as the primary correctness metric. The reserved registry subset
(train + primary eval keys) lives in `igc/modules/base/metric_keys.py`
(`PHASE2_WANDB_METRIC_KEYS`); the remaining families below are the W&B
acceptance-gate instrumentation this phase must produce before it is accepted.

- `phase2_goal_extraction/train/loss`
- `phase2_goal_extraction/train/perplexity`
- `phase2_goal_extraction/train/optimizer_step`
- `phase2_goal_extraction/eval/loss`
- `phase2_goal_extraction/eval/perplexity`
- `phase2_goal_extraction/eval/token_accuracy`
- `phase2_goal_extraction/eval/set_match_rate`
- `phase2_goal_extraction/eval/precision`
- `phase2_goal_extraction/eval/recall`
- `phase2_goal_extraction/eval/f1`
- `phase2_goal_extraction/eval/invalid_rest_rate`
- `phase2_goal_extraction/eval/hard_negative_accuracy`
- `phase2_goal_extraction/eval/top_k_api_accuracy`
- `phase2_goal_extraction/eval/missing_required_api_rate`
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
- the finite bounds (`max_accepted_rows`, `max_candidates`,
  `max_accepted_per_combination`, `max_attempts_per_combination`,
  `max_accepted_per_api`) are enforced, not advisory;
- Phase 3 argument extraction remains outside this builder.

A later approved run can point the YAML provider metadata at the restored
`model_x` artifact and private Pro judge. That run is outside the local CPU
plumbing gate and must not be simulated with live GPU inference, live W&B, live
Redfish crawls, model downloads, cluster jobs, or dataset-scale generation
until the provider and launch gates have passed.

Author:
Mus mbayramo@stanford.edu
