# Phase 1: Redfish JSON Pretraining

Phase 1 trains `model_x` on Redfish JSON structure. The stored D0 target remains the complete
canonical Redfish document, but the serious baseline uses selective structural loss instead of
rewarding a copy of every target token. This is the RedfishBackbone
pretraining/fine-tuning step; StateEncoder, goal extraction, argument extraction, rewards, and RL
policy training are separate consumers with separate weights. This phase only teaches the chosen LLM
the shape of Redfish resources, URI grammar, method context, and JSON completion.

Names are fixed as follows:

- `model_x`: the chosen base LLM after this Redfish JSON pretraining step.
- `weights_role`: `model_x`; this run writes a separate Phase 1 checkpoint.
- `profile`: a `phase1_*` training profile from `igc/modules/train/profiles.py`.
- `corpus_objective`: `phase1_pretrain`, the D0 Redfish JSON objective.
- `phase1_structural_loss_profile`: the YAML-owned rule selecting which target spans receive loss.
- `x`: the input context shown to the model.
- `y_true`: the exact target JSON the model should emit.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete Redfish URI.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `json`: full Redfish JSON resource body.

The starting data fact is intentionally narrow: Phase 1 has Redfish REST API paths, allowed methods,
and JSON bodies. It does not have human operator text such as "mount an ISO and boot the server".
Phase 1 therefore does not train goal extraction. It trains a Redfish-aware `model_x` so the next
phase can use that checkpoint to draft plausible human text from machine-side API evidence.

The current serious Phase 1 profile family is the Qwen2.5 7B rsLoRA path. The historical
structural-mask baseline is `phase1_7b_rslora_r32_structural_mask` in the executable profile
registry; `phase1_7b_rslora_r32` remains the directly comparable full-completion arm. Run names may use
`phase1-finetune-qwen2_5-7b-rslora`. GPT-2 remains a path smoke only.

## Concrete Bindings

| Role | Binding |
| --- | --- |
| Source registry | `configs/data/redfish_sources.yaml` |
| Corpus materializer | `igc.ds.source_registry.materialize_phase1_registry_corpus` |
| Dataset | `igc.ds.corpus_dataset.CorpusJSONLDataset` |
| Renderer | `igc.ds.phase1_render.render_phase1_prompt` |
| Structural loss | `igc.ds.phase1_structural_loss.build_phase1_structural_loss_view` |
| Structural-loss spec | `configs/training/phase1_structural_loss.yaml` |
| Shared token dataset | `igc.ds.sft_dataset.tokenize_prompt_completion` |
| Trainer | `igc.modules.train.sft.SFTTrainer` |
| Model/profile | `phase1_7b_rslora_r32_structural_mask` in `configs/training/profiles.yaml` |
| Task/prompt spec | `redfish_json_reconstruction` in `configs/training/sft_tasks.yaml` |
| Machine contract | `configs/contracts/phase1.yaml` |
| Output checkpoint | `model_x` |
| Promotion gate | `configs/inference/phase1_golden_acceptance.yaml` |

The required source registry combines the real `redfish_ctl` full corpora and the DSP2043
reference-BMC corpus produced through the same discovery artifact contract. DSP2043 broadens
training coverage; real captures remain the held-out acceptance anchor. The exact source registry,
upstream manifests, train rows, held-out rows, and their SHAs are immutable run evidence.
Every selected corpus must include a non-empty `rest_api_map.v1.json` or `rest_api_map.npy` with
both `url_file_mapping` and `allowed_methods_mapping`; the canonical materializer fails closed when
that method evidence is missing.

Training is causal-LM next-token learning over a profile-selected target view. The rendered prompt
contains `x`; prompt and padding labels are always `-100`. With
`historical_structural_mask_v1`, one available family is selected per row and epoch, that evidence
is hidden from the prompt, and only overlapping completion tokens receive labels. The locked
families are `@odata.id`, action targets, target keys, JSON objects, JSON arrays, Redfish allowable
values, and `/redfish/v1/` API prefixes. Held-out family selection is fixed by row. D0 and
`y_true.json` never change; enabled structural loss fails closed unless the materialized row has
`x.json == y_true.json` before masking.

The conditional objective is therefore:

```text
P(selected structural span of y_true.json | masked x.rest_api,
  x.allowed_methods, masked x.json)
```

The `none` structural-loss profile preserves full-completion labeling for controlled comparison;
it is not the historical structural-mask baseline.

Checkpoint rule: Phase 1 writes `model_x` only. Phase 2 initializes `goal_extractor` from the
promoted `model_x`; Phase 3 then initializes `argument_extractor` from the promoted Phase 2
`goal_extractor`. Every phase writes to a distinct output directory and W&B group.

## Pro Usage

Phase 1 uses the private Shared Brain Pro backend only as a helper for code drafting, review,
judging, planning, and acceptance checks. Use Pro with Think Max for substantial changes to the
dataset renderer, trainer loop, launcher, metrics, checkpoint/report flow, or artifact-publishing
path. Pro does not train `model_x`, replace GB300 DDP/FSDP2 sanity, replace W&B readback, or count as
evidence that Phase 1 converged.

## JSONL Row

This is the stored shape for one Phase 1 row. The example is a small collection resource; real rows
come from full Redfish corpora plus the same-run method map.

```json
{
  "phase": 1,
  "dataset": "D0",
  "task": "redfish_json_reconstruction",
  "x": {
    "rest_api": "/redfish/v1/Fabrics/PCIe/Switches",
    "allowed_methods": [
      "GET",
      "HEAD"
    ],
    "json": {
      "@odata.context": "/redfish/v1/$metadata#SwitchCollection.SwitchCollection",
      "@odata.id": "/redfish/v1/Fabrics/PCIe/Switches",
      "@odata.type": "#SwitchCollection.SwitchCollection",
      "Members": [],
      "Members@odata.count": 0,
      "Name": "Switch Collection"
    }
  },
  "y_true": {
    "json": {
      "@odata.context": "/redfish/v1/$metadata#SwitchCollection.SwitchCollection",
      "@odata.id": "/redfish/v1/Fabrics/PCIe/Switches",
      "@odata.type": "#SwitchCollection.SwitchCollection",
      "Members": [],
      "Members@odata.count": 0,
      "Name": "Switch Collection"
    }
  }
}
```

## Rendered Training Text

The JSONL row can be rendered for a causal LLM like this:

```text
### REST API
/redfish/v1/Fabrics/PCIe/Switches

### Allowed Methods
GET, HEAD

### Redfish JSON Input
{
  "@odata.context": "/redfish/v1/$metadata#SwitchCollection.SwitchCollection",
  "@odata.id": "/redfish/v1/Fabrics/PCIe/Switches",
  "@odata.type": "#SwitchCollection.SwitchCollection",
  "Members": [],
  "Members@odata.count": 0,
  "Name": "Switch Collection"
}

### Complete Redfish JSON
{
  "@odata.context": "/redfish/v1/$metadata#SwitchCollection.SwitchCollection",
  "@odata.id": "/redfish/v1/Fabrics/PCIe/Switches",
  "@odata.type": "#SwitchCollection.SwitchCollection",
  "Members": [],
  "Members@odata.count": 0,
  "Name": "Switch Collection"
}
```

`x` is everything before `### Complete Redfish JSON`. `y_true` is the JSON after
`### Complete Redfish JSON`. For the structural-mask profile, the selected evidence in `x` is
replaced by `<|redfish_mask|>` (or a selected target key is removed), while the completion remains
canonical. Shifted labels compute cross-entropy only on the selected completion span. A valid row
has `phase == 1`, `dataset == D0`, task
`redfish_json_reconstruction`, a non-empty string `x.rest_api`, unique uppercase
`x.allowed_methods`, object-valued `x.json` and `y_true.json`, no committed `y_pred`, and no missing
target. Prompt plus completion overflow fails closed; target bytes and selected spans are never
silently truncated.

## Phase 1 W&B Metrics

Use the `PHASE1_WANDB_METRIC_KEYS` registry, defined in
`igc/modules/base/metric_keys.py`, for metrics that the current Phase 1 training
surface tracks directly.
The live registry keeps Phase 1 curves separate from goal extraction or RL:

Current Phase 1 has the W&B namespace, basic training/eval metrics, and the
offline held-out prediction producer for reconstruction, throughput, data-shape,
calibration, and test-time evidence.

- `phase1_finetune/train/loss`
- `phase1_finetune/train/epoch_loss`
- `phase1_finetune/train/perplexity`
- `phase1_finetune/train/epoch_perplexity`
- `phase1_finetune/train/optimizer_step`
- `phase1_finetune/train/tokens_processed`
- `phase1_finetune/eval/loss`
- `phase1_finetune/eval/perplexity`
- `phase1_finetune/eval/token_accuracy`
- `phase1_finetune/throughput/train_tokens_per_sec`
- `phase1_finetune/throughput/train_samples_per_sec`

The held-out producer in `scripts/phase1_inference_gate.py` consumes existing
baseline and `model_x` prediction JSONL artifacts, compares them under the
spec in `configs/inference/phase1_golden_acceptance.yaml`, and writes compact
metrics/evidence to caller-supplied paths. These keys are listed in
`PHASE1_ACCEPTANCE_METRIC_KEYS`, defined in
`igc/modules/base/metric_keys.py`, and emitted by the producer:

- `phase1_finetune/eval/top_k_accuracy`
- `phase1_finetune/eval/json_parse_rate`
- `phase1_finetune/eval/json_exact_match_rate`
- `phase1_finetune/eval/odata_id_match_rate`
- `phase1_finetune/throughput/eval_tokens_per_sec`
- `phase1_finetune/throughput/eval_samples_per_sec`
- `phase1_finetune/data/padding_ratio`
- `phase1_finetune/data/mean_sequence_length`
- `phase1_finetune/data/max_sequence_length`
- `phase1_finetune/calibration/log_prob_per_token`
- `phase1_finetune/calibration/ece`
- `phase1_finetune/test/latency_sec_p50`
- `phase1_finetune/test/latency_sec_p95`
- `phase1_finetune/test/memory_peak_mb`

## Phase 1 Stopping Rule

Full Phase 1 fine-tuning selects `model_x` by validation loss, not by the test split and not by token
accuracy alone:

- primary metric: `phase1_finetune/eval/loss`
- mode: minimize
- patience: 3 evaluation calls
- min delta: 0.005 to 0.01 validation loss
- maximum epochs or steps, evaluation cadence, and save cadence: resolved from the named YAML
  profile and counted in optimizer steps

Gradient accumulation must not silently alter evaluation or save frequency. Every evaluation may
produce a checkpoint, the run tracks the lowest validation loss, and promotion uses that best
checkpoint rather than merely the last checkpoint.

Secondary and diagnostic metrics:

- secondary: `phase1_finetune/eval/perplexity`
- diagnostic: `phase1_finetune/eval/token_accuracy`

## Acceptance Gate

Phase 1 is accepted only after:

- `model_x` trains on the complete approved source manifest, not only fixture data.
- baseline and `model_x` run against the same untouched held-out manifest with per-corpus results;
  the source registry holds out at least 100 rows per real source, or every available row when the
  complete source is smaller than 100. Replay and DSP2043-derived rows remain training-only.
- checkpoint reload succeeds; no target or prediction row is missing; all promotion metrics are
  finite; the report records dataset, foundation-model, tokenizer, and training-code identities.
- W&B contains Phase 1 loss, perplexity, throughput, reconstruction, retention, calibration, and
  test-time plots.
- The best checkpoint and evaluation report are in the approved shared model store.
- The repository stores only reviewed Git LFS artifact pointers or metadata for the checkpoint; raw
  weights are not copied into the source tree.

Starting promotion floors are config-driven: JSON parse and resource-identity match at least
`0.995`, exact-match improvement over the foundation model at least `0.02`, and instruction-judge
acceptance drop at most `0.03`. A small deterministic golden set requires JSON parse and resource
identity rates of exactly `1.0`.

Because `model_x` drafts D1 text, instruction retention is checked before and after Phase 1 on fixed
`k=1`, `k=2`, and `k=3` API combinations. It reports natural-command, judge-acceptance,
missing-intent, extra-intent, and nonsense rates.

## Evaluation

For Phase 1, evaluation should check:

- `y_pred` parses as JSON.
- `y_pred.json["@odata.id"]` equals `x.rest_api`.
- `y_pred.json` exactly matches `y_true.json` for full-document diagnostic runs.
- Loss is computed only on profile-selected `y_true` tokens, never prompt, padding, or unselected
  completion tokens.
- Each historical structural family receives deterministic train coverage and a fixed held-out
  view; the run report records the profile name and exact structural-loss spec SHA.

Author:
Mus mbayramo@stanford.edu
