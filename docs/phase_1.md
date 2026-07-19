# Phase 1: Redfish JSON Pretraining

Phase 1 trains `model_x` on Redfish JSON reconstruction. This is the RedfishBackbone
pretraining/fine-tuning step; StateEncoder, goal extraction, argument extraction, rewards, and RL
policy training are separate consumers with separate weights. This phase only teaches the chosen LLM
the shape of Redfish resources, URI grammar, method context, and JSON completion.

Names are fixed as follows:

- `model_x`: the chosen base LLM after this Redfish JSON pretraining step.
- `weights_role`: `model_x`; this run writes a separate Phase 1 checkpoint.
- `profile`: a `phase1_*` training profile from `igc/modules/train/profiles.py`.
- `corpus_objective`: `phase1_pretrain`, the Redfish JSON reconstruction objective.
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

The current serious Phase 1 profile family is the Qwen2.5 7B rsLoRA path:
`phase1_7b_rslora_r32` in the executable profile registry, with run names that may use
`phase1-finetune-qwen2_5-7b-rslora`. GPT-2 remains a path smoke only.

Training is normal causal-LM next-token learning. The rendered prompt contains `x`; labels are
`-100` over the prompt tokens and actual token IDs over the `y_true` completion tokens. In other
words, this phase trains:

```text
P(y_true.json | x.rest_api, x.allowed_methods, x.json)
```

Checkpoint rule: Phase 1 writes `model_x` only. Later Phase 2/3 runs may initialize from that
checkpoint, but they must write `goal_extractor` and `argument_extractor` checkpoints in distinct
output directories and W&B groups.

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
`### Complete Redfish JSON`. The shifted labels should mask the `x` tokens and compute
cross-entropy only on the `y_true` JSON completion.

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

Full Phase 1 fine-tuning should select `model_x` by validation loss, not by the
test split and not by token accuracy alone:

- primary metric: `phase1_finetune/eval/loss`
- mode: minimize
- patience: 3 evaluation calls
- min delta: 0.005 to 0.01 validation loss
- max epochs: 5 to 10 for small corpora unless the validation curve still improves
- evaluation cadence: 4 evaluations per epoch as the starting point
- save cadence: every evaluation call

For example, if one epoch has 100 optimizer steps, start with `eval_steps=25`
and `save_steps=25`. Save a checkpoint at every evaluation, track the lowest
validation loss, stop after three evaluations without a meaningful improvement,
and restore the checkpoint with the lowest validation loss.

Secondary and diagnostic metrics:

- secondary: `phase1_finetune/eval/perplexity`
- diagnostic: `phase1_finetune/eval/token_accuracy`

## Acceptance Gate

Phase 1 is accepted only after:

- `model_x` trains on the approved full Redfish corpora, not only fixture data.
- W&B shows clear Phase 1 loss, perplexity, throughput, reconstruction, and test-time plots.
- The final checkpoint and evaluation report are in the approved shared model store.
- The repository stores only reviewed Git LFS artifact pointers or metadata for the checkpoint; raw
  weights are not copied into the source tree.

## Evaluation

For Phase 1, evaluation should check:

- `y_pred` parses as JSON.
- `y_pred.json["@odata.id"]` equals `x.rest_api`.
- `y_pred.json` exactly matches `y_true.json` for exact reconstruction runs.
- Loss is computed only on `y_true` tokens, not on prompt/context tokens.

Author:
Mus mbayramo@stanford.edu
