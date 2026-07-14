# Phase 1: Redfish JSON Pretraining

Phase 1 trains `model_x` on Redfish JSON reconstruction. This phase does not train goal extraction,
argument extraction, ordering, rewards, or RL behavior. It only teaches the chosen LLM the shape of
Redfish resources, URI grammar, method context, and JSON completion.

Names are fixed as follows:

- `model_x`: the chosen base LLM after this Redfish JSON pretraining step.
- `x`: the input context shown to the model.
- `y_true`: the exact target JSON the model should emit.
- `y_pred`: the model output during inference or evaluation.
- `rest_api`: one concrete Redfish URI.
- `allowed_methods`: methods from the same discovery run's `rest_api_map.npy`.
- `json`: full Redfish JSON resource body.

Training is normal causal-LM next-token learning. The rendered prompt contains `x`; labels are
`-100` over the prompt tokens and actual token IDs over the `y_true` completion tokens. In other
words, this phase trains:

```text
P(y_true.json | x.rest_api, x.allowed_methods, x.json)
```

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

Use a dedicated namespace so Phase 1 curves never mix with goal extraction or RL:

- `phase1_pretrain/train/loss`
- `phase1_pretrain/train/perplexity`
- `phase1_pretrain/train/optimizer_step`
- `phase1_pretrain/train/tokens_processed`
- `phase1_pretrain/eval/loss`
- `phase1_pretrain/eval/perplexity`
- `phase1_pretrain/eval/token_accuracy`
- `phase1_pretrain/eval/top_k_accuracy`
- `phase1_pretrain/eval/json_parse_rate`
- `phase1_pretrain/eval/json_exact_match_rate`
- `phase1_pretrain/eval/odata_id_match_rate`
- `phase1_pretrain/throughput/train_tokens_per_sec`
- `phase1_pretrain/throughput/train_samples_per_sec`
- `phase1_pretrain/throughput/eval_tokens_per_sec`
- `phase1_pretrain/throughput/eval_samples_per_sec`
- `phase1_pretrain/data/padding_ratio`
- `phase1_pretrain/data/mean_sequence_length`
- `phase1_pretrain/data/max_sequence_length`
- `phase1_pretrain/calibration/log_prob_per_token`
- `phase1_pretrain/calibration/ece`
- `phase1_pretrain/test/latency_sec_p50`
- `phase1_pretrain/test/latency_sec_p95`
- `phase1_pretrain/test/memory_peak_mb`

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
