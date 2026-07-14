# Phase 1: Redfish JSON Pretraining

Phase 1 trains `model_x` on Redfish JSON reconstruction. This phase does not train goal extraction,
argument extraction, ordering, rewards, or RL behavior. It only teaches the chosen LLM the shape of
Redfish resources, URI grammar, method context, and JSON completion.

The executable training contract is `configs/phase_training/profiles.yaml`. Use the
`phase1_pretrain` phase and one of its profiles, such as `phase1_gpt2_smoke` for a path smoke or
`phase1_qwen2_5_7b_rslora` for the 7B adapter run.

## Dataset Row

Phase 1 rows are stored as JSONL. `x` carries the Redfish context, and `y_true` carries the exact JSON
completion target:

```json
{
  "phase": 1,
  "task": "redfish_json_reconstruction",
  "x": {
    "rest_api": "/redfish/v1/Fabrics/PCIe/Switches",
    "allowed_methods": ["GET", "HEAD"],
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

## Training Objective

Training is causal-LM next-token learning. The rendered prompt contains `x`; labels are masked over
the prompt tokens and active only on the `y_true` JSON completion tokens:

```text
P(y_true.json | x.rest_api, x.allowed_methods, x.json)
```

## Metrics

The canonical metric names are defined in `configs/phase_training/profiles.yaml` under
`metric_sets.phase1_pretrain`. Required plot families include:

- `phase1_dataset/*` for record, method, and token-shape counts.
- `phase1_pretrain/*` for loss, perplexity, optimizer step, throughput, padding, and memory.
- `phase1_eval/*` for JSON validity, schema validity, exact reconstruction, URI match, method match,
  and inference latency.

## Acceptance Gate

Phase 1 is accepted only after:

- `model_x` trains on approved full Redfish corpora, not only fixtures.
- W&B shows clear Phase 1 loss, perplexity, throughput, reconstruction, and test-time plots.
- The final checkpoint and evaluation report are stored in the approved shared model store.
- The repository stores only reviewed Git LFS artifact pointers or metadata for checkpoints; raw
  weights are not copied into the source tree.

Author:
Mus mbayramo@stanford.edu
