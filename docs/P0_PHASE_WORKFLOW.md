# P0 Phase Workflow

This is the current P0 split for Redfish language-model work. It keeps the immediate pretraining
goal separate from later goal/argument extraction, so Phase 1 can finish without being blocked by
Phase 2 and Phase 3 dataset generation.

Naming rule: current docs use phase names for launch surfaces and component names for weights. Do
not introduce historical model-number aliases for profiles; those aliases were retired because they
hide the dataset objective. Launchable profiles use `phase1_*`, `phase2_*`, and `phase3_*` names,
while model artifacts use explicit roles such as `model_x`, `goal_extractor`, and
`argument_extractor`.

## Pro Usage Across Phases

The private Shared Brain Pro backend is the reviewer/judge/code-drafting helper for IGC; it is
not a Phase 1/2/3 training checkpoint and it is not used to run GPU training. Use Pro with Think Max
when live Brain metadata advertises the required long context. If Pro is unreachable for required
private-source review or judging, report `BLOCKED:` rather than substituting an external reviewer.

| Phase | When Pro Is Required | What Pro Must Not Replace |
| --- | --- | --- |
| 1 | Substantial changes to the Redfish JSON renderer, trainer, launcher, W&B metrics, checkpoint/report flow, artifact publishing, or acceptance review. | The actual `model_x` training run, W&B evidence, corpus validation, DDP/FSDP2 sanity, or local/offline gates. |
| 2 | `D1` text cleanup/judging, `rest_api_list` schema review, hard-negative/set-coverage metric design, dataset-builder changes, and acceptance review. | The `goal_extractor` checkpoint, accepted `D1` rows, or metric evidence. |
| 3 | Method/argument schema review, row judging for method/argument evidence, unsafe/unsupported argument rejection review, metric design, and acceptance review. | The `argument_extractor` checkpoint, exact argument labels, guardrail evidence, or metric evidence. |

| Phase | Profile prefix | Weight role | W&B namespace | Dataset objective | Checkpoint rule |
| --- | --- | --- | --- | --- | --- |
| 1 | `phase1_*` | `model_x` | `phase1_finetune/*` | `phase1_pretrain` / Redfish JSON reconstruction | writes a separate Phase 1 checkpoint |
| 2 | `phase2_goal_extractor_*` (planned) | `goal_extractor` | `phase2_goal_extraction/*` | `text_to_rest_api_list` | initializes from `model_x` only when requested; never overwrites it |
| 3 | `phase3_argument_extractor_*` (planned) | `argument_extractor` | `phase3_argument_extraction/*` | `text_and_rest_api_list_to_calls` | initializes from `model_x` or `goal_extractor` only when requested; never overwrites either |

Phase 2 and Phase 3 profile names are planned until their trainers land. Do not add live profiles
unless `igc.modules.train.launch.profile_to_argv` can emit a runnable command with a distinct
checkpoint/output directory and W&B namespace.

## Phase 1: Pretrain `model_x`

Purpose: train `model_x` on Redfish JSON reconstruction using full Redfish corpora and same-run
method maps. This creates the Redfish-aware base model used by later phases.

Deliverables:

- Phase 1 dataset renderer: `x = {rest_api, allowed_methods, json}` and `y_true = {json}`.
- Causal-LM labels masked over `x` and active only on the `y_true` JSON completion.
- GPU launch ladder: local/offline unit tests, cheap smoke, short large-model smoke, then full
  Phase 1 run on the approved multi-node GPU training surface.
- W&B metrics under `phase1_finetune/*`, with train/eval/throughput/data/calibration/test
  subgroups.
- Checkpoint and evaluation report showing JSON parse rate, exact-match rate, and `@odata.id`
  match rate.

Acceptance gate:

- Full Redfish JSON pretraining has completed over the approved full corpora, not just fixtures.
- W&B contains readable Phase 1 plots for loss, perplexity, token accuracy, throughput, JSON
  validity, exact reconstruction, calibration, and test-time evaluation.
- The checkpoint and evaluation report are stored in the approved shared model store.
- The repo receives only reviewed artifact metadata through Git LFS pointers; bulky weights are not
  copied back into the source tree.

The first serious model profile should use a 7B instruct backbone with rsLoRA unless a smoke gate
proves that profile cannot run. GPT-2 remains a path smoke only; it is not evidence that the Redfish
representation is useful.

## Phase 2: Build And Train `D1`

Purpose: after Phase 1, use `model_x` plus a review/judging pass to build `D1`, then fine-tune a
specialized model that maps operator text and current Redfish context to a canonical `rest_api_list`.
Phase 2 is primarily set coverage: generated text does not need to mention the sampled APIs in the
canonical row order unless it uses explicit ordering language.

The corpus itself has no human operator sentence labels. It only provides Redfish REST API paths,
allowed methods, and JSON bodies. `D1.x.text` is therefore a synthetic label: sample one, two, or
three API records from the corpus, let the Phase 1 Redfish-tuned `model_x` draft a plausible human
request, then keep that row only if Pro judges that the text maps back to exactly the sampled API set.
This is how a roughly five-thousand-record API corpus can produce a much larger supervised text
dataset without asking a human to inspect every synthetic sentence. Rows must stay human-plausible:
one small request or a short related bundle, not a sentence that asks for twenty unrelated REST
operations.

Current P0 scope: mock-dataset plumbing and offline tests only.

Real dataset build waits for the Phase 1 checkpoint. The accepted row shape is:

```json
{
  "phase": 2,
  "dataset": "D1",
  "task": "text_to_rest_api_list",
  "x": {
    "text": "check the task queue and list the available computer systems",
    "json": [],
    "allowed_methods": {}
  },
  "y_true": {
    "rest_api_list": [
      "/redfish/v1/TaskService/Tasks",
      "/redfish/v1/Systems"
    ]
  }
}
```

Phase 2 training uses token-level cross entropy over one canonical serialized JSON target because
causal-LM fine-tuning needs a single sequence. That canonical order is a serialization convention,
not the main correctness rule. Evaluation must report REST API set match as primary: `[B, A]` is
correct for canonical target `[A, B]` when the human text does not explicitly require order, and
`[]` is correct for a hard-negative/no-action row whose true `rest_api_list` is empty. Ordered exact
match is only an auxiliary metric for rows with explicit order evidence. W&B metrics live under
`phase2_goal_extraction/*`, with train/eval/throughput/data/calibration/test subgroups.

## Phase 3: Method And Argument Extraction

Purpose: after Phase 2, fine-tune a specialized model that maps operator text plus the canonical
`rest_api_list` to one call per `rest_api`, each with `allowed_methods`, `method`, and `arguments`.
Phase 3 reuses the same accepted `D1.x.text`; it does not generate a new human sentence. Here
`arguments` means the request-body or action-parameter bindings for a selected `rest_api` and
`method`, for example `Address = "192.168.1.1"` in an EthernetInterface PATCH body. For read-only
`GET`/`HEAD` calls, `arguments` is `{}`.

Current P0 scope: mock-dataset plumbing and offline tests only.
W&B metrics live under `phase3_argument_extraction/*`, with train/eval/order/throughput/data/calibration/test
subgroups.

The accepted row shape is:

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
    "json": [],
    "allowed_methods": {
      "/redfish/v1/TaskService/Tasks": [
        "GET",
        "HEAD"
      ],
      "/redfish/v1/Systems": [
        "GET",
        "HEAD"
      ]
    }
  },
  "y_true": {
    "calls": [
      {
        "rest_api": "/redfish/v1/TaskService/Tasks",
        "allowed_methods": [
          "GET",
          "HEAD"
        ],
        "method": "GET",
        "arguments": {}
      },
      {
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": [
          "GET",
          "HEAD"
        ],
        "method": "GET",
        "arguments": {}
      }
    ]
  }
}
```

## Inference Contract

After Phase 2 and Phase 3, a sentence should produce testable target calls:

```json
{
  "text": "check the task queue and list the available computer systems",
  "target_calls": [
    {
      "rest_api": "/redfish/v1/TaskService/Tasks",
      "allowed_methods": [
        "GET",
        "HEAD"
      ],
      "method": "GET",
      "arguments": {}
    },
    {
      "rest_api": "/redfish/v1/Systems",
      "allowed_methods": [
        "GET",
        "HEAD"
      ],
      "method": "GET",
      "arguments": {}
    }
  ]
}
```

These target calls are not a forced execution script. RL receives them as the visible goal
specification and still acts over the legal action catalog from the current Redfish state. It may add
reads, waits, retries, verification, prerequisites, or recovery calls outside the target list. The
full simulator/HER contract for hidden dependencies, such as ejecting an existing ISO before mounting
a new one, lives in [RL_SCALING_PLAN.md](RL_SCALING_PLAN.md).

After Phase 3, goal representation becomes an RL research choice. Same-surface experiments may use
concrete `target_calls` directly. Transfer/HER experiments should try a low-dimensional semantic
`z_goal` / `z_sub_goal` derived from `target_calls`, with goal items treated as unordered by default
and dependency edges carried only when explicit evidence says order matters.

Treat RL enablement after Phase 3 as a refactor gate across SIM, HER, DQN/candidate scoring, TD/replay
targets, evaluator/reward, and rollout record shape. Phase 3 target-call quality does not by itself
make RL curves trustworthy.

Author:
Mus mbayramo@stanford.edu
