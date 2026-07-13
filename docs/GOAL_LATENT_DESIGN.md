# Goal latent design

This document defines the public, code-facing contract for natural-language goal extraction and
latent goal alignment in `igc`. It replaces the ambiguous "planner" framing with a narrower job:
extract what the operator wants, align that intent with concrete Redfish state surfaces, and leave
execution strategy to the goal-conditioned RL policy.

## Purpose

`GoalExtractor` maps operator text to one or more atomic sub-goal references. `GoalEncoder` maps both
operator text and concrete Redfish goal surfaces into a shared latent sub-goal space. `GoalVerifier`
checks whether a hidden concrete sub-goal is satisfied by an observation. A compound instruction may
carry a goal set, but the policy normally conditions on one active `z_sub_goal` at a time. The
simulator/evaluator keeps the concrete hidden payload needed for reward and HER.

The boundary is intentional:

- The extractor does not choose API calls.
- The extractor does not invent an ordered execution plan.
- The encoder does not decide success by latent distance.
- The verifier uses concrete state facts and action results, not model confidence.

## Core Records

For a chat interface, `GoalExtractor` returns an envelope with four pieces:

```json
{
  "text": "set NTP then boot the server",
  "atomic_goal_refs": [
    {"goal_id": "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True"},
    {"goal_id": "power.computer_system.PowerState.eq.On"}
  ],
  "relations": [
    {
      "before_goal_id": "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
      "after_goal_id": "power.computer_system.PowerState.eq.On",
      "relation": "before",
      "evidence": "then"
    }
  ],
  "evidence": {
    "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True": "set NTP",
    "power.computer_system.PowerState.eq.On": "boot the server"
  }
}
```

The same sentence without an ordering word, for example "boot the server and set NTP", has the same
two `atomic_goal_refs` but an empty `relations` list. A single-goal sentence has one
`atomic_goal_refs` item and no relations. RL/HER uses the active atomic item as `desired_sub_goal`
and `z_sub_goal`; the episode can carry the whole envelope for progress tracking and UI reporting.

`GoalRef` is the semantic, vendor-neutral atomic sub-goal target. It is small enough to be used as a
class/slot label and stable enough to align examples across vendors.

```json
{
  "goal_id": "power.system.power_state:on",
  "family": "power",
  "resource_type": "ComputerSystem",
  "property_path": "PowerState",
  "operator": "eq",
  "target_value": "On",
  "mode": "state",
  "constraints": {}
}
```

`GoalSurface` is one vendor/world-specific realization of one atomic `GoalRef`. It comes from captured
Redfish JSON and carries enough context to verify the sub-goal in that world.

```json
{
  "goal_ref": {"goal_id": "power.system.power_state:on"},
  "vendor": "dell",
  "source": "real_dell",
  "resource_uri": "/redfish/v1/Systems/1",
  "resource_type": "#ComputerSystem.v1_20_0.ComputerSystem",
  "fact_path": "PowerState",
  "target_value": "On",
  "current_value": "Off",
  "allowed_values": ["On", "Off"],
  "verifier": {
    "kind": "state_eq",
    "resource_uri": "/redfish/v1/Systems/1",
    "property_path": "PowerState",
    "operator": "eq",
    "target_value": "On"
  }
}
```

`GoalTextExample` is the supervised extraction row. The target is a list because an operator sentence
can mention a compound goal. Each list item is an atomic sub-goal; a downstream scheduler or
curriculum chooses which `z_sub_goal` is active at a decision point.

```json
{
  "text": "clear the logs and reboot the server",
  "goal_refs": [
    {"goal_id": "log.event_log.clear"},
    {"goal_id": "power.system.reset:graceful_restart"}
  ],
  "dependencies": [],
  "text_source": "llm_paraphrase",
  "split": "train"
}
```

`GoalDependency` records only dependency information explicitly stated in text. It is not a learned
execution plan and it does not make the whole compound instruction one vector. If the text says
"then", "after", or "before", the dataset can carry a partial order:

```json
{"before": "firmware.component.update", "after": "boot.os.ubuntu2204"}
```

RL still learns which concrete `ToolAction` candidates satisfy each atomic goal in the active
environment.

## Dataset Construction

The dataset is built in two layers.

1. Atomic rows are deterministic. Code scans captured `SourceRecord` streams and emits single goal
   surfaces from observable scalar leaves, inline `@Redfish.AllowableValues` annotations, generic
   `Actions` blocks, and richer Redfish-specific surfaces such as power states, reset transition
   intents, boot override settings, BIOS attribute values, virtual-media states, log clear targets,
   and task terminal states.
2. Text rows are generated or curated from those atomic rows. A paraphrase backend may propose
   operator sentences, but deterministic code owns the labels. During bootstrapping, generated text is
   written as an unvalidated draft row with its JSON-derived `true_y`; after a trained extractor
   exists, a validation pass can discard text that adds a goal, removes a goal, changes a value,
   inserts an unsupported dependency, or changes risk.

Compound text examples are assembled from atomic rows. The raw Redfish capture does not need to
contain a human compound task; it only needs enough atomic surfaces that the builder can combine
compatible goals into a target list.

The safe bootstrap generation loop is:

1. Build atomic `GoalSurface` rows from `SourceRecord` streams.
2. Choose one or more `GoalRef` rows as `true_y`.
3. Ask a configurable paraphrase backend for candidate `x` strings.
4. Write `GoalTextExample` rows where `x` is model-generated text and `true_y` is the selected
   deterministic `GoalRef` set.
5. Train the first real `GoalExtractor`.
6. Rerun that extractor over candidate `x` rows and keep only rows whose `atomic_goal_refs` set and
   `relations` set exactly match `true_y`.

This is how LLM-backed generation is used: it writes candidate language, not labels. A handwritten
keyword parser is intentionally not part of the label path.

The first implementation path is `scripts/build_goal_dataset.py`, which consumes one or more captured
JSON directories through `RedfishFixtureSource` and writes `GoalSurface` rows. This is the small,
local smoke-test path; it proves the builder on tiny fixtures and does not attempt to pull the full
corpus:

```bash
python scripts/build_goal_dataset.py \
  --capture-root /path/to/captured/redfish-json \
  --vendor dell \
  --source real_dell \
  --surfaces-out /path/to/goal_surfaces.jsonl
```

To request text examples, provide selected `--goal-id` targets and a paraphrase mode. The public code
uses generic OpenAI-compatible environment variables and does not hardcode private endpoints:

```bash
GOAL_PARAPHRASE_BASE_URL=<openai-compatible-base-url> \
GOAL_PARAPHRASE_MODEL=<model-name> \
python scripts/build_goal_dataset.py \
  --capture-root /path/to/captured/redfish-json \
  --surfaces-out /path/to/goal_surfaces.jsonl \
  --text-out /path/to/goal_text_examples.jsonl \
  --paraphrase-mode openai \
  --goal-id power.computer_system.PowerState.eq.On
```

The full dataset build runs inside the NV72 Docker lab, not on a laptop. The lab wrapper
`scripts/build_goal_dataset_lab.sh` initializes the `redfish_ctl`/`idrac_ctl` submodule in the Docker
checkout, runs `git lfs pull` inside that submodule so the full Redfish JSON corpus is present, verifies
JSON files exist, and then calls `scripts/build_goal_dataset.py`. Capture roots are discovered from
the LFS-backed vendor fixture directories and `~/.json_responses`, or supplied with
`IGC_CAPTURE_ROOTS`. The model endpoint comes only from environment variables or a private env file;
the script never hardcodes private hosts.

For a lab-side full-corpus draft pass, generate one text batch per discovered atomic goal and write a
manifest with the counts:

```bash
IGC_GOAL_DATASET_OUT=/private/or/lfs/path/goal_dataset \
GOAL_PARAPHRASE_BASE_URL=<openai-compatible-base-url> \
GOAL_PARAPHRASE_MODEL=<model-name> \
bash scripts/build_goal_dataset_lab.sh
```

The generated `goal_surfaces.jsonl`, `goal_text_examples.jsonl`, and `goal_dataset_manifest.json`
are private dataset artifacts. They are not public documentation and should not be committed to this
repository.

## Initial Goal Families

Power goals come from `#ComputerSystem` resources:

- `PowerState == On`
- `PowerState == Off`
- reset transition intents from `ResetType` allowable values

Boot goals come from the `Boot` object on `#ComputerSystem`:

- `Boot.BootSourceOverrideTarget == <allowed value>`
- `Boot.BootSourceOverrideEnabled == Once | Continuous | Disabled`
- boot option references from `#BootOption` resources

BIOS goals come from `#Bios`, `#Bios/Settings`, and attribute registries:

- current attribute facts from `Attributes`
- pending attribute facts from settings resources
- allowable attribute values from `#AttributeRegistry`

Virtual-media goals come from virtual-media resources:

- image inserted/ejected state
- inserted image URI when a safe approved fixture surface exists

Log/task goals come from log-service and task resources:

- log cleared state when the resource exposes a clear action and measurable count/state
- task terminal state such as `Completed`, `Exception`, or `Killed`

Every family must define a verifier before it can train RL rewards.

## Encoders And Objectives

`TextGoalEncoder(text_or_atomic_span) -> z_sub_goal` and
`GoalSurfaceEncoder(surface) -> z_sub_goal` share the same latent space. A compound instruction can
also have a pooled `z_goal_set` for bookkeeping, but RL rewards, HER relabels, and candidate scoring
should be defined over atomic `z_sub_goal` items unless a later experiment proves a set encoder is
needed. A training batch should contain multiple vendors and multiple text paraphrases per semantic
goal. The default objective is multi-positive contrastive alignment plus fact preservation:

```text
L_goal =
  L_supervised_contrastive(text/surface positives, hard negatives)
  + lambda_decode * L_goal_fact_decoder
  + lambda_collapse * L_variance_or_covariance_guard
```

The decoder head predicts family, resource type, property path, operator, target value bucket,
current-vs-pending mode, and dependency hints. It exists to prevent collapse, not to replace the
verifier.

Hard negatives are mandatory:

- same property, different value;
- same value, different property;
- same action family, different reset type;
- current BIOS value versus pending BIOS setting;
- boot override once versus continuous;
- same text family on a different resource type.

## Verifier And Reward Boundary

`GoalVerifier` receives the hidden concrete goal payload, the next observation, and optional action
result metadata. It returns whether the goal is satisfied and any dense components the reward table
defines. A `2xx` action response is evidence that the request was accepted; it is not proof that the
goal was reached. The verifier must re-check measurable state when the goal is a state goal.

For HER, the achieved-goal projector derives future achieved atomic goals from observations:

```text
achieved_goal_refs = AchievedGoalProjector(obs_future)
z_her_sub_goal = GoalEncoder(surface_for(achieved_goal_ref))
reward_t_her = GoalVerifier(achieved_goal_ref, obs_{t+1})
```

The reward is recomputed against the transition's `obs_{t+1}`. A future state that satisfies a goal
does not make every earlier transition successful.

## Metrics

Extraction metrics:

- `goal_extractor/atomic_goal_exact_match`
- `goal_extractor/compound_goal_set_f1`
- `goal_extractor/dependency_edge_f1`
- `goal_extractor/invalid_goal_ref_rate`
- `goal_extractor/value_slot_exact_match`

Latent alignment metrics:

- `goal_encoder/text_to_surface_recall_at_1`
- `goal_encoder/text_to_surface_recall_at_5`
- `goal_encoder/surface_to_text_recall_at_1`
- `goal_encoder/hard_negative_confusion_rate`
- `goal_encoder/effective_rank`
- `goal_encoder/vendor_leakage_probe_accuracy`

Verifier/HER metrics:

- `goal_verifier/state_eq_accuracy`
- `goal_verifier/false_success_rate`
- `goal_verifier/false_failure_rate`
- `her/achieved_goal_projection_count`
- `her/relabel_reward_recompute_error_count`

RL metrics must use the active latent sub-goal seen by the policy and the hidden concrete verifier
payload used by the environment. Mixing those two surfaces in plots hides bugs.

## Implementation Names

Use descriptive names in code and docs:

- `igc.ds.goal_dataset` for records and JSONL IO.
- `igc.ds.goal_dataset_builder` for deterministic Redfish-to-goal extraction and compound assembly.
- `igc.modules.goal_extractor` for text-to-goal-ref parsing interfaces and metrics.
- `igc.modules.goal_encoder` for text/surface latent encoders and objective helpers.
- `igc.modules.goal_verifier` for deterministic state/reward verification.

Do not name new public code after phase labels. Phase labels can remain in roadmaps, but source files
should say what they do.

## Failure Modes

The design is wrong if any of these happen:

- A compound sentence is collapsed into one opaque goal and loses atomic targets.
- A sentence with no explicit ordering produces an ordered plan label.
- A policy success check compares only latent vectors.
- A generated paraphrase changes `Once` to `Continuous`, `GracefulRestart` to `ForceRestart`, or an
  observation/read goal into a mutation goal.
- Vendor-specific names are used as the semantic label instead of as goal-surface evidence.
- HER copies the future reward backward without recomputing against each transition's next
  observation.
- A goal family is used for RL before its verifier exists.

# Author: Mus mbayramo@stanford.edu
