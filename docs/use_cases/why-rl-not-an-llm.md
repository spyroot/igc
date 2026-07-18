# Why an RL agent, and not "just prompt an LLM"

This is the single most important page in the set. If IGC were something a good prompt could do, it
would not be worth building. It is worth building because the thing operators actually need —
**reach a goal on real hardware, provably, safely, and get better at it** — is exactly the thing a
language model, used as a language model, cannot give you.

To be clear up front: **IGC contains an LLM.** Phase 1 fine-tunes a backbone on captured Redfish
JSON to produce `model_x` (the Phase 1 checkpoint defined in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md)), and Phases 2/3 are language tasks — nothing understands
a messy JSON API body like a model trained on language and code. The argument here is not "LLMs are
bad." It is: **an LLM is the right *prior* and the wrong *decision-maker*.** IGC uses the model for
understanding — which APIs a request touches, which method and arguments each needs — and constrains
every *consequential decision* (ordering, retries, waiting, recovery) to a **separate
reinforcement-learning policy** operating in a real environment.

## The concrete failure: prompt a chatbot to run Redfish

Ask a strong model, with no tools: *"Power on the server at this BMC and set it to PXE-boot once."*
It will confidently produce something like:

```
POST /redfish/v1/Systems/1/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
PATCH /redfish/v1/Systems/1
{ "Boot": { "BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once" } }
```

That looks right. On a specific machine it may be **wrong in ways that matter**, and the model has no
way to know:

- The system member may not be `/Systems/1`. On Supermicro or HPE it is frequently a different id
  (`/Systems/Self`, a UUID); the model **guessed** the URL.
- `ComputerSystem.Reset` may not allow `POST` from the current power state, or the resource may not
  advertise that action at all on this firmware. The model cannot see the host's actual
  `allowed_methods`.
- `BootSourceOverrideEnabled: "Once"` may need to be applied at the next reboot as a *pending*
  setting, not a live one — patching the live resource silently no-ops on some vendors.
- After sending both calls, the model has **no idea whether the server is actually powering on**. It
  will tell you "Done!" whether it worked or not.

None of these are prompt-engineering problems. They are structural: the model is reasoning about a
*generic* Redfish from training data, not about *this* controller's *current* state. Every extra bit
of confidence makes it more dangerous, because a wrong write to a BMC can knock real hardware
offline.

## The three ways to solve it, compared

| Property | Prompt an LLM | Hand-written script / runbook | **IGC (pipeline + RL policy)** |
| --- | --- | --- | --- |
| Knows *this* host's real interface | No — reasons from a generic prior | Only what the author hard-coded | **Yes** — trained on the captured interface (D0); observations are real reads |
| Can invent a non-existent endpoint/method | **Yes, routinely** | No (but brittle if the API differs) | **No** — legal methods per URL come from the `allowed_methods_mapping` contract in `rest_api_map.npy` |
| Handles order, prerequisites, waiting | Guesses a plausible order | The author's fixed path | **Yes** — the separate RL policy learns order/retry/wait/recovery from consequences |
| Knows whether it *succeeded* | Self-reports, unverifiable | Only what the author asserted | **Yes** — an evaluator re-reads the resource and checks the target state |
| Recovers from a failed step | No memory across attempts | Only pre-written error branches | **Yes** — recovery is learned from experience (HER) |
| Gets better over time | No | No | **Yes** — every episode is training signal |
| Safe to point at real hardware | No | Depends entirely on the author | **By design** — `dry-run → approval → execute` guardrail |

A hand-written script is honest but frozen: it encodes one operator's path for one vendor, and it
rots the moment the API, the firmware, or the vendor changes. An LLM is flexible but ungrounded. IGC
is built to be **both grounded and adaptive** — which is the combination the job actually requires.

## The division of labor (the current pipeline, precisely)

1. **Phase 2 — which APIs.** From the operator's text, extract `rest_api_list: list[str]` — an
   **unordered unique set** of REST APIs drawn from the captured interface. `[A, B] == [B, A]`; a
   request that maps to nothing yields `[]`; a single API is still a length-1 list. No ordering, no
   plan, no schedule lives here.
2. **Phase 3 — how to call each one.** For each selected API, bind an explicit `http_method` and an
   `arguments` object (`{}` for read-only calls), emitting `calls: list[Call]` — still
   **unordered**. Exact argument *values* stay raw.
3. **Two separate encoders.** `z_rest` encodes the resolved API selection; `z_method` encodes
   method/argument structure. They are **separate latents in v1** — no shared space, no unified
   encoder — and argument values stay outside both.
4. **The separate RL policy decides.** Ordering, retries, waiting for long-running tasks,
   prerequisite reads, and recovery are learned by the RL policy against the JSON simulator
   (replaying captured Redfish state). Known-good order is supplied as **separate RL-oracle
   evidence (`expert_call_order`)**, never smuggled into the Phase 2/3 contract.

Take any stage away and you are back to a confident guess. Keep all of them and you have an agent an
operator can actually audit.

## The properties that kill the failure modes

### 1. A grounded action surface — it cannot hallucinate a call
The APIs the pipeline can name come from the walked, captured interface (D0), and the methods legal
on each URL come from `allowed_methods_mapping` in `rest_api_map.npy` — the binding legal-action
contract produced by `redfish_ctl` discovery. A URL or verb the interface never advertised is not
filtered out after the fact; it is outside the trained selection space and rejected by the contract
gate.

### 2. Verified success — the reward is measured, not self-reported
"Did the power state become `On`?" is answered by *re-reading the resource*, not by asking the model
how it thinks it did. A `2xx` on a write means *accepted*, never *achieved*. **IGC's notion of
"done" is a fact about the (simulated or real) hardware, not an opinion about the transcript.**

### 3. Order, retries, and recovery are learned, not asserted
Reaching a goal is usually multi-step, with hidden prerequisites (stage a pending setting before the
reset; wait for a task to reach a terminal state). The Phase 2/3 language contract deliberately says
nothing about order — the RL policy learns it from reward and from `expert_call_order` oracle
evidence, and may legally insert reads, waits, retries, and recovery calls the target set never
mentioned.

### 4. It learns *in the environment*, including from failure
Most attempts do not go perfectly. **Hindsight Experience Replay** relabels a trajectory that missed
its intended goal with the goal it actually reached, so failed or partial episodes still make the
policy better. This runs offline over captured JSON in the simulator — learning from consequences
without touching hardware.

### 5. Transfer is measured, never assumed
The multi-vendor fixture corpora in the repo (Dell iDRAC, Supermicro, HPE iLO, generic DMTF) exist
to *measure* generalization. The honest record so far: a frozen-encoder zero-shot ranking experiment
came in **under the acceptance bar** on a large held-out corpus (numbers in
[`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md)) — which is
exactly why v1 claims **no** zero-shot universal-REST capability and no shared cross-task latent.
Redfish is the first proof environment; anything beyond it is future evidence, not a present claim.

## The one-sentence takeaway

A prompted LLM answers *"what call would plausibly do this?"*; IGC answers *"which calls does this
request actually name on this interface, with which methods and arguments — and, separately, in what
learned order do I execute, verify, and recover?"* Those are different questions, and only the
second one is safe to automate.

Author:
Mus mbayramo@stanford.edu
