# IGC use cases — what the agent does, end to end

This directory is the **product north-star for IGC**: a concrete, human-readable picture of what the
Infrastructure Goal-Condition Reinforce Learner *does in the real world* once trained — the goals an
operator hands it, how the pipeline turns human text into verified REST execution, and why the
execution side is a **reinforcement-learning policy** rather than a chatbot wrapped around an API.

> **Status — read this first.** These pages are **illustrative end-to-end episodes**, not shipped
> product surfaces. The current, locked pipeline (authoritative in
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md)) is:
>
> **D0** (captured Redfish REST-interface records from `redfish_ctl` discovery) → **Phase 1**
> fine-tune → **`model_x`** → **D1** inverse-label generation (judge-verified) → **Phase 2**
> `rest_api_list` (an **unordered** unique set) → **Phase 3** `calls: list[Call]` (**unordered**,
> explicit `http_method` + `arguments`) → two **separate** encoders **`z_rest`** + **`z_method`** →
> a **separate RL policy** that owns order / retry / wait / recovery → the **JSON simulator**.
>
> The machine-readable schema (`configs/contracts/*.yaml`, enforced by
> `igc/ds/rest_goal_contract.py`) is authoritative; every example in these pages is illustrative.
> **Phase 2 and Phase 3 outputs are unordered; order is separate RL-oracle evidence
> (`expert_call_order`), never part of the language contract.** Redfish is the **first proof
> environment**, not a permanent ontology.

## The one-paragraph version

Cloud and datacenter hardware exposes itself over the **Redfish REST API** — a self-describing tree
of resources (systems, BIOS, storage, firmware, power, logs) where each resource advertises the HTTP
methods it legally allows. IGC captures that interface as **D0**: per-URL JSON plus
`rest_api_map.npy` (written by `redfish_ctl` discovery) holding `url_file_mapping` and
`allowed_methods_mapping` — the **binding legal-action contract**. Phase 1 fine-tunes a backbone on
those records to produce `model_x`. D1 inverts the data: a known unordered API combination is
sampled, `model_x` drafts the human command that would need it, and an independent judge accepts the
row only if the text covers **all and only** the sampled APIs. Phase 2 learns text → the unordered
`rest_api_list` set; Phase 3 binds each selected API to an explicit `http_method` and an `arguments`
object (`{}` for reads). Two separate encoders — `z_rest` for the API selection, `z_method` for
method/argument structure, with exact argument **values** kept raw outside both — feed a separate RL
policy that learns ordering, retries, waiting, recovery, and hidden prerequisites against a JSON
simulator replaying captured Redfish state.

## Why this matters (the short version — full argument in the next file)

A modern LLM, prompted "power this server on via Redfish," will happily emit a URL and a JSON body.
On real hardware that is a **liability**: it does not know *this* host's actual resource tree, it can
invent an endpoint or method that does not exist, it cannot tell you whether the action *worked*, and
it has no way to learn ordering, retries, or recovery from consequences. IGC removes those failure
modes by construction — grounded APIs from the captured interface, verified success by re-reading
state, and a separate RL policy for every consequential decision. That is the subject of
[`why-rl-not-an-llm.md`](why-rl-not-an-llm.md).

## How to read this directory

Start with the two conceptual pages, then pick the scenarios relevant to you:

| File | What it gives you |
| --- | --- |
| [`why-rl-not-an-llm.md`](why-rl-not-an-llm.md) | The core argument: what a prompted LLM cannot do here, and the properties the pipeline guarantees instead. **Start here for the "aha".** |
| [`anatomy-of-an-episode.md`](anatomy-of-an-episode.md) | One goal, followed step by step: text → Phase 2 set → Phase 3 calls → RL execution with guarded writes and verified success. Includes the generic k=1/2/3 contract examples. |
| [`consumption-surfaces.md`](consumption-surfaces.md) | How you drive IGC: today's training/eval CLI, and the target terminal / UI / SDK / reconciler surfaces, each labelled implemented-vs-target. |
| [`uc-01-power-and-boot.md`](uc-01-power-and-boot.md) | Power on / off and one-shot boot override. The simplest end-to-end goal. |
| [`uc-02-bios-attribute-convergence.md`](uc-02-bios-attribute-convergence.md) | Drive BIOS attributes to a target state, handling *pending vs current* settings and apply-at-reboot. |
| [`uc-03-firmware-update.md`](uc-03-firmware-update.md) | A long-running task: push an update, then monitor the job/task phase to real completion. |
| [`uc-04-inventory-and-health.md`](uc-04-inventory-and-health.md) | The safe, read-only lane: crawl to a structured inventory + health roll-up, no mutation. |
| [`uc-05-storage-and-virtual-media.md`](uc-05-storage-and-virtual-media.md) | Multi-step configuration: mount virtual media / create a volume, verifying each precondition. |
| [`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md) | Heterogeneous fleet (Dell + Supermicro + HPE) driven to one desired state — with the honest, measured transfer evidence. |
| [`uc-07-log-and-telemetry-triage.md`](uc-07-log-and-telemetry-triage.md) | Collect/clear service logs and read telemetry — bounded, auditable operational hygiene. |

## Who this is for

- **Operators / SREs** deciding whether to trust an autonomous agent against real BMCs — the guarded
  execution and verified-success model is designed for exactly that scrutiny.
- **ML / RL engineers** who want the problem framing (the language contracts, the encoders, the
  separate RL policy) before the module details in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
- **Reviewers and newcomers** who want, in fifteen minutes, a true picture of where IGC is going and
  why the RL framing is load-bearing rather than decorative.

## The vocabulary these pages share

- **D0** — the captured Redfish REST-interface records: per-URL JSON responses plus
  `rest_api_map.npy` (from `redfish_ctl` discovery) with `url_file_mapping` and
  `allowed_methods_mapping`. The methods map is the **binding legal-action contract**.
- **`model_x`** — the Phase 1 checkpoint: a backbone fine-tuned by causal-LM JSON reconstruction
  over D0.
- **D1** — the inverse-label dataset: `model_x` drafts a human command for a known unordered API
  combination; an independent judge accepts the row only if the text covers **all and only** the
  sampled APIs (deterministic accept).
- **`rest_api_list`** — Phase 2 output: an **unordered unique** `list[str]` of REST APIs
  (`[A, B] == [B, A]`; `[] == []` for hard negatives; a single item is still a length-1 list).
- **`calls: list[Call]`** — Phase 3 output: **unordered**; each `Call` carries `rest_api`, an
  explicit `http_method`, and an `arguments` object (`{}` for read-only calls). Exact argument
  values stay raw.
- **`expert_call_order`** — separate RL-oracle evidence of a known-good execution order. It lives
  outside the Phase 2/3 language contract; the RL policy consumes it as supervision.
- **`z_rest` / `z_method`** — two **separate** encoders: `z_rest` encodes the resolved unordered
  API selection, `z_method` the method/argument structure. Exact argument values stay raw, outside
  both latents. In v1 there is no shared latent space and no unified encoder.
- **RL policy** — the separate execution stage. It owns ordering, retries, waiting, prerequisite
  discovery, recovery, and hidden-state handling, learned against the JSON simulator.
- **JSON simulator** — the offline environment replaying captured Redfish JSON state (from D0);
  the default execution surface for training and evaluation.
- **Evaluator / verified success** — success is *measured* by re-reading the resource and comparing
  it to the goal; a `2xx` response is never trusted as success.
- **Guardrail** — the `dry-run → approval → execute` path that stands between a chosen mutation and
  a real write to hardware.
- **HER (Hindsight Experience Replay)** — turns a trajectory that missed its goal into learning
  signal for the goal it *did* reach; part of the separate RL policy's training, over captured JSON.

Author:
Mus mbayramo@stanford.edu
