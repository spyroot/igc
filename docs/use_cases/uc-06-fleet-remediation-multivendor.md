# UC-06 — Fleet remediation across vendors

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery) → JSON simulator. Contract examples are
> illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the **first proof
> environment**, and in v1 there is **no** proven zero-shot universal-REST capability — this page
> ends with the honest, measured evidence on exactly that point.

## The goal, and what the pipeline extracts from it

You have a rack of machines from several vendors and one desired state: every node reimages from the
network on its next boot, and every node is powered on. You do not want to write a Dell script, a
Supermicro script, an HPE script, and a "generic" script — you want to state the target once:
*"Bring every node to a reimage-ready state: persistent PXE boot, powered on."*

Per node, the language side extracts the same contract (URL templates resolve per host):

```jsonc
// Phase 2 — UNORDERED unique set (k=2)
{ "rest_api_list": [
    "/redfish/v1/Systems/{id}",
    "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset"
] }

// Phase 3 — UNORDERED; explicit http_method + arguments per Call
{ "calls": [
    { "rest_api": "/redfish/v1/Systems/{id}",
      "http_method": "PATCH",
      "arguments": { "Boot": { "BootSourceOverrideEnabled": "Continuous",
                               "BootSourceOverrideTarget": "Pxe" } } },
    { "rest_api": "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset",
      "http_method": "POST",
      "arguments": { "ResetType": "On" } }
] }
```

The desired end state — `Boot.BootSourceOverrideTarget == "Pxe"`,
`Boot.BootSourceOverrideEnabled == "Continuous"`, `PowerState == "On"` — is what the evaluator
re-reads per node. Which calls each node *actually needs* (a node already powered on needs no
reset), and in what order, is the separate RL policy's per-node decision; known-good order is
separate `expert_call_order` oracle evidence, never part of this JSON. Scope guards for the run:
one ComputerSystem member per node, no firmware/BIOS/virtual-media writes, boot values must stay
within what the node advertises as allowable, at most one power action per node.

## Why a script or a chatbot struggles here

A per-vendor script hard-codes the one thing that differs most: the ComputerSystem member id —
`/redfish/v1/Systems/System.Embedded.1` on one box, `/redfish/v1/Systems/1` on another,
`/redfish/v1/Systems/Self` on a generic DMTF service, a bare UUID on a fourth. One new vendor, or
one node that enumerates its systems differently, and the script silently patches nothing or patches
the wrong member. A chatbot will happily emit a `BootSourceOverrideTarget` value or an `Oem` field
the node never advertised — a plausible string that the BMC rejects, or worse, accepts into a stuck
pending state. Neither one *verifies*: both report "done" from having issued a call, not from
re-reading that the state took.

## Observation and the legal methods

The RL policy's observation is the Redfish GET result — the raw JSON body: resource identity
(`@odata.id`, `@odata.type`), the `Boot` object, `PowerState`, `Status`, the action surface
(`Actions[*].target`, `*@Redfish.AllowableValues`), and pending-vs-current settings where present.

Legal methods per URL come from the captured interface's `allowed_methods_mapping` in
`rest_api_map.npy` (the binding contract from `redfish_ctl` discovery) — the policy cannot name an
endpoint that was never walked or a method the resource forbids. For a Supermicro node
(`/Systems/1`) the legal surface around this goal is:

| endpoint (walked)                                  | method | resource_type            | why it is legal                          |
|----------------------------------------------------|--------|--------------------------|------------------------------------------|
| `/redfish/v1/Systems`                              | GET    | `ComputerSystemCollection` | read-only lane, auto-proceeds          |
| `/redfish/v1/Systems/1`                            | GET    | `ComputerSystem`         | read-only lane, auto-proceeds            |
| `/redfish/v1/Systems/1`                            | PATCH  | `ComputerSystem`         | `Boot` is a writable property            |
| `/redfish/v1/Systems/1/Actions/ComputerSystem.Reset` | POST | `ComputerSystem`         | body carries `Actions[*].target`         |

## The trajectory (RL execution, per node)

Observe → choose → dry-run/approve → execute → observe → evaluate, abbreviated. Read lanes
auto-proceed; the two writes each stop at the guardrail.

**1. Observe the collection (read, auto).**

```
GET /redfish/v1/Systems
→ 200  { "Members": [ { "@odata.id": "/redfish/v1/Systems/1" } ] }
```

**2. Observe the member (read, auto).** The policy reads current state and the allowable set it must
stay inside.

```
GET /redfish/v1/Systems/1
→ 200  {
    "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
    "PowerState": "Off",
    "Boot": {
      "BootSourceOverrideEnabled": "Disabled",
      "BootSourceOverrideTarget":  "None",
      "BootSourceOverrideTarget@Redfish.AllowableValues": ["None","Pxe","Hdd","Cd","BiosSetup"]
    },
    "Actions": { "#ComputerSystem.Reset": {
      "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
      "ResetType@Redfish.AllowableValues": ["On","ForceOff","GracefulRestart"] } }
  }
```

`Pxe` is in the allowable set — the binding is legal on this box. Two target facts are unmet
(`Boot` target and `PowerState`), so two writes are needed.

**3. Boot PATCH → dry-run → approve → execute (mutating).**

```
# dry-run (no call issued): the rendered body the agent WILL send
PATCH /redfish/v1/Systems/1
{ "Boot": { "BootSourceOverrideEnabled": "Continuous", "BootSourceOverrideTarget": "Pxe" } }
--- guardrail: mutation on ComputerSystem, pauses for approval ---
# on approval, execute
→ 200  (settings accepted)
```

**4. Power action → dry-run → approve → execute (mutating).**

```
POST /redfish/v1/Systems/1/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
--- guardrail: mutation, pauses for approval ---
→ 204
```

**5. Re-observe and evaluate.**

```
GET /redfish/v1/Systems/1
→ 200  { "PowerState": "On",
         "Boot": { "BootSourceOverrideEnabled": "Continuous",
                   "BootSourceOverrideTarget": "Pxe" } }
```

Now the second vendor, **same goal, different tree** — a generic DMTF service exposing its system as
`/redfish/v1/Systems/Self`, already powered on:

```
GET  /redfish/v1/Systems           → { "Members":[{"@odata.id":"/redfish/v1/Systems/Self"}] }
GET  /redfish/v1/Systems/Self      → PowerState "On"; Boot target "None"
PATCH /redfish/v1/Systems/Self     { "Boot": { "BootSourceOverrideEnabled":"Continuous",
                                                "BootSourceOverrideTarget":"Pxe" } }   # approve → 200
# PowerState already "On" → no Reset issued; the shorter path is the correct path
GET  /redfish/v1/Systems/Self      → Boot target "Pxe", PowerState "On"
```

Different member id, different starting state, different number of steps — one goal statement, one
pipeline. Dell iDRAC (`.../Systems/System.Embedded.1`, `Oem/Dell` present) and HPE iLO
(`.../Systems/1`, `Oem/Hpe` present) resolve the same way; `Oem` sections are observed but never
required by the contract.

## What "done" means

The evaluator does not trust that a PATCH returned 200 or that Reset returned 204. It re-reads the
ComputerSystem member and checks each fact against the fresh body:
`Boot.BootSourceOverrideTarget == "Pxe"`, `Boot.BootSourceOverrideEnabled == "Continuous"`,
`PowerState == "On"`. A node is done only when all three hold on the re-read; a node whose PATCH
landed in a *pending* settings resource but has not applied is **not** done — current-vs-pending is
read from the resource itself, exactly so this cannot be self-reported away. Success is a measured
property of the tree, per node, or it did not happen.

## Constraints, risk, and the guardrail

Both writes are mutating and both pause. The `Boot` PATCH is low-to-moderate risk (it changes next
boot behavior, not firmware or BIOS attributes — those are excluded by the run's scope guard). The
`ComputerSystem.Reset` is higher risk: it changes the power state of a physical machine. So every
mutation runs the guardrail in order — **dry-run** renders the exact endpoint, method, and body;
**approval** is where the operator sees that body before any call touches the BMC; **execute** only
then. Read lanes (`GET /Systems`, the member GET, the re-read) carry no such pause and auto-proceed.
The at-most-one-power-action-per-node guard bounds blast radius: a run that tries to Reset a node
twice is illegal before it is ever risky.

## What transfers — the honest, measured record

The desire is obvious: one trained pipeline, many vendors. The evaluator's per-node verdict feeds
**HER**: a run that overshot (issued a Reset on a node that was already On, or PATCHed a mode the
node did not advertise) is relabeled by its *achieved* state, so the RL policy is rewarded toward
the shortest safe path — the `/Systems/Self` node above learns the one-write solution rather than
blindly replaying the two-write one. The multi-vendor fixture corpora shipped in the repo
(`idrac`/`supermicro`/`hpe`/`generic`) are exactly the material for measuring cross-vendor behavior.

**The honest bar (historical D-002 evidence, from the superseded candidate-representation design;
kept because the numbers are real and the lesson stands).** A zero-shot go/no-go — a frozen encoder
with no learned projection, ranking each state's true graph neighbors — measured
**k=5 = 0.293 on the 1,499-node Supermicro corpus and 0.754 on the 167-node HPE iLO corpus**, both
under the ≥0.80 top-5 bar (NO-GO), though ~30× over random. The read: on a large host, global text
similarity fills the top-5 with look-alike sibling sensor leaves rather than true transitions.
Representation similarity alone is **not** enough; *trained* components are load-bearing.

That experiment belonged to a pointer/candidate action-selection design that the current
architecture **supersedes** — today's stack is Phase 2/3 extraction feeding two separate encoders
(`z_rest`, `z_method`) and a separate RL policy. But its lesson is baked into the current claims
discipline: **v1 asserts no zero-shot universal-REST transfer and no shared cross-vendor latent.**
Redfish is the first proof environment; every transfer claim must arrive with a measured number, the
way this one did.


# Author: Mus mbayramo@stanford.edu
