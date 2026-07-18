# UC-01 — Power & boot orchestration

> **Illustrative episode, not shipped code.** The current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 captured Redfish records → Phase 1 → `model_x` → D1 judge-verified
> inverse labels → Phase 2 **unordered** `rest_api_list` → Phase 3 **unordered** `calls: list[Call]`
> (explicit `http_method` + `arguments`) → two separate encoders `z_rest` + `z_method` → a
> **separate RL policy** owning order/retry/wait/recovery → the JSON simulator. Contract examples
> below are illustrative; the machine-readable schema (`configs/contracts/*.yaml`) is authoritative.
> Redfish is the first proof environment, not a permanent ontology.

## The goal, and what the pipeline extracts from it

An operator does not say "PATCH `BootSourceOverrideTarget` then POST `ComputerSystem.Reset`." They
say: *"Boot this node from PXE one time, then bring it up."* The language side extracts **what** the
request touches — never the order:

```jsonc
// Phase 2 — UNORDERED unique set (k=2 here; a single API would still be a length-1 list)
{ "rest_api_list": [
    "/redfish/v1/Systems/{id}",
    "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset"
] }

// Phase 3 — UNORDERED; each Call = { rest_api, explicit http_method, arguments }
{ "calls": [
    { "rest_api": "/redfish/v1/Systems/{id}",
      "http_method": "PATCH",
      "arguments": { "Boot": { "BootSourceOverrideTarget": "Pxe",
                               "BootSourceOverrideEnabled": "Once" } } },
    { "rest_api": "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset",
      "http_method": "POST",
      "arguments": { "ResetType": "On" } }
] }
```

Nothing here says *when*. Sequencing ("stage the override before the reset"), idempotence, waiting,
and recovery belong to the separate RL policy; a known-good order is recorded only as separate
RL-oracle evidence (`expert_call_order`), outside this contract.

## Why a script or a chatbot struggles here

A shell script hard-codes one vendor's URL and one `ResetType`, and cheerfully sends `On` to a node
that is already on — or `GracefulShutdown` to a node that is already off, then waits forever for a
state change that will never come. A chatbot will happily invent `ResetType: "Reboot"` or
`BootSourceOverrideTarget: "Network"` — plausible strings that no schema on that box accepts.
Neither one *checks* the box afterward: both report success from having sent the request, not from
the machine reaching the state. Power and boot look trivial until idempotence, the right
graceful-vs-forced reset, and vendor enum drift all bite at once.

## Observation and the legal methods

The RL policy's observation is a Redfish GET of the ComputerSystem resource — `PowerState`, the
`Boot` object, `Actions`, and the `@Redfish.AllowableValues` annotations the box itself publishes:

```jsonc
// GET /redfish/v1/Systems/{id}
{
  "@odata.type": "#ComputerSystem.v1_x_x.ComputerSystem",
  "PowerState": "Off",
  "Boot": {
    "BootSourceOverrideTarget": "None",
    "BootSourceOverrideEnabled": "Disabled",
    "BootSourceOverrideTarget@Redfish.AllowableValues": ["None","Pxe","Hdd","Cd","BiosSetup"]
  },
  "Actions": {
    "#ComputerSystem.Reset": {
      "target": "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset",
      "ResetType@Redfish.AllowableValues": ["On","ForceOff","GracefulShutdown","GracefulRestart","ForceRestart"]
    }
  }
}
```

Which methods are legal on which URL comes from the captured interface — `allowed_methods_mapping`
in `rest_api_map.npy`, the binding contract written by `redfish_ctl` discovery. Argument values are
bound only from the box's own allowable-values list — the pipeline cannot name a `ResetType` the
machine did not advertise.

| Endpoint | Method | Legal argument (from box) | Effect |
|---|---|---|---|
| `/redfish/v1/Systems/{id}` | `GET` | — | read PowerState + Boot (observation) |
| `/redfish/v1/Systems/{id}` | `PATCH` | `Boot.BootSourceOverrideTarget ∈ {Pxe,Hdd,Cd,BiosSetup}` | set boot override |
| `/redfish/v1/Systems/{id}` | `PATCH` | `Boot.BootSourceOverrideEnabled ∈ {Once,Continuous,Disabled}` | one-shot vs sticky |
| `/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset` | `POST` | `ResetType ∈ AllowableValues` | power/reset transition |

`Pxe` + `Once` = "next boot only, then revert." `Continuous` would make PXE sticky across every boot
— a different goal, and `Once` is what the evaluator will hold the run to.

## The trajectory (RL execution)

**Observe.** `GET /redfish/v1/Systems/{id}` → `PowerState: "Off"`, `BootSourceOverrideTarget: "None"`.

**Idempotence check (up front).** The evaluator diffs the current observation against the goal state
*before* any mutation. If every field already matched, the episode is done with an empty trajectory
— a verified no-op, not a re-issued reset. Here two facts are unsatisfied, so the policy proceeds.

**Execute the boot override first (RL-chosen order).** The policy has learned — from simulator
reward and `expert_call_order` evidence — that staging the override before powering on stays within
a one-reset budget:

```http
PATCH /redfish/v1/Systems/{id}
{ "Boot": { "BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once" } }
```

**Dry-run → approve → execute.** This mutates hardware, so the guardrail intercepts: it renders the
exact request, shows the predicted diff (`Boot: None/Disabled → Pxe/Once`), and pauses for approval.
On approval the PATCH executes; a read-only GET would have auto-proceeded with no pause.

**Observe.** `GET` again → `BootSourceOverrideTarget: "Pxe"`, `BootSourceOverrideEnabled: "Once"`.
Boot half satisfied.

**Execute the power transition.** The node is `Off`, so the correct transition to reach
`PowerState: On` is `ResetType: "On"` — *not* `GracefulRestart` (there is no OS to restart) and
*not* `ForceOff` (wrong direction). A prefer-graceful constraint only kicks in when the machine is
already `On` and must come down: then the policy picks `GracefulShutdown` over `ForceOff`, and
`GracefulRestart` over `ForceRestart`, unless a stuck state forces the hard variant.

```http
POST /redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
```

**Dry-run → approve → execute → observe.** Guardrail pauses (mutation), approval, POST, then poll
`GET` until `PowerState: "On"`.

**Evaluate.** All target facts now measured true against the re-read state → goal reached.

## What "done" means

Done is a measurement, never a claim. The evaluator re-reads `GET /redfish/v1/Systems/{id}` and
asserts each fact against the fresh body:

- `PowerState == "On"` — read back from the resource, not inferred from "the POST returned 204."
- `Boot.BootSourceOverrideTarget == "Pxe"` and `BootSourceOverrideEnabled == "Once"`.

A `2xx` on the Reset action means *accepted*, not *reached*; the box may still be transitioning. The
policy keeps observing until the observed state equals the target (or a timeout ends the episode as
failure). Success is defined by the world, not by the agent's confidence.

## Constraints, risk, and the guardrail

Every action here except the GETs is a **mutation** — a `PATCH` to `Boot` and a `POST` to
`ComputerSystem.Reset`. A `ForceOff` or `ForceRestart` on a running node can lose in-flight work, so
those carry the highest risk weight.

The guardrail is fixed: **dry-run → approval → execute** for anything that changes state. It renders
the concrete request and the predicted state diff, then blocks until an operator approves. Read-only
lanes (the observation GETs, the idempotence pre-check) auto-proceed with no pause. Approval is
per-mutation, so a run that would flip a sticky `Continuous` override or issue a hard reset cannot
slip through as a side effect of a "just power it on" instruction.

## What transfers / what it learned

The same episode, run on Dell iDRAC, Supermicro, HPE iLO, or a generic DMTF service, differs only in
the enum sets and OEM extras each box advertises — which is exactly why argument values are read
from `@Redfish.AllowableValues` on the resource instead of a hard-coded table. The call *shape*
(`ComputerSystem.Reset` POST, `Boot` PATCH on the ComputerSystem) is standard schema; the *values*
are drawn per-box, so a vendor that omits `Pxe` or adds an OEM reset type is handled without a code
edit. (Cross-vendor transfer is a measured question, not an assumption — see the recorded evidence
in `uc-06-fleet-remediation-multivendor.md`.)

From HER, even a run that overshot teaches: an episode that ended in `PowerState: On` with a
`Continuous` override — wrong for *this* goal — is relabeled as a successful trajectory for the goal
it *did* reach, so the failed rollout still yields gradient. Over many episodes the RL policy
converges on the **shortest safe path**: skip already-satisfied fields (idempotence), pick the
minimal-risk `ResetType` that reaches the target power state, and combine the override fields in one
PATCH rather than two round-trips. The simplest goal is the one that teaches the loop its spine —
verify first, act only on the delta, prove it by re-reading.


# Author: Mus mbayramo@stanford.edu
