# UC-01 — Power & boot orchestration

> **Target loop, not shipped code.** This page describes the intended end-to-end IGC agent loop, grounded in `docs/ARCHITECTURE.md` (target interaction model) and `docs/DECISIONS.md` (D-001/D-002 action-candidate design). Today the code is a Phase-0 Redfish MDP shell: a mock server replays captured responses and a legacy one-hot policy runs a smoke DQN/HER loop. The behavior below is where that shell is going.

## The goal (in the operator's words, and the machine-checkable spec)

An operator does not say "PATCH `BootSourceOverrideTarget` then POST `ComputerSystem.Reset`." They say: *"Boot this node from PXE once, then power it on."* IGC takes that instruction and pins it to a spec it can verify by re-reading the machine — no interpretation left to chance.

```python
Goal(
    instruction="Boot the node from PXE one time, then bring it up.",
    spec={
        "Boot.BootSourceOverrideTarget": "Pxe",
        "Boot.BootSourceOverrideEnabled": "Once",   # one-shot, not sticky
        "PowerState": "On",
    },
    constraints={
        "prefer": "graceful",        # avoid ForceOff/ForceRestart unless required
        "no_data_loss": True,        # a running OS must be shut down cleanly
        "idempotent": True,          # re-running an already-satisfied goal is a no-op
    },
    plan=None,   # discovered from the walked tree, not hard-coded
)
```

The spec is the contract. Every field is a Redfish property IGC can GET back and compare. Nothing here says *how* — the "how" is discovered from the resource tree and the endpoint's own allowed methods.

## Why a script or a chatbot struggles here

A shell script hard-codes one vendor's URL and one `ResetType`, and cheerfully sends `On` to a node that is already on — or `GracefulShutdown` to a node that is already off, then waits forever for a state change that will never come. A chatbot will happily invent `ResetType: "Reboot"` or `BootSourceOverrideTarget: "Network"` — plausible strings that no schema on that box accepts. Neither one *checks* the box afterward: both report success from having sent the request, not from the machine reaching the state. Power and boot look trivial until idempotence, the right graceful-vs-forced reset, and vendor enum drift all bite at once.

## Observation and the legal actions

The observation is a Redfish GET of the ComputerSystem resource — `PowerState`, the `Boot` object, `Actions`, and the `@Redfish.AllowableValues` annotations the box itself publishes:

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

The legal action catalog is the cross-product of **endpoints from the walked tree** and **methods that endpoint actually allows** (`allowed_methods_mapping` from the `rest_api_map.npy` contract). Argument values come only from the box's own allowable-values list — the agent cannot name a `ResetType` the machine did not advertise.

| Endpoint | Method | Legal argument (from box) | Effect |
|---|---|---|---|
| `/redfish/v1/Systems/{id}` | `GET` | — | read PowerState + Boot (observation) |
| `/redfish/v1/Systems/{id}` | `PATCH` | `Boot.BootSourceOverrideTarget ∈ {Pxe,Hdd,Cd,BiosSetup}` | set boot override |
| `/redfish/v1/Systems/{id}` | `PATCH` | `Boot.BootSourceOverrideEnabled ∈ {Once,Continuous,Disabled}` | one-shot vs sticky |
| `/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset` | `POST` | `ResetType ∈ AllowableValues` | power/reset transition |

`Pxe` + `Once` = "next boot only, then revert." `Continuous` would make PXE sticky across every boot — a different goal, and the spec's `Once` is what the evaluator will hold IGC to.

## The trajectory

**Observe.** `GET /redfish/v1/Systems/{id}` → `PowerState: "Off"`, `BootSourceOverrideTarget: "None"`.

**Idempotence check (up front).** The evaluator diffs the current observation against `spec` *before* any mutation. If every field already matched, the goal is done and the trajectory is empty — a verified no-op, not a re-issued reset. Here two fields are unsatisfied, so IGC plans.

**Choose (boot override).** From the legal catalog, the pointer policy selects `PATCH /redfish/v1/Systems/{id}`; the argument decoder fills slots only from the box's allowable values.

```http
PATCH /redfish/v1/Systems/{id}
{ "Boot": { "BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once" } }
```

**Dry-run → approve → execute.** This mutates hardware, so the guardrail intercepts: it renders the exact request, shows the predicted diff (`Boot: None/Disabled → Pxe/Once`), and pauses for approval. On approval the PATCH executes; a read-only GET would have auto-proceeded with no pause.

**Observe.** `GET` again → `BootSourceOverrideTarget: "Pxe"`, `BootSourceOverrideEnabled: "Once"`. Boot half satisfied.

**Choose (power).** The node is `Off`, so the correct transition to reach `PowerState: On` is `ResetType: "On"` — *not* `GracefulRestart` (there is no OS to restart) and *not* `ForceOff` (wrong direction). The `constraints.prefer = graceful` bias only kicks in when the machine is already `On` and must come down: then IGC picks `GracefulShutdown` over `ForceOff`, and `GracefulRestart` over `ForceRestart`, unless a constraint or a stuck state forces the hard variant.

```http
POST /redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
```

**Dry-run → approve → execute → observe.** Guardrail pauses (mutation), approval, POST, then poll `GET` until `PowerState: "On"`.

**Evaluate.** All three spec fields now measured true against the re-read state → goal reached.

## What "done" means

Done is a measurement, never a claim. The evaluator re-reads `GET /redfish/v1/Systems/{id}` and asserts each spec field against the fresh body:

- `PowerState == "On"` — read back from the resource, not inferred from "the POST returned 204."
- `Boot.BootSourceOverrideTarget == "Pxe"` and `BootSourceOverrideEnabled == "Once"`.

A `2xx` on the Reset action means *accepted*, not *reached*; the box may still be transitioning. IGC keeps observing until the observed state equals the spec (or a timeout ends the episode as failure). Success is defined by the world, not by the agent's confidence.

## Constraints, risk, and the guardrail

Every action here except the GETs is a **mutation** — a `PATCH` to `Boot` and a `POST` to `ComputerSystem.Reset`. A `ForceOff` or `ForceRestart` on a running node can lose in-flight work, so those carry the highest risk weight and the `no_data_loss` constraint steers away from them.

The guardrail is fixed: **dry-run → approval → execute** for anything that changes state. It renders the concrete request and the predicted state diff, then blocks until an operator approves. Read-only lanes (the observation GETs, the idempotence pre-check) auto-proceed with no pause. Approval is per-mutation, so a plan that flips a sticky `Continuous` override or issues a hard reset cannot slip through as a side effect of a "just power it on" instruction.

## What transfers / what it learned

The same trajectory, run on Dell iDRAC, Supermicro, HPE iLO, or a generic DMTF service, differs only in the enum sets and OEM extras each box advertises — which is exactly why IGC reads `@Redfish.AllowableValues` off the resource instead of hard-coding a table. The action *shape* (`ComputerSystem.Reset` POST, `Boot` PATCH on the ComputerSystem) is standard schema and transfers unchanged; the *values* are drawn per-box, so a vendor that omits `Pxe` or adds an OEM reset type is handled without a code edit.

From HER, even a run that overshot teaches: an episode that ended in `PowerState: On` with a `Continuous` override — wrong for *this* goal — is relabeled as a successful trajectory for the goal it *did* reach, so the failed rollout still yields gradient. Over many episodes the policy converges on the **shortest safe path**: skip already-satisfied fields (idempotence), pick the minimal-risk `ResetType` that reaches the target power state, and set the override in the same PATCH rather than two round-trips. The simplest goal is the one that teaches the loop its spine — verify first, act only on the delta, prove it by re-reading.


# Author: Mus mbayramo@stanford.edu
