# UC-02 — BIOS / attribute convergence

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery) → JSON simulator. Contract examples are
> illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the first proof environment.

## The goal, and what the pipeline extracts from it

An operator wants a node's firmware settings brought to a known-good baseline: boot mode UEFI,
virtualization and IOMMU on, and — because these are performance nodes — hyper-threading disabled.
In natural language: *"Make sure this box's BIOS matches our compute baseline, and reboot it once if
you have to."* The language side extracts the touched APIs and their bindings — never the sequence:

```jsonc
// Phase 2 — UNORDERED unique set (k=3)
{ "rest_api_list": [
    "/redfish/v1/Systems/{id}/Bios",
    "/redfish/v1/Systems/{id}/Bios/Settings",
    "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset"
] }

// Phase 3 — UNORDERED; read-only rows carry arguments {}
{ "calls": [
    { "rest_api": "/redfish/v1/Systems/{id}/Bios",
      "http_method": "GET",  "arguments": {} },
    { "rest_api": "/redfish/v1/Systems/{id}/Bios/Settings",
      "http_method": "PATCH",
      "arguments": { "Attributes": { "BootMode": "Uefi", "ProcVirtualization": "Enabled",
                                     "Iommu": "Enabled", "LogicalProc": "Disabled" } } },
    { "rest_api": "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset",
      "http_method": "POST", "arguments": { "ResetType": "GracefulRestart" } }
] }
```

The stage-before-reset ordering, the one-reset budget, and the wait-for-apply behavior are the RL
policy's job, learned from reward and from separate `expert_call_order` oracle evidence — they are
deliberately absent from the JSON above. Success is checked against **current** attributes, not the
settings buffer; that single choice is what keeps the agent honest.

## Why a script or a chatbot struggles here

BIOS on Redfish is not a live register — writing an attribute does not change it. A PATCH lands in a
*pending* settings buffer and only takes effect on the next qualifying reset. A naive script (or an
LLM told "set LogicalProc=Disabled") PATCHes the resource, gets `200 OK`, and reports success —
while the running machine is unchanged. The trap is a **false Markov collapse**: the agent treats
the acknowledgement as the new state, so its model of the world and the actual firmware diverge
silently until someone reboots weeks later and gets a surprise. Getting this right needs staging,
exactly-one reset, and a re-read *after apply* — a sequence, and sequences are exactly what the
separate RL policy exists to learn.

## Observation and the legal methods

The RL policy's observation is a Redfish GET of the Bios resource. What it reads:

- `Attributes` — the **current** effective values.
- `@Redfish.Settings` — the annotation whose `SettingsObject.@odata.id` points at the pending
  settings resource, plus `SupportedApplyTimes` and any prior-apply `Messages`.
- On the settings object: `@Redfish.SettingsApplyTime` (or the `ApplyTime` in the PATCH payload),
  typically `OnReset` for BIOS.

Which methods are legal on which URL comes from the captured interface's `allowed_methods_mapping`
in `rest_api_map.npy` (the binding contract from `redfish_ctl` discovery). Nothing is invented:

| Endpoint (from the walk) | Method | Why it is legal here |
|---|---|---|
| `/redfish/v1/Systems/{id}/Bios` | `GET` | read current `Attributes` + `@Redfish.Settings` |
| `/redfish/v1/Systems/{id}/Bios/Settings` | `PATCH` | settings object advertises PATCH; stage pending values |
| `/redfish/v1/Systems/{id}` | `POST` (`ComputerSystem.Reset`) | `Actions` target present; applies OnReset settings |
| `/redfish/v1/Systems/{id}/Bios/Settings` | `GET` | confirm pending buffer state after apply |

`GET`s are read-only lanes (auto-proceed). The `PATCH` and the `Reset` `POST` are mutations and each
must clear the guardrail.

## The trajectory (RL execution)

Abbreviated observe → choose → dry-run/approve → execute → observe → evaluate loop, with real
standard Redfish calls:

1. **Observe.** `GET /redfish/v1/Systems/{id}/Bios`. Current `Attributes` show
   `LogicalProc: "Enabled"`, `BootMode: "Uefi"` already correct, `Iommu: "Disabled"`. The
   `@Redfish.Settings` annotation resolves the settings object to
   `/redfish/v1/Systems/{id}/Bios/Settings`; `SupportedApplyTimes` includes `OnReset`.

2. **Choose.** Target ≠ current on two attributes. The only legal way to change them is a PATCH to
   the settings object; the policy selects it (staging first is the learned order).

3. **Dry-run.** The guardrail renders the exact request and diff without sending it:

   ```http
   PATCH /redfish/v1/Systems/{id}/Bios/Settings
   Content-Type: application/json

   {
     "@Redfish.SettingsApplyTime": {
       "@odata.type": "#Settings.v1_3_5.PreferredApplyTime",
       "ApplyTime": "OnReset"
     },
     "Attributes": {
       "Iommu": "Enabled",
       "LogicalProc": "Disabled"
     }
   }
   ```
   Dry-run summary: *stages 2 attributes, ApplyTime OnReset, no immediate effect, requires 1 reset.*

4. **Approve → execute.** Operator approves the stage. PATCH returns `200`/`202`; the buffer now
   holds the two pending values.

5. **Observe (mid-flight).** `GET …/Bios` again. **Current `Attributes` are still
   `Iommu: Disabled`, `LogicalProc: Enabled`.** This is the exact point where a naive agent declares
   victory. The evaluator does not — current ≠ target, and the settings object now shows a non-empty
   pending set. Not done.

6. **Choose the reset.** Settings are staged and the one-reset budget is unspent. Legal action:
   `(…/Systems/{id}, POST ComputerSystem.Reset)`.

7. **Dry-run → approve → execute** the single reset:

   ```http
   POST /redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset
   { "ResetType": "GracefulRestart" }
   ```
   Guardrail flags this as a higher-risk power action; operator approves. One reset spent; budget
   now zero.

8. **Observe after apply.** Poll the reset `Task` to completion (no further mutation while the task
   runs — a waiting behavior the RL policy learns), then `GET …/Bios` once more.

9. **Evaluate.** Against the *re-read* current state — see below.

## What "done" means

The evaluator verifies against freshly re-read state, never against the PATCH acknowledgement or the
agent's own claim:

- `GET /redfish/v1/Systems/{id}/Bios` → assert **current** `Attributes` now equal every target value
  (`BootMode: Uefi`, `ProcVirtualization: Enabled`, `Iommu: Enabled`, `LogicalProc: Disabled`).
- `GET /redfish/v1/Systems/{id}/Bios/Settings` → assert the pending `Attributes` set is empty (or
  its post-apply `Messages` report success) — nothing left staged after apply.

Only when both hold — measured, after reset — does the episode terminate as success. If current
attributes still lag, the buffer is consumed but wrong, or a second reset would be required, it is a
failure, not a partial credit. Success is never self-reported; the read-back is the reward signal.

## Constraints, risk, and the guardrail

- **Mutations and their risk.** The `Bios/Settings` PATCH is low/medium risk (staged, reversible by
  re-PATCHing before apply, no immediate effect). The `ComputerSystem.Reset` is the real risk: it
  interrupts the running workload. Both pause at **dry-run → approval → execute**; the reset is
  surfaced as the higher-severity gate.
- **Where approval pauses.** Every write stops before it leaves the agent. The operator sees the
  rendered request and diff, not a summary the model wrote about itself.
- **Budget as a hard rail.** At-most-one-reset is enforced by the environment/guard, not by hoping
  the policy learned restraint — a second reset is simply illegal once the budget is spent, which
  also defends the BMC from a reset-storm.
- **Ordering as learned behavior.** Rebooting before anything is staged burns the one reset for
  nothing; that is exactly the negative-reward pattern the RL policy trains away, with
  `expert_call_order` evidence supervising the known-good sequence.

## What transfers / what it learned

The convergence shape here — **stage → apply-via-reset → verify current** — is a reusable execution
skill, and HER is what makes the failures pay. An episode that PATCHed the wrong attribute, or
rebooted before staging, gets relabeled: *the trajectory that produced the state it actually reached
becomes a demonstration for that (accidental) goal.* The policy thereby learns the shortest safe
path — one PATCH, one reset, one verify — instead of the flailing PATCH-reboot-PATCH-reboot loop a
fresh policy tends toward.

The mechanism is DMTF-standard: `@Redfish.Settings` + a SettingsObject + `ApplyTime: OnReset` is how
BIOS convergence works on Dell iDRAC, Supermicro, HPE iLO, and generic DMTF implementations alike.
Attribute *names* differ across vendors (a Dell `LogicalProc` is not spelled the same everywhere),
but those names come from the walked tree and the operator's goal, never from a hardcoded map. And
the anti-pattern the policy internalizes — *never trust the write, verify the re-read* — is exactly
the false-Markov-collapse defense that generalizes to every pending-settings resource in Redfish
(NIC, storage, manager config), not just BIOS. Whether that carries across vendors without
retraining is a measured question — see the recorded evidence in
`uc-06-fleet-remediation-multivendor.md`.


# Author: Mus mbayramo@stanford.edu
