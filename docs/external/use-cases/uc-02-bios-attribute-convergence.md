# UC-02 — BIOS / attribute convergence

> Target loop, grounded in `docs/external/architecture/overview.md` + `docs/external/roadmap/decisions.md`. Today the code is a Phase-0 Redfish MDP shell (captured-replay env, one-hot actions); the observe → choose → guard → verify loop below is the behavior we are building toward, not what ships in Phase 0.

## The goal (in the operator's words, and the machine-checkable spec)

An operator wants a node's firmware settings brought to a known-good baseline: boot mode UEFI, virtualization and IOMMU on, and — because these are performance nodes — hyper-threading disabled. In natural language: *"Make sure this box's BIOS matches our compute baseline, and reboot it once if you have to."*

```python
Goal(
    instruction="Bring BIOS attributes to the compute baseline; one reset permitted.",
    spec={
        # machine-checkable: read back from /Systems/{id}/Bios .Attributes
        "current_attributes_equal": {
            "BootMode": "Uefi",
            "ProcVirtualization": "Enabled",
            "Iommu": "Enabled",
            "LogicalProc": "Disabled",   # hyper-threading off
        },
        "pending_settings_empty": True,   # nothing left staged after apply
    },
    constraints=[
        "at_most_one_reset",
        "reset_only_after_settings_staged",
        "no_power_action_while_task_running",
    ],
    plan=None,  # discovered from the walked resource tree, not hand-authored
)
```

The spec is deliberately written against **current** attributes, not the settings object. That single choice is what keeps the agent honest.

## Why a script or a chatbot struggles here

BIOS on Redfish is not a live register — writing an attribute does not change it. A PATCH lands in a *pending* settings buffer and only takes effect on the next qualifying reset. A naive script (or an LLM told "set LogicalProc=Disabled") PATCHes the resource, gets `200 OK`, and reports success — while the running machine is unchanged. The trap is a **false Markov collapse**: the agent treats the acknowledgement as the new state, so its model of the world and the actual firmware diverge silently until someone reboots weeks later and gets a surprise. Getting this right needs staging, exactly-one reset, and a re-read *after apply* — a sequence, not a call.

## Observation and the legal actions

The agent's observation is a Redfish GET of the Bios resource. What it reads:

- `Attributes` — the **current** effective values.
- `@Redfish.Settings` — the annotation whose `SettingsObject.@odata.id` points at the pending settings resource, plus `SupportedApplyTimes` and any prior-apply `Messages`.
- On the settings object: `@Redfish.SettingsApplyTime` (or the `ApplyTime` in the payload you PATCH), typically `OnReset` for BIOS.

Actions are drawn only from the **legal catalog**: an endpoint that actually appeared in the walked tree, paired with a method that endpoint's `allowed_methods` (from HTTP `Allow` / the resource's advertised operations) permits. Nothing is invented.

| Endpoint (from the walk) | Method | Why it is legal here |
|---|---|---|
| `/redfish/v1/Systems/{id}/Bios` | `GET` | read current `Attributes` + `@Redfish.Settings` |
| `/redfish/v1/Systems/{id}/Bios/Settings` | `PATCH` | settings object advertises PATCH; stage pending values |
| `/redfish/v1/Systems/{id}` | `POST` (`ComputerSystem.Reset`) | `Actions` target present; applies OnReset settings |
| `/redfish/v1/Systems/{id}/Bios/Settings` | `GET` | confirm pending buffer state after apply |

`GET`s are read-only lanes (auto-proceed). The `PATCH` and the `Reset` `POST` are mutations and each must clear the guardrail.

## The trajectory

Abbreviated observe → choose → dry-run/approve → execute → observe → evaluate loop, with real standard Redfish calls:

1. **Observe.** `GET /redfish/v1/Systems/{id}/Bios`. Current `Attributes` show `LogicalProc: "Enabled"`, `BootMode: "Uefi"` already correct, `Iommu: "Disabled"`. The `@Redfish.Settings` annotation resolves the settings object to `/redfish/v1/Systems/{id}/Bios/Settings`; `SupportedApplyTimes` includes `OnReset`.

2. **Choose.** Spec ≠ current on two attributes. The only legal way to change them is a PATCH to the settings object. The agent selects `(…/Bios/Settings, PATCH)` from the catalog.

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

4. **Approve → execute.** Operator approves the stage. PATCH returns `200`/`202`; the buffer now holds the two pending values.

5. **Observe (mid-flight).** `GET …/Bios` again. **Current `Attributes` are still `Iommu: Disabled`, `LogicalProc: Enabled`.** This is the exact point where a naive agent declares victory. The evaluator does not — current ≠ target, and the settings object now shows a non-empty pending set. Not done.

6. **Choose the reset.** Constraint `reset_only_after_settings_staged` is satisfied, `at_most_one_reset` still has budget. Legal action: `(…/Systems/{id}, POST ComputerSystem.Reset)`.

7. **Dry-run → approve → execute** the single reset:

   ```http
   POST /redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset
   { "ResetType": "GracefulRestart" }
   ```
   Guardrail flags this as a higher-risk power action; operator approves. One reset spent; budget now zero.

8. **Observe after apply.** Poll the reset `Task` to completion (`no_power_action_while_task_running` blocks any further mutation until then), then `GET …/Bios` once more.

9. **Evaluate.** Against the *re-read* current state — see below.

## What "done" means

The Evaluator verifies the spec against freshly re-read state, never against the PATCH acknowledgement or the agent's own claim:

- `GET /redfish/v1/Systems/{id}/Bios` → assert **current** `Attributes` now equal every value in `current_attributes_equal` (`BootMode: Uefi`, `ProcVirtualization: Enabled`, `Iommu: Enabled`, `LogicalProc: Disabled`).
- `GET /redfish/v1/Systems/{id}/Bios/Settings` → assert the pending `Attributes` set is empty (or its post-apply `Messages` report success), satisfying `pending_settings_empty`.

Only when both hold — measured, after reset — does the episode terminate as success. If current attributes still lag, the buffer is consumed but wrong, or a second reset would be required, it is a failure, not a partial credit. Success is never self-reported; the read-back is the reward signal.

## Constraints, risk, and the guardrail

- **Mutations and their risk.** The `Bios/Settings` PATCH is low/medium risk (staged, reversible by re-PATCHing before apply, no immediate effect). The `ComputerSystem.Reset` is the real risk: it interrupts the running workload. Both pause at **dry-run → approval → execute**; the reset is surfaced as the higher-severity gate.
- **Where approval pauses.** Every write stops before it leaves the agent. The operator sees the rendered request and diff, not a summary the model wrote about itself.
- **Budget as a hard rail.** `at_most_one_reset` is enforced by the catalog/guard, not by hoping the policy learned restraint — a second reset action is simply illegal once the budget is spent, which also defends the BMC from a reset-storm.
- **Ordering.** `reset_only_after_settings_staged` prevents the classic waste: rebooting before anything is staged, burning the one reset for nothing.

## What transfers / what it learned

The convergence shape here — **stage → apply-via-reset → verify current** — is a reusable skill, and HER is what makes the failures pay. An episode that PATCHed the wrong attribute, or rebooted before staging, gets relabeled: *the trajectory that produced the state it actually reached becomes an optimal demonstration for that (accidental) goal.* The agent thereby learns the shortest safe path — one PATCH, one reset, one verify — instead of the flailing PATCH-reboot-PATCH-reboot loop a fresh policy tends toward.

The transferable lesson is vendor-independent because the mechanism is DMTF-standard: `@Redfish.Settings` + a SettingsObject + `ApplyTime: OnReset` is how BIOS convergence works on Dell iDRAC, Supermicro, HPE iLO, and generic DMTF implementations alike. Attribute *names* differ across vendors (a Dell `LogicalProc` is not spelled the same everywhere), but those names come from the walked tree and the goal spec, never from a hardcoded map — so the same policy that converges BIOS on one vendor converges it on the next with no retraining of the loop, only the per-vendor attribute vocabulary discovered at observe time. And the anti-pattern it internalizes — *never trust the write, verify the re-read* — is exactly the false-Markov-collapse defense that generalizes to every pending-settings resource in Redfish (NIC, storage, manager config), not just BIOS.


# Author: Mus mbayramo@stanford.edu
