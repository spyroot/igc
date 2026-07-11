# UC-03 — Firmware update workflow

> Target loop, grounded in `docs/ARCHITECTURE.md` + `docs/DECISIONS.md`. Today the code is a Phase-0 Redfish MDP shell (captured-data replay, one-hot actions); the behavior below is the end-to-end loop IGC is being built to run.

## The goal (in the operator's words, and the machine-checkable spec)

An operator does not want to babysit a firmware flash. They want to say what version should be running and walk away — and be sure that when the agent claims "done," the component is actually running that build and nothing else moved.

```python
Goal(
    instruction="Update the BMC firmware to build 7.10.30 on this system, and nothing else.",
    spec={
        "target_component": "BMC",                 # SoftwareInventory member for the manager
        "target_version": "7.10.30",               # exact Version string after update
        "task_terminal_state": "Completed",        # UpdateService task must reach Completed
    },
    constraints=[
        "single_component",                        # exactly one inventory member changes
        "no_reset_of_unrelated_targets",           # do not touch other Managers/Systems
        "no_config_writes",                        # firmware only; no BIOS attribute / boot changes
        "mutation_requires_approval",              # SimpleUpdate pauses for human OK
    ],
    plan=[
        "read UpdateService + FirmwareInventory to confirm current version and update capability",
        "submit SimpleUpdate for exactly the target component",
        "monitor the returned Task to a terminal state",
        "re-read SoftwareInventory and verify the running Version equals the target",
    ],
)
```

The spec is the contract. Every clause is machine-checkable against a re-read of the box — not against what the update service reported at submit time.

## Why a script or a chatbot struggles here

A shell script that POSTs `SimpleUpdate` and prints the HTTP 202 has done the *easy* 5% — it declares victory at submit, before a single byte is flashed. The hard part is the long tail: the update is asynchronous and multi-phase (staged → verified → applied → sometimes a controller reset), it can sit at `Running` for many minutes, and it can fail *after* accepting the job with a `TaskState: Exception`. A chatbot that "sounds confident" has no way to distinguish a job that completed from one that stalled at 60% or silently rolled back — and no notion that flashing the *wrong* target, or resetting a neighbor, violated the goal.

## Observation and the legal actions

The agent observes Redfish `GET` results and never invents an endpoint or a verb. Every candidate action is a pair — an endpoint from the walked resource tree, and a method advertised by that endpoint's `Allow` header or `@Redfish.ActionInfo`. First reads:

- `GET /redfish/v1/UpdateService` — is `SimpleUpdate` in `Actions`? What `TransferProtocol` values does `@Redfish.ActionInfo` allow?
- `GET /redfish/v1/UpdateService/FirmwareInventory` → the `SoftwareInventory` member for the BMC, to read its current `Version`.

From that walk, the legal action catalog for this goal:

| Endpoint | Method | Redfish meaning | Lane |
|---|---|---|---|
| `/redfish/v1/UpdateService` | GET | read update capability + action info | read-only |
| `/redfish/v1/UpdateService/FirmwareInventory/BMC` | GET | read current firmware `Version` | read-only |
| `/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate` | POST | stage + apply an image (mutation) | **guarded** |
| `/redfish/v1/TaskService/Tasks/{id}` | GET | poll `TaskState` / `PercentComplete` | read-only |
| `/redfish/v1/UpdateService/FirmwareInventory/BMC` | GET | re-read `Version` after apply | read-only |

Note what is *absent*: no `ComputerSystem.Reset`, no `Bios` `SettingsObject` write. Those endpoints exist in the walk but are not in this goal's catalog, so the policy cannot select them.

## The trajectory

Abbreviated observe → choose → dry-run/approve → execute → observe → evaluate loop with real standard Redfish calls.

**1. Observe (read-only, auto-proceed).** Confirm the service supports the action and read the baseline version.

```http
GET /redfish/v1/UpdateService/FirmwareInventory/BMC
→ 200 { "@odata.type": "#SoftwareInventory.v1_x.SoftwareInventory",
        "Id": "BMC", "Version": "7.10.06", "Updateable": true }
```

Baseline `7.10.06` ≠ target `7.10.30`, and `Updateable: true`. The goal is not yet satisfied, and it is achievable.

**2. Choose.** The policy scores the catalog against the goal and selects `SimpleUpdate` on exactly the BMC target. Because it is a `POST` that mutates hardware, it enters the guardrail rather than executing.

**3. Dry-run → approval.** The agent renders the exact request it intends to send and pauses:

```http
POST /redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate
{
  "ImageURI": "<operator-supplied image reference>",
  "TransferProtocol": "HTTPS",
  "Targets": [ "/redfish/v1/UpdateService/FirmwareInventory/BMC" ]
}
```

The dry-run shows a single-element `Targets` list — the guardrail's `single_component` check passes — and no reset action bundled in. The operator approves.

**4. Execute.** The service accepts the job and returns a Task, not a result:

```http
→ 202 Accepted
   Location: /redfish/v1/TaskService/Tasks/42
   { "@odata.type": "#Task.v1_x.Task", "Id": "42", "TaskState": "New" }
```

**5. Observe the long-running task (read-only poll loop).** The agent does **not** stop here. It polls until a terminal state, treating `New`/`Running` as "keep observing":

```http
GET /redfish/v1/TaskService/Tasks/42
→ { "TaskState": "Running", "PercentComplete": 35 }
   ... "Running", 72 ...
   ... "Running", 100 ...
→ { "TaskState": "Completed", "PercentComplete": 100,
    "TaskStatus": "OK" }
```

**The Exception path.** If a poll returns `TaskState: Exception` (or `Killed`), the agent does not retry blindly and never reports success. It reads the Task `Messages[]` for the failure detail, marks the goal unmet, and surfaces the failure for the operator — a failed flash is a terminal outcome to report, not to paper over.

**6. Re-read and evaluate.** Reaching `Completed` is necessary but not sufficient. The agent re-reads the inventory:

```http
GET /redfish/v1/UpdateService/FirmwareInventory/BMC
→ 200 { "Id": "BMC", "Version": "7.10.30" }
```

## What "done" means

The Evaluator verifies the spec against freshly-read state — never against the submit response or the agent's own narration:

- **Task terminality:** the polled `Task` reached `TaskState == "Completed"` (an `Exception`/`Killed` terminal state fails the goal outright).
- **Version match:** the re-read `SoftwareInventory` member for the BMC reports `Version == "7.10.30"`. A job that "completed" but left the old build running is a **failure**, and only this re-read catches it.
- **Single-component:** a diff of `FirmwareInventory` versions before vs. after shows exactly one member changed. Any second version delta violates `single_component`.
- **No collateral:** no unrelated `Reset` or config write was issued (the catalog made those unselectable, and the evaluator confirms nothing else in scope moved).

Only when all clauses hold does the episode terminate as success. "Completed" from the update service is one input to that judgment, not the judgment itself.

## Constraints, risk, and the guardrail

Firmware flashing is among the highest-risk Redfish mutations: a bad image or an interrupted apply can brick a management controller, and a stray controller reset can drop a live host. So:

- **`SimpleUpdate` is a guarded mutation.** It always runs dry-run → approval → execute. The dry-run exposes the literal `ImageURI`, `TransferProtocol`, and `Targets` for human inspection; approval is where the operator catches a wrong image or a fat-fingered target *before* anything is written.
- **Read lanes auto-proceed.** The `UpdateService`/`FirmwareInventory` reads and the whole `TaskService/Tasks/{id}` poll loop carry no approval cost, so monitoring is free and continuous.
- **Scope is fenced by the catalog, not by good intentions.** `ComputerSystem.Reset` and BIOS `SettingsObject` writes are out of this goal's action set, so the policy structurally cannot bundle them into a firmware job.
- Credentials and image references are operator-supplied at run time — never baked into the goal, the policy, or logs.

## What transfers / what it learned

**HER turns the slow path into a lesson.** A firmware episode is long and mostly waiting. When a run overshoots — extra redundant `GET`s, a poll interval that was too eager, or an abandoned attempt that landed on a *different* valid version — Hindsight Experience Replay relabels that trajectory with the goal it *did* achieve ("reached version X, one component, task Completed"). The agent extracts a clean, near-shortest recipe (read capability → submit once → poll to terminal → verify) from otherwise wasted episodes, and stops padding the path with unnecessary reads.

**The shape is vendor-portable.** The pattern — `UpdateService.SimpleUpdate` returns a `Task`, poll `TaskState`/`PercentComplete` to a terminal state, then verify a `SoftwareInventory.Version` — is standard DMTF Redfish. Only the concrete endpoints and inventory member ids differ across Dell iDRAC, Supermicro, HPE iLO, and generic DMTF implementations. Because the agent discovers the catalog from each machine's own walked tree rather than hardcoding paths, the "submit → monitor to Completed → verify version" competency transfers to a new vendor's controller with little or no retraining — the async-task discipline is the reusable skill, not any one URL.


# Author: Mus mbayramo@stanford.edu
