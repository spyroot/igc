# UC-03 — Firmware update workflow

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery) → JSON simulator. Contract examples are
> illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the first proof environment.

## The goal, and what the pipeline extracts from it

An operator does not want to babysit a firmware flash. They say: *"Update the BMC firmware to build
7.10.30 on this system, and nothing else"* — and expect that when the agent claims "done," the
component is actually running that build and nothing else moved. The language side extracts:

```jsonc
// Phase 2 — UNORDERED unique set (k=3)
{ "rest_api_list": [
    "/redfish/v1/UpdateService",
    "/redfish/v1/UpdateService/FirmwareInventory/BMC",
    "/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate"
] }

// Phase 3 — UNORDERED; reads carry arguments {}
{ "calls": [
    { "rest_api": "/redfish/v1/UpdateService",
      "http_method": "GET",  "arguments": {} },
    { "rest_api": "/redfish/v1/UpdateService/FirmwareInventory/BMC",
      "http_method": "GET",  "arguments": {} },
    { "rest_api": "/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate",
      "http_method": "POST",
      "arguments": { "ImageURI": "<operator-supplied image reference>",
                     "TransferProtocol": "HTTPS",
                     "Targets": ["/redfish/v1/UpdateService/FirmwareInventory/BMC"] } }
] }
```

Read-capability-first, submit-once, poll-to-terminal, verify-after — that whole *sequence* is the
separate RL policy's competency (order, waiting, retries, recovery), supervised by separate
`expert_call_order` oracle evidence, not by anything in the JSON above. The task-poll reads the
policy inserts during execution are legal calls it learned to add; they never need to appear in the
language contract.

## Why a script or a chatbot struggles here

A shell script that POSTs `SimpleUpdate` and prints the HTTP 202 has done the *easy* 5% — it
declares victory at submit, before a single byte is flashed. The hard part is the long tail: the
update is asynchronous and multi-phase (staged → verified → applied → sometimes a controller reset),
it can sit at `Running` for many minutes, and it can fail *after* accepting the job with a
`TaskState: Exception`. A chatbot that "sounds confident" has no way to distinguish a job that
completed from one that stalled at 60% or silently rolled back — and no notion that flashing the
*wrong* target, or resetting a neighbor, violated the goal.

## Observation and the legal methods

The RL policy observes Redfish `GET` results and never invents an endpoint or a verb: legal methods
per URL come from the captured interface's `allowed_methods_mapping` in `rest_api_map.npy` (the
binding contract from `redfish_ctl` discovery), and action capability from `@Redfish.ActionInfo`.
First reads:

- `GET /redfish/v1/UpdateService` — is `SimpleUpdate` in `Actions`? What `TransferProtocol` values
  does `@Redfish.ActionInfo` allow?
- `GET /redfish/v1/UpdateService/FirmwareInventory` → the `SoftwareInventory` member for the BMC, to
  read its current `Version`.

The legal surface for this goal:

| Endpoint | Method | Redfish meaning | Lane |
|---|---|---|---|
| `/redfish/v1/UpdateService` | GET | read update capability + action info | read-only |
| `/redfish/v1/UpdateService/FirmwareInventory/BMC` | GET | read current firmware `Version` | read-only |
| `/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate` | POST | stage + apply an image (mutation) | **guarded** |
| `/redfish/v1/TaskService/Tasks/{id}` | GET | poll `TaskState` / `PercentComplete` | read-only |
| `/redfish/v1/UpdateService/FirmwareInventory/BMC` | GET | re-read `Version` after apply | read-only |

Note what is *absent*: no `ComputerSystem.Reset`, no `Bios` `SettingsObject` write. Those endpoints
exist in the walk but the operator's request never named them, so Phase 2 does not select them — and
an execution that drifted into them would fail the no-collateral check below.

## The trajectory (RL execution)

**1. Observe (read-only, auto-proceed).** Confirm the service supports the action and read the
baseline version.

```http
GET /redfish/v1/UpdateService/FirmwareInventory/BMC
→ 200 { "@odata.type": "#SoftwareInventory.v1_x.SoftwareInventory",
        "Id": "BMC", "Version": "7.10.06", "Updateable": true }
```

Baseline `7.10.06` ≠ target `7.10.30`, and `Updateable: true`. The goal is not yet satisfied, and it
is achievable.

**2. Choose.** The policy selects the `SimpleUpdate` POST on exactly the BMC target. Because it is a
mutation, it enters the guardrail rather than executing.

**3. Dry-run → approval.** The agent renders the exact request it intends to send and pauses:

```http
POST /redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate
{
  "ImageURI": "<operator-supplied image reference>",
  "TransferProtocol": "HTTPS",
  "Targets": [ "/redfish/v1/UpdateService/FirmwareInventory/BMC" ]
}
```

The dry-run shows a single-element `Targets` list — exactly one component — and no reset action
bundled in. The operator approves.

**4. Execute.** The service accepts the job and returns a Task, not a result:

```http
→ 202 Accepted
   Location: /redfish/v1/TaskService/Tasks/42
   { "@odata.type": "#Task.v1_x.Task", "Id": "42", "TaskState": "New" }
```

**5. Observe the long-running task (read-only poll loop).** The policy does **not** stop here —
waiting on an async task is a learned behavior. It polls until a terminal state, treating
`New`/`Running` as "keep observing":

```http
GET /redfish/v1/TaskService/Tasks/42
→ { "TaskState": "Running", "PercentComplete": 35 }
   ... "Running", 72 ...
   ... "Running", 100 ...
→ { "TaskState": "Completed", "PercentComplete": 100,
    "TaskStatus": "OK" }
```

**The Exception path.** If a poll returns `TaskState: Exception` (or `Killed`), the policy does not
retry blindly and never reports success. It reads the Task `Messages[]` for the failure detail,
marks the goal unmet, and surfaces the failure for the operator — a failed flash is a terminal
outcome to report, not to paper over. Distinguishing retryable from terminal failures is part of the
recovery behavior the RL stage trains.

**6. Re-read and evaluate.** Reaching `Completed` is necessary but not sufficient. The policy
re-reads the inventory:

```http
GET /redfish/v1/UpdateService/FirmwareInventory/BMC
→ 200 { "Id": "BMC", "Version": "7.10.30" }
```

## What "done" means

The evaluator verifies against freshly-read state — never against the submit response or the agent's
own narration:

- **Task terminality:** the polled `Task` reached `TaskState == "Completed"` (an `Exception`/
  `Killed` terminal state fails the goal outright).
- **Version match:** the re-read `SoftwareInventory` member for the BMC reports
  `Version == "7.10.30"`. A job that "completed" but left the old build running is a **failure**,
  and only this re-read catches it.
- **Single-component:** a diff of `FirmwareInventory` versions before vs. after shows exactly one
  member changed. Any second version delta violates the "and nothing else" clause.
- **No collateral:** no unrelated `Reset` or config write was issued — the extracted contract never
  contained one, and the evaluator confirms nothing else in scope moved.

Only when all clauses hold does the episode terminate as success. "Completed" from the update
service is one input to that judgment, not the judgment itself.

## Constraints, risk, and the guardrail

Firmware flashing is among the highest-risk Redfish mutations: a bad image or an interrupted apply
can brick a management controller, and a stray controller reset can drop a live host. So:

- **`SimpleUpdate` is a guarded mutation.** It always runs dry-run → approval → execute. The dry-run
  exposes the literal `ImageURI`, `TransferProtocol`, and `Targets` for human inspection; approval
  is where the operator catches a wrong image or a fat-fingered target *before* anything is written.
- **Read lanes auto-proceed.** The `UpdateService`/`FirmwareInventory` reads and the whole
  `TaskService/Tasks/{id}` poll loop carry no approval cost, so monitoring is free and continuous.
- **Scope is fenced by the contract, not by good intentions.** The extracted `rest_api_list` for
  this request contains no reset and no BIOS write, so the execution has no business issuing one —
  and the evaluator's no-collateral check catches any drift.
- Credentials and image references are operator-supplied at run time — never baked into the goal,
  the policy, or logs.

## What transfers / what it learned

**HER turns the slow path into a lesson.** A firmware episode is long and mostly waiting. When a run
overshoots — extra redundant `GET`s, a poll interval that was too eager, or an abandoned attempt
that landed on a *different* valid version — Hindsight Experience Replay relabels that trajectory
with the goal it *did* achieve ("reached version X, one component, task Completed"). The policy
extracts a clean, near-shortest recipe (read capability → submit once → poll to terminal → verify)
from otherwise wasted episodes, and stops padding the path with unnecessary reads.

**The shape is standard DMTF.** `UpdateService.SimpleUpdate` returns a `Task`; poll
`TaskState`/`PercentComplete` to a terminal state; then verify a `SoftwareInventory.Version`. Only
the concrete endpoints and inventory member ids differ across Dell iDRAC, Supermicro, HPE iLO, and
generic DMTF implementations — the async-task discipline is the reusable execution skill. How well
it carries to an unseen vendor is a measured question, not a promise: see the recorded transfer
evidence in `uc-06-fleet-remediation-multivendor.md`.


# Author: Mus mbayramo@stanford.edu
