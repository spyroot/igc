# UC-07 — Log & telemetry triage

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery) → JSON simulator. Contract examples are
> illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the first proof environment.

Operational hygiene is the friendliest place to meet an RL agent that acts on real hardware, because
the blast radius is small and the audit trail is everything. This scenario is almost entirely READ —
pull the event log, snapshot a few sensors — wrapped around exactly **one** gated mutation: clearing
the log. That single `ClearLog` is where dry-run → approval → execute earns its keep, and where a
verified, replayable trajectory beats a human clicking through a BMC UI.

## The goal, and what the pipeline extracts from it

The operator says: *"Before we hand this node back, capture everything the System Event Log has
recorded since the maintenance window opened, then clear the log so the next run starts clean — but
don't clear it until the entries are safely archived."* (`T0`, the window cutoff, is
operator-supplied; the agent does not invent it.)

```jsonc
// Phase 2 — UNORDERED unique set (k=3 shown; the set carries no ordering)
{ "rest_api_list": [
    "/redfish/v1/Managers/{id}/LogServices/SEL/Entries",
    "/redfish/v1/Chassis/{id}/Thermal",
    "/redfish/v1/Managers/{id}/LogServices/SEL/Actions/LogService.ClearLog"
] }

// Phase 3 — UNORDERED; read-only rows carry arguments {}
{ "calls": [
    { "rest_api": "/redfish/v1/Managers/{id}/LogServices/SEL/Entries",
      "http_method": "GET",  "arguments": {} },
    { "rest_api": "/redfish/v1/Chassis/{id}/Thermal",
      "http_method": "GET",  "arguments": {} },
    { "rest_api": "/redfish/v1/Managers/{id}/LogServices/SEL/Actions/LogService.ClearLog",
      "http_method": "POST", "arguments": {} }
] }
```

**The archive-before-clear ordering is deliberately absent from this JSON.** It is exactly the kind
of prerequisite the separate RL policy owns: the known-good sequence is recorded as separate
RL-oracle evidence (`expert_call_order`), the evaluator rejects a run that cleared first, and the
policy learns from both. Nothing in the Phase 2/3 language contract carries order.

## Why a script or a chatbot struggles here

A hard-coded script assumes one vendor's SEL lives at one fixed path and one fixed `ClearLog` target
— point it at a different BMC and it either 404s or, worse, clears the wrong log. A chatbot will
happily *say* "the log is cleared" without ever re-reading it, and will paste a `ClearLog` POST with
no gate between "collect" and "destroy." Neither one guarantees the ordering constraint — archive
*before* clear — because neither verifies state; they narrate intent. And neither leaves an artifact
you could hand to an auditor: what was on the log, at what timestamps, who approved the wipe, and
proof the wipe took. The whole value here is a machine that refuses to report success until the
re-read proves it.

## Observation and the legal methods

The observation is a Redfish GET result. The policy starts from the service root and the walked
resource tree, then reads the LogService that owns the SEL and its `Entries` collection. It never
guesses a URL: legal methods per endpoint come from the captured interface's
`allowed_methods_mapping` in `rest_api_map.npy` (the binding contract from `redfish_ctl`
discovery). If a host does not expose a `TelemetryService`, that row simply is not in the legal
surface, and the policy falls back to the `Thermal` / `Power` reads that are; it cannot select an
endpoint the walk never found.

A few legal rows for this scenario:

| Endpoint (from the walked tree) | Method | Kind | Legal because |
| --- | --- | --- | --- |
| `/redfish/v1/Managers/{id}/LogServices/SEL` | `GET` | read | LogService resource, `GET` allowed |
| `/redfish/v1/Managers/{id}/LogServices/SEL/Entries` | `GET` | read | LogEntryCollection, `GET` allowed |
| `/redfish/v1/Chassis/{id}/Thermal` | `GET` | read | Thermal (`Temperatures[]`), `GET` allowed |
| `/redfish/v1/Chassis/{id}/Power` | `GET` | read | Power (`PowerControl[]`), `GET` allowed |
| `/redfish/v1/TelemetryService/MetricReports/{id}` | `GET` | read | MetricReport, `GET` allowed |
| `.../LogServices/SEL/Actions/LogService.ClearLog` | `POST` | **mutate** | `#LogService.ClearLog` advertised in `Actions` |

Only the last row is a mutation. Everything above it is an auto-proceed read lane.

Two properties worth naming:

- **The method is the resource's, not the agent's.** `POST` is legal on the `ClearLog` target only
  because that endpoint's `allowed_methods` advertises it. The agent cannot conjure a `DELETE` on a
  collection that forbids it.
- **The surface is host-specific.** It is rebuilt from *this* controller's walk, so a policy trained
  against one vendor's tree does not carry a stale URL into another's — it re-derives the legal set
  on the machine in front of it.

## The trajectory (RL execution)

Abbreviated observe → choose → (dry-run / approve) → execute → observe → evaluate, with real
standard Redfish calls.

1. **Observe** the LogService so we know the SEL exists and where its clear action lives.
   ```http
   GET /redfish/v1/Managers/{id}/LogServices/SEL
   ```
   ```jsonc
   {
     "@odata.type": "#LogService.v1_x_x.LogService",
     "Id": "SEL",
     "OverWritePolicy": "WrapsWhenFull",
     "Entries": { "@odata.id": "/redfish/v1/Managers/{id}/LogServices/SEL/Entries" },
     "Actions": {
       "#LogService.ClearLog": {
         "target": "/redfish/v1/Managers/{id}/LogServices/SEL/Actions/LogService.ClearLog"
       }
     }
   }
   ```

2. **Observe + archive** the entries collection (paged via `Members` / `Members@odata.nextLink`),
   keeping only `Created >= T0`.
   ```http
   GET /redfish/v1/Managers/{id}/LogServices/SEL/Entries
   ```
   Each `LogEntry` carries the fields the archive needs — `Created`, `Severity`, `MessageId`,
   `Message`, `EntryType`, `SensorType`. The agent writes them to the run artifact and records the
   observed count `N`.

3. **Observe** the sensor snapshots the goal asks for (read lane, auto-proceeds):
   ```http
   GET /redfish/v1/Chassis/{id}/Thermal   # -> Temperatures[].ReadingCelsius
   GET /redfish/v1/Chassis/{id}/Power     # -> PowerControl[].PowerConsumedWatts
   ```

4. **Choose** the one mutation — `POST` to the `ClearLog` target — but only after the archive is
   written and counted. The RL policy has learned (reward + `expert_call_order` evidence) not to
   select `ClearLog` while the archive fact is still false; the ordering is learned and
   evaluator-enforced, not hard-coded in the language contract.

5. **Dry-run** the mutation: resolve the target, confirm `POST` is in that endpoint's
   `allowed_methods`, render the exact request, and surface it for approval — no bytes sent yet.
   ```http
   POST /redfish/v1/Managers/{id}/LogServices/SEL/Actions/LogService.ClearLog
   Content-Type: application/json

   {}
   ```

6. **Approve** — a human okays the single destructive step. Only then does the agent **execute** the
   `POST` (a compliant BMC returns `204 No Content` or a task/`200`).

7. **Observe again** — re-read the collection to see the effect:
   ```http
   GET /redfish/v1/Managers/{id}/LogServices/SEL/Entries
   ```

8. **Evaluate** against the goal (next section).

## What "done" means

Done is measured, never self-reported. The evaluator re-reads state and checks each fact against it:

- **Archived + counted** — the run artifact contains every entry whose `Created >= T0` that was
  observed in step 2, and the archived count equals the observed `N`.
- **Cleared** — the step-7 re-read of `.../Entries` shows the log emptied:
  `Members@odata.count == 0` (or the vendor's post-clear state, e.g. a single "log cleared" marker
  entry — the evaluator checks the re-read, not the POST's return code).
- **Snapshots captured** — `Temperatures[].ReadingCelsius` and `PowerControl[].PowerConsumedWatts`
  are present in the artifact.
- **Ordering** — the archive write and its count precede the `ClearLog` timestamp in the trajectory;
  a run that cleared first fails even if the log is now empty.

If any fact is false — say the clear returned `204` but the re-read still lists entries — the goal
is **not** done, regardless of what the action reported. This split matters: the POST's return code
is what the *controller claims*; the step-7 GET is what the *controller shows*. The evaluator trusts
the second, always. A green trajectory is one it confirmed by GET, so the run is auditable end to
end: every read, the archived entries, the dry-run, the human approval, and the verified post-state
are in the record, in order.

## Constraints, risk, and the guardrail

Risk is deliberately lopsided. The reads (`LogService`, `Entries`, `Thermal`, `Power`,
`MetricReports`) are **read-only** and auto-proceed — no pause, no approval. The one mutation,
`LogService.ClearLog`, is **destructive-but-bounded**: it erases the event history on that
controller, which is unrecoverable if the archive step was skipped — exactly why the ordering
constraint and the gate exist. Every mutating action passes the guardrail — **dry-run → approval →
execute** — and this scenario has precisely one, so approval pauses exactly once, right before the
log is cleared. That is the entire human-in-the-loop surface for the run.

## What transfers / what it learned

**HER** turns a run that stopped early — collected and snapshotted but never got approval to clear —
into a labeled trajectory for the sub-goal it *did* reach ("SEL archived and sensors captured"). The
policy learns the collect-then-clear ordering and the shortest safe path to it from both full and
partial runs, instead of only from clean successes.

It also learns the negative: selecting `ClearLog` before the archive fact flips true is a dead end
the evaluator will reject, so the path it converges on is the *shortest* one that still satisfies
every constraint — collect, snapshot, clear, verify — with no wasted reads and no premature
mutation.

**Cross-vendor** is where the walked legal surface pays off. `LogService`, its `Entries` collection,
the `#LogService.ClearLog` action, `Thermal`, and `Power` are DMTF-standard types, but their exact
placement differs — SEL under `/Managers/{id}/LogServices` on one platform, an event log under
`/Systems/{id}/LogServices` on another; the clear-action target path is vendor-shaped. Because the
concrete `target` and its legal methods are resolved from each host's own walked tree
(`rest_api_map.npy`), the same pipeline triages a Dell iDRAC, a Supermicro BMC, an HPE iLO, or a
generic DMTF implementation without a per-vendor script — the schema is the interface, and the
guardrail is the same everywhere. (How far the *trained* components carry across vendors is a
measured question — see the recorded evidence in `uc-06-fleet-remediation-multivendor.md`.)


# Author: Mus mbayramo@stanford.edu
