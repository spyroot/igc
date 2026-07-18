# UC-05 — Storage & virtual media

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery/preconditions) → JSON simulator. Contract
> examples are illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the first proof
> environment.

## The goal, and what the pipeline extracts from it

An operator wants a node prepped to reinstall: *"Mount the install ISO as virtual CD, and make sure
there is a RAID1 data volume."* Two mutating sub-goals, both gated on real hardware preconditions.
The language side extracts the touched APIs and bindings — never the precondition logic or the
sequence:

```jsonc
// Phase 2 — UNORDERED unique set (k=2 mutating targets)
{ "rest_api_list": [
    "/redfish/v1/Managers/{id}/VirtualMedia/{slot}/Actions/VirtualMedia.InsertMedia",
    "/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes"
] }

// Phase 3 — UNORDERED; each Call = { rest_api, explicit http_method, arguments }
{ "calls": [
    { "rest_api": "/redfish/v1/Managers/{id}/VirtualMedia/{slot}/Actions/VirtualMedia.InsertMedia",
      "http_method": "POST",
      "arguments": { "Image": "<operator-supplied ISO reference>",
                     "Inserted": true, "WriteProtected": true } },
    { "rest_api": "/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes",
      "http_method": "POST",
      "arguments": { "RAIDType": "RAID1", "Drives": ["<eligible-drive-refs>"] } }
] }
```

The precondition reads (is the media slot free? are the drives unconfigured? is capacity
sufficient?), the idempotency check, and the ordering are the separate RL policy's job — legal reads
it learned to insert, supervised by reward and separate `expert_call_order` oracle evidence. They do
not belong in the language contract.

## Why a script or a chatbot struggles here

A shell script `POST`s the `InsertMedia` action and the `Volumes` collection in a fixed order, then
exits 0 — it never notices the slot already had an ISO mounted, or that the two "spare" drives are
already members of another volume, so it either clobbers a peer's media or gets a 400 it can't
interpret. A chatbot will happily *narrate* a create-volume call with a plausible-looking body and a
drive URI it invented, because nothing grounds it in the walked interface. Neither one verifies
capacity or configuration state *before* the mutating call, and neither re-reads the tree afterward
to prove the volume actually materialized.

## Observation and the legal methods

The observation is a Redfish GET result — here, the `Storage` subtree and the manager's
`VirtualMedia` collection, plus the `Drives` members with their `CapacityBytes`, `MediaType`, and
`Links.Volumes` (empty ⇒ unconfigured). Legal methods per URL come from the captured interface's
`allowed_methods_mapping` in `rest_api_map.npy` (the binding contract from `redfish_ctl` discovery)
plus `@Redfish.ActionInfo`. The legal surface for this goal:

| Endpoint (walked)                                            | Method | What it does                              |
|-------------------------------------------------------------|--------|-------------------------------------------|
| `/redfish/v1/Managers/{id}/VirtualMedia/{slot}`             | GET    | read `Inserted`, `Image`, `MediaTypes`    |
| `.../VirtualMedia/{slot}/Actions/VirtualMedia.InsertMedia`  | POST   | attach ISO (`Image`, `Inserted`, `WriteProtected`) |
| `.../VirtualMedia/{slot}/Actions/VirtualMedia.EjectMedia`   | POST   | detach media (guarded — constraint blocks)|
| `/redfish/v1/Systems/{id}/Storage/{ctrl}`                   | GET    | read controller + `Drives` links          |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Drives/{d}`        | GET    | read `CapacityBytes`, `Links.Volumes`     |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes`           | GET    | list existing volumes (idempotency check) |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes`           | POST   | create Volume (`RAIDType`, `Drives`)      |

If `POST` is not among a collection's advertised methods, the create call is illegal on that URL —
the contract gate rejects it. No hallucinated URIs, no invented action names.

## The trajectory (RL execution)

**1 — Observe (virtual media slot).** GET the slot; check the precondition, not just try-and-catch.

```json
GET /redfish/v1/Managers/1/VirtualMedia/CD
{ "Inserted": false, "Image": null, "MediaTypes": ["CD", "DVD"],
  "ConnectedVia": "NotConnected", "WriteProtected": true }
```

`Inserted: false` — no one else's media to clobber. Precondition met → proceed.

**2 — Dry-run (InsertMedia).** The chosen mutating call renders to a concrete body first, executed
in dry-run (no write) so the guardrail and operator see exactly what will hit the hardware:

```http
POST /redfish/v1/Managers/1/VirtualMedia/CD/Actions/VirtualMedia.InsertMedia
{ "Image": "https://images.example/os/install.iso",
  "Inserted": true, "WriteProtected": true }
```

**3 — Approve → execute.** Mutation on a management controller → the guardrail pauses for approval,
then issues the real POST. Response `204 No Content` (or a `Task` reference to poll).

**4 — Observe (storage preconditions) — and adapt when one fails.** Before any volume POST, GET the
drive members:

```json
GET /redfish/v1/Systems/System.Embedded.1/Storage/RAID.0/Drives/Disk.0
{ "CapacityBytes": 960197124096, "MediaType": "SSD",
  "Links": { "Volumes": [] } }               // unconfigured, ~960 GB  → eligible
GET .../Drives/Disk.1
{ "CapacityBytes": 960197124096, "MediaType": "SSD",
  "Links": { "Volumes": [ {"@odata.id": ".../Volumes/Volume.1"} ] } }   // already a member
```

`Disk.1` is already a member of another volume. The policy does **not** POST with a bad member set.
It re-scans the walked `Drives` list for another unconfigured member of sufficient `CapacityBytes`;
if it finds `Disk.2` (empty `Links.Volumes`, enough capacity) it pairs `Disk.0` + `Disk.2`. If no
second eligible drive exists, it refuses the volume sub-goal and reports `precondition_unmet` rather
than building a degraded or wrong-capacity volume — this adapt-or-refuse behavior is exactly the
recovery competency the RL stage trains.

**5 — Idempotency check, then create.** GET `Volumes`; if a `RAID1` volume over the target drives
already exists, the sub-goal is already satisfied — skip the POST. Otherwise dry-run → approve →
execute:

```http
POST /redfish/v1/Systems/System.Embedded.1/Storage/RAID.0/Volumes
{ "RAIDType": "RAID1",
  "Drives": [ {"@odata.id": ".../Drives/Disk.0"},
              {"@odata.id": ".../Drives/Disk.2"} ] }
```

Response `202 Accepted` with a `@odata.id` to a `Task` — the policy polls `TaskState` until
`Completed` (waiting on async tasks is learned behavior, same as UC-03).

**6 — Observe again & evaluate.** Re-read both subtrees and hand the fresh state to the evaluator.

## What "done" means

The evaluator never trusts the action's return code as success. It re-reads state and measures each
target fact against it:

- Virtual media ← re-GET the slot: `Inserted == true`, `Image` equals the requested URI,
  `WriteProtected == true`, `ConnectedVia != "NotConnected"`.
- Volume ← re-GET the `Volumes` collection and the created Volume: `RAIDType == "RAID1"`, member
  drive count == 2, and `CapacityBytes` at or above the required usable capacity.

All facts pass against the re-read state ⇒ terminal success. A `204`/`202` with a volume that never
appears, or a mirror whose capacity falls short, is a **failure** — measured, not self-reported.

## Constraints, risk, and the guardrail

Both sub-goals are **mutating and physically consequential**: `InsertMedia` changes what the node
boots from; volume creation writes controller metadata and can wipe member drives if aimed wrong. So
every POST here runs the full lane: **dry-run → approval pause → execute**. The guardrail pauses at
each rendered mutating body, showing the exact `Image` URI or drive member set an operator is
authorizing. The read-only lane — every GET that gathers preconditions and every re-read the
evaluator does — auto-proceeds without a pause. Constraints are hard filters, not hints: a drive
with a non-empty `Links.Volumes` is never an eligible member, and `EjectMedia` is blocked whenever a
slot is already `Inserted` by someone else.

## What transfers / what it learned

The first time, the policy may reach a valid volume the long way — probing several drives, or trying
an `InsertMedia` on an already-occupied slot and being blocked. HER relabels those detours: the
failed "insert into occupied slot" transition becomes a labeled example that *checking `Inserted`
first* is the shorter safe path, and the drive-precondition scan becomes the learned prefix to any
volume create. Over episodes the policy converges on the shortest safe ordering — precondition GETs,
then the single correct POST — instead of trial-and-error.

The structural angle: `Volume` with `RAIDType` + `Drives` members, and `VirtualMedia` with
`InsertMedia`/`EjectMedia`, are **standard DMTF schema**; endpoint ids and OEM extensions differ per
vendor, but the legal-method + precondition + evaluator loop is identical. The repo's multi-vendor
fixture corpora (Dell iDRAC, Supermicro, HPE iLO, generic DMTF) are the material for *measuring* how
far that carries — see the recorded transfer evidence in
`uc-06-fleet-remediation-multivendor.md`; v1 claims no zero-shot universal-REST capability.


# Author: Mus mbayramo@stanford.edu
