# UC-05 — Storage & virtual media

> Target-loop description, grounded in `docs/external/architecture/overview.md` + `docs/external/roadmap/decisions.md`. Today the code is a Phase-0 Redfish MDP shell (captured-replay env, one-hot actions); the observe → choose → dry-run → execute → evaluate loop below is the target behavior those documents commit to, not what the current trainer emits.

## The goal (in the operator's words, and the machine-checkable spec)

An operator wants a node prepped to reinstall: mount the install ISO as virtual media, and make sure
the OS has a mirrored data volume to land on. Two mutating sub-goals, both gated on real hardware
preconditions.

```python
Goal(
    instruction="Mount the install ISO as virtual CD, and ensure a RAID1 data volume exists.",
    spec={
        # sub-goal A: virtual media
        "virtual_media.image_attached": True,
        "virtual_media.inserted": True,
        "virtual_media.write_protected": True,
        # sub-goal B: storage volume
        "volume.raid_type": "RAID1",
        "volume.min_capacity_bytes": 900_000_000_000,   # ~900 GB usable
        "volume.member_drive_count": 2,
    },
    constraints=[
        "no_existing_media_eject",        # never eject media already mounted by someone else
        "only_unconfigured_drives",       # never consume a drive already in a volume
        "no_destructive_reinit",          # no controller reset / drive secure-erase
        "idempotent",                     # re-running must not create a second volume
    ],
    plan=None,   # the agent discovers the ordering from the walked tree
)
```

The spec is machine-checkable: every key is a field the Evaluator can re-read from Redfish and
compare. Nothing here is "looks done."

## Why a script or a chatbot struggles here

A shell script `POST`s the `InsertMedia` action and the `Volumes` collection in a fixed order, then
exits 0 — it never notices the slot already had an ISO mounted, or that the two "spare" drives are
already members of another volume, so it either clobbers a peer's media or gets a 400 it can't
interpret. A chatbot will happily *narrate* a create-volume call with a plausible-looking body and a
drive URI it invented, because it has no walked action catalog to constrain it. Neither one verifies
capacity or configuration state *before* the mutating call, and neither re-reads the tree afterward to
prove the volume actually materialized.

## Observation and the legal actions

The agent's Observation is a Redfish GET result — here, the `Storage` subtree and the manager's
`VirtualMedia` collection, plus the `Drives` members with their `CapacityBytes`, `MediaType`, and
`Links.Volumes` (empty ⇒ unconfigured). Actions are never free-form: each candidate is an endpoint
harvested from the walked resource tree paired with a method drawn from *that* endpoint's
`allowed_methods` (the `Allow` header / `@Redfish.ActionInfo`). The legal catalog for this goal:

| Endpoint (walked)                                            | Method | What it does                              |
|-------------------------------------------------------------|--------|-------------------------------------------|
| `/redfish/v1/Managers/{id}/VirtualMedia/{slot}`             | GET    | read `Inserted`, `Image`, `MediaTypes`    |
| `.../VirtualMedia/{slot}/Actions/VirtualMedia.InsertMedia`  | POST   | attach ISO (`Image`, `Inserted`, `WriteProtected`) |
| `.../VirtualMedia/{slot}/Actions/VirtualMedia.EjectMedia`   | POST   | detach media (guarded — constraint blocks)|
| `/redfish/v1/Systems/{id}/Storage/{ctrl}`                   | GET    | read controller + `Drives` links          |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Drives/{d}`        | GET    | read `CapacityBytes`, `Links.Volumes`     |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes`           | GET    | list existing volumes (idempotency check) |
| `/redfish/v1/Systems/{id}/Storage/{ctrl}/Volumes`           | POST   | create Volume (`RAIDType`, `Drives`)      |

If `POST` is not in a collection's advertised methods, the create candidate simply is not in the
catalog — the agent cannot select an action the endpoint does not permit. No hallucinated URIs, no
invented action names.

## The trajectory

**1 — Observe (virtual media slot).** GET the slot; check the precondition, not just try-and-catch.

```json
GET /redfish/v1/Managers/1/VirtualMedia/CD
{ "Inserted": false, "Image": null, "MediaTypes": ["CD", "DVD"],
  "ConnectedVia": "NotConnected", "WriteProtected": true }
```

`Inserted: false` satisfies `no_existing_media_eject`. Precondition met → proceed.

**2 — Choose + dry-run (InsertMedia).** The chosen mutating action renders to a concrete body first,
executed in dry-run (no write) so the guardrail and operator see exactly what will hit the hardware:

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

`Disk.1` violates `only_unconfigured_drives`. The agent does **not** POST with a bad member set. It
re-scans the walked `Drives` list for another unconfigured member of sufficient `CapacityBytes`; if
it finds `Disk.2` (empty `Links.Volumes`, ≥ `min_capacity_bytes`) it pairs `Disk.0` + `Disk.2`. If no
second eligible drive exists, it refuses sub-goal B and reports `precondition_unmet` rather than
building a degraded or wrong-capacity volume.

**5 — Idempotency check, then create.** GET `Volumes`; if a `RAID1` volume over the target drives
already exists, sub-goal B is already satisfied — skip the POST. Otherwise dry-run → approve →
execute:

```http
POST /redfish/v1/Systems/System.Embedded.1/Storage/RAID.0/Volumes
{ "RAIDType": "RAID1",
  "Drives": [ {"@odata.id": ".../Drives/Disk.0"},
              {"@odata.id": ".../Drives/Disk.2"} ] }
```

Response `202 Accepted` with a `@odata.id` to a `Task` — the agent polls `TaskState` until
`Completed`.

**6 — Observe again & evaluate.** Re-read both subtrees and hand the fresh state to the Evaluator.

## What "done" means

The Evaluator never trusts the action's return code as success. It re-reads state and measures each
spec key against it:

- `virtual_media.image_attached` / `inserted` / `write_protected` ← re-GET the slot: `Inserted == true`,
  `Image` equals the requested URI, `WriteProtected == true`, `ConnectedVia != "NotConnected"`.
- `volume.raid_type == "RAID1"`, `volume.member_drive_count == 2`, and the new volume's
  `CapacityBytes >= min_capacity_bytes` ← re-GET the `Volumes` collection and the created Volume.

All keys pass against the re-read state ⇒ terminal success. A `204`/`202` with a volume that never
appears, or a mirror whose capacity falls short, is a **failure** — measured, not self-reported.

## Constraints, risk, and the guardrail

Both sub-goals are **mutating and physically consequential**: `InsertMedia` changes what the node
boots from; volume creation writes controller metadata and can wipe member drives if aimed wrong. So
every POST here runs the full lane: **dry-run → approval pause → execute**. The guardrail pauses at
each rendered mutating body, showing the exact `Image` URI or drive member set an operator is
authorizing. The read-only lane — every GET that gathers preconditions and every re-read the
Evaluator does — auto-proceeds without a pause. Constraints are hard filters, not hints:
`only_unconfigured_drives` removes any candidate member with a non-empty `Links.Volumes` from the
selectable set, and `no_existing_media_eject` removes the `EjectMedia` candidate whenever a slot is
already `Inserted` by someone else.

## What transfers / what it learned

The first time, the agent may reach a valid volume the long way — probing several drives, or trying an
`InsertMedia` on an already-occupied slot and being blocked. HER relabels those detours: the failed
"insert into occupied slot" transition becomes a labeled example that *checking `Inserted` first* is
the shorter safe path, and the drive-precondition scan becomes the learned prefix to any volume
create. Over episodes the policy converges on the shortest safe ordering — precondition GETs, then the
single correct POST — instead of trial-and-error.

The transfer angle is structural, not hardcoded: `Volume` with `RAIDType` + `Drives` members, and
`VirtualMedia` with `InsertMedia`/`EjectMedia`, are **standard DMTF schema**. The agent keys off
resource types and advertised actions discovered in the walked tree, so the same policy drives Dell
iDRAC, Supermicro, HPE iLO, and generic DMTF controllers — the endpoint ids and OEM extensions differ,
but the legal-catalog + precondition + evaluator loop is identical. What it learned about *mirror this
node safely* carries to hardware it has never seen.


# Author: Mus mbayramo@stanford.edu
