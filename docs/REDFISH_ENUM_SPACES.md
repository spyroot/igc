# Redfish Enum Spaces — Offline Ingestion of BIOS / Boot / ActionInfo Value Spaces

Design note for `igc/ds/sources/redfish_enum_space.py`, the extraction layer that turns
the richer resource types captured by `redfish_ctl` discovery (the data-collection tool
that writes per-resource JSON under `~/.json_responses/<host>/`) into an offline catalog
of raw argument **value spaces**. The machine-readable contract schemas under
`configs/contracts/*.yaml` are authoritative for record shapes; every example in this
document is illustrative only.

## Purpose — value spaces for Phase 3 argument binding

Phase 3 (ordered method/argument extraction, per the Phase 1/2/3 pipeline docs) emits
`calls: list[Call]`, where each `Call` carries an explicit `http_method` and an explicit
`arguments` object (`{}` for reads). One call is still a list of length one; the field is
never a scalar. In the locked v1 architecture, exact argument **values** stay raw —
outside both the `z_rest` encoder (REST-goal latent) and the separate `z_method` encoder
(method latent); neither encoder embeds concrete values, and there is no shared or
unified latent. A value-space catalog is therefore exactly the right offline artifact:
it enumerates, per writable slot, which raw values a corpus actually allows, so Phase 3
argument binding can fill `arguments` from real per-slot choice sets instead of
hand-written schemas.

Illustrative generic examples of the Phase 3 output this catalog supports
(k = number of calls; the set is UNORDERED — execution order is separate RL-oracle
evidence recorded as `expert_call_order`, not part of the Phase 3 target):

```yaml
# k=1 — "set x to 1"
calls:
  - {http_method: PATCH, arguments: {x: 1}}

# k=2 — "set x to 1 and read z"
calls:
  - {http_method: PATCH, arguments: {x: 1}}
  - {http_method: GET,   arguments: {}}

# k=3 — "set x to 1, set y to 2, and read z"
calls:
  - {http_method: PATCH, arguments: {x: 1}}
  - {http_method: PATCH, arguments: {y: 2}}
  - {http_method: GET,   arguments: {}}
```

Redfish-shaped values (`ResetType`, `BootMode`, …) appear below only because Redfish is
the current test environment. No planner, scheduler, or curriculum lives in this layer or
in Phase 2/3; ordering, prerequisites, retries, waiting, and recovery belong to the
separate RL policy stage.

## What discovery captures vs. what was ingested

`redfish_ctl` discovery preserves the legacy output contract — one JSON file per resource plus
`rest_api_map.npy` holding `url_file_mapping` and `allowed_methods_mapping` (the binding `.npy`
contract loaded by `igc/ds/ds_rest_trajectories.py`, unchanged by this work). Materialized dataset
artifacts also provide `rest_api_map.v1.json` with relative paths for portable consumers. The generic
adapter
`RedfishFixtureSource` (in `igc/ds/sources/redfish_fixture_source.py`) already streams
every captured body as a provenance-tagged `SourceRecord`; what was missing is semantic
extraction of the value spaces inside these resource types:

| Resource type (`@odata.type`) | Where captured | Value space it carries |
| --- | --- | --- |
| `#AttributeRegistry` (e.g. Dell `Bios/BiosRegistry`, the `BiosAttributeRegistry` under `/redfish/v1/Registries`) | Dell + Supermicro corpora | Per-BIOS-attribute type, `ReadOnly` flag, and — for `Enumeration` attributes — the `Value[].ValueName` choice list (e.g. 360 attributes / 78 enumerations on the Supermicro GB300 target) |
| `#Bios` (`Attributes` dict + `@Redfish.Settings` pointer) | all vendors | Current attribute values; the `.../Bios/Settings` child is the pending-settings resource (`bios-pending` in `redfish_ctl` terms) |
| `#ActionInfo` (`Parameters[].AllowableValues`) | referenced via `@Redfish.ActionInfo` from `Actions` blocks | Authoritative per-action parameter names, `Required`, `DataType`, allowable values |
| Inline `<prop>@Redfish.AllowableValues` annotations | `Boot` and `Actions` blocks of `#ComputerSystem`, all vendors | Lightweight vendor alternative to ActionInfo (e.g. `BootSourceOverrideTarget`, `ResetType`) |
| `#BootOption` / `#BootOptionCollection` | Dell, Supermicro, HPE (iLO) | `BootOptionReference` ids — the value space bounding a `Boot.BootOrder` write |
| `#MessageRegistryFile` | HPE (iLO) | A registry *pointer*: `Location[].Uri` names an external payload; carries no entries itself |

## Ingestion path (offline, fixture-driven)

Reuse over duplication: the layer does **not** walk the filesystem and does **not**
crawl. It consumes the existing `SourceRecord` stream, so the trust tagging, URL
canonicalization (`@odata.id`-first), and skip accounting of `RedfishFixtureSource`
apply unchanged, and any future adapter (DMTF mockup replay, emulator) feeds it for
free.

```
RedfishFixtureSource(capture dir)          # existing generic adapter
        │  SourceRecord stream (trust-ordered: real first)
        ▼
EnumSpaceIndex.from_records(...)           # igc/ds/sources/redfish_enum_space.py
        │  classify_resource() by @odata.type, URL fallback
        ├── #AttributeRegistry  → slots_from_attribute_registry()  → registry_slots
        ├── #ActionInfo         → slots_from_action_info()          → action_slots
        ├── #BootOption         → BootOptionReference               → boot_references
        ├── #MessageRegistryFile→ counted (payload is external)
        └── every body          → slots_from_inline_annotations()   → inline action / settings slots
        ▼
index.patch_arg_schema()  →  {"PATCH": {attr:  {"type", "enum", "required"}}}
index.post_arg_schema()   →  {"POST":  {param: {"type", "enum", "required"}}}
index.boot_reference_space()  →  ordered BootOptionReference ids
```

The exported fragments are per-slot raw value spaces keyed by HTTP method — the shape
Phase 3 argument binding draws candidate values from. For the dataset side,
`normalize_enriched()` wraps the existing `normalize_record()` (in
`igc/ds/sources/training_object.py`) and stamps
`expected_semantics["resource_kind"]`, so BIOS/boot/registry observations become
distinguishable `TrainingExample` records without re-parsing bodies.

## Precedence and dedup rules

* **First-seen wins** per slot name inside each bucket — same convention as `SourceMix`
  dedup, so callers feed higher-trust records first.
* **ActionInfo beats inline annotations** on a parameter name clash regardless of stream
  order: DMTF makes `#ActionInfo` the authoritative parameter description; inline
  `Actions` annotations are the fallback tier.
* **Read-only registry attributes never enter the PATCH schema** — a settings write
  cannot carry them.
* Registry pointers (`#MessageRegistryFile`) contribute no slots; they are counted in
  `num_registry_pointers` so a corpus report can say how many registries live outside
  the capture.

## Validation on real corpora

Running the index over the vendor fixture corpora
(`tests/{dell,supermicro,hpe}_fixtures`) plus one live HPE capture tree
(2,030 records, zero parse skips) yields 380 patchable slots — 97 of them categorical
with real enum spaces (e.g. `ProcCStates`, `BootMode`) — 9 action parameters
(`ResetType` with the full DMTF reset set, Dell OEM job parameters, SecureBoot /
storage-initialize parameters), 24 boot references across vendor naming styles, and 17
external registry pointers.

## Deliberately out of scope / follow-ups

* **No live crawl.** Everything here reads captured JSON. Live capture of missing
  pieces (below) stays gated on an approved non-production host.
* **ActionInfo bodies are not yet in the captures**: discovery does not follow
  `@Redfish.ActionInfo` references, so no `#ActionInfo` body exists in any current tree
  (only its JSON Schema). The extractor is built and tested against the DMTF shape; a
  `redfish_ctl`-side enhancement to follow those references would light it up on real
  data.
* **HPE registry payloads are external** (`Location[].Uri` under a registry store);
  ingesting them needs either a targeted capture of those URIs or the vendor tree they
  ship in.
* The legacy `.npy` contract is retained while portable dataset maps are added.
