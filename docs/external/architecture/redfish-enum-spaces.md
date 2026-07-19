# Redfish Enum Spaces — Offline Ingestion of BIOS / Boot / ActionInfo Value Spaces

Design note for `igc/ds/sources/redfish_enum_space.py`, the extraction layer that turns
the richer resource types captured by `redfish_ctl` discovery (the data-collection tool
that writes per-resource JSON under `~/.json_responses/<host>/`) into the per-slot
argument value spaces consumed by the stage-2 argument decoder
(`igc/modules/policy/argument_decoder.py`).

## Motivation

Stage 1 of the policy (the pointer policy) selects a value-independent action template;
stage 2 fills argument values. For a categorical argument, the decoder scores the slot's
OWN allowable values, read from `ToolSpec.arg_schema[op][slot]["enum"]` — so the head
width tracks one slot's choice count and never a global vocabulary. Until now those enum
spaces existed only as hand-written schemas in tests. The captured corpora already
contain the real value spaces; this layer extracts them offline.

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

Reuse over duplication: the layer does **not** walk the filesystem. It consumes the
existing `SourceRecord` stream, so the trust tagging, URL canonicalization
(`@odata.id`-first), and skip accounting of `RedfishFixtureSource` apply unchanged, and
any future adapter (DMTF mockup replay, emulator) feeds it for free.

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

The exported fragments are exactly what `arg_slots_for` in the argument decoder reads
(pinned by a round-trip test through a real `ToolSpec`). For the dataset side,
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
