"""
Offline tests for the Redfish enum-space extraction layer.

Pins that captured resource bodies are classified by ``@odata.type`` (with URL
fallbacks), that the three extractors reproduce the shapes observed in the real
vendor captures — Dell/Supermicro ``#AttributeRegistry`` bodies, DMTF
``#ActionInfo`` parameters, and inline ``@Redfish.AllowableValues`` annotations
— and that :class:`EnumSpaceIndex` folds a record stream into ``arg_schema``
fragments the stage-2 argument decoder consumes verbatim. Fixture bodies are
minimal copies of the captured Dell / Supermicro / HPE shapes; pure stdlib, no
network, no fixture trees on disk.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

from igc.core.types import ToolSpec
from igc.ds.sources import (
    EnumSpaceIndex,
    RedfishFixtureSource,
    ResourceKind,
    SourceRecord,
    TrustLevel,
    classify_resource,
    normalize_enriched,
    slots_from_action_info,
    slots_from_attribute_registry,
    slots_from_inline_annotations,
    to_arg_schema,
)
from igc.modules.policy.argument_decoder import arg_slots_for


def _record(url: str, body: dict, source: str = "real_dell",
            trust: TrustLevel = TrustLevel.REAL) -> SourceRecord:
    """Build a minimal GET-observation record around ``body``."""
    return SourceRecord(url=url, response=body, source=source, trust_level=trust)


def _registry_body() -> dict:
    """A minimal ``#AttributeRegistry`` body in the captured Dell/Supermicro shape."""
    return {
        "@odata.type": "#AttributeRegistry.v1_3_0.AttributeRegistry",
        "RegistryEntries": {
            "Attributes": [
                {"AttributeName": "ProcCStates", "Type": "Enumeration",
                 "CurrentValue": "Disabled", "ReadOnly": False,
                 "Value": [{"ValueName": "Enabled"}, {"ValueName": "Disabled"}]},
                {"AttributeName": "ActiveCores", "Type": "Integer", "ReadOnly": False},
                {"AttributeName": "SystemServiceTag", "Type": "String", "ReadOnly": True},
            ]
        },
    }


def _action_info_body() -> dict:
    """A minimal DMTF ``#ActionInfo`` body (``Parameters[].AllowableValues``)."""
    return {
        "@odata.type": "#ActionInfo.v1_4_2.ActionInfo",
        "Id": "ResetActionInfo",
        "Parameters": [
            {"Name": "ResetType", "Required": True, "DataType": "String",
             "AllowableValues": ["On", "ForceOff", "GracefulRestart"]},
        ],
    }


def _system_body() -> dict:
    """A minimal ``#ComputerSystem`` with inline annotations in the captured shape."""
    return {
        "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
        "Boot": {
            "BootSourceOverrideTarget": "None",
            "BootSourceOverrideTarget@Redfish.AllowableValues": ["None", "Pxe", "Hdd"],
        },
        "Actions": {
            "#ComputerSystem.Reset": {
                "ResetType@Redfish.AllowableValues": ["On", "ForceOff"],
                "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
            }
        },
    }


def test_classify_by_odata_type() -> None:
    """Each captured @odata.type prefix maps to its resource kind."""
    cases = [
        ("#Bios.v1_2_0.Bios", "/redfish/v1/Systems/1/Bios", ResourceKind.BIOS),
        ("#AttributeRegistry.v1_3_0.AttributeRegistry", "/x", ResourceKind.ATTRIBUTE_REGISTRY),
        ("#MessageRegistryFile.v1_0_4.MessageRegistryFile", "/x", ResourceKind.REGISTRY_FILE),
        ("#BootOptionCollection.BootOptionCollection", "/x", ResourceKind.BOOT_OPTION_COLLECTION),
        ("#BootOption.v1_0_4.BootOption", "/x", ResourceKind.BOOT_OPTION),
        ("#ActionInfo.v1_4_2.ActionInfo", "/x", ResourceKind.ACTION_INFO),
        ("#ComputerSystem.v1_20_0.ComputerSystem", "/x", ResourceKind.COMPUTER_SYSTEM),
        ("#Chassis.v1_25_0.Chassis", "/x", ResourceKind.OTHER),
    ]
    for odata_type, url, expected in cases:
        assert classify_resource(url, {"@odata.type": odata_type}) is expected


def test_classify_bios_pending_by_settings_url() -> None:
    """A #Bios body at .../Bios/Settings is the pending-settings resource (any case)."""
    body = {"@odata.type": "#Bios.v1_2_0.Bios"}
    assert classify_resource(
        "/redfish/v1/Systems/1/Bios/Settings", body) is ResourceKind.BIOS_PENDING
    assert classify_resource(
        "/redfish/v1/systems/1/bios/settings", body) is ResourceKind.BIOS_PENDING
    assert classify_resource("/redfish/v1/Systems/1/Bios", body) is ResourceKind.BIOS


def test_classify_url_fallback_without_odata_type() -> None:
    """Old-firmware captures without @odata.type classify from the URL."""
    assert classify_resource("/redfish/v1/systems/1/bios", {}) is ResourceKind.BIOS
    assert classify_resource(
        "/redfish/v1/systems/1/bios/settings", {}) is ResourceKind.BIOS_PENDING
    assert classify_resource(
        "/redfish/v1/Systems/1/BootOptions", {}) is ResourceKind.BOOT_OPTION_COLLECTION
    assert classify_resource(
        "/redfish/v1/Systems/1/BootOptions/1", {}) is ResourceKind.BOOT_OPTION
    assert classify_resource("/redfish/v1/Chassis/1", {}) is ResourceKind.OTHER


def test_registry_enumeration_yields_categorical_slot() -> None:
    """An Enumeration registry attribute becomes a categorical slot with its values."""
    slots = {s.name: s for s in slots_from_attribute_registry("/reg", _registry_body())}
    slot = slots["ProcCStates"]
    assert slot.choices == ("Enabled", "Disabled")
    assert slot.type == "string" and not slot.read_only
    assert slot.current_value == "Disabled"
    assert slot.extracted_from == "registry"


def test_registry_non_enum_types_yield_typed_freeform_slots() -> None:
    """Integer/String registry attributes become typed slots without choices."""
    slots = {s.name: s for s in slots_from_attribute_registry("/reg", _registry_body())}
    assert slots["ActiveCores"].type == "integer" and slots["ActiveCores"].choices is None
    assert slots["SystemServiceTag"].type == "string"
    assert slots["SystemServiceTag"].read_only


def test_registry_pointer_and_malformed_entries_yield_nothing() -> None:
    """An HPE-style registry pointer (no RegistryEntries) and junk entries yield no slots."""
    pointer = {"@odata.type": "#MessageRegistryFile.v1_0_4.MessageRegistryFile",
               "Location": [{"Language": "en", "Uri": "/registrystore/x"}]}
    assert slots_from_attribute_registry("/reg", pointer) == []
    junk = {"RegistryEntries": {"Attributes": ["not-a-dict", {"Type": "Enumeration"}]}}
    assert slots_from_attribute_registry("/reg", junk) == []
    assert slots_from_attribute_registry("/reg", {"RegistryEntries": {}}) == []


def test_registry_enum_without_value_names_degrades_to_freeform() -> None:
    """An Enumeration attribute with an empty/malformed Value list has no choices."""
    body = {"RegistryEntries": {"Attributes": [
        {"AttributeName": "Odd", "Type": "Enumeration", "Value": []},
        {"AttributeName": "Odder", "Type": "Enumeration", "Value": ["bare-string"]},
    ]}}
    slots = {s.name: s for s in slots_from_attribute_registry("/reg", body)}
    assert slots["Odd"].choices is None
    assert slots["Odder"].choices is None


def test_action_info_parameters_become_required_categorical_slots() -> None:
    """ActionInfo Parameters carry name, required, type, and allowable values."""
    (slot,) = slots_from_action_info("/ai", _action_info_body())
    assert slot.name == "ResetType" and slot.required
    assert slot.choices == ("On", "ForceOff", "GracefulRestart")
    assert slot.extracted_from == "action_info"
    assert slots_from_action_info("/ai", {"Parameters": []}) == []
    assert slots_from_action_info("/ai", {}) == []


def test_inline_annotations_record_dotted_path() -> None:
    """@Redfish.AllowableValues anywhere in a body yield slots with their location."""
    slots = {s.path: s for s in slots_from_inline_annotations("/sys", _system_body())}
    boot = slots["Boot.BootSourceOverrideTarget"]
    assert boot.choices == ("None", "Pxe", "Hdd")
    assert boot.current_value == "None"
    reset = slots["Actions.#ComputerSystem.Reset.ResetType"]
    assert reset.choices == ("On", "ForceOff")
    assert slots_from_inline_annotations("/sys", {"Boot": {}}) == []


def test_index_routes_kinds_and_splits_action_vs_settings() -> None:
    """The index buckets registry, action, and patchable-property slots separately."""
    index = EnumSpaceIndex.from_records([
        _record("/redfish/v1/Systems/1/Bios/BiosRegistry", _registry_body()),
        _record("/redfish/v1/Systems/1", _system_body()),
        _record("/redfish/v1/Systems/1/ResetActionInfo", _action_info_body()),
    ])
    patch = index.patch_arg_schema()["PATCH"]
    assert patch["ProcCStates"]["enum"] == ["Enabled", "Disabled"]
    assert patch["BootSourceOverrideTarget"]["enum"] == ["None", "Pxe", "Hdd"]
    assert "SystemServiceTag" not in patch  # read-only never patchable
    post = index.post_arg_schema()["POST"]
    assert post["ResetType"]["enum"] == ["On", "ForceOff", "GracefulRestart"]
    assert post["ResetType"]["required"] is True
    assert index.by_kind == {"ATTRIBUTE_REGISTRY": 1, "COMPUTER_SYSTEM": 1, "ACTION_INFO": 1}


def test_index_counts_registry_pointers_without_slots() -> None:
    """HPE registry pointers are counted but contribute no slots."""
    pointer = {"@odata.type": "#MessageRegistryFile.v1_0_4.MessageRegistryFile",
               "Location": [{"Language": "en", "Uri": "/registrystore/x"}]}
    index = EnumSpaceIndex.from_records([_record("/redfish/v1/Registries/U58", pointer)])
    assert index.num_registry_pointers == 1
    assert index.patch_arg_schema() == {"PATCH": {}}


def test_boot_option_references_dedupe_in_order() -> None:
    """BootOption records build the ordered unique BootOrder id space."""
    def boot(ref: str) -> dict:
        return {"@odata.type": "#BootOption.v1_0_4.BootOption",
                "Id": ref, "BootOptionReference": ref}
    index = EnumSpaceIndex.from_records([
        _record("/redfish/v1/Systems/1/BootOptions/1", boot("Boot0005")),
        _record("/redfish/v1/Systems/1/BootOptions/2", boot("Boot0002")),
        _record("/redfish/v1/Systems/1/BootOptions/3", boot("Boot0005")),  # dup
        _record("/redfish/v1/Systems/1/BootOptions/4",
                {"@odata.type": "#BootOption.v1_0_4.BootOption", "Id": "Boot0009"}),
    ])
    assert index.boot_reference_space() == ("Boot0005", "Boot0002", "Boot0009")


def test_first_seen_wins_on_duplicate_slot_names() -> None:
    """Feeding higher-trust records first keeps their fragment on a name clash."""
    real = _registry_body()
    sim = {"RegistryEntries": {"Attributes": [
        {"AttributeName": "ProcCStates", "Type": "Enumeration",
         "Value": [{"ValueName": "Simulated"}]},
    ]}}
    index = EnumSpaceIndex.from_records([
        _record("/real/reg", real),
        _record("/sim/reg", sim, source="sim", trust=TrustLevel.SIM_GENERIC),
    ])
    assert index.patch_arg_schema()["PATCH"]["ProcCStates"]["enum"] == ["Enabled", "Disabled"]


def test_extracted_schema_round_trips_through_argument_decoder() -> None:
    """The exported arg_schema is consumed verbatim by the stage-2 decoder."""
    index = EnumSpaceIndex.from_records([
        _record("/redfish/v1/Systems/1/ResetActionInfo", _action_info_body()),
    ])
    spec = ToolSpec(tool_name="redfish", ops=["POST"], arg_schema=index.post_arg_schema())
    (slot,) = arg_slots_for(spec, "POST")
    assert slot.name == "ResetType" and slot.is_categorical and slot.required
    assert slot.choices == ("On", "ForceOff", "GracefulRestart")


def test_index_over_fixture_source_stream(tmp_path: Path) -> None:
    """End-to-end: a capture directory feeds the index through RedfishFixtureSource."""
    root = tmp_path / "capture"
    root.mkdir()
    files = {
        "_redfish_v1_Systems_1_Bios_BiosRegistry.json": _registry_body(),
        "_redfish_v1_Systems_1.json": dict(_system_body(),
                                           **{"@odata.id": "/redfish/v1/Systems/1"}),
    }
    for name, body in files.items():
        (root / name).write_text(json.dumps(body))
    source = RedfishFixtureSource(str(root), "real_dell", TrustLevel.REAL, vendor="dell")
    index = EnumSpaceIndex.from_records(source)
    patch = index.patch_arg_schema()["PATCH"]
    assert set(patch) == {"ProcCStates", "ActiveCores", "BootSourceOverrideTarget"}
    assert index.post_arg_schema()["POST"]["ResetType"]["enum"] == ["On", "ForceOff"]


def test_normalize_enriched_stamps_resource_kind() -> None:
    """Enriched normalization tags each example with its classified kind."""
    examples = normalize_enriched([
        _record("/redfish/v1/Systems/1/Bios",
                {"@odata.type": "#Bios.v1_2_0.Bios", "Attributes": {"BootMode": "Uefi"}}),
        _record("/redfish/v1/Chassis/1", {"@odata.type": "#Chassis.v1_25_0.Chassis"}),
    ])
    assert examples[0].expected_semantics["resource_kind"] == "bios"
    assert examples[1].expected_semantics["resource_kind"] == "other"
    # the base normalization contract is preserved
    assert examples[0].expected_semantics["read_only"] is True
    assert examples[0].to_dict()["expected_semantics"]["resource_kind"] == "bios"


def test_to_arg_schema_fragment_shape() -> None:
    """Fragments carry type/required and a list enum only when categorical."""
    from igc.ds.sources import EnumSlot
    schema = to_arg_schema("PATCH", [
        EnumSlot(name="A", type="string", choices=("x", "y")),
        EnumSlot(name="B", type="integer"),
    ])
    assert schema == {"PATCH": {"A": {"type": "string", "required": False, "enum": ["x", "y"]},
                                "B": {"type": "integer", "required": False}}}


# Author: Mus mbayramo@stanford.edu
