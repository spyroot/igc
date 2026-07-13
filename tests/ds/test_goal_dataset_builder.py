"""Offline tests for the latent sub-goal dataset builder.

These tests pin the corrected GoalExtractor target: operator text maps to an
unordered set of atomic ``GoalRef`` sub-goals, while concrete Redfish JSON
surfaces remain hidden verifier/reward evidence. No network, GPU, or live
Redfish host is required.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.ds.goal_dataset import GoalDependency, GoalTextExample, read_goal_text_examples
from igc.ds.goal_dataset_builder import build_goal_surfaces, make_goal_text_example
from igc.ds.sources import SourceRecord, TrustLevel


def _record(url: str, body: dict, vendor: str = "dell") -> SourceRecord:
    """Build a REAL-tier SourceRecord with vendor provenance."""
    return SourceRecord(
        url=url,
        response=body,
        source=f"real_{vendor}",
        trust_level=TrustLevel.REAL,
        vendor=vendor,
        schema_version=str(body.get("@odata.type", "")),
    )


def _system_body() -> dict:
    """Tiny ComputerSystem body with power, boot, and reset surfaces."""
    return {
        "@odata.id": "/redfish/v1/Systems/1",
        "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
        "PowerState": "Off",
        "PowerState@Redfish.AllowableValues": ["On", "Off"],
        "Boot": {
            "BootSourceOverrideTarget": "None",
            "BootSourceOverrideTarget@Redfish.AllowableValues": ["None", "Pxe", "Hdd"],
            "BootSourceOverrideEnabled": "Disabled",
            "BootSourceOverrideEnabled@Redfish.AllowableValues": [
                "Disabled",
                "Once",
                "Continuous",
            ],
        },
        "Actions": {
            "#ComputerSystem.Reset": {
                "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                "ResetType@Redfish.AllowableValues": [
                    "On",
                    "GracefulRestart",
                    "ForceRestart",
                ],
            }
        },
    }


def _network_protocol_body() -> dict:
    """Tiny ManagerNetworkProtocol body with NTP state."""
    return {
        "@odata.id": "/redfish/v1/Managers/1/NetworkProtocol",
        "@odata.type": "#ManagerNetworkProtocol.v1_10_0.ManagerNetworkProtocol",
        "NTP": {
            "ProtocolEnabled": False,
            "NTPServers": ["time-a.example.test", "time-b.example.test"],
        },
    }


def _chassis_body() -> dict:
    """Generic non-ComputerSystem body with nested state and actions."""
    return {
        "@odata.id": "/redfish/v1/Chassis/1",
        "@odata.type": "#Chassis.v1_25_0.Chassis",
        "Name": "Main Chassis",
        "PowerState": "On",
        "Thermal": {
            "Fans": [
                {"Name": "Fan 1", "Reading": 42, "Status": {"State": "Enabled"}},
            ],
        },
        "Actions": {
            "#Chassis.Reset": {
                "target": "/redfish/v1/Chassis/1/Actions/Chassis.Reset",
                "ResetType@Redfish.AllowableValues": ["ForceRestart"],
            },
            "#LogService.ClearLog": {
                "target": "/redfish/v1/Chassis/1/LogServices/EventLog/Actions/LogService.ClearLog",
            },
        },
    }


def test_computer_system_record_yields_atomic_power_boot_and_reset_surfaces() -> None:
    """One JSON resource yields many atomic sub-goal surfaces, never a plan."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Systems/1", _system_body()),
    ])

    by_id = {surface.goal_ref.goal_id: surface for surface in surfaces}

    power = by_id["power.computer_system.PowerState.eq.On"]
    assert power.goal_ref.family == "power"
    assert power.goal_ref.target_value == "On"
    assert power.current_value == "Off"
    assert power.verifier["kind"] == "state_eq"
    assert power.verifier["property_path"] == "PowerState"

    boot_target = by_id["boot.computer_system.Boot.BootSourceOverrideTarget.eq.Pxe"]
    assert boot_target.goal_ref.family == "boot"
    assert boot_target.allowed_values == ("None", "Pxe", "Hdd")
    assert boot_target.goal_ref.target_value == "Pxe"

    reset = by_id["action.computer_system.ComputerSystem.Reset.ResetType.eq.GracefulRestart"]
    assert reset.goal_ref.family == "action"
    assert reset.goal_ref.mode == "transition"
    assert reset.goal_ref.action_name == "ComputerSystem.Reset"
    assert reset.goal_ref.arguments == {"ResetType": "GracefulRestart"}


def test_generic_redfish_record_yields_nested_state_and_action_surfaces() -> None:
    """Unknown Redfish resource types still produce state and transition labels."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Chassis/1", _chassis_body(), vendor="supermicro"),
    ])

    power = next(
        surface for surface in surfaces
        if surface.goal_ref.resource_type == "chassis"
        and surface.goal_ref.property_path == "PowerState"
        and surface.goal_ref.target_value == "On"
    )
    assert power.goal_ref.family == "power"
    assert power.current_value == "On"
    assert power.verifier["kind"] == "state_eq"

    fan = next(
        surface for surface in surfaces
        if surface.goal_ref.property_path == "Thermal.Fans[].Reading"
    )
    assert fan.goal_ref.family == "thermal"
    assert fan.goal_ref.target_value == 42
    assert fan.verifier["property_path"] == "Thermal.Fans[].Reading"

    reset = next(
        surface for surface in surfaces
        if surface.goal_ref.action_name == "Chassis.Reset"
        and surface.goal_ref.arguments == {"ResetType": "ForceRestart"}
    )
    assert reset.goal_ref.mode == "transition"
    assert reset.verifier["kind"] == "transition"

    clear_log = next(
        surface for surface in surfaces
        if surface.goal_ref.action_name == "LogService.ClearLog"
    )
    assert clear_log.goal_ref.arguments == {}
    assert clear_log.goal_ref.goal_id.endswith(".invoke")


def test_inline_allowable_values_on_any_resource_yield_state_targets() -> None:
    """Allowable values outside Actions become atomic target values."""
    body = {
        "@odata.id": "/redfish/v1/Managers/1",
        "@odata.type": "#Manager.v1_16_0.Manager",
        "CommandShell": {
            "ConnectTypesSupported": "SSH",
            "ConnectTypesSupported@Redfish.AllowableValues": ["SSH", "IPMI", "Oem"],
        },
    }

    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Managers/1", body, vendor="hpe"),
    ])

    targets = {
        surface.goal_ref.target_value
        for surface in surfaces
        if surface.goal_ref.property_path == "CommandShell.ConnectTypesSupported"
    }
    assert targets == {"SSH", "IPMI", "Oem"}


def test_free_form_string_targets_do_not_enter_goal_ids() -> None:
    """Private instance strings stay verifier payloads, not semantic ids."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Managers/1/NetworkProtocol", _network_protocol_body()),
    ])

    ntp_servers = [
        surface for surface in surfaces
        if surface.goal_ref.property_path == "NTP.NTPServers"
    ]
    assert {surface.target_value for surface in ntp_servers} == {
        "time-a.example.test",
        "time-b.example.test",
    }
    assert {surface.goal_ref.goal_id for surface in ntp_servers} == {
        "network.manager_network_protocol.NTP.NTPServers.eq.value",
    }
    assert all("time-" not in surface.goal_ref.goal_id for surface in ntp_servers)
    assert {
        surface.verifier["target_value"] for surface in ntp_servers
    } == {"time-a.example.test", "time-b.example.test"}


def test_metadata_and_link_leaves_do_not_become_goal_labels() -> None:
    """Dataset rows avoid identity/link noise while covering real state leaves."""
    body = {
        "@odata.id": "/redfish/v1/Managers/1",
        "@odata.type": "#Manager.v1_16_0.Manager",
        "Id": "1",
        "Name": "Manager",
        "Links": {"ManagerForServers": [{"@odata.id": "/redfish/v1/Systems/1"}]},
        "Status": {"State": "Enabled", "Health": "OK"},
    }

    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Managers/1", body),
    ])

    paths = {surface.goal_ref.property_path for surface in surfaces}
    assert "Status.State" in paths
    assert "Status.Health" in paths
    assert "@odata.id" not in paths
    assert "Links.ManagerForServers[].@odata.id" not in paths


def test_sensitive_identity_leaves_do_not_become_prompt_labels() -> None:
    """Secrets and account identity fields are not allowed into LLM prompts."""
    body = {
        "@odata.id": "/redfish/v1/AccountService/Accounts/1",
        "@odata.type": "#ManagerAccount.v1_12_0.ManagerAccount",
        "UserName": "admin",
        "Password": "plain-text",
        "SSHKey": "ssh-rsa AAAA...",
        "Token": "secret-token",
        "RoleId": "Administrator",
        "PermanentMACAddress": "12:34:56:78:90:ab",
        "HostName": "private-host",
        "Enabled": True,
        "Actions": {
            "#Bios.ChangePassword": {
                "target": "/redfish/v1/Systems/1/Bios/Actions/Bios.ChangePassword",
            }
        },
    }

    surfaces = build_goal_surfaces([
        _record("/redfish/v1/AccountService/Accounts/1", body),
    ])

    paths = {surface.goal_ref.property_path for surface in surfaces}
    assert "Enabled" in paths
    assert "UserName" not in paths
    assert "Password" not in paths
    assert "SSHKey" not in paths
    assert "Token" not in paths
    assert "RoleId" not in paths
    assert "PermanentMACAddress" not in paths
    assert "HostName" not in paths
    assert all(surface.goal_ref.action_name != "Bios.ChangePassword" for surface in surfaces)


def test_schema_and_registry_documents_do_not_become_goal_surfaces() -> None:
    """Schema/registry corpus JSON is metadata, not an RL goal surface."""
    schema_body = {
        "@odata.id": "/redfish/v1/JsonSchemas/ComputerSystem",
        "@odata.type": "#JsonSchemaFile.v1_0_0.JsonSchemaFile",
        "Definitions": {
            "PowerState": {
                "enum": ["On", "Off"],
                "description": "Power state documentation",
            }
        },
    }
    registry_body = {
        "@odata.id": "/redfish/v1/Registries/Base",
        "@odata.type": "#MessageRegistry.v1_6_0.MessageRegistry",
        "RegistryPrefix": "Base",
        "Messages": {
            "Success": {
                "Description": "Operation succeeded",
                "Severity": "OK",
            }
        },
    }

    surfaces = build_goal_surfaces([
        _record("/redfish/v1/JsonSchemas/ComputerSystem", schema_body),
        _record("/redfish/v1/Registries/Base", registry_body),
    ])

    assert surfaces == ()


def test_boot_server_and_set_ntp_text_has_two_unordered_sub_goals() -> None:
    """The example x maps to true_y as a set of atomic z_sub_goal targets."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Systems/1", _system_body(), vendor="dell"),
        _record(
            "/redfish/v1/Managers/1/NetworkProtocol",
            _network_protocol_body(),
            vendor="dell",
        ),
    ])
    by_id = {surface.goal_ref.goal_id: surface for surface in surfaces}
    example = make_goal_text_example(
        text="boot server and set ntp",
        goal_refs=[
            by_id["power.computer_system.PowerState.eq.On"].goal_ref,
            by_id["network.manager_network_protocol.NTP.ProtocolEnabled.eq.True"].goal_ref,
        ],
        text_source="human",
    )

    assert isinstance(example, GoalTextExample)
    assert example.text == "boot server and set ntp"
    assert [ref.goal_id for ref in example.goal_refs] == [
        "power.computer_system.PowerState.eq.On",
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
    ]
    assert example.dependencies == ()


def test_explicit_then_adds_dependency_hint_without_becoming_a_plan() -> None:
    """Ordering words create dependency edges, not a concrete action sequence."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Systems/1", _system_body()),
        _record("/redfish/v1/Managers/1/NetworkProtocol", _network_protocol_body()),
    ])
    by_id = {surface.goal_ref.goal_id: surface for surface in surfaces}
    example = make_goal_text_example(
        text="set ntp then boot server",
        goal_refs=[
            by_id["network.manager_network_protocol.NTP.ProtocolEnabled.eq.True"].goal_ref,
            by_id["power.computer_system.PowerState.eq.On"].goal_ref,
        ],
        dependencies=[
            GoalDependency(
                before_goal_id="network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
                after_goal_id="power.computer_system.PowerState.eq.On",
                relation="before",
                evidence="then",
            )
        ],
    )

    assert [ref.goal_id for ref in example.goal_refs] == [
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
        "power.computer_system.PowerState.eq.On",
    ]
    assert example.dependencies[0].before_goal_id.startswith("network.")
    assert example.dependencies[0].after_goal_id.startswith("power.")


def test_goal_text_examples_jsonl_round_trip(tmp_path: Path) -> None:
    """Dataset rows serialize without losing sub-goal IDs or dependency hints."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Systems/1", _system_body()),
        _record("/redfish/v1/Managers/1/NetworkProtocol", _network_protocol_body()),
    ])
    by_id = {surface.goal_ref.goal_id: surface for surface in surfaces}
    example = make_goal_text_example(
        text="set ntp then boot server",
        goal_refs=[
            by_id["network.manager_network_protocol.NTP.ProtocolEnabled.eq.True"].goal_ref,
            by_id["power.computer_system.PowerState.eq.On"].goal_ref,
        ],
        dependencies=[
            GoalDependency(
                before_goal_id="network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
                after_goal_id="power.computer_system.PowerState.eq.On",
                relation="before",
                evidence="then",
            )
        ],
        split="train",
    )

    path = tmp_path / "goal_text_examples.jsonl"
    example.write_jsonl(path, [example])
    (loaded,) = read_goal_text_examples(path)

    assert loaded == example
