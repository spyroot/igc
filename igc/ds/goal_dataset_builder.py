"""Build GoalExtractor / GoalEncoder rows from captured Redfish JSON.

This module owns deterministic ``Y`` construction. A paraphrase model may later
generate operator text ``X``, but code here decides which atomic sub-goal refs
are true. Most Redfish captures expose one property/action surface at a time;
compound text examples are assembled by grouping those atomic rows.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from igc.ds.goal_dataset import GoalDependency, GoalRef, GoalSurface, GoalTextExample
from igc.ds.sources import SourceRecord

_ALLOWABLE_SUFFIX = "@Redfish.AllowableValues"
_SCALAR_TYPES = (str, int, float, bool)
_SKIP_SUBTREES = frozenset({"Actions", "Links", "Members"})
_SKIP_LEAF_KEYS = frozenset({
    "@odata.context",
    "@odata.etag",
    "@odata.id",
    "@odata.type",
    "Created",
    "Description",
    "Id",
    "Modified",
    "Name",
    "Password",
    "RoleId",
    "Token",
    "UserName",
})
_SENSITIVE_NAME_PARTS = frozenset({
    "apikey",
    "api_key",
    "certificate",
    "credential",
    "key",
    "macaddress",
    "mac_address",
    "password",
    "privatekey",
    "private_key",
    "roleid",
    "role_id",
    "secret",
    "serialnumber",
    "serial_number",
    "sshkey",
    "ssh_key",
    "token",
    "username",
    "user_name",
})
_SENSITIVE_NAME_SUBSTRINGS = (
    "asset_tag",
    "change_password",
    "credential",
    "host_name",
    "hostname",
    "key",
    "macaddress",
    "mac_address",
    "password",
    "private",
    "secret",
    "serial",
    "token",
    "uuid",
)
_METADATA_RESOURCE_TYPES = frozenset({
    "AttributeRegistry",
    "JsonSchemaFile",
    "MessageRegistry",
    "PrivilegeRegistry",
})
_METADATA_URI_PARTS = (
    "/jsonschemas/",
    "/registries/",
)


def _bool_title(value: bool) -> str:
    """Return stable bool labels for goal IDs."""
    return "True" if value else "False"


def _id_part(value: Any) -> str:
    """Normalize a value into a readable goal-id component."""
    if isinstance(value, bool):
        return _bool_title(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "empty"


def _goal_value_id(value: Any) -> str:
    """Return the public-safe value component for a semantic goal id."""
    if isinstance(value, bool):
        return _bool_title(value)
    if isinstance(value, (int, float)):
        return _id_part(value)
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_ -]{1,64}", value):
        return _id_part(value)
    return "value"


def _snake(value: str) -> str:
    """Convert a Redfish class/property name into a readable id token."""
    if not value:
        return "resource"
    with_separators = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", with_separators)
    return normalized.strip("_").lower() or "resource"


def _resource_type_id(odata_type: str) -> str:
    """Map a Redfish ``@odata.type`` to a stable, readable resource-type id."""
    if "ComputerSystem" in odata_type:
        return "computer_system"
    if "ManagerNetworkProtocol" in odata_type:
        return "manager_network_protocol"
    if "Bios" in odata_type:
        return "bios"
    if "VirtualMedia" in odata_type:
        return "virtual_media"
    if "LogService" in odata_type:
        return "log_service"
    if "Task" in odata_type:
        return "task"
    if "." in odata_type:
        tail = odata_type.rsplit(".", 1)[-1]
        if tail and not tail.startswith("v"):
            return _snake(tail)
    return "resource"


def _is_metadata_document(record: SourceRecord) -> bool:
    """Return whether a corpus record documents schema, not mutable state."""
    body = record.response
    if not isinstance(body, Mapping):
        return False

    odata = str(body.get("@odata.type", record.schema_version))
    if any(resource_type in odata for resource_type in _METADATA_RESOURCE_TYPES):
        return True

    normalized_uri = record.url.lower()
    if any(uri_part in normalized_uri for uri_part in _METADATA_URI_PARTS):
        return True

    if "$schema" in body or "Definitions" in body or "definitions" in body:
        return True
    return "RegistryPrefix" in body and (
        "Messages" in body or "RegistryEntries" in body
    )


def _get_path(body: Mapping[str, Any], path: str) -> Any:
    """Read a dotted property path from a mapping."""
    current: Any = body
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _state_goal_ref(
    family: str,
    resource_type: str,
    property_path: str,
    target_value: Any,
) -> GoalRef:
    """Create a state-equality atomic sub-goal ref."""
    goal_id = ".".join((
        family,
        resource_type,
        property_path,
        "eq",
        _goal_value_id(target_value),
    ))
    return GoalRef(
        goal_id=goal_id,
        family=family,
        resource_type=resource_type,
        property_path=property_path,
        operator="eq",
        target_value=target_value,
        mode="state",
    )


def _transition_goal_ref(
    family: str,
    resource_type: str,
    action_name: str,
    argument_name: str,
    target_value: Any,
) -> GoalRef:
    """Create an action/transition atomic sub-goal ref."""
    goal_id = ".".join((
        family,
        resource_type,
        action_name,
        argument_name,
        "eq",
        _goal_value_id(target_value),
    ))
    return GoalRef(
        goal_id=goal_id,
        family=family,
        resource_type=resource_type,
        operator="eq",
        target_value=target_value,
        mode="transition",
        action_name=action_name,
        arguments={argument_name: target_value},
    )


def _surface(
    record: SourceRecord,
    goal_ref: GoalRef,
    fact_path: str,
    target_value: Any,
    current_value: Any,
    allowed_values: Sequence[Any] = (),
    verifier_kind: str = "state_eq",
) -> GoalSurface:
    """Create a vendor-specific goal surface around one atomic sub-goal."""
    verifier = {
        "kind": verifier_kind,
        "resource_uri": record.url,
        "property_path": fact_path,
        "operator": goal_ref.operator,
        "target_value": target_value,
    }
    if goal_ref.action_name:
        verifier["action_name"] = goal_ref.action_name
        verifier["arguments"] = dict(goal_ref.arguments)
    provenance = dict(record.provenance)
    if record.allowed_methods:
        provenance["allowed_methods"] = list(record.allowed_methods)
    return GoalSurface(
        goal_ref=goal_ref,
        vendor=record.vendor or "",
        source=record.source,
        resource_uri=record.url,
        resource_type=str(record.response.get("@odata.type", record.schema_version)),
        fact_path=fact_path,
        target_value=target_value,
        current_value=current_value,
        allowed_values=tuple(allowed_values),
        verifier=verifier,
        provenance=provenance,
    )


def _family_for_path(resource_type: str, path: str) -> str:
    """Infer a coarse semantic family from a resource type and property path."""
    lowered = f"{resource_type}.{path}".lower()
    if "power" in lowered:
        return "power"
    if "boot" in lowered:
        return "boot"
    if "bios" in lowered or "attribute" in lowered:
        return "bios"
    if "ntp" in lowered or "network" in lowered or "ethernet" in lowered:
        return "network"
    if "thermal" in lowered or "fan" in lowered or "temperature" in lowered:
        return "thermal"
    if "status" in lowered or "health" in lowered:
        return "status"
    if "log" in lowered:
        return "log"
    first = path.split(".", 1)[0].replace("[]", "")
    return _snake(first) if first else "state"


def _family_for_action(action_name: str) -> str:
    """Infer a coarse semantic family from a Redfish action name."""
    lowered = action_name.lower()
    if "reset" in lowered or "power" in lowered:
        return "action"
    if "log" in lowered or "clearlog" in lowered:
        return "log"
    if "virtualmedia" in lowered:
        return "virtual_media"
    if "." in action_name:
        return _snake(action_name.split(".", 1)[0])
    return "action"


def _is_scalar(value: Any) -> bool:
    """Whether a JSON value is a useful atomic state target."""
    return isinstance(value, _SCALAR_TYPES) and value is not None


def _is_sensitive_path(path: str) -> bool:
    """Whether a property path could carry secret or identity material."""
    normalized_path = _snake(path.replace("[]", " "))
    if any(token in normalized_path for token in _SENSITIVE_NAME_SUBSTRINGS):
        return True
    for part in path.split("."):
        normalized = _snake(part.replace("[]", ""))
        if normalized in _SENSITIVE_NAME_PARTS:
            return True
    return False


def _walk_scalar_leaves(
    node: Any,
    path: str = "",
) -> Iterable[tuple[str, Any]]:
    """Yield scalar leaves from a Redfish body, skipping metadata/link noise."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key.endswith(_ALLOWABLE_SUFFIX):
                continue
            if key in _SKIP_SUBTREES or key in _SKIP_LEAF_KEYS:
                continue
            child_path = f"{path}.{key}" if path else key
            if _is_sensitive_path(child_path):
                continue
            if _is_scalar(value):
                yield child_path, value
            elif isinstance(value, (Mapping, list)):
                yield from _walk_scalar_leaves(value, child_path)
    elif isinstance(node, list):
        for value in node:
            child_path = f"{path}[]" if path else "[]"
            if _is_scalar(value):
                yield child_path, value
            elif isinstance(value, (Mapping, list)):
                yield from _walk_scalar_leaves(value, child_path)


def _walk_allowable_values(
    node: Any,
    path: str = "",
    under_actions: bool = False,
) -> Iterable[tuple[str, tuple[Any, ...], bool]]:
    """Yield ``(property_path, allowed_values, under_actions)`` annotations."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            current_under_actions = under_actions or key == "Actions"
            if key.endswith(_ALLOWABLE_SUFFIX) and isinstance(value, list) and value:
                prop = key[: -len(_ALLOWABLE_SUFFIX)]
                prop_path = f"{path}.{prop}" if path else prop
                if _is_sensitive_path(prop_path):
                    continue
                yield prop_path, tuple(value), under_actions
                continue
            child_path = f"{path}.{key}" if path else key
            yield from _walk_allowable_values(value, child_path, current_under_actions)
    elif isinstance(node, list):
        for value in node:
            child_path = f"{path}[]" if path else "[]"
            yield from _walk_allowable_values(value, child_path, under_actions)


def _action_name_from_path(path: str) -> str:
    """Extract ``ComputerSystem.Reset`` from an ``Actions.#...`` path."""
    parts = path.split(".")
    for index, part in enumerate(parts):
        if part == "Actions" and index + 1 < len(parts):
            return parts[index + 1].lstrip("#")
    return ""


def _action_surfaces(record: SourceRecord) -> list[GoalSurface]:
    """Extract generic transition goals from a Redfish ``Actions`` block."""
    body = record.response
    actions = body.get("Actions")
    if not isinstance(actions, Mapping):
        return []
    resource_type = _resource_type_id(str(body.get("@odata.type", "")))
    surfaces: list[GoalSurface] = []
    for action_key, action_body in sorted(actions.items()):
        if not isinstance(action_body, Mapping):
            continue
        action_name = str(action_key).lstrip("#")
        if _is_sensitive_path(action_name):
            continue
        allowed_slots = [
            (path.rsplit(".", 1)[-1], values)
            for path, values, under_actions in _walk_allowable_values(
                action_body,
                f"Actions.{action_key}",
                under_actions=True,
            )
            if under_actions
        ]
        if allowed_slots:
            for argument_name, allowed_values in allowed_slots:
                for value in allowed_values:
                    surfaces.append(_surface(
                        record,
                        _transition_goal_ref(
                            _family_for_action(action_name),
                            resource_type,
                            action_name,
                            argument_name,
                            value,
                        ),
                        f"Actions.{action_key}.{argument_name}",
                        value,
                        None,
                        allowed_values,
                        verifier_kind="transition",
                    ))
            continue
        goal_id = ".".join((
            _family_for_action(action_name),
            resource_type,
            action_name,
            "invoke",
        ))
        surfaces.append(_surface(
            record,
            GoalRef(
                goal_id=goal_id,
                family=_family_for_action(action_name),
                resource_type=resource_type,
                mode="transition",
                action_name=action_name,
                arguments={},
            ),
            f"Actions.{action_key}",
            None,
            None,
            (),
            verifier_kind="transition",
        ))
    return surfaces


def _generic_state_surfaces(record: SourceRecord) -> list[GoalSurface]:
    """Extract state goals from arbitrary Redfish scalar leaves and enums."""
    body = record.response
    resource_type = _resource_type_id(str(body.get("@odata.type", "")))
    surfaces: list[GoalSurface] = []
    emitted: set[tuple[str, str, str]] = set()

    for path, allowed_values, under_actions in _walk_allowable_values(body):
        if under_actions:
            continue
        for value in allowed_values:
            if value in (None, ""):
                continue
            ref = _state_goal_ref(_family_for_path(resource_type, path), resource_type, path, value)
            key = (ref.goal_id, path, _id_part(value))
            if key in emitted:
                continue
            emitted.add(key)
            surfaces.append(_surface(
                record,
                ref,
                path,
                value,
                _get_path(body, path),
                allowed_values,
            ))

    for path, value in _walk_scalar_leaves(body):
        if value in (None, ""):
            continue
        ref = _state_goal_ref(_family_for_path(resource_type, path), resource_type, path, value)
        key = (ref.goal_id, path, _id_part(value))
        if key in emitted:
            continue
        emitted.add(key)
        surfaces.append(_surface(
            record,
            ref,
            path,
            value,
            value,
            (),
        ))
    return surfaces


def _computer_system_surfaces(record: SourceRecord) -> list[GoalSurface]:
    """Extract power, boot, and reset sub-goal surfaces from a ComputerSystem body."""
    body = record.response
    resource_type = _resource_type_id(str(body.get("@odata.type", "")))
    surfaces: list[GoalSurface] = []

    allowed_power = body.get(f"PowerState{_ALLOWABLE_SUFFIX}") or ["On", "Off"]
    for value in allowed_power:
        surfaces.append(_surface(
            record,
            _state_goal_ref("power", resource_type, "PowerState", value),
            "PowerState",
            value,
            body.get("PowerState"),
            allowed_power,
        ))

    boot = body.get("Boot")
    if isinstance(boot, Mapping):
        for prop in ("BootSourceOverrideTarget", "BootSourceOverrideEnabled"):
            allowed = boot.get(f"{prop}{_ALLOWABLE_SUFFIX}") or ()
            for value in allowed:
                if value in (None, ""):
                    continue
                surfaces.append(_surface(
                    record,
                    _state_goal_ref("boot", resource_type, f"Boot.{prop}", value),
                    f"Boot.{prop}",
                    value,
                    boot.get(prop),
                    allowed,
                ))

    existing = {surface.goal_ref.goal_id for surface in surfaces}
    for surface in _action_surfaces(record):
        if surface.goal_ref.goal_id not in existing:
            surfaces.append(surface)

    return surfaces


def _manager_network_protocol_surfaces(record: SourceRecord) -> list[GoalSurface]:
    """Extract NTP-related atomic sub-goal surfaces from ManagerNetworkProtocol."""
    body = record.response
    resource_type = _resource_type_id(str(body.get("@odata.type", "")))
    ntp = body.get("NTP")
    if not isinstance(ntp, Mapping):
        return []

    surfaces = []
    for value in (True, False):
        surfaces.append(_surface(
            record,
            _state_goal_ref(
                "network",
                resource_type,
                "NTP.ProtocolEnabled",
                value,
            ),
            "NTP.ProtocolEnabled",
            value,
            ntp.get("ProtocolEnabled"),
            (True, False),
        ))
    servers = ntp.get("NTPServers")
    if isinstance(servers, list):
        for server in servers:
            surfaces.append(_surface(
                record,
                _state_goal_ref(
                    "network",
                    resource_type,
                    "NTP.NTPServers",
                    server,
                ),
                "NTP.NTPServers",
                server,
                tuple(servers),
                tuple(servers),
                verifier_kind="contains",
            ))
    return surfaces


def _bios_surfaces(record: SourceRecord) -> list[GoalSurface]:
    """Extract BIOS attribute facts as atomic sub-goal surfaces."""
    body = record.response
    resource_type = _resource_type_id(str(body.get("@odata.type", "")))
    attrs = body.get("Attributes")
    if not isinstance(attrs, Mapping):
        return []
    return [
        _surface(
            record,
            _state_goal_ref("bios", resource_type, f"Attributes.{name}", value),
            f"Attributes.{name}",
            value,
            value,
            (),
        )
        for name, value in sorted(attrs.items())
    ]


def build_goal_surfaces(records: Iterable[SourceRecord]) -> tuple[GoalSurface, ...]:
    """Build deterministic atomic goal surfaces from captured Redfish records.

    :param records: already-captured Redfish JSON records.
    :return: atomic, vendor-specific goal surfaces.
    """
    surfaces: list[GoalSurface] = []
    for record in records:
        if not isinstance(record.response, Mapping):
            continue
        if _is_metadata_document(record):
            continue
        odata = str(record.response.get("@odata.type", record.schema_version))
        surfaces.extend(_generic_state_surfaces(record))
        surfaces.extend(_action_surfaces(record))
        if "ComputerSystem" in odata:
            surfaces.extend(_computer_system_surfaces(record))
        elif "ManagerNetworkProtocol" in odata:
            surfaces.extend(_manager_network_protocol_surfaces(record))
        elif "Bios" in odata:
            surfaces.extend(_bios_surfaces(record))

    deduped: list[GoalSurface] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for surface in surfaces:
        key = (
            surface.goal_ref.goal_id,
            surface.vendor,
            surface.source,
            surface.resource_uri,
            surface.fact_path,
            _id_part(surface.target_value),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(surface)
    return tuple(deduped)


def make_goal_text_example(
    text: str,
    goal_refs: Sequence[GoalRef],
    dependencies: Sequence[GoalDependency] = (),
    text_source: str = "template",
    split: str = "train",
    metadata: Mapping[str, Any] | None = None,
) -> GoalTextExample:
    """Create a text example for atomic or compound sub-goal extraction.

    The order of ``goal_refs`` is preserved only for stable serialization.
    Semantics are set-like unless ``dependencies`` explicitly add a text-level
    partial order.
    """
    return GoalTextExample(
        text=text,
        goal_refs=tuple(goal_refs),
        dependencies=tuple(dependencies),
        text_source=text_source,
        split=split,
        metadata=dict(metadata or {}),
    )


# Author: Mus mbayramo@stanford.edu
