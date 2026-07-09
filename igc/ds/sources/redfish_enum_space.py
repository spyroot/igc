"""
Extract per-slot argument value spaces from captured Redfish resources.

The stage-2 argument decoder (:mod:`igc.modules.policy.argument_decoder`) scores a
categorical argument over its OWN allowable values, read from
``ToolSpec.arg_schema[op][slot]["enum"]``. This module builds those enum spaces
offline from the resource types ``redfish_ctl`` discovery captures under
``~/.json_responses/<host>/`` (and the vendor fixture corpora): BIOS attribute
registries (``#AttributeRegistry`` bodies with ``RegistryEntries.Attributes``),
``#ActionInfo`` bodies (``Parameters[].AllowableValues``), inline
``<prop>@Redfish.AllowableValues`` annotations (e.g. ``Boot`` and ``Actions``
blocks of a ``#ComputerSystem``), and ``#BootOption`` entries (the
``BootOptionReference`` id space that bounds ``Boot.BootOrder``).

Everything here is pure stdlib and consumes already-captured
:class:`~igc.ds.sources.base.SourceRecord` streams (typically from
:class:`~igc.ds.sources.redfish_fixture_source.RedfishFixtureSource`); it never
touches a live controller. HPE-style registry *pointers*
(``#MessageRegistryFile`` bodies whose ``Location[].Uri`` names an external
registry) are counted but yield no slots — the pointed-to payload is a separate
capture.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from igc.ds.sources.base import SourceRecord
from igc.ds.sources.training_object import TrainingExample, normalize_record

# suffix of the standard Redfish annotation carrying a property's allowable values.
ALLOWABLE_SUFFIX = "@Redfish.AllowableValues"

# Redfish registry / ActionInfo type names -> JSON-schema-style type names used
# in ``arg_schema`` fragments (see igc.modules.policy.argument_decoder.arg_slots_for).
_TYPE_MAP = {
    "enumeration": "string",
    "string": "string",
    "password": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
}


class ResourceKind(enum.Enum):
    """Semantic type of a captured Redfish resource, keyed off ``@odata.type``.

    ``classify_resource`` falls back to URL heuristics when a capture carries no
    ``@odata.type`` (older iLO firmware omits it on some resources).
    """
    BIOS = "bios"
    BIOS_PENDING = "bios_pending"
    ATTRIBUTE_REGISTRY = "attribute_registry"
    REGISTRY_FILE = "registry_file"
    BOOT_OPTION_COLLECTION = "boot_option_collection"
    BOOT_OPTION = "boot_option"
    ACTION_INFO = "action_info"
    COMPUTER_SYSTEM = "computer_system"
    OTHER = "other"


# (odata.type prefix, kind) in match order; BIOS is refined to BIOS_PENDING by URL.
_ODATA_PREFIXES: Tuple[Tuple[str, ResourceKind], ...] = (
    ("#AttributeRegistry.", ResourceKind.ATTRIBUTE_REGISTRY),
    ("#MessageRegistryFile.", ResourceKind.REGISTRY_FILE),
    ("#BootOptionCollection.", ResourceKind.BOOT_OPTION_COLLECTION),
    ("#BootOption.", ResourceKind.BOOT_OPTION),
    ("#ActionInfo.", ResourceKind.ACTION_INFO),
    ("#Bios.", ResourceKind.BIOS),
    ("#ComputerSystem.", ResourceKind.COMPUTER_SYSTEM),
)


def _is_pending_settings_url(url: str) -> bool:
    """True when *url* names the pending-settings child of a BIOS resource.

    Matches the standard ``.../Bios/Settings`` (Dell) and lowercase
    ``.../bios/settings`` (iLO) forms.

    :param url: canonical resource URL.
    :return: whether the URL is a BIOS pending-settings resource.
    """
    parts = url.rstrip("/").lower().rsplit("/", 2)
    return len(parts) == 3 and parts[1] == "bios" and parts[2] == "settings"


def classify_resource(url: str, body: Dict[str, Any]) -> ResourceKind:
    """Classify a captured resource by ``@odata.type``, with URL fallbacks.

    :param url: canonical resource URL (``@odata.id`` or filename-derived).
    :param body: decoded JSON resource body.
    :return: the :class:`ResourceKind`; ``OTHER`` when nothing matches.
    """
    odata_type = body.get("@odata.type", "") if isinstance(body, dict) else ""
    for prefix, kind in _ODATA_PREFIXES:
        if isinstance(odata_type, str) and odata_type.startswith(prefix):
            if kind is ResourceKind.BIOS and _is_pending_settings_url(url):
                return ResourceKind.BIOS_PENDING
            return kind
    # URL fallbacks for captures without @odata.type.
    lowered = url.rstrip("/").lower()
    if _is_pending_settings_url(url):
        return ResourceKind.BIOS_PENDING
    if lowered.endswith("/bios"):
        return ResourceKind.BIOS
    if "/bootoptions/" in lowered:
        return ResourceKind.BOOT_OPTION
    if lowered.endswith("/bootoptions"):
        return ResourceKind.BOOT_OPTION_COLLECTION
    return ResourceKind.OTHER


@dataclass(frozen=True)
class EnumSlot:
    """One argument slot extracted from a captured resource.

    Mirrors what :func:`igc.modules.policy.argument_decoder.arg_slots_for` needs:
    a slot with ``choices`` is categorical (finite enum space); without, it is a
    typed free-form slot for the generative head.

    :param name: slot key (attribute / parameter / property name).
    :param type: JSON-schema-style type name (``string``/``integer``/...).
    :param choices: allowable values, or ``None`` for a free-form slot.
    :param required: whether the op is invalid without this slot.
    :param read_only: registry ``ReadOnly`` flag (read-only slots are excluded
        from PATCH schemas).
    :param current_value: value observed at capture time, when the source carries one.
    :param source_url: URL of the resource the slot was extracted from.
    :param extracted_from: which extractor produced it (``registry`` /
        ``action_info`` / ``inline``).
    :param path: dotted location inside the body for inline annotations
        (e.g. ``Boot.BootSourceOverrideTarget``), else ``""``.
    """
    name: str
    type: str = "string"
    choices: Optional[Tuple] = None
    required: bool = False
    read_only: bool = False
    current_value: Any = None
    source_url: str = ""
    extracted_from: str = ""
    path: str = ""

    def to_schema_fragment(self) -> Dict[str, Any]:
        """Return the ``arg_schema[op][slot]`` fragment for this slot.

        The fragment shape (``type`` / ``enum`` / ``required``) is exactly what
        ``arg_slots_for`` reads when sizing the stage-2 decoder head.

        :return: dict with ``type``, ``required``, and ``enum`` when categorical.
        """
        fragment: Dict[str, Any] = {"type": self.type, "required": self.required}
        if self.choices:
            fragment["enum"] = list(self.choices)
        return fragment


def _map_type(type_name: Any) -> str:
    """Map a Redfish registry/ActionInfo type name to a JSON-schema-style name.

    :param type_name: e.g. ``"Enumeration"`` / ``"Integer"`` / ``"String"``.
    :return: mapped name; unknown or missing types default to ``"string"``.
    """
    if not isinstance(type_name, str):
        return "string"
    return _TYPE_MAP.get(type_name.lower(), "string")


def slots_from_attribute_registry(url: str, body: Dict[str, Any]) -> List[EnumSlot]:
    """Extract per-attribute slots from an ``#AttributeRegistry`` body.

    Walks ``RegistryEntries.Attributes``: an ``Enumeration`` attribute yields a
    categorical slot whose choices come from ``Value[].ValueName``; other types
    yield typed free-form slots. Entries without an ``AttributeName`` are
    skipped. A registry *pointer* (no ``RegistryEntries``, HPE style) yields [].

    :param url: URL of the registry resource.
    :param body: decoded registry body.
    :return: extracted slots in registry order.
    """
    entries = body.get("RegistryEntries")
    if not isinstance(entries, dict):
        return []
    slots: List[EnumSlot] = []
    for attr in entries.get("Attributes", []) or []:
        if not isinstance(attr, dict):
            continue
        name = attr.get("AttributeName")
        if not isinstance(name, str) or not name:
            continue
        choices: Optional[Tuple] = None
        if attr.get("Type") == "Enumeration":
            names = [
                v.get("ValueName") for v in attr.get("Value", []) or []
                if isinstance(v, dict) and v.get("ValueName") is not None
            ]
            choices = tuple(names) if names else None
        slots.append(EnumSlot(
            name=name,
            type=_map_type(attr.get("Type")),
            choices=choices,
            read_only=bool(attr.get("ReadOnly", False)),
            current_value=attr.get("CurrentValue"),
            source_url=url,
            extracted_from="registry",
        ))
    return slots


def slots_from_action_info(url: str, body: Dict[str, Any]) -> List[EnumSlot]:
    """Extract parameter slots from an ``#ActionInfo`` body.

    Follows the DMTF ActionInfo shape: ``Parameters[]`` entries with ``Name``,
    ``Required``, ``DataType``, and ``AllowableValues``. Discovery does not yet
    follow ``@Redfish.ActionInfo`` references, so bodies come from targeted
    captures; the shape is pinned by the fixture tests either way.

    :param url: URL of the ActionInfo resource.
    :param body: decoded ActionInfo body.
    :return: extracted slots in parameter order.
    """
    slots: List[EnumSlot] = []
    for param in body.get("Parameters", []) or []:
        if not isinstance(param, dict):
            continue
        name = param.get("Name")
        if not isinstance(name, str) or not name:
            continue
        values = param.get("AllowableValues")
        choices = tuple(values) if isinstance(values, list) and values else None
        slots.append(EnumSlot(
            name=name,
            type=_map_type(param.get("DataType")),
            choices=choices,
            required=bool(param.get("Required", False)),
            source_url=url,
            extracted_from="action_info",
        ))
    return slots


def slots_from_inline_annotations(url: str, body: Dict[str, Any]) -> List[EnumSlot]:
    """Extract slots from ``<prop>@Redfish.AllowableValues`` annotations anywhere in a body.

    Vendors annotate writable properties inline instead of (or besides)
    publishing ActionInfo — e.g. ``Boot.BootSourceOverrideTarget`` and the
    ``ResetType`` of an ``Actions`` block. The dotted ``path`` records where the
    annotation sat, so callers can split action parameters (path under
    ``Actions``) from patchable properties.

    :param url: URL of the annotated resource.
    :param body: decoded resource body.
    :return: extracted slots in depth-first key order.
    """
    slots: List[EnumSlot] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith(ALLOWABLE_SUFFIX) and isinstance(value, list) and value:
                    prop = key[: -len(ALLOWABLE_SUFFIX)]
                    slots.append(EnumSlot(
                        name=prop,
                        type="string",
                        choices=tuple(value),
                        current_value=node.get(prop),
                        source_url=url,
                        extracted_from="inline",
                        path=f"{path}.{prop}" if path else prop,
                    ))
                else:
                    _walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for item in node:
                _walk(item, path)

    _walk(body, "")
    return slots


def to_arg_schema(op: str, slots: Iterable[EnumSlot]) -> Dict[str, Dict[str, Any]]:
    """Assemble slots into the ``{op: {slot: fragment}}`` shape ``ToolSpec.arg_schema`` takes.

    First-seen wins on a duplicate slot name (mirrors the adapter-order
    convention in :class:`~igc.ds.sources.mixer.SourceMix` dedup), so feed
    higher-trust records first.

    :param op: the operation the slots belong to (e.g. ``"PATCH"`` / ``"POST"``).
    :param slots: slots to assemble.
    :return: an ``arg_schema``-shaped dict for one op.
    """
    fragments: Dict[str, Any] = {}
    for slot in slots:
        if slot.name not in fragments:
            fragments[slot.name] = slot.to_schema_fragment()
    return {op: fragments}


class EnumSpaceIndex:
    """Accumulate per-slot value spaces from a provenance-tagged record stream.

    Routes each :class:`SourceRecord` by :func:`classify_resource`: attribute
    registries and ActionInfo bodies feed their dedicated extractors, every body
    is scanned for inline annotations, and ``#BootOption`` entries contribute
    their ``BootOptionReference`` to the boot-order id space. Feed records in
    trust order (real first) — schema exports keep the first-seen fragment per
    slot name.
    """

    def __init__(self) -> None:
        #: slots from registry bodies, keyed by registry URL.
        self.registry_slots: Dict[str, List[EnumSlot]] = {}
        #: action-parameter slots from ``#ActionInfo`` bodies (authoritative per DMTF).
        self.action_slots: Dict[str, List[EnumSlot]] = {}
        #: action-parameter slots from inline ``Actions`` annotations (fallback tier).
        self.inline_action_slots: Dict[str, List[EnumSlot]] = {}
        #: patchable-property slots (inline annotations outside ``Actions``).
        self.settings_slots: Dict[str, List[EnumSlot]] = {}
        #: ordered unique ``BootOptionReference`` ids seen across boot options.
        self.boot_references: List[str] = []
        #: record count per :class:`ResourceKind` name (all records, not just slot-bearing).
        self.by_kind: Dict[str, int] = {}
        #: registry pointers seen (kind ``REGISTRY_FILE``) — their payload is external.
        self.num_registry_pointers = 0

    @classmethod
    def from_records(cls, records: Iterable[SourceRecord]) -> "EnumSpaceIndex":
        """Build an index from a record stream.

        :param records: provenance-tagged records, highest trust first.
        :return: the populated index.
        """
        index = cls()
        for record in records:
            index.add_record(record)
        return index

    def add_record(self, record: SourceRecord) -> ResourceKind:
        """Classify one record and fold its value spaces into the index.

        :param record: a captured GET observation.
        :return: the kind the record classified as.
        """
        kind = classify_resource(record.url, record.response)
        self.by_kind[kind.name] = self.by_kind.get(kind.name, 0) + 1

        if kind is ResourceKind.ATTRIBUTE_REGISTRY:
            slots = slots_from_attribute_registry(record.url, record.response)
            if slots:
                self.registry_slots.setdefault(record.url, []).extend(slots)
        elif kind is ResourceKind.REGISTRY_FILE:
            self.num_registry_pointers += 1
        elif kind is ResourceKind.ACTION_INFO:
            slots = slots_from_action_info(record.url, record.response)
            if slots:
                self.action_slots.setdefault(record.url, []).extend(slots)
        elif kind is ResourceKind.BOOT_OPTION:
            reference = record.response.get("BootOptionReference") or record.response.get("Id")
            if isinstance(reference, str) and reference and reference not in self.boot_references:
                self.boot_references.append(reference)

        # any body may carry inline annotations; split them by location.
        for slot in slots_from_inline_annotations(record.url, record.response):
            top_segment = slot.path.split(".", 1)[0]
            bucket = self.inline_action_slots if top_segment == "Actions" else self.settings_slots
            bucket.setdefault(record.url, []).append(slot)
        return kind

    def _merged(self, buckets: Iterable[Dict[str, List[EnumSlot]]],
                writeable_only: bool = False) -> List[EnumSlot]:
        """Flatten slot buckets in insertion order, optionally dropping read-only slots.

        :param buckets: bucket dicts to merge.
        :param writeable_only: when true, exclude ``read_only`` slots.
        :return: the flattened slot list.
        """
        merged: List[EnumSlot] = []
        for bucket in buckets:
            for slots in bucket.values():
                merged.extend(s for s in slots if not (writeable_only and s.read_only))
        return merged

    def patch_arg_schema(self, op: str = "PATCH") -> Dict[str, Dict[str, Any]]:
        """The ``arg_schema`` for settings writes: registry attributes + patchable properties.

        Read-only registry attributes are excluded — a PATCH can never carry
        them. First-seen wins per slot name across registries and hosts.

        :param op: schema key to emit under (default ``"PATCH"``).
        :return: ``{op: {attribute: fragment}}``.
        """
        return to_arg_schema(op, self._merged(
            (self.registry_slots, self.settings_slots), writeable_only=True))

    def post_arg_schema(self, op: str = "POST") -> Dict[str, Dict[str, Any]]:
        """The ``arg_schema`` for action invocations.

        ``#ActionInfo`` parameters take precedence over inline ``Actions``
        annotations on a name clash, whatever order the records streamed in —
        DMTF makes ActionInfo the authoritative parameter description.

        :param op: schema key to emit under (default ``"POST"``).
        :return: ``{op: {parameter: fragment}}``.
        """
        return to_arg_schema(op, self._merged((self.action_slots, self.inline_action_slots)))

    def boot_reference_space(self) -> Tuple[str, ...]:
        """The ordered ``BootOptionReference`` id space bounding ``Boot.BootOrder``.

        :return: unique references in first-seen order.
        """
        return tuple(self.boot_references)


def normalize_enriched(records: Iterable[SourceRecord]) -> List[TrainingExample]:
    """Normalize records into TrainingExamples stamped with their resource kind.

    Delegates to :func:`~igc.ds.sources.training_object.normalize_record` and adds
    ``expected_semantics["resource_kind"]`` so the enriched resource types (BIOS
    attributes, boot options, registries) are distinguishable downstream without
    re-parsing bodies.

    :param records: iterable of GET-observation records.
    :return: one enriched :class:`TrainingExample` per record.
    """
    examples: List[TrainingExample] = []
    for record in records:
        kind = classify_resource(record.url, record.response)
        example = normalize_record(record)
        example.expected_semantics["resource_kind"] = kind.value
        examples.append(example)
    return examples


# Author: Mus mbayramo@stanford.edu
