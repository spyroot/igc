"""Offline, pure-stdlib tests for igc.core.action_render (no GPU/torch/network).

Locks the canonical ToolAction rendering used by the pointer / candidate-scoring
policy: deterministic, order-stable, and value-independent — the template key
embeds the action TYPE (tool/op/target/arg-shape), never concrete argument values.

Author:
Mus mbayramo@stanford.edu
"""
from igc.core.action_render import action_template_key, action_to_prompt
from igc.core.types import RiskLevel, ToolAction, ToolSpec


def test_golden_render_get():
    """A simple GET action renders to the exact canonical string."""
    a = ToolAction(tool_name="redfish", op="GET", target="/redfish/v1/Systems")
    assert action_to_prompt(a) == "tool=redfish op=GET target=/redfish/v1/Systems args=[]"


def test_render_includes_arg_shape_not_values():
    """Argument keys+types appear (sorted); concrete values never do."""
    a = ToolAction(
        tool_name="redfish",
        op="PATCH",
        target="/redfish/v1/Systems/Bios/Settings",
        arguments={"Attributes": {"BootMode": "Uefi"}},
        risk_level=RiskLevel.MUTATING,
    )
    rendered = action_to_prompt(a)
    assert rendered == (
        "tool=redfish op=PATCH target=/redfish/v1/Systems/Bios/Settings args=[Attributes:object]"
    )
    assert "Uefi" not in rendered  # concrete values are never rendered


def test_arg_key_order_is_stable():
    """Different argument insertion order yields identical rendering and key."""
    a = ToolAction("sql", "INSERT", arguments={"a": 1, "b": "x"})
    b = ToolAction("sql", "INSERT", arguments={"b": "y", "a": 2})
    assert action_to_prompt(a) == action_to_prompt(b)
    assert action_template_key(a) == action_template_key(b)


def test_template_key_is_value_independent():
    """Actions differing only in concrete argument VALUES share a template key."""
    a = ToolAction("fs", "write", target="/tmp/x", arguments={"content": "hello"})
    b = ToolAction("fs", "write", target="/tmp/x", arguments={"content": "world"})
    assert action_template_key(a) == action_template_key(b)


def test_template_key_distinguishes_target_and_op():
    """A different target or op yields a different template key."""
    base = ToolAction("redfish", "GET", target="/redfish/v1/Systems")
    other_target = ToolAction("redfish", "GET", target="/redfish/v1/Chassis")
    other_op = ToolAction("redfish", "HEAD", target="/redfish/v1/Systems")
    assert action_template_key(base) != action_template_key(other_target)
    assert action_template_key(base) != action_template_key(other_op)


def test_equal_actions_render_and_key_identically():
    """ToolActions equal under __eq__/__hash__ render and key identically."""
    a = ToolAction("github", "POST", target="/repos/x/y/pulls", arguments={"title": "t"})
    b = ToolAction("github", "POST", target="/repos/x/y/pulls", arguments={"title": "t"})
    assert a == b
    assert action_to_prompt(a) == action_to_prompt(b)
    assert action_template_key(a) == action_template_key(b)


def test_spec_arg_schema_overrides_inferred_types():
    """When a ToolSpec is given, arg types come from its schema, not the values."""
    spec = ToolSpec(
        tool_name="sql",
        ops=["INSERT"],
        arg_schema={"INSERT": {"row": {"type": "object"}, "table": {"type": "string"}}},
    )
    a = ToolAction("sql", "INSERT", arguments={"row": {"id": 1}, "table": "users"})
    assert action_to_prompt(a, spec) == "tool=sql op=INSERT args=[row:object,table:string]"


def test_bool_typed_before_int():
    """A boolean argument renders as boolean, not integer."""
    a = ToolAction("fs", "chmod", arguments={"recursive": True})
    assert action_to_prompt(a) == "tool=fs op=chmod args=[recursive:boolean]"


def test_schema_id_appears_when_set():
    """schema_id is appended only when truthy."""
    a = ToolAction("redfish", "PATCH", target="/x", schema_id="set_bios")
    assert action_to_prompt(a).endswith("schema=set_bios")
    b = ToolAction("redfish", "GET", target="/x")
    assert "schema=" not in action_to_prompt(b)


def test_edge_empty_args_none_target_none_schema():
    """Empty args, None target, None schema_id all render without error."""
    a = ToolAction("fs", "ls")
    assert action_to_prompt(a) == "tool=fs op=ls args=[]"
    assert isinstance(action_template_key(a), str) and len(action_template_key(a)) == 32


# Author: Mus mbayramo@stanford.edu
