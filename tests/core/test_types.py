"""Offline, pure-stdlib tests for igc.core.types (no GPU/torch/network).

Covers the keystone ToolAction (hashability, JSON round-trip, equality/hash
consistency, defensive copying) and the other core dataclasses' defaults.

Author:
Mus mbayramo@stanford.edu
"""
import json

import pytest

from igc.core.types import (
    Goal,
    Observation,
    RiskLevel,
    SimResult,
    ToolAction,
    ToolSpec,
    Transition,
)


def test_risk_level_is_monotone():
    """Risk levels must order READ_ONLY < MUTATING < DESTRUCTIVE for the gate."""
    assert RiskLevel.READ_ONLY < RiskLevel.MUTATING < RiskLevel.DESTRUCTIVE
    assert int(RiskLevel.DESTRUCTIVE) == 2


def test_toolaction_defaults_are_read_only_and_empty():
    """A bare ToolAction defaults to read-only with empty, independent args."""
    a = ToolAction(tool_name="redfish", op="GET")
    assert a.risk_level is RiskLevel.READ_ONLY
    assert a.arguments == {}
    assert a.target is None
    # default_factory gives each instance its own dict
    b = ToolAction(tool_name="redfish", op="GET")
    assert a.arguments is not b.arguments


def test_toolaction_is_hashable_despite_dict_field():
    """The frozen dataclass with a dict field is still usable in sets/dicts."""
    a = ToolAction("fs", "write", {"path": "/tmp/x", "content": "hi"}, target="/tmp/x")
    s = {a, ToolAction("fs", "write", {"path": "/tmp/x", "content": "hi"}, target="/tmp/x")}
    assert len(s) == 1  # equal actions collapse


def test_equal_actions_have_equal_hash():
    """Equality and hashing must agree (arguments key order independent)."""
    a = ToolAction("sql", "INSERT", {"a": 1, "b": 2})
    b = ToolAction("sql", "INSERT", {"b": 2, "a": 1})
    assert a == b
    assert hash(a) == hash(b)


def test_toolaction_json_round_trip():
    """from_dict(to_dict(a)) == a, and to_dict is JSON-serializable."""
    a = ToolAction(
        tool_name="github",
        op="POST",
        arguments={"title": "fix", "labels": ["bug"]},
        target="/repos/x/y/pulls",
        risk_level=RiskLevel.MUTATING,
        schema_id="open_pr",
    )
    d = a.to_dict()
    assert json.loads(json.dumps(d)) == d  # serializable
    assert d["risk_level"] == "MUTATING"  # enum by name
    assert ToolAction.from_dict(d) == a


def test_to_dict_copies_arguments():
    """Mutating the dict returned by to_dict must not affect the frozen action."""
    a = ToolAction("fs", "write", {"path": "/tmp/x"})
    d = a.to_dict()
    d["arguments"]["path"] = "/tmp/HACKED"
    assert a.arguments["path"] == "/tmp/x"


def test_from_dict_tolerates_missing_optionals():
    """from_dict fills missing optional keys with sane defaults."""
    a = ToolAction.from_dict({"tool_name": "fs", "op": "ls", "risk_level": "READ_ONLY"})
    assert a == ToolAction("fs", "ls")


@pytest.mark.parametrize("bad", [{}, {"tool_name": "x"}, {"op": "GET"}])
def test_from_dict_requires_core_keys(bad):
    """Missing required keys (tool_name/op/risk_level) raise KeyError."""
    with pytest.raises(KeyError):
        ToolAction.from_dict(bad)


def test_transition_carries_done_flags():
    """Transition exposes the terminated/truncated flags the 4-tuple buffer lacked."""
    t = Transition(observation=Observation(text="{}"), reward=1.0, terminated=True, truncated=False)
    assert t.terminated and not t.truncated
    assert t.info == {}


def test_observation_and_simresult_defaults():
    """Observation/SimResult expose sensible status defaults."""
    assert Observation(text="x").status == 200
    assert SimResult(body=None, status=404).error is False


def test_goal_plan_and_constraints_default_empty():
    """A Goal starts with no plan and independent empty constraints."""
    g = Goal(instruction="boot ubuntu")
    assert g.plan is None
    assert g.constraints == []
    assert Goal(instruction="x").constraints is not g.constraints


def test_toolspec_defaults_enumerated_read_only():
    """A ToolSpec defaults to an enumerated target space and read-only risk."""
    spec = ToolSpec(tool_name="redfish", ops=["GET", "HEAD"])
    assert spec.target_space == "enumerated"
    assert spec.risk_level is RiskLevel.READ_ONLY


# Author: Mus mbayramo@stanford.edu
