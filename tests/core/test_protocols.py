"""Offline, pure-stdlib tests for igc.core.protocols (no GPU/torch/network).

Verifies the Protocols are runtime_checkable and that a complete duck-typed stub
satisfies ``isinstance`` while an incomplete one does not — the structural contract
the env registry relies on.
"""
from igc.core.protocols import Evaluator, GoalEnvironment, Simulator, ToolCatalog
from igc.core.types import Observation, SimResult, Transition


class GoodSim:
    def reset(self, seed=None):
        pass

    def execute(self, target, op, args=None):
        return SimResult(body=None, status=200)

    def snapshot(self):
        return None

    def restore(self, snap):
        pass

    def is_live(self):
        return False


class GoodCatalog:
    def specs(self):
        return []

    def available_actions(self, obs):
        return []

    def validate(self, action):
        return True


class GoodEvaluator:
    def verify(self, goal, obs):
        return (False, 0.0)


class GoodEnv:
    name = "redfish"

    def reset(self, goal):
        return Observation(text="{}")

    def available_actions(self, obs):
        return []

    def step(self, action):
        return Transition(
            observation=Observation(text="{}"), reward=0.0, terminated=False, truncated=False
        )

    def verify(self, obs, goal):
        return (False, 0.0)


def test_complete_stubs_satisfy_protocols():
    """A class implementing every member structurally passes isinstance."""
    assert isinstance(GoodSim(), Simulator)
    assert isinstance(GoodCatalog(), ToolCatalog)
    assert isinstance(GoodEvaluator(), Evaluator)
    assert isinstance(GoodEnv(), GoalEnvironment)


def test_incomplete_stub_fails_isinstance():
    """A class missing required methods is not an instance of the Protocol."""

    class PartialSim:
        def reset(self, seed=None):
            pass  # missing execute/snapshot/restore/is_live

    assert not isinstance(PartialSim(), Simulator)


def test_protocols_are_runtime_checkable():
    """isinstance against each Protocol must not raise (i.e. runtime_checkable)."""
    for proto in (Simulator, ToolCatalog, Evaluator, GoalEnvironment):
        assert isinstance(object(), proto) is False
