"""Offline tests for safe_probe_actions (tool-teaching safe probing, ARCHITECTURE §12.4-D).

Cold-start probing of an unknown tool must stay at or below a RiskLevel ceiling and
respect the catalog's own legality/validation — a card may narrow the probes to its
tool but can never lift the ceiling. Uses a tiny in-memory catalog; pure stdlib.

Author:
Mus mbayramo@stanford.edu
"""

from igc.core.tool_card import ToolCard
from igc.core.types import Observation, RiskLevel, ToolAction
from igc.modules.teach.safe_probe import safe_probe_actions


class FakeCatalog:
    """A minimal ToolCatalog: fixed candidate list + an allow-set for validate."""

    def __init__(self, actions, invalid=None):
        self._actions = actions
        self._invalid = set(invalid or [])

    def available_actions(self, obs):
        return list(self._actions)

    def validate(self, action):
        return (action.tool_name, action.op) not in self._invalid


_OBS = Observation(text="{}")


def test_excludes_actions_above_ceiling() -> None:
    """Mutating/destructive candidates are excluded under the READ_ONLY ceiling."""
    catalog = FakeCatalog([
        ToolAction("ComputerSystem", "GET", risk_level=RiskLevel.READ_ONLY),
        ToolAction("ComputerSystem", "Reset", risk_level=RiskLevel.MUTATING),
        ToolAction("ComputerSystem", "Delete", risk_level=RiskLevel.DESTRUCTIVE),
    ])
    probes = safe_probe_actions(None, catalog, _OBS)
    assert [a.op for a in probes] == ["GET"]


def test_respects_catalog_validate() -> None:
    """A candidate the catalog rejects is not offered as a probe."""
    catalog = FakeCatalog(
        [
            ToolAction("ComputerSystem", "GET", risk_level=RiskLevel.READ_ONLY),
            ToolAction("ComputerSystem", "HEAD", risk_level=RiskLevel.READ_ONLY),
        ],
        invalid=[("ComputerSystem", "HEAD")],
    )
    probes = safe_probe_actions(None, catalog, _OBS)
    assert [a.op for a in probes] == ["GET"]


def test_card_narrows_probes_to_its_tool() -> None:
    """With a card, only that card's tool is probed (other safe tools excluded)."""
    catalog = FakeCatalog([
        ToolAction("ComputerSystem", "GET", risk_level=RiskLevel.READ_ONLY),
        ToolAction("Chassis", "GET", risk_level=RiskLevel.READ_ONLY),
    ])
    card = ToolCard(env_name="redfish", tool_name="Chassis", op="GET")
    probes = safe_probe_actions(card, catalog, _OBS)
    assert [a.tool_name for a in probes] == ["Chassis"]


def test_custom_ceiling_allows_mutating() -> None:
    """An explicit MUTATING ceiling admits mutating probes (but never via a card)."""
    catalog = FakeCatalog([
        ToolAction("ComputerSystem", "Reset", risk_level=RiskLevel.MUTATING),
        ToolAction("ComputerSystem", "Delete", risk_level=RiskLevel.DESTRUCTIVE),
    ])
    probes = safe_probe_actions(None, catalog, _OBS, ceiling=RiskLevel.MUTATING)
    assert [a.op for a in probes] == ["Reset"]


# Author: Mus mbayramo@stanford.edu
