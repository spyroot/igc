"""Safe cold-start probing for an unknown tool (``docs/ARCHITECTURE.md`` §12.4 seam D).

When the agent meets a tool it does not know, it should *learn* it from low-risk
reads before risking a mutating call. :func:`safe_probe_actions` returns only the
candidates that are simultaneously legal now (:meth:`ToolCatalog.available_actions`),
argument-valid (:meth:`ToolCatalog.validate`), and at or below a
:class:`~igc.core.types.RiskLevel` ceiling (``READ_ONLY`` by default) — so on a real
BMC a probe physically cannot issue a mutating op. This is the binding-safety side of
tool-teaching: a card is advisory only and may never lift this ceiling (§12.5).

Pure stdlib; the catalog is duck-typed (the :class:`~igc.core.protocols.ToolCatalog`
protocol), so this runs in the offline CPU subset against the mock REST env.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import List, Optional

from igc.core.tool_card import ToolCard
from igc.core.types import Observation, RiskLevel, ToolAction


def safe_probe_actions(
    card: Optional[ToolCard],
    catalog,
    obs: Observation,
    ceiling: RiskLevel = RiskLevel.READ_ONLY,
) -> List[ToolAction]:
    """The legal, valid, at-or-below-``ceiling`` candidate actions for probing.

    The returned actions are a subset of ``catalog.available_actions(obs)`` (so the
    catalog — never the card — remains the sole authority on legality and risk). When
    a ``card`` is given, the result is narrowed to that card's tool, i.e. safe ways to
    probe *that* unknown tool; with ``card=None`` it is every safe candidate.

    :param card: the card whose tool is being probed, or ``None`` for all tools.
    :param catalog: a :class:`~igc.core.protocols.ToolCatalog` (duck-typed:
        ``available_actions`` + ``validate``).
    :param obs: the current observation, passed to ``available_actions``.
    :param ceiling: the maximum allowed :class:`~igc.core.types.RiskLevel`
        (default ``READ_ONLY``); a card can never raise this.
    :return: the safe probe actions, in ``available_actions`` order.
    """
    safe = [
        action
        for action in catalog.available_actions(obs)
        if action.risk_level <= ceiling and catalog.validate(action)
    ]
    if card is not None:
        safe = [action for action in safe if action.tool_name == card.tool_name]
    return safe


# Author: Mus mbayramo@stanford.edu
