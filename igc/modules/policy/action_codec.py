"""Encode a list of ToolActions into candidate key embeddings for the pointer policy.

Bridges :mod:`igc.core.action_render` (the canonical, value-independent text + template
key for an action) and a :class:`~igc.modules.llm.llm_encoder.TextEncoder`. For a state's
legal candidates (``GoalEnvironment.available_actions(obs)``) it returns a ``[N, H]``
key matrix the pointer policy scores against the state query. Embeddings are cached by
:func:`igc.core.action_render.action_template_key`, so identical action *types* (which
differ only in concrete argument values) are encoded exactly once — the cache that makes
scoring a large, dynamic candidate set cheap.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

from igc.core.action_render import action_template_key, action_to_prompt
from igc.core.types import ToolAction, ToolSpec
from igc.modules.llm.llm_encoder import TextEncoder

if TYPE_CHECKING:  # card is duck-typed through action_render; no runtime import needed
    from igc.core.tool_card import ToolCard


class ActionCodec:
    """Render + embed ToolActions into ``[N, H]`` keys, cached by action template.

    :param encoder: the shared text encoder (real or stub).
    :param specs: optional tool specs, so argument types come from each tool's declared
        ``arg_schema`` rather than being inferred from concrete values.
    """

    def __init__(self, encoder: TextEncoder, specs: Optional[Sequence[ToolSpec]] = None):
        self.encoder = encoder
        self._specs: Dict[str, ToolSpec] = {s.tool_name: s for s in (specs or [])}
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(
        self,
        actions: List[ToolAction],
        cards: Optional[Dict[Tuple[str, str], "ToolCard"]] = None,
    ) -> torch.Tensor:
        """Encode candidate actions to a ``[N, H]`` key matrix (in action order).

        Only cache-missing action *types* are passed to the encoder, so repeated or
        value-only-different candidates cost nothing.

        When ``cards`` is given (tool-teaching, ``docs/ARCHITECTURE.md`` §12.4 seam
        A), the :class:`~igc.core.tool_card.ToolCard` for a candidate's
        ``(tool_name, op)`` is folded into both its template key and its rendering, so
        a card-enriched candidate gets its own cache entry and embedding while every
        other candidate stays byte-identical to the cardless path.

        :param actions: the legal candidate actions for one state.
        :param cards: optional ``(tool_name, op) -> ToolCard`` map for the current env.
        :return: ``[N, H]`` keys (``[0, H]`` for an empty candidate set).
        """
        if not actions:
            return torch.empty(0, self.encoder.hidden_size)

        def _card_for(action: ToolAction):
            return cards.get((action.tool_name, action.op)) if cards else None

        keys = [
            action_template_key(a, self._specs.get(a.tool_name), _card_for(a))
            for a in actions
        ]

        # Collect the unique cache-misses (dedup so the encoder runs once per type).
        missing: Dict[str, str] = {}
        for action, key in zip(actions, keys):
            if key not in self._cache and key not in missing:
                missing[key] = action_to_prompt(
                    action, self._specs.get(action.tool_name), _card_for(action)
                )

        if missing:
            miss_keys = list(missing)
            embeddings = self.encoder.encode([missing[k] for k in miss_keys])
            for key, embedding in zip(miss_keys, embeddings):
                self._cache[key] = embedding

        return torch.stack([self._cache[key] for key in keys], dim=0)


# Author: Mus mbayramo@stanford.edu
