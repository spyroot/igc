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

from typing import Dict, List, Optional, Sequence

import torch

from igc.core.action_render import action_template_key, action_to_prompt
from igc.core.types import ToolAction, ToolSpec
from igc.modules.llm.llm_encoder import TextEncoder


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

    def encode(self, actions: List[ToolAction]) -> torch.Tensor:
        """Encode candidate actions to a ``[N, H]`` key matrix (in action order).

        Only cache-missing action *types* are passed to the encoder, so repeated or
        value-only-different candidates cost nothing.

        :param actions: the legal candidate actions for one state.
        :return: ``[N, H]`` keys (``[0, H]`` for an empty candidate set).
        """
        if not actions:
            return torch.empty(0, self.encoder.hidden_size)

        keys = [action_template_key(a, self._specs.get(a.tool_name)) for a in actions]

        # Collect the unique cache-misses (dedup so the encoder runs once per type).
        missing: Dict[str, str] = {}
        for action, key in zip(actions, keys):
            if key not in self._cache and key not in missing:
                missing[key] = action_to_prompt(action, self._specs.get(action.tool_name))

        if missing:
            miss_keys = list(missing)
            embeddings = self.encoder.encode([missing[k] for k in miss_keys])
            for key, embedding in zip(miss_keys, embeddings):
                self._cache[key] = embedding

        return torch.stack([self._cache[key] for key in keys], dim=0)


# Author: Mus mbayramo@stanford.edu
