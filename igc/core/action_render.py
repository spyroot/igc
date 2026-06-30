"""Canonical rendering of a ToolAction for candidate encoding and cache keys.

The pointer / candidate-scoring policy embeds each legal ToolAction (the ones
:meth:`igc.core.protocols.GoalEnvironment.available_actions` yields for the
current state) and scores it against the state. For that to be efficient and
stable, an action must render to a canonical, order-stable, *value-independent*
string: it captures the action TYPE — tool, op, target, and the argument SHAPE
(sorted ``key:type`` pairs) — but never concrete argument values, so two actions
that differ only in their argument values share one rendering, one cache key, and
one embedding. :func:`action_template_key` is the stable hash used as that key.

Pure standard library on purpose (no torch/numpy/HF): this is the deterministic
foundation the candidate encoder, the embedding cache, HER re-scoring, and the
``action_repr`` parity test all build on, so it must stay CPU/offline-testable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

from igc.core.types import ToolAction, ToolSpec

if TYPE_CHECKING:  # avoid a runtime import cycle; the card is duck-typed below
    from igc.core.tool_card import ToolCard


def _infer_type(value) -> str:
    """Map a Python argument value to a JSON-schema-style type name (shape, not value).

    :param value: an argument value.
    :return: a stable type name. ``bool`` is checked before ``int`` because ``bool``
        is a subclass of ``int``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _arg_types(action: ToolAction, spec: Optional[ToolSpec]) -> dict:
    """Derive the ``slot -> type`` map describing an action's argument shape.

    Prefers the declared ``spec.arg_schema[op]`` (so the signature is independent of
    which optional arguments a concrete action carries); otherwise infers the types
    from the action's own argument values.

    :param action: the action whose argument shape is described.
    :param spec: optional tool spec providing the declared argument schema.
    :return: a dict mapping argument slot name to a type name.
    """
    if spec is not None and action.op in spec.arg_schema:
        schema = spec.arg_schema[action.op] or {}
        return {
            slot: (frag.get("type", "string") if isinstance(frag, dict) else "string")
            for slot, frag in schema.items()
        }
    return {slot: _infer_type(value) for slot, value in (action.arguments or {}).items()}


def action_to_prompt(
    action: ToolAction,
    spec: Optional[ToolSpec] = None,
    card: Optional["ToolCard"] = None,
) -> str:
    """Render a ToolAction into a canonical, order-stable, value-independent string.

    The string encodes the action TYPE — tool, op, target, the argument shape
    (``key:type`` pairs sorted by key), and ``schema_id`` when set — and is the text
    a backbone embeds for candidate scoring. Concrete argument *values* are never
    included, so actions differing only in values render identically.

    When a ``card`` (a :class:`~igc.core.tool_card.ToolCard` for this op,
    ``docs/ARCHITECTURE.md`` §12.4 seam A) is supplied, its bounded
    :meth:`~igc.core.tool_card.ToolCard.render_clause` is appended so the backbone
    embeds a tool-aware candidate; this re-keys exactly that candidate (see
    :func:`action_template_key`). With ``card=None`` the rendering is byte-identical
    to the cardless form, so the passive path is unchanged.

    :param action: the action to render.
    :param spec: optional tool spec; when given and it declares a schema for the
        action's op, argument types come from ``spec.arg_schema[op]`` rather than
        being inferred from the values.
    :param card: optional grounded ToolCard whose clause refines the rendering.
    :return: the canonical rendering.
    """
    parts = [f"tool={action.tool_name}", f"op={action.op}"]
    if action.target is not None:
        parts.append(f"target={action.target}")
    arg_types = _arg_types(action, spec)
    if arg_types:
        body = ",".join(f"{slot}:{arg_types[slot]}" for slot in sorted(arg_types))
        parts.append(f"args=[{body}]")
    else:
        parts.append("args=[]")
    if action.schema_id:
        parts.append(f"schema={action.schema_id}")
    if card is not None:
        parts.append(card.render_clause())
    return " ".join(parts)


def action_template_key(
    action: ToolAction,
    spec: Optional[ToolSpec] = None,
    card: Optional["ToolCard"] = None,
) -> str:
    """Stable hash of :func:`action_to_prompt`, used as an embedding-cache key.

    Uses ``hashlib`` (not the salted built-in ``hash``) so the key is identical
    across processes and runs. Passing a ``card`` mixes its clause into the key, so a
    card-enriched candidate gets its own cache entry while every other candidate is
    untouched (``card=None`` reproduces the original key exactly).

    :param action: the action to key.
    :param spec: optional tool spec, forwarded to :func:`action_to_prompt`.
    :param card: optional grounded ToolCard, forwarded to :func:`action_to_prompt`.
    :return: a hex digest uniquely identifying the (optionally card-enriched) action TYPE.
    """
    rendered = action_to_prompt(action, spec, card)
    return hashlib.blake2b(rendered.encode("utf-8"), digest_size=16).hexdigest()


# Author: Mus mbayramo@stanford.edu
