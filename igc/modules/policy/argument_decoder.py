"""Second-stage argument decoder for mutating tool actions.

Stage 1 (the pointer policy in :mod:`igc.modules.policy.pointer_policy`) scores
value-independent action *templates*: by design
:func:`igc.core.action_render.action_to_prompt` drops argument values, so a read
action (``GET /redfish/v1/Systems/1``) is fully specified by its template, but a
*mutating* action is not — e.g. ``POST .../Actions/ComputerSystem.Reset`` needs a
``ResetType`` value and ``PATCH .../Bios/Settings`` needs ``Attributes``. This module
is stage 2: given the selected template's :class:`~igc.core.types.ToolSpec` and op
plus the encoded state, it fills the argument *values*.

No-explosion property (same principle as stage 1): a categorical argument is chosen
by scoring its OWN candidate values — output width equals the number of choices for
that one slot (read from the Redfish ``@Redfish.ActionInfo`` allowable values, carried
in ``spec.arg_schema[op][slot]["enum"]``) — and slots are decoded independently, so
the head never enumerates the cross-product of all slot values. Free-form slots
(no enum) are left as a documented stub for a generative value head.

The pure functions (slot enumeration, selection, assembly) are standard-library only
and CPU/offline-testable; only :class:`CategoricalArgScorer` imports torch.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from igc.core.types import ToolAction, ToolSpec


@dataclass(frozen=True)
class ArgumentSlot:
    """One argument to fill for a chosen action template.

    :param name: argument key (e.g. ``"ResetType"``).
    :param type: JSON-schema-style type name (``string``/``integer``/...).
    :param choices: the allowable categorical values (the enum), or ``None`` for a
        free-form slot that a generative head must produce.
    :param required: whether the action is invalid without this argument.
    """

    name: str
    type: str = "string"
    choices: Optional[tuple] = None
    required: bool = False

    @property
    def is_categorical(self) -> bool:
        """True when the slot has a finite candidate set to score over."""
        return bool(self.choices)


def arg_slots_for(spec: ToolSpec, op: str) -> list:
    """Enumerate the argument slots an op declares in a tool's schema.

    Reads ``spec.arg_schema[op]`` (the per-op argument schema): each entry's
    ``type``, its ``enum``/``choices`` (mapped to :attr:`ArgumentSlot.choices`), and
    its ``required`` flag. A non-dict entry degrades to a free-form string slot.

    :param spec: the tool spec for the selected template.
    :param op: the operation/verb being decoded.
    :return: the ordered list of :class:`ArgumentSlot` to fill (empty for a read op
        with no declared arguments).
    """
    schema = (spec.arg_schema or {}).get(op, {}) or {}
    slots = []
    for name, frag in schema.items():
        if not isinstance(frag, dict):
            slots.append(ArgumentSlot(name=name))
            continue
        choices = frag.get("enum") or frag.get("choices")
        slots.append(
            ArgumentSlot(
                name=name,
                type=frag.get("type", "string"),
                choices=tuple(choices) if choices else None,
                required=bool(frag.get("required", False)),
            )
        )
    return slots


def select_categorical(slot: ArgumentSlot, scores) -> Any:
    """Pick a categorical slot's value as the argmax over its choice scores.

    :param slot: a categorical slot (``slot.is_categorical`` must hold).
    :param scores: a 1-D score per choice, length ``len(slot.choices)``; the width is
        the slot's choice count, never a global vocabulary.
    :return: the chosen value from ``slot.choices``.
    :raises ValueError: if the slot is free-form or the score width is wrong.
    """
    if not slot.is_categorical:
        raise ValueError(f"slot {slot.name!r} is free-form; categorical selection needs choices")
    values = list(scores)
    if len(values) != len(slot.choices):
        raise ValueError(
            f"score width {len(values)} != {len(slot.choices)} choices for slot {slot.name!r}"
        )
    best = max(range(len(values)), key=lambda i: values[i])
    return slot.choices[best]


def assemble_arguments(slots: list, chosen: dict) -> dict:
    """Build the ``arguments`` dict from per-slot chosen values, enforcing required.

    :param slots: the slots from :func:`arg_slots_for`.
    :param chosen: ``slot name -> value`` (a missing or ``None`` value is treated as
        "not provided").
    :return: the arguments dict for a :class:`~igc.core.types.ToolAction`.
    :raises ValueError: if a required slot has no value.
    """
    args = {}
    for slot in slots:
        value = chosen.get(slot.name)
        if value is not None:
            args[slot.name] = value
        elif slot.required:
            raise ValueError(f"required argument {slot.name!r} not provided")
    return args


def apply_arguments(template: ToolAction, arguments: dict) -> ToolAction:
    """Return the concrete action: the chosen template with its argument values set.

    The template carries the value-independent fields the pointer stage selected
    (tool/op/target/risk); stage 2 only fills ``arguments``.

    :param template: the action template selected by the pointer policy.
    :param arguments: the values from :func:`assemble_arguments`.
    :return: a new :class:`~igc.core.types.ToolAction` with ``arguments`` applied.
    """
    return replace(template, arguments=dict(arguments))


# torch is only needed for the learned scorer; keep the import local so the pure
# slot/selection logic above stays import-light and CPU/offline-testable.
try:  # pragma: no cover - exercised only where torch is installed
    import torch
    import torch.nn as nn

    from igc.modules.policy.pointer_policy import score_candidates

    class CategoricalArgScorer(nn.Module):
        """Score a categorical slot's candidate values against the state query.

        Reuses the pointer policy's dot-product scoring: the output width is the
        slot's choice count ``K`` (not a global value vocabulary), so adding an
        allowable value never widens any other slot's head.

        :param h_dim: backbone hidden size ``H`` (state/value embedding width).
        :param q_dim: query/key dimension ``d``.
        """

        def __init__(self, h_dim: int, q_dim: int = 256):
            super().__init__()
            self.query = nn.Sequential(nn.Linear(h_dim, q_dim))
            self.value_proj = nn.Sequential(nn.Linear(h_dim, q_dim))

        def forward(
            self,
            state_h: "torch.Tensor",
            value_h: "torch.Tensor",
            mask: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            """:param state_h: ``[B, H]`` pooled state (optionally state+goal upstream).
            :param value_h: ``[B, K, H]`` embeddings of the slot's candidate values.
            :param mask: optional ``[B, K]`` legality mask (1 real, 0 padding).
            :return: ``[B, K]`` scores; width tracks ``K``, the slot's choice count.
            """
            q = torch.nn.functional.normalize(self.query(state_h), dim=-1)
            k = torch.nn.functional.normalize(self.value_proj(value_h), dim=-1)
            return score_candidates(q, k, mask)

except ImportError:  # pragma: no cover - torch absent (pure-logic install)
    CategoricalArgScorer = None


# Author: Mus mbayramo@stanford.edu
