"""Ground a teacher-induced ToolCard against real evidence before it may up-rank.

The LLM teacher (``docs/ARCHITECTURE.md`` §12.3) *will* fabricate; nothing it
claims is trusted on assertion. :class:`ToolCardGrounder` runs the offline-checkable
gates of §12.5 over a :class:`~igc.core.tool_card.ToolCard`:

* **Gate 1 — evidence:** drop any error-taxonomy rule that cites no real
  :class:`~igc.core.types.Transition`.
* **Gate 2 — schema + enum:** drop ``effective_signature`` slots absent from the
  discovery-time :class:`~igc.core.types.ToolSpec` ``arg_schema[op]``, and clip any
  declared enum to the catalog's allowable values. **The catalog (``.npy`` /
  ``@Redfish.ActionInfo``, §9.3) overrides the teacher on any enum conflict.**
* **Gate 4 — online falsification:** fold each real observation into the card's
  :class:`~igc.core.tool_card.Grounding` tallies, promoting a confirmed card to
  ``GROUNDED`` and a contradicted one to ``CONTRADICTED`` (prior zeroed).

Gate 3 (M4 replay via :class:`~igc.core.protocols.Evaluator`) lands in a later
slice; this module stays pure stdlib so it runs in the offline CPU subset.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from igc.core.tool_card import ErrorRule, ToolCard
from igc.core.types import ToolSpec


def confirm_against_observation(
    card: ToolCard, status: int, body: object
) -> Optional[bool]:
    """Judge whether one real observation confirms or contradicts a card's claims.

    A *testable* prediction is one the card actually makes about this observation:

    * error (``status >= 400``): confirmed when a card error rule matches the
      observation (the card anticipated this failure); no judgement otherwise.
    * success (``status < 400``): confirmed when every declared
      ``expected_response`` field is present in a dict body; contradicted when a
      declared field is missing; no judgement when nothing is declared.

    :param card: the card whose predictions are tested.
    :param status: the observed normalized status.
    :param body: the observed body.
    :return: ``True`` (confirmed), ``False`` (contradicted), or ``None`` (no
        testable prediction — leave the tallies untouched).
    """
    if status >= 400:
        return True if card.classify_error(status, body) is not None else None
    if not card.expected_response:
        return None
    if not isinstance(body, dict):
        return None
    missing = [f for f in card.expected_response if f not in body]
    return len(missing) == 0


class ToolCardGrounder:
    """Apply the offline grounding gates to a teacher-induced ToolCard.

    :param require_evidence: when True (default), gate 1 drops error rules that
        cite no evidence id.
    """

    def __init__(self, require_evidence: bool = True):
        self.require_evidence = require_evidence

    def ground(
        self,
        card: ToolCard,
        spec: Optional[ToolSpec] = None,
        actioninfo: Optional[dict] = None,
        observed_fields: Optional[set] = None,
    ) -> ToolCard:
        """Return a cleaned copy of ``card`` with ungrounded claims removed.

        Applies gate 1 (evidence) and gate 2 (schema + enum). The returned card is
        still ``PROVISIONAL``: only real observations (via :meth:`observe`) or M4
        replay (a later slice) promote it to ``GROUNDED``. A card that is stale
        relative to ``spec`` (:meth:`ToolCard.is_stale`) is emptied and marked
        contradicted — its schema moved out from under it.

        :param card: the raw induced card.
        :param spec: the discovery-time tool spec; ``effective_signature`` slots not
            in ``spec.arg_schema[op]`` are dropped when given.
        :param actioninfo: optional ``slot -> [allowed values]`` from the catalog
            (``.npy``/ActionInfo); enum claims are clipped to it (catalog overrides).
        :param observed_fields: optional set of field names actually seen in a real
            body; ``expected_response`` is clipped to it when given.
        :return: the cleaned card (a new instance; the input is not mutated).
        """
        grounded = replace(
            card,
            effective_signature=dict(card.effective_signature),
            expected_response=dict(card.expected_response),
            error_taxonomy=list(card.error_taxonomy),
            preconditions=list(card.preconditions),
            usage_tips=list(card.usage_tips),
            provenance=dict(card.provenance),
            grounding=replace(card.grounding),
        )

        if spec is not None and grounded.is_stale(spec):
            grounded.effective_signature = {}
            grounded.expected_response = {}
            grounded.error_taxonomy = []
            from igc.core.tool_card import GroundingStatus

            grounded.grounding.status = GroundingStatus.CONTRADICTED
            return grounded

        # Gate 1 — evidence: uncited error rules cannot be trusted.
        if self.require_evidence:
            grounded.error_taxonomy = [
                r for r in grounded.error_taxonomy
                if (r.evidence_ids if isinstance(r, ErrorRule) else (r.get("evidence_ids")))
            ]

        # Gate 2 — schema: an arg slot the catalog never declared is a hallucination.
        if spec is not None:
            allowed_slots = set((spec.arg_schema or {}).get(grounded.op, {}).keys())
            grounded.effective_signature = {
                slot: frag
                for slot, frag in grounded.effective_signature.items()
                if slot in allowed_slots
            }

        # Gate 2 — enum: clip declared enums to the catalog's allowable values.
        if actioninfo:
            for slot, frag in grounded.effective_signature.items():
                if isinstance(frag, dict) and frag.get("enum") and slot in actioninfo:
                    allowed = set(actioninfo[slot])
                    frag["enum"] = [v for v in frag["enum"] if v in allowed]

        # Schema gate for response fields, when we know what a real body contained.
        if observed_fields is not None:
            grounded.expected_response = {
                f: t for f, t in grounded.expected_response.items() if f in observed_fields
            }

        return grounded

    def observe(self, card: ToolCard, status: int, body: object) -> ToolCard:
        """Fold one real observation into a card's grounding tallies (gate 4).

        Mutates and returns ``card``: a confirmed prediction promotes it toward
        ``GROUNDED``; a contradicted one toward ``CONTRADICTED`` (prior zeroed once
        contradictions dominate). A non-testable observation is a no-op.

        :param card: the card to update in place.
        :param status: the observed normalized status.
        :param body: the observed body.
        :return: the same card, with updated grounding.
        """
        verdict = confirm_against_observation(card, status, body)
        if verdict is not None:
            card.grounding.record(verdict)
        return card


# Author: Mus mbayramo@stanford.edu
