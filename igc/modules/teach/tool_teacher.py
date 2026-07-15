"""Induce a ToolCard for an unknown tool from a few observed interactions.

The teacher (``docs/ARCHITECTURE.md`` §12.3) turns the agent's own ``k`` real
``(call -> result/error)`` :class:`~igc.core.types.Transition` records into a
:class:`~igc.core.tool_card.ToolCard`. Two implementations share the
:class:`ToolTeacher` interface:

* :class:`StubTeacher` — a deterministic, offline, **no-LLM** teacher that reads the
  observed transitions directly (success bodies -> expected fields, error statuses ->
  an error taxonomy, the spec -> the signature). It is the offline test double *and*
  the honest rule-based baseline the curve eval (§12.7) measures the LLM teacher
  against; it never reaches the network.
* A live LLM teacher (the gated bootstrap path from §11.4) is the
  follow-on — it shares this interface so the offline plumbing is identical.

:func:`teach_tool` is the induce -> ground -> confirm -> store pipeline used by both.
Pure stdlib in this slice (no torch/network); the live teacher's HTTP client is the
only piece behind a download/network gate.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from igc.core.tool_card import ErrorClass, ErrorRule, ToolCard
from igc.core.types import ToolSpec, Transition
from igc.modules.teach.grounding import ToolCardGrounder


def _json_type(value) -> str:
    """Map a Python value to a JSON-schema-style type name (shape, not value)."""
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


def _error_class_for(status: int) -> ErrorClass:
    """Heuristic class for an observed error status (the rule-based teacher's prior).

    ``409`` (conflict) and ``412`` (precondition failed) read as an unmet
    precondition; ``429`` and ``5xx`` as transient/retriable; other ``4xx`` as a
    fatal client error (the call itself is wrong).
    """
    if status in (409, 412):
        return ErrorClass.PRECONDITION_UNMET
    if status == 429 or status >= 500:
        return ErrorClass.RETRIABLE
    return ErrorClass.FATAL


@runtime_checkable
class ToolTeacher(Protocol):
    """Induces a ToolCard for one ``op`` from observed transitions."""

    def induce(
        self,
        env_name: str,
        spec: ToolSpec,
        op: str,
        transitions: List[Transition],
        actioninfo: Optional[dict] = None,
    ) -> ToolCard: ...


class StubTeacher:
    """A deterministic, offline ToolCard inducer — the no-LLM baseline + test double.

    Reads the observed interactions for ``(spec.tool_name, op)`` directly: the
    discovery ``spec`` (plus ``actioninfo`` enums) gives the signature, success
    bodies give the expected-response fields, and each error status becomes an
    evidence-cited :class:`~igc.core.tool_card.ErrorRule`. No network, no model — so
    every claim it makes is already backed by a real transition.
    """

    source = "stub"

    def induce(
        self,
        env_name: str,
        spec: ToolSpec,
        op: str,
        transitions: List[Transition],
        actioninfo: Optional[dict] = None,
    ) -> ToolCard:
        """Induce a ToolCard for ``(spec.tool_name, op)`` from ``transitions``.

        :param env_name: the environment the tool belongs to.
        :param spec: the discovery-time tool spec (signature source).
        :param op: the op being learned.
        :param transitions: observed transitions; only those whose action matches
            ``(spec.tool_name, op)`` are read, and their index is the evidence id.
        :param actioninfo: optional ``slot -> [allowed values]`` enum source.
        :return: a PROVISIONAL card citing the observed evidence (ungrounded until
            :func:`teach_tool` confirms it).
        """
        relevant = [
            (i, tr)
            for i, tr in enumerate(transitions)
            if tr.action.tool_name == spec.tool_name and tr.action.op == op
        ]

        # Signature: the declared arg schema, with enums pulled from ActionInfo.
        signature = {}
        op_schema = (spec.arg_schema or {}).get(op, {}) or {}
        for slot, frag in op_schema.items():
            entry = {"type": frag.get("type", "string") if isinstance(frag, dict) else "string"}
            if isinstance(frag, dict) and frag.get("required"):
                entry["required"] = True
            if actioninfo and slot in actioninfo:
                entry["enum"] = list(actioninfo[slot])
            elif isinstance(frag, dict) and frag.get("enum"):
                entry["enum"] = list(frag["enum"])
            signature[slot] = entry

        # Expected response: top-level fields seen in any successful (2xx) body.
        expected = {}
        error_rules = {}
        evidence_ids = []
        for idx, tr in relevant:
            eid = f"t{idx}"
            evidence_ids.append(eid)
            status = tr.next_observation.status
            body = tr.next_observation.structured
            if status < 400:
                if isinstance(body, dict):
                    for field_name, value in body.items():
                        expected.setdefault(field_name, _json_type(value))
            else:
                # Dedup error rules by status; first occurrence wins, all cite evidence.
                if status not in error_rules:
                    error_rules[status] = ErrorRule(
                        match={"status": status},
                        error_class=_error_class_for(status),
                        evidence_ids=[eid],
                    )
                elif eid not in error_rules[status].evidence_ids:
                    error_rules[status].evidence_ids.append(eid)

        return ToolCard(
            env_name=env_name,
            tool_name=spec.tool_name,
            op=op,
            spec_fingerprint=ToolCard.compute_spec_fingerprint(spec, op),
            effective_signature=signature,
            expected_response=expected,
            error_taxonomy=[error_rules[s] for s in sorted(error_rules)],
            provenance={"source": self.source, "evidence_ids": evidence_ids, "k_observed": len(relevant)},
        )


def teach_tool(
    teacher: ToolTeacher,
    grounder: ToolCardGrounder,
    store,
    env_name: str,
    spec: ToolSpec,
    op: str,
    transitions: List[Transition],
    actioninfo: Optional[dict] = None,
) -> ToolCard:
    """Induce, ground, confirm against evidence, and store a card for one op.

    The pipeline of §12.3-12.5: the teacher induces a raw card, the grounder drops
    ungrounded claims (evidence/schema/enum gates), each observed transition for this
    op is folded in as online confirmation, and the result is stored per-trial.

    :param teacher: a :class:`ToolTeacher` (stub or live).
    :param grounder: the :class:`~igc.modules.teach.grounding.ToolCardGrounder`.
    :param store: a :class:`~igc.core.tool_card.ToolCardStore` to put the card in.
    :param env_name: the environment name.
    :param spec: the discovery-time tool spec.
    :param op: the op being learned.
    :param transitions: the observed transitions (evidence).
    :param actioninfo: optional enum source for the schema/enum gate.
    :return: the grounded, stored card.
    """
    raw = teacher.induce(env_name, spec, op, transitions, actioninfo)
    card = grounder.ground(raw, spec, actioninfo=actioninfo)
    for tr in transitions:
        if tr.action.tool_name == spec.tool_name and tr.action.op == op:
            grounder.observe(card, tr.next_observation.status, tr.next_observation.structured)
    store.put(card)
    return card


# Author: Mus mbayramo@stanford.edu
