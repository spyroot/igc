"""The ToolCard — an evidence-grounded, versioned spec of one tool/op.

A ``ToolCard`` is what the tool-teaching layer (``docs/ARCHITECTURE.md`` §12)
induces for a tool the agent does not yet know: from a few real ``(call ->
result/error)`` interactions, the gated LLM teacher distills the op's effective
argument signature, the response fields to expect, an error taxonomy (what each
status *means* — retriable / fatal / precondition-unmet), and preconditions. The
card is a *refinement* of the
discovery-time :class:`~igc.core.types.ToolSpec`, never a replacement, and it is
only ever advisory to scoring: per the safety invariant it may RAISE caution but
can never lower a :class:`~igc.core.types.RiskLevel` or unlock a destructive op.

The card flows into the agent through the existing candidate-rendering seam:
:func:`igc.core.action_render.action_to_prompt` appends :meth:`ToolCard.render_clause`
so the pointer policy embeds a tool-aware candidate (no head resize), and
:meth:`ToolCard.content_hash` re-keys that one candidate's embedding cache when the
learned content changes.

Pure standard library on purpose (no torch/numpy/transformers), mirroring
:mod:`igc.core.types`: this is imported across the teach + policy layers and must
stay cheap and CPU/offline-testable. Teacher claims are NOT trusted on assertion —
:mod:`igc.modules.teach.grounding` gates every field before a card may up-rank.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from igc.core.types import ToolAction, ToolSpec


class GroundingStatus(str, Enum):
    """How much of a card has survived grounding (:mod:`igc.modules.teach.grounding`).

    ``PROVISIONAL`` may only widen exploration; ``GROUNDED`` (passed the schema /
    evidence / verifier gates) may up-rank a candidate; ``CONTRADICTED`` has been
    falsified by real observations and its prior is zeroed.
    """

    PROVISIONAL = "provisional"
    GROUNDED = "grounded"
    CONTRADICTED = "contradicted"


class ErrorClass(str, Enum):
    """What an observed error status means for *this* op, so a failed call teaches.

    ``RETRIABLE`` — transient, retry as-is; ``FATAL`` — the call is wrong, do not
    retry blindly; ``PRECONDITION_UNMET`` — a prerequisite (a pending job, a power
    state) must be satisfied first.
    """

    RETRIABLE = "retriable"
    FATAL = "fatal"
    PRECONDITION_UNMET = "precondition_unmet"


@dataclass
class ErrorRule:
    """One entry in a card's error taxonomy: a ``match`` predicate over an observed
    result and the :class:`ErrorClass` plus guidance it implies.

    ``match`` keys are any of ``status`` (exact int), ``code`` (a backend error code
    string found in the body) and ``body_substring`` (a substring of the serialized
    body). Every rule must cite ``evidence_ids`` (ids of real
    :class:`~igc.core.types.Transition` records) — the evidence gate drops uncited
    rules, so a hallucinated error class cannot survive.
    """

    match: dict = field(default_factory=dict)
    error_class: ErrorClass = ErrorClass.FATAL
    meaning: str = ""
    suggested_fix: str = ""
    evidence_ids: list = field(default_factory=list)

    def matches(self, status: Optional[int], body: object) -> bool:
        """True when an observed ``(status, body)`` satisfies this rule's ``match``.

        :param status: the observed normalized status (HTTP code or equivalent).
        :param body: the observed body; stringified for the substring/code checks.
        :return: whether every present ``match`` key is satisfied.
        """
        if not self.match:
            return False
        if "status" in self.match and self.match["status"] != status:
            return False
        body_text = body if isinstance(body, str) else json.dumps(body, sort_keys=True, default=str)
        if "code" in self.match and str(self.match["code"]) not in body_text:
            return False
        if "body_substring" in self.match and str(self.match["body_substring"]) not in body_text:
            return False
        return True

    def to_dict(self) -> dict:
        """Serialize to a plain dict (``error_class`` by its string value)."""
        return {
            "match": dict(self.match),
            "error_class": self.error_class.value,
            "meaning": self.meaning,
            "suggested_fix": self.suggested_fix,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorRule":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            match=dict(d.get("match") or {}),
            error_class=ErrorClass(d.get("error_class", ErrorClass.FATAL.value)),
            meaning=d.get("meaning", ""),
            suggested_fix=d.get("suggested_fix", ""),
            evidence_ids=list(d.get("evidence_ids") or []),
        )


@dataclass
class Grounding:
    """The trust state of a card: its :class:`GroundingStatus` plus the running
    tallies the online falsifier updates from real ``StepResult`` outcomes.

    A card flips ``PROVISIONAL -> GROUNDED`` once the grounder confirms it and
    ``-> CONTRADICTED`` once contradictions outweigh confirmations, at which point
    the injection layer must drop its prior.
    """

    status: GroundingStatus = GroundingStatus.PROVISIONAL
    n_confirmations: int = 0
    n_contradictions: int = 0
    m4_checks_passed: int = 0
    m4_checks_failed: int = 0

    def record(self, confirmed: bool) -> None:
        """Fold one real observation into the tallies and recompute the status.

        Contradictions dominate: once they exceed confirmations the card is
        ``CONTRADICTED``; otherwise any confirmation lifts it to ``GROUNDED``.

        :param confirmed: whether the observation agreed with the card.
        """
        if confirmed:
            self.n_confirmations += 1
        else:
            self.n_contradictions += 1
        if self.n_contradictions > self.n_confirmations:
            self.status = GroundingStatus.CONTRADICTED
        elif self.n_confirmations > 0:
            self.status = GroundingStatus.GROUNDED

    def to_dict(self) -> dict:
        """Serialize to a plain dict (``status`` by its string value)."""
        return {
            "status": self.status.value,
            "n_confirmations": self.n_confirmations,
            "n_contradictions": self.n_contradictions,
            "m4_checks_passed": self.m4_checks_passed,
            "m4_checks_failed": self.m4_checks_failed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grounding":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            status=GroundingStatus(d.get("status", GroundingStatus.PROVISIONAL.value)),
            n_confirmations=int(d.get("n_confirmations", 0)),
            n_contradictions=int(d.get("n_contradictions", 0)),
            m4_checks_passed=int(d.get("m4_checks_passed", 0)),
            m4_checks_failed=int(d.get("m4_checks_failed", 0)),
        )


# Upper bound on the card clause appended to a candidate rendering, so a learned
# card cannot blow the encoder's token budget for one candidate.
_CLAUSE_MAX_LEN = 240


@dataclass
class ToolCard:
    """An evidence-grounded, versioned spec of one ``(env_name, tool_name, op)``.

    Refines the discovery-time :class:`~igc.core.types.ToolSpec` with what the
    teacher induced and the grounder confirmed: the ``effective_signature``
    (``slot -> {type, enum?, required, ...}``, enums bounded by the catalog), the
    ``expected_response`` field types, the ``error_taxonomy``, ``preconditions``,
    and ``usage_tips``. ``spec_fingerprint`` pins the schema the card was induced
    against so a moved op schema auto-invalidates the card (:meth:`is_stale`).
    """

    env_name: str
    tool_name: str
    op: str
    spec_fingerprint: str = ""
    effective_signature: dict = field(default_factory=dict)
    expected_response: dict = field(default_factory=dict)
    error_taxonomy: list = field(default_factory=list)
    preconditions: list = field(default_factory=list)
    usage_tips: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    grounding: Grounding = field(default_factory=Grounding)
    version: int = 1

    @property
    def key(self) -> tuple:
        """The store key: ``(env_name, tool_name, op)``."""
        return (self.env_name, self.tool_name, self.op)

    def _semantic_body(self) -> dict:
        """The canonical, hashable view of the card's *learned content* only.

        Excludes the volatile trust state (``grounding``), ``version``, and
        ``provenance`` so :meth:`content_hash` (an embedding-cache key) changes when
        the agent's knowledge changes, not when a counter ticks.
        """
        return {
            "tool_name": self.tool_name,
            "op": self.op,
            "effective_signature": self.effective_signature,
            "expected_response": self.expected_response,
            "error_taxonomy": [r.to_dict() if isinstance(r, ErrorRule) else r for r in self.error_taxonomy],
            "preconditions": sorted(self.preconditions),
            "usage_tips": sorted(self.usage_tips),
        }

    def content_hash(self) -> str:
        """Stable hex digest of the learned content (``hashlib``, cross-process).

        Used to re-key exactly this candidate's embedding when the card's content
        changes; two cards with identical content hash equally.
        """
        canonical = json.dumps(self._semantic_body(), sort_keys=True, default=str)
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

    def render_clause(self, max_len: int = _CLAUSE_MAX_LEN) -> str:
        """A bounded, deterministic, value-independent ``card=[...]`` clause.

        Encodes the card's *shape* — required slots (with enum membership when
        declared), the status->class error map, and the expected response field
        names — so a backbone embeds a more tool-aware candidate. Never includes a
        concrete argument *value*, matching the value-independence of
        :func:`igc.core.action_render.action_to_prompt`. Truncated to ``max_len``.

        :param max_len: hard cap on the clause length (token-budget guard).
        :return: the ``card=[...]`` clause, or ``"card=[]"`` when empty.
        """
        segments = []
        if self.effective_signature:
            slots = []
            for slot in sorted(self.effective_signature):
                frag = self.effective_signature[slot] or {}
                typ = frag.get("type", "string") if isinstance(frag, dict) else "string"
                tag = slot if not (isinstance(frag, dict) and frag.get("required")) else f"{slot}!"
                if isinstance(frag, dict) and frag.get("enum"):
                    tag = f"{tag}:enum"
                else:
                    tag = f"{tag}:{typ}"
                slots.append(tag)
            segments.append("sig:" + ",".join(slots))
        if self.error_taxonomy:
            errs = []
            for rule in self.error_taxonomy:
                r = rule if isinstance(rule, ErrorRule) else ErrorRule.from_dict(rule)
                status = r.match.get("status")
                if status is not None:
                    errs.append(f"{status}={r.error_class.value}")
            if errs:
                segments.append("err:" + ",".join(sorted(set(errs))))
        if self.expected_response:
            segments.append("exp:" + ",".join(sorted(self.expected_response)))
        clause = "card=[" + ";".join(segments) + "]"
        if len(clause) > max_len:
            clause = clause[: max_len - 1] + "]"
        return clause

    def classify_error(self, status: Optional[int], body: object) -> Optional[ErrorClass]:
        """Classify an observed error via the first matching :class:`ErrorRule`.

        :param status: observed normalized status.
        :param body: observed body.
        :return: the matched :class:`ErrorClass`, or ``None`` if no rule matches.
        """
        for rule in self.error_taxonomy:
            r = rule if isinstance(rule, ErrorRule) else ErrorRule.from_dict(rule)
            if r.matches(status, body):
                return r.error_class
        return None

    def is_stale(self, spec: ToolSpec) -> bool:
        """True when ``spec``'s op schema no longer matches the card's fingerprint.

        A stale card was induced against a different schema and must not be trusted
        (the spec-fingerprint auto-invalidation gate).

        :param spec: the current tool spec to check against.
        :return: whether the card is stale relative to ``spec``.
        """
        if not self.spec_fingerprint:
            return False
        return self.spec_fingerprint != self.compute_spec_fingerprint(spec, self.op)

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-safe dict (``from_dict(c.to_dict()) == c``)."""
        return {
            "env_name": self.env_name,
            "tool_name": self.tool_name,
            "op": self.op,
            "spec_fingerprint": self.spec_fingerprint,
            "effective_signature": self.effective_signature,
            "expected_response": self.expected_response,
            "error_taxonomy": [r.to_dict() if isinstance(r, ErrorRule) else dict(r) for r in self.error_taxonomy],
            "preconditions": list(self.preconditions),
            "usage_tips": list(self.usage_tips),
            "provenance": dict(self.provenance),
            "grounding": self.grounding.to_dict(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCard":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            env_name=d["env_name"],
            tool_name=d["tool_name"],
            op=d["op"],
            spec_fingerprint=d.get("spec_fingerprint", ""),
            effective_signature=dict(d.get("effective_signature") or {}),
            expected_response=dict(d.get("expected_response") or {}),
            error_taxonomy=[ErrorRule.from_dict(r) for r in (d.get("error_taxonomy") or [])],
            preconditions=list(d.get("preconditions") or []),
            usage_tips=list(d.get("usage_tips") or []),
            provenance=dict(d.get("provenance") or {}),
            grounding=Grounding.from_dict(d.get("grounding") or {}),
            version=int(d.get("version", 1)),
        )

    @staticmethod
    def compute_spec_fingerprint(spec: ToolSpec, op: str) -> str:
        """Hex fingerprint of ``spec.arg_schema[op]`` for stale-card detection.

        :param spec: the tool spec the card is (or was) induced against.
        :param op: the op whose argument schema is fingerprinted.
        :return: a stable hex digest of the op's declared argument schema.
        """
        schema = (spec.arg_schema or {}).get(op, {})
        canonical = json.dumps(schema, sort_keys=True, default=str)
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


class ToolCardStore:
    """A per-trial namespace of cards keyed ``(env_name, tool_name, op)``.

    One store lives per adaptation trial so a held-out-API trial's induced cards
    never leak across trials (which would invalidate the few-shot claim). It is a
    thin dict wrapper with action lookup and JSON round-trip.
    """

    def __init__(self, trial_id: str = ""):
        """:param trial_id: an opaque per-trial tag (kept out of card content)."""
        self.trial_id = trial_id
        self._cards: dict = {}

    def put(self, card: ToolCard) -> None:
        """Insert or replace a card by its :attr:`ToolCard.key`."""
        self._cards[card.key] = card

    def get(self, env_name: str, tool_name: str, op: str) -> Optional[ToolCard]:
        """Return the card for ``(env_name, tool_name, op)`` or ``None``."""
        return self._cards.get((env_name, tool_name, op))

    def for_action(self, action: ToolAction, env_name: str) -> Optional[ToolCard]:
        """Return the card matching an action in ``env_name`` (by tool + op), or ``None``.

        :param action: the candidate action.
        :param env_name: the environment the action belongs to.
        :return: the matching card or ``None``.
        """
        return self._cards.get((env_name, action.tool_name, action.op))

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, key) -> bool:
        return key in self._cards

    def to_dict(self) -> dict:
        """Serialize the whole store (``trial_id`` + every card)."""
        return {
            "trial_id": self.trial_id,
            "cards": [c.to_dict() for c in self._cards.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCardStore":
        """Reconstruct from :meth:`to_dict` output."""
        store = cls(trial_id=d.get("trial_id", ""))
        for cd in d.get("cards") or []:
            store.put(ToolCard.from_dict(cd))
        return store


# Author: Mus mbayramo@stanford.edu
