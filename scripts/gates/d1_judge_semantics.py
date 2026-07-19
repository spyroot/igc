"""Gate: d1.judge_semantics — the locked D1 structured-judge acceptance formula.

Phase 2 D1 rows are ``text -> rest_api_list`` (an *unordered unique set*): model_x
drafts operator ``text`` for an API set that was **selected before generation**, and
a private Pro judge decides whether that text maps back to *exactly* that set. Only
rows the judge accepts under the locked contract may enter the accepted D1 dataset.

This gate encodes that contract as pure, offline logic so the acceptance decision is
deterministic, auditable, and separated from any model/network:

* ``parse_judge_result`` — parse and strictly validate a **structured** judge result
  (``accepted``, ``natural``, ``nonsense``, ``ambiguous``, ``duplicate_intent``,
  ``method_semantics_valid``, ``reason`` plus ``coverage`` items carrying
  ``rest_api``/``text_span``/``supported`` and an ``extra_intents`` list). Malformed
  or mistyped judge output raises ``D1JudgeResultError`` — a judge that cannot be
  parsed must never silently accept a row.
* ``decide_accept`` — the EXACT acceptance formula: a row is accepted only when the
  supported-coverage API set equals the selected set AND every guard
  (accepted/natural/method_semantics_valid true; nonsense/ambiguous/duplicate_intent/
  extra_intents absent) passes. It returns the specific failing reason codes so a
  reject is explainable, never a bare boolean.
* ``target_set_matches`` — the D1 *target* eval: a prediction matches only when it is
  the same set AND carries no duplicates.

Separation of concerns (locked): the generator and judge may see APIs, JSON bodies,
methods, descriptions, and examples; the Phase 2 *training* model sees only
``x.text``; and the D1 label is the API set chosen before generation. This module
touches none of the model-facing surface — it only judges the (structured verdict,
selected set) pair.

Offline (CI): ``run`` scores a hand-labelled suite (``load_suite``) of
``(judge_result, selected_rest_api_list, expected_accepted)`` cases against
``decide_accept`` so a regression in the acceptance formula is caught deterministically
with no model or network. The suite is operator-supplied (``--suite``); the real
model_x + Pro-judge run is BLOCKED while the Brain/GB300 surface is unavailable.

Used by the Phase 2 D1 acceptance path: whatever admits accepted D1 rows must call
``decide_accept`` on the structured judge verdict, and the offline gate CLI/``run``
guards the formula in CI. If this logic drifts, a lenient judge silently poisons the
labelled dataset, so the reasons list and the zero-false-accept suite are the tripwire.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class D1JudgeResultError(ValueError):
    """Raised when a structured judge result is missing fields or is mistyped.

    A judge result that cannot be parsed into the locked shape must never be treated
    as an accept; callers convert this into a reject rather than admitting the row.
    """


# Boolean judge fields that must be present and of type ``bool`` in every result.
_REQUIRED_BOOL_FIELDS = (
    "accepted",              # judge's top-level accept/reject decision.
    "natural",               # text reads as a natural operator request.
    "nonsense",              # text is junk / not an operator request.
    "ambiguous",             # text is too vague to map to a concrete API set.
    "duplicate_intent",      # text repeats the same intent / API.
    "method_semantics_valid",  # text's implied HTTP method matches allowed methods.
)

# Reason codes returned by ``decide_accept`` for each way a row can fail. Stable
# tokens so callers/tests can assert on the exact failing condition.
REASON_NOT_ACCEPTED = "not_accepted"                    # verdict.accepted is False.
REASON_NOT_NATURAL = "not_natural"                      # verdict.natural is False.
REASON_NONSENSE = "nonsense"                            # verdict.nonsense is True.
REASON_AMBIGUOUS = "ambiguous"                          # verdict.ambiguous is True.
REASON_DUPLICATE_INTENT = "duplicate_intent"            # verdict.duplicate_intent is True.
REASON_METHOD_SEMANTICS_INVALID = "method_semantics_invalid"  # method_semantics_valid False.
REASON_EXTRA_INTENTS = "extra_intents"                  # verdict.extra_intents non-empty.
REASON_COVERAGE_MISMATCH = "coverage_mismatch"          # supported coverage != selected set.


@dataclass(frozen=True)
class CoverageItem:
    """One judge coverage claim: a selected API grounded in a span of the text.

    :param rest_api: the concrete Redfish API the judge claims the text covers.
    :param text_span: the substring of the operator text that supports the API.
    :param supported: whether the judge actually found support for the API in text.
    """

    rest_api: str      # concrete Redfish API the coverage item is about.
    text_span: str     # span of operator text the judge cites as support.
    supported: bool    # True only if the text genuinely supports this API.


@dataclass(frozen=True)
class JudgeVerdict:
    """Parsed, validated structured judge result for one D1 draft.

    Field meanings mirror the locked D1 judge contract; only ``supported`` coverage
    items count toward the covered API set in ``decide_accept``.

    :param accepted: judge's top-level accept flag.
    :param natural: text reads as a natural operator request.
    :param nonsense: text is junk / not an operator request.
    :param ambiguous: text is too vague to map to a concrete API set.
    :param duplicate_intent: text repeats the same intent / API.
    :param method_semantics_valid: implied HTTP method matches allowed methods.
    :param coverage: per-API grounding claims (rest_api/text_span/supported).
    :param extra_intents: intents the text expresses beyond the selected API set.
    :param reason: short non-secret judge reason string.
    """

    accepted: bool                          # judge's top-level accept decision.
    natural: bool                           # text is a natural operator request.
    nonsense: bool                          # text is junk / not a request.
    ambiguous: bool                         # text too vague for a concrete set.
    duplicate_intent: bool                  # text repeats an intent / API.
    method_semantics_valid: bool            # implied method matches allowed methods.
    coverage: tuple[CoverageItem, ...]      # per-API grounding claims.
    extra_intents: tuple[str, ...]          # intents beyond the selected set.
    reason: str = ""                        # short non-secret judge reason.

    @property
    def covered_api_set(self) -> set[str]:
        """API set the judge actually supports (only ``supported`` coverage items)."""
        return {item.rest_api for item in self.coverage if item.supported}


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    """Return ``mapping[key]`` requiring it to be present and a real ``bool``."""
    if key not in mapping:
        raise D1JudgeResultError(f"judge result missing required field: {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise D1JudgeResultError(f"judge field {key!r} must be a bool, got {type(value).__name__}")
    return value


def _parse_coverage(raw: Any) -> tuple[CoverageItem, ...]:
    """Validate and convert the ``coverage`` list into ``CoverageItem`` objects."""
    if not isinstance(raw, list):
        raise D1JudgeResultError("judge field 'coverage' must be a list")
    items: list[CoverageItem] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise D1JudgeResultError(f"coverage[{index}] must be a mapping")
        for field_name in ("rest_api", "text_span", "supported"):
            if field_name not in entry:
                raise D1JudgeResultError(f"coverage[{index}] missing field: {field_name}")
        rest_api = entry["rest_api"]
        text_span = entry["text_span"]
        supported = entry["supported"]
        if not isinstance(rest_api, str):
            raise D1JudgeResultError(f"coverage[{index}].rest_api must be a str")
        if not isinstance(text_span, str):
            raise D1JudgeResultError(f"coverage[{index}].text_span must be a str")
        if not isinstance(supported, bool):
            raise D1JudgeResultError(f"coverage[{index}].supported must be a bool")
        items.append(CoverageItem(rest_api=rest_api, text_span=text_span, supported=supported))
    return tuple(items)


def _parse_extra_intents(raw: Any) -> tuple[str, ...]:
    """Validate and convert the ``extra_intents`` list into a tuple of strings."""
    if not isinstance(raw, list):
        raise D1JudgeResultError("judge field 'extra_intents' must be a list")
    if not all(isinstance(item, str) for item in raw):
        raise D1JudgeResultError("judge field 'extra_intents' must contain only strings")
    return tuple(raw)


def parse_judge_result(raw: Mapping[str, Any] | str) -> JudgeVerdict:
    """Parse and strictly validate a structured judge result.

    Unlike a lenient parser, this raises on any malformed or mistyped field: an
    unparseable judge result must be surfaced, never coerced into a silent accept.

    :param raw: the judge result as a mapping or a JSON string.
    :return: a validated :class:`JudgeVerdict`.
    :raises D1JudgeResultError: if the JSON is invalid or any required field is
        missing or of the wrong type.
    """
    if isinstance(raw, str):
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise D1JudgeResultError(f"invalid judge JSON: {exc.msg}") from exc
    else:
        parsed = raw
    if not isinstance(parsed, Mapping):
        raise D1JudgeResultError("judge result must be a mapping")

    if "coverage" not in parsed:
        raise D1JudgeResultError("judge result missing required field: coverage")
    if "extra_intents" not in parsed:
        raise D1JudgeResultError("judge result missing required field: extra_intents")

    bools = {name: _require_bool(parsed, name) for name in _REQUIRED_BOOL_FIELDS}
    coverage = _parse_coverage(parsed["coverage"])
    extra_intents = _parse_extra_intents(parsed["extra_intents"])

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        raise D1JudgeResultError("judge field 'reason' must be a str")

    return JudgeVerdict(
        accepted=bools["accepted"],
        natural=bools["natural"],
        nonsense=bools["nonsense"],
        ambiguous=bools["ambiguous"],
        duplicate_intent=bools["duplicate_intent"],
        method_semantics_valid=bools["method_semantics_valid"],
        coverage=coverage,
        extra_intents=extra_intents,
        reason=reason,
    )


def decide_accept(
    verdict: JudgeVerdict,
    selected_rest_api_list: Sequence[str],
) -> tuple[bool, list[str]]:
    """Apply the EXACT locked D1 acceptance formula to a structured verdict.

    A D1 row is accepted only when every guard passes AND the judge's supported
    coverage set equals the API set selected before generation::

        covered = {c.rest_api for c in coverage if c.supported}
        accepted = accepted and natural and not nonsense and not ambiguous
                   and not duplicate_intent and method_semantics_valid
                   and not extra_intents and covered == set(selected_rest_api_list)

    The selected set is compared as a set (D1 targets are unordered unique sets), so
    reordering the selected list does not change the decision.

    :param verdict: the parsed, validated structured judge verdict.
    :param selected_rest_api_list: the API set selected before generation (the label).
    :return: ``(accepted, reasons)`` where ``reasons`` lists the stable failing
        reason code(s); an empty list means accepted.
    """
    reasons: list[str] = []

    if not verdict.accepted:
        reasons.append(REASON_NOT_ACCEPTED)
    if not verdict.natural:
        reasons.append(REASON_NOT_NATURAL)
    if verdict.nonsense:
        reasons.append(REASON_NONSENSE)
    if verdict.ambiguous:
        reasons.append(REASON_AMBIGUOUS)
    if verdict.duplicate_intent:
        reasons.append(REASON_DUPLICATE_INTENT)
    if not verdict.method_semantics_valid:
        reasons.append(REASON_METHOD_SEMANTICS_INVALID)
    if verdict.extra_intents:
        reasons.append(f"{REASON_EXTRA_INTENTS}:{sorted(verdict.extra_intents)}")

    covered = verdict.covered_api_set
    selected = set(selected_rest_api_list)
    if covered != selected:
        missing = sorted(selected - covered)   # selected APIs the text does not support.
        unselected = sorted(covered - selected)  # supported APIs that were not selected.
        reasons.append(f"{REASON_COVERAGE_MISMATCH}:missing={missing},unselected={unselected}")

    return (not reasons, reasons)


def target_set_matches(pred: Sequence[str], expected: Sequence[str]) -> bool:
    """D1 target eval: prediction matches only as the same set with no duplicates.

    Implements ``set(pred) == set(expected) and len(pred) == len(set(pred))`` — order
    is ignored, but a duplicated prediction fails even when the set is right.

    :param pred: predicted ``rest_api_list``.
    :param expected: expected (selected) ``rest_api_list``.
    :return: True only if the sets are equal and ``pred`` has no duplicates.
    """
    pred_list = list(pred)
    return set(pred_list) == set(expected) and len(pred_list) == len(set(pred_list))


def load_suite(path: Path) -> dict[str, Any]:
    """Load and minimally validate a hand-labelled D1 acceptance suite.

    :param path: YAML suite path with a top-level ``cases`` list.
    :return: the parsed suite mapping.
    :raises ValueError: if the file is not a mapping with a ``cases`` list.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"d1 judge-semantics suite malformed: {path}")
    return data


def score_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Score one suite case: ``decide_accept`` vs the hand-labelled expectation.

    :param case: a suite case with ``judge_result`` (mapping or JSON string),
        ``selected_rest_api_list``, and ``expected_accepted``.
    :return: a row describing the decision, expectation, and agreement.
    """
    verdict = parse_judge_result(case["judge_result"])
    accepted, reasons = decide_accept(verdict, case["selected_rest_api_list"])
    expected = bool(case["expected_accepted"])
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "accepted": accepted,
        "expected_accepted": expected,
        "reasons": reasons,
        "agree": accepted == expected,
        "false_accept": accepted and not expected,   # admitted a row that must be rejected.
        "false_reject": (not accepted) and expected,  # rejected a row that must be accepted.
    }


def run(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Score a hand-labelled suite against the D1 acceptance formula.

    :param suite: mapping with a ``cases`` list (see :func:`score_case`).
    :return: a report with per-case rows, accuracy, and false-accept/reject ids.
    """
    rows = [score_case(case) for case in suite["cases"]]
    total = len(rows)
    agree = sum(1 for row in rows if row["agree"])
    return {
        "gate": "d1.judge_semantics",
        "total": total,
        "agreement": agree,
        "accuracy": (agree / total) if total else 0.0,
        "false_accepts": [row["id"] for row in rows if row["false_accept"]],
        "false_rejects": [row["id"] for row in rows if row["false_reject"]],
        "rows": rows,
    }


def is_consistent(report: Mapping[str, Any], *, min_accuracy: float = 1.0) -> bool:
    """The formula is consistent with the suite: no false-accepts and full accuracy.

    :param report: a :func:`run` report.
    :param min_accuracy: minimum agreement fraction required (default 1.0).
    :return: True only with zero false-accepts and accuracy >= ``min_accuracy``.
    """
    return not report["false_accepts"] and report["accuracy"] >= min_accuracy


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the D1 judge-semantics gate (offline suite mode)."""
    parser = argparse.ArgumentParser(
        description="Check the locked D1 acceptance formula against a hand-labelled suite.",
    )
    parser.add_argument("--suite", required=True, help="YAML suite of judge-result cases.")
    parser.add_argument("--out", default="reports/gate-report-d1-judge-semantics.json")
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    args = parser.parse_args(argv)

    report = run(load_suite(Path(args.suite)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(
        {k: report[k] for k in ("accuracy", "false_accepts", "false_rejects")},
        indent=2,
        sort_keys=True,
    ))
    if not is_consistent(report, min_accuracy=args.min_accuracy):
        print("BLOCKER: D1 acceptance formula disagrees with the calibration suite.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
