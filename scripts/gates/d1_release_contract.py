"""Gate: contract.d1-release -- freeze the Phase 2 D1 dataset row contract.

Phase 2 (REST-goal extraction) trains ``text -> a SET of Redfish REST APIs``. The
Redfish corpus has API paths, allowed methods, and JSON bodies but no human
request text, so a D1 row is DRAFTED from a D0 row: ``model_x`` writes the missing
operator ``text`` for a pre-selected API set, and a Pro judge decides whether the
draft maps back to EXACTLY that selected set before the row is accepted.

Two invariants this gate freezes, independent of any single example:

* **View separation / no input leakage.** The generator and judge may see APIs,
  JSON, allowed methods, and descriptions, but the Phase 2 TRAINING model sees
  ONLY ``x.text``. So a valid D1 ``x`` is TEXT-ONLY; a ``json``/``allowed_methods``/
  ``rest_api`` carried in ``x`` is Phase-2 input leakage and fails the gate.
* **Unordered unique set target.** ``y_true.rest_api_list`` is the API set selected
  BEFORE generation. A prediction is correct only when ``set(pred) == set(expected)``
  AND it carries no duplicates; order never matters (``[B, A] == [A, B]``).

``validate_d1_row`` is a pure structural check (returns a list of human-readable
violations; empty means valid). ``evaluate_target`` is the pure set-match target
metric. ``compute_acceptance`` mirrors the EXACT judge acceptance logic in the
committed contract so the same rule is enforced in one place. All three are pure
Python; the gate loads the committed contract
(``configs/contracts/d1_contract.yaml``) and asserts its ILLUSTRATIVE reference
row/judge result are self-consistent -- no corpus, model, or network, and no Phase
1/2 render import.

Used by ``.github/workflows/ci.yml`` (gate job) and the platform gate registry.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BOOTSTRAP = 3

DEFAULT_CONTRACT = "configs/contracts/d1_contract.yaml"

# Exact top-level contract. A D1 row has these keys and no others.
_TOP_LEVEL_KEYS = {
    "phase",            # int 2
    "dataset",          # "D1"
    "source_dataset",   # "D0"
    "model_x",          # drafting model tag
    "task",             # "text_to_rest_api_list"
    "target_semantics", # "unordered_unique_set"
    "x",                # Phase 2 training input (text only)
    "y_true",           # the selected REST API set label
    "validation",       # judge/review provenance flags
}
# x is TEXT-ONLY: exactly this key set.
_X_KEYS = {"text"}
# Keys that, if present in x, are Phase-2 input leakage (must never appear).
_X_LEAKAGE_KEYS = {"json", "allowed_methods", "rest_api"}
# y_true carries exactly the label list.
_Y_TRUE_KEYS = {"rest_api_list"}
# validation: one str field + eight bool fields (the 9 provenance fields).
_VALIDATION_STR_KEYS = {"text_source"}
_VALIDATION_BOOL_KEYS = {
    "review_judged",          # a Pro judge reviewed this row
    "natural",                # text reads as a natural request
    "exact_api_coverage",     # text covers exactly the selected set
    "extra_intent",           # text asks for an API outside the set
    "duplicate_intent",       # text repeats the same API intent
    "ambiguous",              # text too vague to map to an API
    "nonsense",               # text is junk
    "method_semantics_valid", # requested action matches an allowed method
}
_VALIDATION_KEYS = _VALIDATION_STR_KEYS | _VALIDATION_BOOL_KEYS

# Required literal values for the tag fields (the locked D1 identity).
_EXPECTED_PHASE = 2
_EXPECTED_DATASET = "D1"
_EXPECTED_SOURCE_DATASET = "D0"
_EXPECTED_TASK = "text_to_rest_api_list"
_EXPECTED_TARGET_SEMANTICS = "unordered_unique_set"

# Structured judge-result contract (what the Pro judge emits per draft).
_JUDGE_BOOL_KEYS = {
    "accepted",               # top-level accept flag
    "natural",                # text reads naturally
    "nonsense",               # text is junk
    "ambiguous",              # text too vague
    "duplicate_intent",       # text repeats an API intent
    "method_semantics_valid", # action matches an allowed method
}
_JUDGE_LIST_KEYS = {"coverage", "extra_intents"}
_JUDGE_STR_KEYS = {"reason"}
_JUDGE_KEYS = _JUDGE_BOOL_KEYS | _JUDGE_LIST_KEYS | _JUDGE_STR_KEYS
_COVERAGE_ENTRY_KEYS = {"rest_api", "text_span", "supported"}


def _type_name(value: Any) -> str:
    """Return a short, stable type name for violation messages."""
    return type(value).__name__


def _is_bool(value: Any) -> bool:
    """True only for a real bool (guards against int/str masquerading)."""
    return isinstance(value, bool)


def _is_str(value: Any) -> bool:
    """True only for a real str."""
    return isinstance(value, str)


def _validate_x(x: Any) -> list[str]:
    """Validate the TEXT-ONLY ``x`` observation block of a D1 row.

    ``x`` must be exactly ``{"text": <str>}``. Any ``json``/``allowed_methods``/
    ``rest_api`` here is Phase-2 input leakage and is reported specifically.

    :param x: the value stored under the ``x`` key.
    :return: list of violation strings; empty when ``x`` is well-formed.
    """
    if not isinstance(x, dict):
        return [f"'x' must be a dict, got {_type_name(x)}"]

    violations: list[str] = []
    keys = set(x)

    # Call out input leakage explicitly (highest-signal failure mode).
    for leak in sorted(_X_LEAKAGE_KEYS & keys):
        violations.append(
            f"Phase-2 input leakage: 'x.{leak}' present; x must be text-only"
        )
    for missing in sorted(_X_KEYS - keys):
        violations.append(f"missing x key: {missing!r}")
    for extra in sorted(keys - _X_KEYS - _X_LEAKAGE_KEYS):
        violations.append(f"unexpected x key: {extra!r}")

    if "text" in x and not _is_str(x["text"]):
        violations.append(f"'x.text' must be str, got {_type_name(x['text'])}")

    return violations


def _validate_y_true(y_true: Any) -> list[str]:
    """Validate the ``y_true`` label block of a D1 row.

    ``y_true.rest_api_list`` must be a list of unique strings (the unordered
    unique set target); duplicates are a contract violation at storage time.

    :param y_true: the value stored under the ``y_true`` key.
    :return: list of violation strings; empty when ``y_true`` is well-formed.
    """
    if not isinstance(y_true, dict):
        return [f"'y_true' must be a dict, got {_type_name(y_true)}"]

    violations: list[str] = []
    keys = set(y_true)
    for extra in sorted(keys - _Y_TRUE_KEYS):
        violations.append(f"unexpected y_true key: {extra!r}")

    if "rest_api_list" not in y_true:
        violations.append("missing y_true key: 'rest_api_list'")
        return violations

    api_list = y_true["rest_api_list"]
    if not isinstance(api_list, list):
        violations.append(
            f"'y_true.rest_api_list' must be a list, got {_type_name(api_list)}"
        )
        return violations

    for index, api in enumerate(api_list):
        if not _is_str(api):
            violations.append(
                f"'y_true.rest_api_list[{index}]' must be str, got {_type_name(api)}"
            )
    # Only test uniqueness when every entry is a hashable string.
    if all(_is_str(a) for a in api_list) and len(api_list) != len(set(api_list)):
        violations.append(
            "'y_true.rest_api_list' has duplicate entries (target is a unique set)"
        )

    return violations


def _validate_validation(validation: Any) -> list[str]:
    """Validate the ``validation`` provenance block of a D1 row.

    Exactly one ``str`` field (``text_source``) and eight ``bool`` flags.

    :param validation: the value stored under the ``validation`` key.
    :return: list of violation strings; empty when well-formed.
    """
    if not isinstance(validation, dict):
        return [f"'validation' must be a dict, got {_type_name(validation)}"]

    violations: list[str] = []
    keys = set(validation)
    for missing in sorted(_VALIDATION_KEYS - keys):
        violations.append(f"missing validation key: {missing!r}")
    for extra in sorted(keys - _VALIDATION_KEYS):
        violations.append(f"unexpected validation key: {extra!r}")

    for key in sorted(_VALIDATION_STR_KEYS & keys):
        if not _is_str(validation[key]):
            violations.append(
                f"'validation.{key}' must be str, got {_type_name(validation[key])}"
            )
    for key in sorted(_VALIDATION_BOOL_KEYS & keys):
        if not _is_bool(validation[key]):
            violations.append(
                f"'validation.{key}' must be bool, got {_type_name(validation[key])}"
            )

    return violations


def validate_d1_row(row: Any) -> list[str]:
    """Validate one Phase 2 D1 row against the exact stored contract.

    Checks the exact top-level key set (no missing, no extra), the required
    literal tag values (``phase``/``dataset``/``source_dataset``/``task``/
    ``target_semantics``), that ``x`` is TEXT-ONLY (json/allowed_methods leakage
    fails), that ``y_true.rest_api_list`` is a duplicate-free list of strings, and
    that ``validation`` carries its one str + eight bool fields.

    :param row: a decoded JSONL row (any value; non-dicts are reported).
    :return: list of human-readable violations; an empty list means the row is a
        valid D1 row.
    """
    if not isinstance(row, dict):
        return [f"row must be a dict, got {_type_name(row)}"]

    violations: list[str] = []
    keys = set(row)
    for missing in sorted(_TOP_LEVEL_KEYS - keys):
        violations.append(f"missing top-level key: {missing!r}")
    for extra in sorted(keys - _TOP_LEVEL_KEYS):
        violations.append(f"unexpected top-level key: {extra!r}")

    if "phase" in row:
        phase = row["phase"]
        # bool is an int subclass; a boolean phase is a type error, not a value.
        if _is_bool(phase) or not isinstance(phase, int):
            violations.append(f"'phase' must be int, got {_type_name(phase)}")
        elif phase != _EXPECTED_PHASE:
            violations.append(f"'phase' must be {_EXPECTED_PHASE}, got {phase}")

    for field, expected in (
        ("dataset", _EXPECTED_DATASET),
        ("source_dataset", _EXPECTED_SOURCE_DATASET),
        ("task", _EXPECTED_TASK),
        ("target_semantics", _EXPECTED_TARGET_SEMANTICS),
    ):
        if field in row:
            value = row[field]
            if not _is_str(value):
                violations.append(f"'{field}' must be str, got {_type_name(value)}")
            elif value != expected:
                violations.append(f"'{field}' must be {expected!r}, got {value!r}")

    if "model_x" in row and not _is_str(row["model_x"]):
        violations.append(f"'model_x' must be str, got {_type_name(row['model_x'])}")

    if "x" in row:
        violations.extend(_validate_x(row["x"]))
    if "y_true" in row:
        violations.extend(_validate_y_true(row["y_true"]))
    if "validation" in row:
        violations.extend(_validate_validation(row["validation"]))

    return violations


def evaluate_target(pred: Sequence[str], expected: Sequence[str]) -> bool:
    """Score a predicted REST API list against the D1 unordered-unique-set target.

    Correct only when ``set(pred) == set(expected)`` AND ``pred`` has no
    duplicates. Order is irrelevant (``[B, A]`` matches ``[A, B]``); a repeated
    API in the prediction fails; ``[]`` matches ``[]`` for no-action rows.

    :param pred: the model's predicted list of REST API path strings.
    :param expected: the canonical selected API set (list form, may be sorted).
    :return: True when the prediction exactly matches the expected set with no
        duplicates.
    """
    pred_list = list(pred)
    if len(pred_list) != len(set(pred_list)):
        return False  # duplicate prediction fails even if the set matches
    return set(pred_list) == set(expected)


def _coverage_supported(coverage: Iterable[Any]) -> set[str]:
    """Return the set of ``rest_api`` values marked ``supported`` in coverage.

    Tolerant of both dicts and objects (attribute access) for each entry, so the
    same logic works on parsed JSON and on a structured judge dataclass.

    :param coverage: iterable of coverage entries.
    :return: set of supported API path strings.
    """
    covered: set[str] = set()
    for entry in coverage:
        if isinstance(entry, Mapping):
            supported = bool(entry.get("supported"))
            rest_api = entry.get("rest_api")
        else:
            supported = bool(getattr(entry, "supported", False))
            rest_api = getattr(entry, "rest_api", None)
        if supported and isinstance(rest_api, str):
            covered.add(rest_api)
    return covered


def compute_acceptance(
    verdict: Mapping[str, Any], selected_rest_api_list: Sequence[str]
) -> bool:
    """Apply the EXACT D1 judge acceptance logic from the committed contract.

    ``covered = {c.rest_api for c in coverage if c.supported}`` and::

        accepted = verdict.accepted and verdict.natural and not verdict.nonsense
                   and not verdict.ambiguous and not verdict.duplicate_intent
                   and verdict.method_semantics_valid and not verdict.extra_intents
                   and covered == set(selected_rest_api_list)

    :param verdict: the structured judge result (see ``validate_judge_result``).
    :param selected_rest_api_list: the API set selected BEFORE generation (the
        D1 label), matched as a set against the supported coverage.
    :return: True when the draft is accepted into D1.
    """
    # Fail-closed: a verdict that OMITS a required flag must reject, never slip
    # through. Positive flags default False (missing -> reject); negative flags
    # default to a present/non-empty sentinel (missing -> reject).
    covered = _coverage_supported(verdict.get("coverage", []))
    return bool(
        verdict.get("accepted", False)
        and verdict.get("natural", False)
        and not verdict.get("nonsense", True)
        and not verdict.get("ambiguous", True)
        and not verdict.get("duplicate_intent", True)
        and verdict.get("method_semantics_valid", False)
        and not verdict.get("extra_intents", ("_missing",))
        and covered == set(selected_rest_api_list)
    )


def validate_judge_result(result: Any) -> list[str]:
    """Validate one structured judge result against the committed shape.

    :param result: the judge result (any value; non-dicts are reported).
    :return: list of violation strings; empty when well-formed.
    """
    if not isinstance(result, dict):
        return [f"judge result must be a dict, got {_type_name(result)}"]

    violations: list[str] = []
    keys = set(result)
    for missing in sorted(_JUDGE_KEYS - keys):
        violations.append(f"missing judge key: {missing!r}")
    for extra in sorted(keys - _JUDGE_KEYS):
        violations.append(f"unexpected judge key: {extra!r}")

    for key in sorted(_JUDGE_BOOL_KEYS & keys):
        if not _is_bool(result[key]):
            violations.append(f"'{key}' must be bool, got {_type_name(result[key])}")
    for key in sorted(_JUDGE_STR_KEYS & keys):
        if not _is_str(result[key]):
            violations.append(f"'{key}' must be str, got {_type_name(result[key])}")
    for key in sorted(_JUDGE_LIST_KEYS & keys):
        if not isinstance(result[key], list):
            violations.append(f"'{key}' must be a list, got {_type_name(result[key])}")

    if isinstance(result.get("coverage"), list):
        for index, entry in enumerate(result["coverage"]):
            if not isinstance(entry, dict):
                violations.append(
                    f"'coverage[{index}]' must be a dict, got {_type_name(entry)}"
                )
                continue
            entry_keys = set(entry)
            for missing in sorted(_COVERAGE_ENTRY_KEYS - entry_keys):
                violations.append(f"missing coverage[{index}] key: {missing!r}")
            for extra in sorted(entry_keys - _COVERAGE_ENTRY_KEYS):
                violations.append(f"unexpected coverage[{index}] key: {extra!r}")
            if "supported" in entry and not _is_bool(entry["supported"]):
                violations.append(
                    f"'coverage[{index}].supported' must be bool, "
                    f"got {_type_name(entry['supported'])}"
                )

    return violations


def load_contract(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load and minimally validate the committed D1 contract.

    :param path: path to ``d1_contract.yaml``.
    :return: the parsed contract mapping.
    :raises ValueError: when the required contract sections are missing.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"row_shape", "acceptance_logic", "reference_row", "reference_judge_result"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(
            f"D1 contract malformed (need {sorted(required)}): {path}"
        )
    return data


def check(path: str | Path = DEFAULT_CONTRACT) -> int:
    """Validate the committed contract's ILLUSTRATIVE reference for self-consistency.

    Confirms the reference row is a valid D1 row, its label evaluates as a target
    match against itself, its judge result is well-formed, and applying the exact
    acceptance logic to the reference judge result + reference label accepts it.

    :param path: path to ``d1_contract.yaml``.
    :return: ``EXIT_OK`` when the reference is self-consistent, ``EXIT_FAIL`` on
        any drift, ``EXIT_BOOTSTRAP`` when the contract file is not committed yet.
    """
    contract_path = Path(path)
    if not contract_path.is_file():
        print(
            "BOOTSTRAP: D1 contract not committed yet; add "
            f"{contract_path} with row_shape, acceptance_logic, and references.",
            file=sys.stderr,
        )
        return EXIT_BOOTSTRAP

    contract = load_contract(contract_path)

    row = contract["reference_row"]
    row_violations = validate_d1_row(row)
    if row_violations:
        print("BLOCKER: committed D1 reference row is not a valid D1 row:", file=sys.stderr)
        for violation in row_violations:
            print(f"  - {violation}", file=sys.stderr)
        return EXIT_FAIL

    selected = row["y_true"]["rest_api_list"]
    if not evaluate_target(selected, selected):
        print("BLOCKER: D1 reference label does not self-match the target metric.", file=sys.stderr)
        return EXIT_FAIL

    verdict = contract["reference_judge_result"]
    judge_violations = validate_judge_result(verdict)
    if judge_violations:
        print("BLOCKER: committed D1 reference judge result is malformed:", file=sys.stderr)
        for violation in judge_violations:
            print(f"  - {violation}", file=sys.stderr)
        return EXIT_FAIL

    if not compute_acceptance(verdict, selected):
        print(
            "BLOCKER: D1 reference judge result + label does not pass the acceptance logic.",
            file=sys.stderr,
        )
        return EXIT_FAIL

    print("OK: Phase 2 D1 contract reference is valid and self-consistent.")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``contract.d1-release`` gate."""
    parser = argparse.ArgumentParser(description="Freeze the Phase 2 D1 dataset contract.")
    parser.add_argument(
        "--contract",
        default=DEFAULT_CONTRACT,
        help="Path to the committed D1 contract (row shape + acceptance logic + references).",
    )
    args = parser.parse_args(argv)
    return check(args.contract)


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
