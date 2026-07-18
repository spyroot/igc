"""Gate: d1.balance-and-dedupe — the accepted Phase 2 D1 set is deduped and balanced.

Phase 2 D1 rows are drafted by ``model_x`` and accepted by the Pro judge one at a
time (the judge/acceptance contract lives in
``scripts/gates/judge_calibration.py`` and ``scripts/gates/generator_qualification.py``).
Judging each draft in isolation guarantees per-row correctness but says nothing
about the *shape of the accepted corpus*: the same operator request can be drafted
many times with trivial wording changes (near-duplicates), and a whole eligible
REST API can end up with zero single-command training rows while common APIs are
over-represented. Either failure quietly poisons the Phase 2 text->rest_api_list
model — it memorises a handful of phrasings and never learns the rare API — while
every per-row gate stays green.

This gate is the corpus-level guard that runs *after* per-row acceptance:

* ``normalize`` collapses a draft to a comparison-stable form (case, whitespace,
  punctuation, and stuttered repeated words), so cosmetic variants compare equal.
* ``is_exact_duplicate`` / ``is_near_duplicate`` catch a re-drafted request before
  it inflates one intent. Exact + token-overlap (Jaccard) dedup exists now; an
  embedding-similarity pass is a later, additive layer — the token pass is the
  offline, CI-safe floor.
* ``balance_report`` summarises the accepted set: how many 1/2/3-API rows, how
  often each API and each API *combination* appears, which APIs have at least one
  single-command row, and (when the build records it) duplicate / near-duplicate /
  retry and judge-acceptance rates.
* ``fail_balance`` turns that report into blocking violations: an eligible API with
  no single-command coverage, or a judge acceptance rate below a floor (the judge
  is rejecting almost everything) or above a ceiling (a rubber-stamp judge — a
  suspiciously high rate is as bad as a low one).

The D1 target is an **unordered unique set** of REST APIs
(``set(pred) == set(expected)`` and no duplicates); this gate treats every
``y_true.rest_api_list`` as a set and flags any stored list that carries a
duplicate as malformed. It reads only ``x.text`` and ``y_true.rest_api_list`` from
each row — never the Phase 2 model input beyond ``x.text`` — so it enforces the
separation contract (the training model sees only ``x.text``; the API label was
selected before generation).

Offline (CI): pure Python over in-memory / JSONL accepted rows plus a YAML list of
eligible APIs and scalar thresholds — no model, corpus tarball, or network. It
imports no Phase 1/2 render or model module.

Used by ``tests/gates/test_d1_balance_and_dedupe.py`` (offline gate) and by the
Phase 2 D1 build's post-acceptance balance check; ``main`` runs it over a built
JSONL for a one-shot audit.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

# Any run of characters that is neither a word char nor whitespace is treated as
# punctuation and flattened to a space (so "boot-config." and "boot config"
# normalise to the same tokens). ``\w`` keeps letters, digits, and underscore.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Optional per-row generation-telemetry block. When the build attaches it, the
# report gains corpus-level rates; when absent, those rates are ``None`` ("if
# present"). Recognised keys are documented on ``_generation_stats``.
_GENERATION_KEY = "generation"


def normalize(text: str) -> str:
    """Collapse a draft to a comparison-stable form for dedup.

    Applies, in order: lowercase, punctuation-to-space, whitespace-collapse, and
    consecutive repeated-word collapse (``"check check boot"`` -> ``"check boot"``).
    ``normalize`` is idempotent, so passing an already-normalised string returns it
    unchanged — callers may keep a set of normalised texts and compare safely.

    :param text: raw operator-request text (``x.text`` of a D1 row).
    :return: the normalised string (may be empty when ``text`` is blank/punctuation).
    """
    lowered = str(text).lower()
    depunctuated = _PUNCT_RE.sub(" ", lowered)
    tokens = depunctuated.split()  # splits on any whitespace and drops empties
    collapsed: list[str] = []
    for token in tokens:
        # Collapse only *consecutive* duplicate words (a stutter), not every repeat.
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return " ".join(collapsed)


def _tokens(text: str) -> set[str]:
    """Return the set of normalised word tokens for token-overlap comparison."""
    return set(normalize(text).split())


def is_exact_duplicate(text: str, index: Iterable[str]) -> bool:
    """Return True when ``text`` normalises to a string already seen in ``index``.

    Comparison is on the normalised form, so cosmetic variants (case, spacing,
    trailing punctuation, stutters) count as exact duplicates. ``index`` is any
    iterable of previously-seen texts; because ``normalize`` is idempotent it may
    hold raw or already-normalised strings.

    :param text: candidate text.
    :param index: iterable of previously-accepted texts (raw or normalised).
    :return: True if a normalised match exists in ``index``.
    """
    target = normalize(text)
    for existing in index:
        if normalize(existing) == target:
            return True
    return False


def is_near_duplicate(
    text: str,
    accepted_texts: Iterable[str],
    token_overlap_threshold: float,
) -> bool:
    """Return True when ``text`` token-overlaps an accepted text at/above threshold.

    Overlap is the Jaccard index of normalised word-token sets
    (``|A ∩ B| / |A ∪ B|``). A threshold of ``1.0`` requires identical token sets
    (order-independent); lower thresholds catch reworded near-duplicates. Empty
    token sets never match (a blank draft is not a near-duplicate of anything). An
    exact duplicate is always a near-duplicate at any threshold ``<= 1.0``.

    :param text: candidate text.
    :param accepted_texts: iterable of already-accepted texts.
    :param token_overlap_threshold: Jaccard threshold in ``[0.0, 1.0]``.
    :return: True if any accepted text meets/exceeds the overlap threshold.
    """
    tokens = _tokens(text)
    if not tokens:
        return False
    for existing in accepted_texts:
        other = _tokens(existing)
        if not other:
            continue
        union = tokens | other
        overlap = len(tokens & other) / len(union) if union else 0.0
        if overlap >= token_overlap_threshold:
            return True
    return False


def _rest_api_list(row: Mapping[str, Any]) -> list[str]:
    """Extract ``y_true.rest_api_list`` from a D1 row (empty list when absent)."""
    y_true = row.get("y_true") or {}
    apis = y_true.get("rest_api_list") if isinstance(y_true, Mapping) else None
    return [str(a) for a in apis] if isinstance(apis, (list, tuple)) else []


def _row_text(row: Mapping[str, Any]) -> str:
    """Extract ``x.text`` from a D1 row (empty string when absent)."""
    x = row.get("x") or {}
    text = x.get("text") if isinstance(x, Mapping) else None
    return str(text) if isinstance(text, str) else ""


def _generation_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Compute corpus-level generation rates from optional per-row telemetry.

    A row may carry a ``"generation"`` block recorded by the D1 build:

    * ``attempts`` (int >= 1): total draft+judge attempts that produced this
      accepted row (``attempts - 1`` earlier drafts were rejected). Absent -> 1.
    * ``duplicate`` (bool): the draft was flagged an exact duplicate during build.
    * ``near_duplicate`` (bool): the draft was flagged a near-duplicate.

    :param rows: the accepted rows.
    :return: ``{acceptance_rate, retry_rate, duplicate_rate, near_duplicate_rate}``
        when at least one row carries a generation block, else ``None`` (the build
        did not record telemetry, so the rates are unknown — "if present").
    """
    blocks = [
        row[_GENERATION_KEY]
        for row in rows
        if isinstance(row.get(_GENERATION_KEY), Mapping)
    ]
    if not blocks:
        return None

    accepted = len(blocks)
    attempts: list[int] = []
    for block in blocks:
        try:
            value = int(block.get("attempts", 1))
        except (TypeError, ValueError):
            value = 1
        attempts.append(value if value >= 1 else 1)

    total_attempts = sum(attempts)
    return {
        # accepted rows / total judge attempts recorded on them: a low value means
        # the judge rejects most drafts; a high value near 1.0 is a rubber stamp.
        "acceptance_rate": (accepted / total_attempts) if total_attempts else 0.0,
        # fraction of accepted rows that needed more than one attempt.
        "retry_rate": sum(1 for a in attempts if a > 1) / accepted,
        # fraction of accepted rows the build had flagged as an exact duplicate.
        "duplicate_rate": sum(1 for b in blocks if b.get("duplicate")) / accepted,
        # fraction flagged as a near-duplicate.
        "near_duplicate_rate": sum(1 for b in blocks if b.get("near_duplicate")) / accepted,
    }


def balance_report(accepted_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise the shape of an accepted D1 set for the balance gate.

    Every ``y_true.rest_api_list`` is treated as an unordered unique set (the D1
    target semantics); a stored list with a duplicate entry is counted as
    ``malformed`` because it violates the ``len(pred) == len(set(pred))`` rule.

    :param accepted_rows: iterable of accepted D1 rows (each with ``x.text`` and
        ``y_true.rest_api_list``); may carry an optional ``generation`` block.
    :return: a report dict with:

        * ``k_counts``          — accepted rows per set size ``k`` (1/2/3, and any
          other observed size).
        * ``per_rest_api``      — accepted rows containing each API.
        * ``per_combination``   — accepted rows per canonical API combination
          (sorted APIs joined by ``" + "``).
        * ``single_api_coverage`` — sorted APIs that have >= 1 single-command
          (``k == 1``) accepted row.
        * ``total_accepted`` / ``malformed`` — corpus size / rows whose stored list
          was empty or carried a duplicate.
        * ``acceptance_rate`` / ``retry_rate`` / ``duplicate_rate`` /
          ``near_duplicate_rate`` — ``None`` unless a ``generation`` block was
          present on at least one row.
    """
    rows = list(accepted_rows)

    k_counts: dict[int, int] = {}                 # set size -> count of accepted rows
    per_rest_api: dict[str, int] = {}             # api -> count of accepted rows using it
    per_combination: dict[str, int] = {}          # canonical combo string -> count
    single_api_coverage: set[str] = set()         # apis with >=1 single-command row
    malformed = 0                                  # empty or duplicate-bearing rest_api_list

    for row in rows:
        apis = _rest_api_list(row)
        unique = sorted(set(apis))
        if not apis or len(apis) != len(unique):
            # Empty set (no target) or a duplicate in the stored list both break
            # the unordered-unique-set contract.
            malformed += 1

        k = len(unique)
        k_counts[k] = k_counts.get(k, 0) + 1
        for api in unique:
            per_rest_api[api] = per_rest_api.get(api, 0) + 1
        combination = " + ".join(unique)  # canonical, order-independent combo key
        per_combination[combination] = per_combination.get(combination, 0) + 1
        if k == 1:
            single_api_coverage.add(unique[0])

    report: dict[str, Any] = {
        "gate": "d1.balance-and-dedupe",
        "total_accepted": len(rows),
        "malformed": malformed,
        "k_counts": dict(sorted(k_counts.items())),
        "per_rest_api": dict(sorted(per_rest_api.items())),
        "per_combination": dict(sorted(per_combination.items())),
        "single_api_coverage": sorted(single_api_coverage),
    }

    stats = _generation_stats(rows)
    for name in ("acceptance_rate", "retry_rate", "duplicate_rate", "near_duplicate_rate"):
        report[name] = stats[name] if stats else None
    return report


def fail_balance(
    report: Mapping[str, Any],
    eligible_apis: Iterable[str],
    min_acceptance: float,
    max_acceptance: float,
) -> list[str]:
    """Turn a balance report into blocking violations (empty list == pass).

    Fails when:

    * any eligible API has no single-command (``k == 1``) accepted row — the model
      would never see that API in isolation; or
    * the judge acceptance rate (when recorded) is below ``min_acceptance`` (the
      judge rejects nearly everything) or above ``max_acceptance`` (a rubber-stamp
      judge — a suspiciously high rate is treated as a failure too).

    The acceptance-rate bounds are only checked when the report carries an
    ``acceptance_rate`` (i.e. the build recorded generation telemetry); without it
    the rate is unknown and cannot fail the gate.

    :param report: output of :func:`balance_report`.
    :param eligible_apis: the APIs that MUST each have single-command coverage.
    :param min_acceptance: lower bound on the judge acceptance rate.
    :param max_acceptance: upper bound on the judge acceptance rate.
    :return: sorted list of human-readable violation strings; empty means pass.
    """
    failures: list[str] = []

    covered = set(report.get("single_api_coverage") or [])
    for api in sorted(set(eligible_apis)):
        if api not in covered:
            failures.append(f"eligible API has no single-command coverage: {api}")

    rate = report.get("acceptance_rate")
    if rate is not None:
        if rate < min_acceptance:
            failures.append(
                f"judge acceptance rate {rate:.3f} < min {min_acceptance} "
                "(judge rejecting almost everything)"
            )
        if rate > max_acceptance:
            failures.append(
                f"judge acceptance rate {rate:.3f} > max {max_acceptance} "
                "(suspiciously high — possible rubber-stamp judge)"
            )
    return failures


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load accepted D1 rows from a JSONL file (one row per non-blank line)."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_eligible_apis(raw: str) -> list[str]:
    """Resolve ``--eligible-apis`` from a YAML file path or a comma-separated list.

    :param raw: either a path to a YAML file (a bare list, or a mapping with an
        ``eligible_apis`` key) or a literal ``"a,b,c"`` string.
    :return: the list of eligible API strings.
    """
    candidate = Path(raw)
    if candidate.is_file():
        data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            data = data.get("eligible_apis", [])
        if not isinstance(data, list):
            raise ValueError(f"eligible-apis file must be a list or have 'eligible_apis': {raw}")
        return [str(a) for a in data]
    return [item.strip() for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``d1.balance-and-dedupe`` gate (offline audit)."""
    parser = argparse.ArgumentParser(
        description="Balance + dedupe audit of an accepted Phase 2 D1 set."
    )
    parser.add_argument("--rows", required=True, help="JSONL of accepted D1 rows.")
    parser.add_argument(
        "--eligible-apis",
        required=True,
        help="YAML file (list or {eligible_apis: [...]}) or a comma-separated list.",
    )
    parser.add_argument("--min-acceptance", type=float, default=0.05)
    parser.add_argument("--max-acceptance", type=float, default=0.95)
    parser.add_argument("--out", default="reports/gate-report-d1-balance.json")
    args = parser.parse_args(argv)

    rows = _load_rows(Path(args.rows))
    eligible = _load_eligible_apis(args.eligible_apis)
    report = balance_report(rows)
    failures = fail_balance(report, eligible, args.min_acceptance, args.max_acceptance)
    report["violations"] = failures
    report["balanced"] = not failures

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "total_accepted": report["total_accepted"],
        "k_counts": report["k_counts"],
        "single_api_coverage": report["single_api_coverage"],
        "acceptance_rate": report["acceptance_rate"],
        "violations": failures,
    }, indent=2, sort_keys=True))
    if failures:
        print("BLOCKER: accepted D1 set is unbalanced or under-covered.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
