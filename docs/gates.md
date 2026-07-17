# Contract-authority gate (Phase 2/3 REST-goal)

The **contract-authority gate** protects the single canonical Phase 2/3 REST-goal
contract — the module `igc/ds/rest_goal_contract.py`, which owns the row shape,
the prompt/target renderers, and the parse/eval used to score model output.

It exists because a green *per-module* test suite is not enough: a past change
shipped a parallel contract module with a forked metric namespace, and its own
isolated tests were green, so the fork slipped through. The gates below are
**repo-level** — they scan the whole live package and freeze the approved JSON
shape — so that class of drift fails loudly.

The runner is `scripts/gates/contract_authority.py`. It composes the checks and
writes a sanitized JSON report to `reports/gate-report-contract-authority.json`.

## Checks

| Check | Source | What it proves | Failure behavior |
| --- | --- | --- | --- |
| `repo.contract-single-source` | `scripts/gates/contract_single_source.py` + `configs/contracts/rest_goal.yaml` | One canonical contract module; no forked namespace tokens and no canonical symbol defined in a second module (no parallel island). | **fail** (blocks) on any forked token or parallel definition. |
| `repo.schema-snapshot` | `scripts/gates/schema_snapshot.py` + `schemas/snapshots/rest_goal_contract.shape.json` | The approved Phase 2/3 row *shape* (field names, nesting, value types — never values) has not drifted. | **fail** on drift; **bootstrap** (not a failure) until the snapshot is first committed with `--update`. |
| `offline.contract-tests` | `tests/ds/test_rest_goal_contract.py` | The canonical row builders, renderers, parsers, and evaluators behave as specified. | **fail** on any red test. |
| `offline.gate-tests` | `tests/gates/` | The gate self-tests plus the **inference-envelope** deterministic exact-match layer (`tests/gates/test_inference_envelope.py`). | **fail** on any red test. |

All four checks run **offline** — no GPU, no network, no HuggingFace download, no
live Redfish host. Real-model behavior is checked only by the envelope test's
`@pytest.mark.gpu` stage, which the offline pytest run excludes.

## Profiles

- **Offline profile (default, required on every Phase 2/3 PR).** All four checks
  above. Runs in CI and in an approved remote container. This is the profile that
  must produce observed output for a PR.
- **Envelope/GPU profile (opt-in, currently BLOCKED).** The real-model/vLLM
  stage in `tests/gates/test_inference_envelope.py`
  (`test_phase2_model_response_envelope_only`) loads an approved Phase 2
  checkpoint and checks only the response **envelope** — model loadability,
  request/response keys present, JSON parse succeeds, output shape — never
  stochastic answer quality. It skips unless a GPU and `IGC_ENVELOPE_MODEL_DIR`
  are both present. It is **BLOCKED while the GB300/NV72 surface is powered off**
  and, when unblocked, runs only inside an approved remote container, never on the
  operator laptop.

## Execution surface

The runner is intended for **CI** (`.github/workflows/ci.yml` gate job) or an
**approved remote GB300/NV72 container**. It performs no GPU work itself and does
no laptop detection; the execution surface is enforced by where it is invoked. The
report is sanitized (check names, statuses, return codes, and a scrubbed
one-line summary only — no IPs, hostnames, tokens, or file bodies).

## Running it

```bash
python scripts/gates/contract_authority.py
# writes reports/gate-report-contract-authority.json; exits non-zero if any check FAILED
```

The first time, generate and commit the schema snapshot on an approved surface so
`repo.schema-snapshot` moves from `bootstrap` to `pass`:

```bash
python scripts/gates/schema_snapshot.py --update
git add schemas/snapshots/rest_goal_contract.shape.json
```

## PR rule

A **Phase 2 or Phase 3 PR** — anything touching the contract, dataset builders,
renderers, parsers, evaluators, or their metric namespaces — **must include the
observed contract-authority gate output** (the `overall: pass` report, or the
console summary) in the PR body. If the gate cannot be run on an approved surface,
the PR is marked **`BLOCKED:`** with the exact reason (for example, GB300 powered
off for the envelope profile). A Phase 2/3 PR with no observed gate output and no
`BLOCKED:` note is not accepted.
