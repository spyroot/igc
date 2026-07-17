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
| `offline.phase123-conformance` | `scripts/gates/phase123_conformance.py` | A tiny deterministic row walks from Phase 1 render/token tensors through D1 judging, Phase 2 parse/set metrics, and Phase 3 `target_calls`. | **fail** on any key/shape/envelope drift. |

All five checks run **offline** — no GPU, no network, no HuggingFace download, no
live Redfish host. Real-model behavior is checked only by the envelope test's
`@pytest.mark.gpu` stage, which the offline pytest run excludes.

## Artifact Cache Gates

The model-artifact cache tooling keeps run outputs under one operator-selected
root, with a default of `/Volumes/k8s-sata-stripe/igc` from
`configs/artifacts/model_cache.yaml`. The structure is:

```text
<root>/
  base_models/<hf-cache-subdir>/                  # original base model, once
  phases/<phase>/<role>/runs/<run_id>/
    artifacts/adapter_model.safetensors
    reports/{report.json,parameters.json,SHA256SUMS.txt}
  cache_manifest.json
```

Use the cache scripts on an approved Kubernetes/remote container, not as laptop
validation:

```bash
python scripts/gates/model_artifact_cache.py plan --root /Volumes/k8s-sata-stripe/igc
python scripts/gates/model_artifact_cache.py init --root /Volumes/k8s-sata-stripe/igc
python scripts/gates/model_artifact_cache.py materialize \
  --download-lfs \
  --base-source-root /models/hf-cache/hub \
  --root /Volumes/k8s-sata-stripe/igc
python scripts/gates/model_artifact_cache.py verify --require-base-model --root /Volumes/k8s-sata-stripe/igc
python scripts/gates/model_artifact_load_gate.py --root /Volumes/k8s-sata-stripe/igc
```

`model_artifact_load_gate.py` always verifies report JSON and adapter tensor
keys/shapes. The heavier `--load-cpu` mode loads base + adapter locally from the
shared cache and refuses laptop execution unless a tiny fixture test explicitly
uses `--allow-local-fixture`.

## Profiles

- **Offline profile (default, required on every Phase 2/3 PR).** All five checks
  above. Runs in CI and in an approved remote container. This is the profile that
  must produce observed output for a PR.
- **Envelope/GPU profile (opt-in).** The real-model/vLLM
  stage in `tests/gates/test_inference_envelope.py`
  (`test_phase2_model_response_envelope_only`) loads an approved Phase 2
  checkpoint and checks only the response **envelope** — model loadability,
  request/response keys present, JSON parse succeeds, output shape — never
  stochastic answer quality. It skips unless a GPU and `IGC_ENVELOPE_MODEL_DIR`
  are both present. It runs only inside an approved remote container, never on
  the operator laptop.

## Execution surface

The runner is intended for **CI** (`.github/workflows/ci.yml` gate job) or an
**approved remote/container validation surface**. It performs no GPU work itself and does
no laptop detection; the execution surface is enforced by where it is invoked. The
report is sanitized (check names, statuses, return codes, and a scrubbed
one-line summary only — no IPs, hostnames, tokens, or file bodies).

## Running it

```bash
python scripts/gates/contract_authority.py
# writes reports/gate-report-contract-authority.json and
# reports/gate-report-phase123-conformance.json; exits non-zero if any check FAILED
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
the PR is marked **`BLOCKED:`** with the exact reason. A Phase 2/3 PR with no observed gate output and no
`BLOCKED:` note is not accepted.
