# redfish_ctl Backend Integration Plan

## Purpose

`redfish_ctl`, the sibling Redfish discovery and simulator project consumed through the `idrac_ctl/`
submodule path today, is the authoritative source for Redfish corpora, simulator capabilities, and
mutation behavior. IGC should consume those contracts instead of maintaining an independent Redfish
mutation model.

This plan prevents the RL path from training against stale `~/.json_responses` assumptions,
filename-derived paths, one-directory capture layouts, or simulator behavior that only exists in
IGC tests. It also gives the implementation team a queue that can land in small PRs while
preserving the existing `MockServer` compatibility tests.

## Current Status

This document is both the public roadmap and the status anchor for the first integration slice. The
first slice adds the backend protocol, response/status/capability normalization, HTTP and optional
in-process adapters, a legacy mock adapter, manifest validation, and offline contract tests. The
current Redfish RL path still uses captured JSON, `rest_api_map.npy`, `MockServer`, and legacy Gym
wrappers. The action-catalog, mutation-capable Gym environment, reset/vectorization, workflow, and
scale gates below must land before Redfish RL defaults move to `redfish_ctl`.

Full acceptance also depends on the provider side exposing stable simulator and corpus-manifest
contracts. Until that provider contract exists, IGC should fail clearly instead of guessing capture
layout, reconstructing Redfish paths from filenames, or importing provider test helpers.

## Hard Contract

- Redfish RL defaults to a `redfish_ctl` backend. IGC keeps `MockServer`, defined in
  `igc/envs/rest_mock_server.py`, only for generic replay compatibility tests and non-Redfish
  fixtures.
- The preferred runtime backend is an in-process `redfish_ctl` simulator API for high-throughput RL.
  A standalone HTTP `redfish_ctl` simulator must be usable through the same IGC backend protocol.
- IGC must not import `redfish_ctl` test-only modules, sandbox mock servers, or raw mutation helpers.
  Only provider-declared library APIs, HTTP APIs, schemas, and materialized manifests are valid
  dependencies.
- IGC must not assume `~/.json_responses`, a capture-IP directory, flattened files, or a single
  directory depth. Corpus discovery must be manifest-driven and recursive.
- IGC must not reconstruct paths from filenames when a Redfish `@odata.id` field or provider mapping
  exists.
- IGC must not infer that every discovered Redfish `Actions` entry is simulatable. The RL action
  space comes from simulator-declared capabilities and their argument schemas.
- IGC must represent payload-bearing actions. An action is at least method, target or capability id,
  JSON payload, and verification metadata; it is not only `(url_index, method_index)`.
- Every RL reset calls the backend reset path and clears overlays, replay-visible state, history, and
  backend RNG for that environment instance.
- In-process and HTTP simulator modes must produce equivalent normalized responses and equivalent
  Gym `info` metadata for the same seeded episode.

## Target Surfaces

### Backend Protocol

IGC should introduce one Redfish environment backend protocol. The exact module path can be adjusted
when implementation starts, but the interface must preserve these operations:

```python
class RedfishEnvironmentBackend(Protocol):
    def reset(self, seed: int | None = None) -> RedfishBackendStatus: ...
    def request(
        self,
        method: str,
        path_or_capability: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RedfishBackendResponse: ...
    def status(self) -> RedfishBackendStatus: ...
    def capabilities(self) -> RedfishBackendCapabilities: ...
    def close(self) -> None: ...
```

The normalized response type should carry HTTP status, JSON body, headers, provider error code,
mutation metadata, task/job metadata, and provenance. The Gym wrapper should place non-policy
diagnostics in `info`, including backend kind, corpus id, simulator contract version, provider
capability id, reset seed, and whether the action was declared valid before execution.

### Backend Implementations

| Backend | Source | Purpose |
| --- | --- | --- |
| `RedfishCtlInProcessBackend` | `redfish_ctl` library API | Fast default for RL rollouts and vectorized training |
| `RedfishCtlHttpBackend` | `redfish_ctl` standalone simulator HTTP API | Parity mode and process-boundary validation |
| `LegacyMockServerBackend` | IGC `MockServer` class | Existing non-Redfish replay tests and compatibility smoke tests |

### Corpus Manifest

The corpus manifest is the authoritative data contract. It should identify the materialized corpus,
vendor/model filters, simulator contract version, provider revision, source capture family, valid
resource roots, allowed methods, and action-capability records. IGC should prefer
`rest_api_map.v1.json`, produced by `redfish_ctl`, and use the existing `rest_api_map.npy` fallback
only for legacy read-only fixtures until the provider fully migrates.

The CLI should add planned flags:

| Flag | Meaning |
| --- | --- |
| `--redfish-corpus-manifest` | Path to the provider manifest that describes a materialized corpus |
| `--corpus-id` | Stable corpus identifier selected from the manifest |
| `--vendor` | Optional vendor filter supplied by the user or experiment profile |
| `--model` | Optional platform-model filter supplied by the user or experiment profile |
| `--corpus-kind` | Defaults to `full`; narrows to explicit manifest-defined corpus variants |

`--json-root`, the legacy raw-capture root flag, should remain accepted for compatibility but emit a
deprecation warning on Redfish RL paths once the manifest backend exists.

## Implementation Queue

### IGC-RF-00: Backend Protocol and Compatibility Shell

Create the backend protocol, response/status dataclasses, and three backend adapters. Wire no RL
default yet.

Files expected to change:

- Create `igc/envs/redfish/backend_types.py` for dataclasses and typed errors.
- Create `igc/envs/redfish/backends.py` for the protocol and backend factory.
- Create `igc/envs/redfish/redfish_ctl_inprocess.py` for provider-library integration.
- Create `igc/envs/redfish/redfish_ctl_http.py` for standalone simulator integration.
- Create `igc/envs/redfish/legacy_mock.py` for `MockServer` compatibility.
- Add tests under `tests/envs/redfish/`.

Required tests:

- In-process and HTTP backends normalize successful JSON, no-body, and error responses into the same
  `RedfishBackendResponse` fields.
- Missing simulator contract, unsupported contract version, unavailable provider library, and
  unreachable HTTP simulator fail with actionable messages.
- Legacy mock backend preserves existing `MockResponse` behavior for replay-only tests.

### IGC-RF-01: Manifest-First Corpus Loading

Add corpus-manifest arguments and manifest loading. Do not change training defaults until the backend
protocol is green.

Files expected to change:

- Modify the shared argument parser where corpus flags are defined.
- Add `igc/ds/sources/redfish_ctl_manifest.py` for schema validation and recursive materialized
  corpus discovery.
- Extend `igc/ds/sources/redfish_fixture_source.py` only if legacy compatibility requires it.
- Add tests under `tests/ds/`.

Required tests:

- Manifest JSON validates required fields and rejects unknown major schema versions.
- Recursive discovery skips manifest/control JSON and includes only observation resources declared
  by the provider manifest.
- `@odata.id` or provider mapping wins over filename-derived URLs.
- `rest_api_map.v1.json` is preferred; `.npy` fallback is explicitly marked legacy in the returned
  provenance.
- Vendor, model, corpus id, and corpus kind filters are deterministic and visible in run metadata.

### IGC-RF-02: Mutation-Capable Redfish Gym Environment

Introduce a Redfish Gym environment that uses structured actions and backend-declared capabilities.
The old one-hot env remains available until parity tests and trainer cutover land.

Files expected to change:

- Create `igc/envs/redfish_env.py` or a narrower package equivalent.
- Add a `ToolCatalog.available_actions(state)` implementation backed by provider capabilities.
- Add structured action encode/decode helpers for method, target or capability id, JSON payload,
  expected observation, and goal verifier hint.
- Add tests under `tests/envs/redfish/`.

Required tests:

- Legal actions and argument domains come only from backend capabilities.
- Payload-bearing actions are executed with their JSON payload intact.
- Unsupported methods, invalid payloads, and undeclared capabilities return deterministic simulator
  failures without mutating state.
- `terminated` and `truncated` are preserved separately in Gym step results.
- Goal completion is computed from post-action state, not from HTTP success alone.

### IGC-RF-03: Reset, Seeding, and Vector Isolation

Make every environment reset call backend reset and prove that vectorized episodes are independent.

Files expected to change:

- Modify the new Redfish env reset path.
- Add vector-env fixtures under `tests/envs/redfish/`.

Required tests:

- `reset(seed)` resets provider state, overlays, replay-visible history, and RNG.
- Two vectorized envs with different seeds do not share mutation overlays or replay history.
- Reusing one training episode with a mutated goal is rejected; goal relabeling must create replay
  samples, not mutate the live episode.
- Failed writes do not leak partial state into the next reset.

### IGC-RF-04: ISO, BIOS, Boot, and Reset Workflow Gate

Add one complete chained workflow against both in-process and HTTP backends. This is the first
end-to-end Redfish mutation acceptance gate.

Workflow:

1. Insert or mount virtual media ISO.
2. Verify virtual media state.
3. Stage BIOS settings.
4. Verify pending settings and live settings are distinct.
5. Arm one-time boot override.
6. Reset the system.
7. Verify BIOS convergence and one-time boot consumption.
8. Reset the episode and verify the original simulator state is restored.

Required tests:

- Correct ordering succeeds in both backend modes.
- Out-of-order writes fail deterministically or remain pending according to provider semantics.
- Unmatched actions fail without mutation.
- Failed writes do not alter unrelated resources.
- HTTP and in-process responses have equivalent normalized response fields and Gym `info` metadata.

### IGC-RF-05: Scale and Concurrency Gate

Benchmark simulator throughput and state isolation before large RL rollout jobs use the backend.

Files expected to change:

- Add `scripts/bench_redfish_backend.py`.
- Add lightweight tests for benchmark parsing and safety defaults.
- Add an opt-in slow benchmark marker for local runs and a separate cluster runbook entry.

Required gates:

- Thousands of in-process requests complete without response corruption, leaked state, or reset
  drift.
- A smaller HTTP parity run proves process-boundary behavior matches the in-process backend.
- Vectorized training smoke runs with independent backend sessions per env.
- Benchmark output includes request throughput and p50/p95/p99 latency for GET, valid mutation,
  invalid mutation, reset, and workflow episode.

### IGC-RF-06: Version Negotiation, Docs, and CI Pins

Make the provider contract explicit and visible in every Redfish RL run.

Files expected to change:

- Add schema validation docs for the simulator contract and corpus manifest.
- Pin the consumed `redfish_ctl` revision in CI or a dedicated provider-contract check.
- Document the exact provider materialization command once the provider PR exposes it.
- Link the upstream provider issue or PR that introduces the simulator and manifest contracts.

Required tests:

- Unsupported simulator-contract versions fail before an episode starts.
- Unsupported corpus-manifest versions fail before dataset loading starts.
- Run metadata records provider revision, corpus id, backend kind, and contract versions.
- CI can run offline with tiny provider fixtures and without live Redfish access.

## Test Matrix

| Gate | Default | Resources | Evidence |
| --- | --- | --- | --- |
| Backend contract unit tests | Yes | CPU only, no network | `pytest -q tests/envs/redfish` |
| Manifest schema and filtering tests | Yes | CPU only, tiny fixtures | `pytest -q tests/ds` focused subset |
| Legacy mock compatibility | Yes | CPU only | Existing `MockServer` tests plus adapter tests |
| In-process workflow | Yes after provider fixture lands | CPU only, provider fixture | Workflow pytest |
| HTTP workflow parity | Opt-in until CI fixture exists | Local simulator process | Marked slow/integration |
| Throughput benchmark | Opt-in | Lab or developer machine | Benchmark JSON plus p50/p95/p99 report |
| Live Redfish canary | Never default | Approved non-production host only | Manual approval and run log |

## Acceptance Criteria

- No default Redfish RL test depends on `~/.json_responses`.
- No independent Redfish mutation semantics remain in IGC.
- Payload-bearing actions are represented and validated.
- Every reset resets simulator state and vector episodes are isolated.
- The ISO, BIOS, boot, and reset workflow passes in both in-process and HTTP backend modes.
- In-process and HTTP backends are equivalent at the normalized-response layer.
- Provenance identifies exact corpus id, provider revision, simulator contract version, and manifest
  version.
- The thousands-request benchmark passes without response corruption, state leakage, or reset drift.

## Non-Goals

- Do not run live Redfish crawls as part of this plan.
- Do not import provider tests, sandbox mock servers, or unversioned helper code.
- Do not make simulator-only success a substitute for held-out corpus evaluation or approved canary
  evidence.
- Do not remove the legacy mock server until the compatibility tests and non-Redfish replay tests no
  longer depend on it.
