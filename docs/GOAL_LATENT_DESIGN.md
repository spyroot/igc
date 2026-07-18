# Goal latent design

**This document supersedes the abandoned goal-latent design** (the earlier goal-reference /
shared sub-goal-latent system; see the superseded entries in [DECISIONS.md](DECISIONS.md)).
That design is dead; do not build against it. This file now defines only one thing: the
**latent boundary** between the Phase 2/3 extraction stages and the RL policy.

The machine-readable schema under `configs/contracts/*.yaml` (the contract files checked into this
repository) is **authoritative**. Every example in this document is **illustrative only**; when an
example and the schema disagree, the schema wins.

## The two latents

v1 uses **two separate encoders**. There is no shared latent space and no unified encoder in v1.

- **`z_rest`** — produced by the Phase 2 encoder — encodes the resolved, **unordered**
  `rest_api_list` selection (which REST operations the request maps to).
- **`z_method`** — produced by the separate Phase 3 encoder — encodes the method/argument
  **structure** of each selected operation (HTTP method plus which argument slots are bound).

Exact argument **values** (addresses, names, sizes, enum choices) stay **raw and outside both
latents**. They travel beside the latents as the compiled calls and raw argument bindings, and the
consumer reads them directly; neither encoder is asked to memorize or reconstruct concrete values.

Both latents feed a **separate RL policy stage**. The policy — not Phase 2/3 — learns ordering,
prerequisites, retries, waiting, recovery, hidden-state discovery, and error handling. Phase 2/3
contain no planner, no scheduler, and no curriculum.

## Encoder sources and the RL freeze

Each encoder initializes from the phase that learned its representation (locked in the
machine-readable contract, `configs/contracts/goal_latent.yaml`):

| Component | Source checkpoint | Why |
| --- | --- | --- |
| StateEncoder backbone | `model_x` (Phase 1) | stable JSON-aware representation of environment state |
| `z_rest` encoder | Phase 2 checkpoint (`goal_extractor`) | strongest representation of resources selected from human language |
| `z_method` encoder | Phase 3 checkpoint (`argument_extractor`) | strongest representation of methods/functions and argument structure |

During RL training:

- **StateEncoder parameters are frozen.**
- **Phase 2/3 (`z_rest`/`z_method`) parameters are frozen.**
- **Only the RL policy learns.**

Fine-tuning an encoder during RL, or initializing one from a different source than this table, is
non-compliant; the `latent.contract` gate fails a contract that changes either.

## Public latent interface

The interface the RL policy consumes (shapes as `[dim, ...]`; `max_operations` is the padded
per-request operation capacity from the contract schema):

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `z_rest` | `[batch, max_operations, d_rest]` | per-operation latent of the unordered REST selection |
| `z_method` | `[batch, max_operations, d_method]` | per-operation latent of method/argument structure |
| `operation_mask` | `[batch, max_operations]` | 1 for real operations, 0 for padding |

Beside the latents, the **exact compiled calls and raw argument bindings remain available** to the
policy, evaluator, and simulator. Latents are for conditioning and scoring; concrete execution and
verification always read the raw calls, never a latent decode.

## Contract examples (illustrative)

Phase 2 output is always `rest_api_list: list[str]`; Phase 3 output is always `calls: list[Call]`
with an explicit `http_method`, an `operation_name` (action/function name, `null` for plain REST verbs), and an `arguments` object (`{}` for reads). One operation is still a
list of length one — never a scalar or a scalar/list union. Both lists are **unordered sets**;
ordering, when an oracle knows it, is separate RL evidence recorded as `expert_call_order` (defined
in the contract schema), never an ordered Phase 2/3 label.

Generic examples:

**k=1** — "set x to 1":

```json
{"rest_api_list": ["/api/x"]}
```

```json
{"calls": [
  {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}}
]}
```

**k=2** — "set x to 1 and read z":

```json
{"rest_api_list": ["/api/x", "/api/z"]}
```

```json
{"calls": [
  {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}},
  {"rest_api": "/api/z", "http_method": "GET", "operation_name": null, "arguments": {}}
]}
```

**k=3** — "set x to 1, set y to 2, and read z":

```json
{"rest_api_list": ["/api/x", "/api/y", "/api/z"]}
```

```json
{"calls": [
  {"rest_api": "/api/x", "http_method": "PATCH", "operation_name": null, "arguments": {"x": 1}},
  {"rest_api": "/api/y", "http_method": "PATCH", "operation_name": null, "arguments": {"y": 2}},
  {"rest_api": "/api/z", "http_method": "GET", "operation_name": null, "arguments": {}}
]}
```

Redfish-shaped paths (e.g. `/redfish/v1/Systems/1`) appear in real data because **Redfish is the
current test environment** — the first proof environment for this contract — not because the
contract is Redfish-specific.

## v1 non-claims

Kept explicitly out of v1; each is a **later experiment**, not a current property:

- No shared latent space between `z_rest` and `z_method`.
- No unified/shared encoder across phases or vendors.
- No proven cross-OEM equivalence in latent space.
- No zero-shot universal-REST claim. Redfish is the first proof environment; generalization beyond
  it is unmeasured.

## Operational notes

Corpus staging, the `rest_api_map.npy` contract (`url_file_mapping` + `allowed_methods_mapping`,
loaded via `np.load(..., allow_pickle=True).item()`), the no-hardcoded-endpoints/credentials rule,
and the do-not-commit-datasets rule are authoritative in the team operating guides and the
`redfish_ctl` contract notes; this document does not restate them.

# Author: Mus mbayramo@stanford.edu
