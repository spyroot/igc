# UC-04 — Inventory & health discovery

> **Illustrative episode, not shipped code.** Current pipeline (authoritative in
> `docs/ARCHITECTURE.md`): D0 → Phase 1 → `model_x` → D1 → Phase 2 **unordered** `rest_api_list` →
> Phase 3 **unordered** `calls: list[Call]` → separate encoders `z_rest` + `z_method` → a
> **separate RL policy** (order/retry/wait/recovery) → JSON simulator. Contract examples are
> illustrative; `configs/contracts/*.yaml` is authoritative. Redfish is the first proof environment.

This is the on-ramp. Before IGC is ever allowed to change a machine, it earns trust by *reading* one
completely — every reachable component, every health field — and proving to an evaluator that it saw
the whole box and nothing is on fire. Every action in this scenario is a `GET` or `HEAD`, so the
guardrail auto-proceeds: no approvals, no risk, no writes. It is the safest possible way to watch an
RL policy learn to navigate a real Redfish tree — and the same walk shape is how the D0 corpus
itself was captured by `redfish_ctl` discovery.

## The goal, and what the pipeline extracts from it

The operator says: *"Walk this machine and give me a full inventory and a health summary — tell me
if anything is Critical."*

For a bounded read request, the language side extracts a read-only contract — e.g. the health
snapshot slice (illustrative, k=2; every `Call` is explicit and reads carry `arguments: {}`):

```jsonc
{ "rest_api_list": [
    "/redfish/v1/Chassis/{id}/Thermal",
    "/redfish/v1/Chassis/{id}/Power"
] }
{ "calls": [
    { "rest_api": "/redfish/v1/Chassis/{id}/Thermal", "http_method": "GET", "arguments": {} },
    { "rest_api": "/redfish/v1/Chassis/{id}/Power",   "http_method": "GET", "arguments": {} }
] }
```

The *full* inventory walk is different in kind: it is not a fixed call set but a traversal whose
frontier grows as links are discovered. That traversal — which link to follow next, when the graph
is exhausted — is the separate RL policy's read-lane behavior, learned in the JSON simulator over
captured trees. The run is bounded by hard operational constraints:

```yaml
# run constraints (operational, enforced by the environment — not part of the language contract)
methods_allowed: [GET, HEAD]     # read-only lane; mutation is out of scope
rate_limit_rps: 4                # pace the BMC; a naive crawl has knocked one offline
max_requests: 2000
```

## Why a script or a chatbot struggles here

A hand-written crawler hard-codes today's tree shape — add an NVMe drive, a second NIC, a vendor OEM
subtree, and it silently walks past the new `@odata.id` it never learned about. A chatbot will
happily *narrate* a healthy-looking summary from the first page it read and call it done, with no
guarantee it reached `Storage/Drives` at all. Redfish is a hypermedia graph, not a fixed API: the
only correct traversal is "follow every link you actually find," and the only honest health claim is
one backed by having read every node. Both the brittle script and the confident chatbot fail the
same way — they report completeness they never verified.

## Observation and the legal methods

The observation is a Redfish `GET` result: the resource body, its `@odata.type`, the `@odata.id`
links it exposes, and its `Status` block. Which methods are legal on which URL comes from the
captured interface's `allowed_methods_mapping` in `rest_api_map.npy` (the binding contract from
`redfish_ctl` discovery) — the policy cannot invent a URL or a verb. In this scenario every legal
action on the frontier is read-only:

| Endpoint (from a walked link) | Method | Why it is legal | Lane |
| --- | --- | --- | --- |
| `/redfish/v1/Systems/<id>` | `GET` | `@odata.id` from the `Systems` collection | read-only |
| `/redfish/v1/Systems/<id>/Processors` | `GET` | link in the System body | read-only |
| `/redfish/v1/Systems/<id>/Memory` | `GET` | link in the System body | read-only |
| `/redfish/v1/Systems/<id>/Storage` | `GET` | link in the System body | read-only |
| `/redfish/v1/Chassis/<id>/Thermal` | `GET` | link in the Chassis body | read-only |
| `/redfish/v1/Chassis/<id>/Power` | `GET` | link in the Chassis body | read-only |
| `/redfish/v1/Managers/<id>` | `HEAD` | reachability probe before a full read | read-only |

There is no mutating candidate on the frontier — the run's `methods_allowed` constraint filters the
surface to `GET`/`HEAD`, so the policy's entire action space is safe reads.

## The trajectory (RL read-lane execution)

Observe → choose → (dry-run is trivial for reads) → execute → observe → evaluate, iterated
breadth-first until the reachable graph is exhausted.

1. **Observe the root.** `GET /redfish/v1/` → the service root, exposing the `Systems`, `Chassis`,
   and `Managers` collections as `@odata.id` links.

2. **Follow a discovered link.**
   ```http
   GET /redfish/v1/Systems
   ```
   ```json
   {
     "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
     "Members": [ { "@odata.id": "/redfish/v1/Systems/Self" } ]
   }
   ```

3. **Dry-run auto-passes.** The guardrail classifies `GET` as read-only, so the dry-run → approval →
   execute pipeline collapses to immediate execution — no human pause.
   ```http
   GET /redfish/v1/Systems/Self
   ```
   ```json
   {
     "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
     "Status": { "State": "Enabled", "Health": "OK" },
     "Processors":  { "@odata.id": "/redfish/v1/Systems/Self/Processors" },
     "Memory":      { "@odata.id": "/redfish/v1/Systems/Self/Memory" },
     "Storage":     { "@odata.id": "/redfish/v1/Systems/Self/Storage" },
     "EthernetInterfaces": { "@odata.id": "/redfish/v1/Systems/Self/EthernetInterfaces" }
   }
   ```

4. **Expand the frontier.** Each newly seen `@odata.id` becomes a legal read. The policy descends
   into member resources:
   ```http
   GET /redfish/v1/Systems/Self/Processors/CPU1
   ```
   ```json
   {
     "@odata.type": "#Processor.v1_16_0.Processor",
     "Status": { "State": "Enabled", "Health": "OK" }
   }
   ```
   ```http
   GET /redfish/v1/Systems/Self/Storage/Ctrl0/Drives/Disk0
   ```
   ```json
   {
     "@odata.type": "#Drive.v1_18_0.Drive",
     "Status": { "State": "Enabled", "Health": "Warning" }
   }
   ```

5. **Cross to the physical view.** `Chassis` links carry `Power` and `Thermal` (`#Power.Power`,
   `#Thermal.Thermal`), whose sensor entries also expose `Status.State`/`Status.Health`. These are
   folded into the same health roll-up.

6. **Terminate on graph exhaustion.** When every reachable `@odata.id` has been read and no unread
   link remains on the frontier, the episode ends and control passes to the evaluator. Rate limiting
   (`rate_limit_rps`) paces the whole walk so the BMC is never hammered.

## What "done" means

Done is measured, never self-reported. The evaluator re-derives the answer from the crawled tree:

- **Completeness** — every expected component class (`Chassis`, `Systems`, `Managers`,
  `Processors`, `Memory`, `Storage`, `Drives`, `EthernetInterfaces`, `NetworkAdapters`, `Power`,
  `Thermal`) has at least one resource in the visited set, and the set of visited `@odata.id`s
  equals the set of links discovered (the frontier drained to empty). A missed subtree leaves a link
  unvisited and the assertion fails.
- **Health roll-up** — the evaluator aggregates `Status.Health` across every visited resource and
  computes the worst severity. A max-severity bar of `Warning` passes only if no component reported
  `Critical`. The `Warning` drive above keeps the run PASS-but-flagged; a single `Critical` fan or
  drive would fail the health assertion and surface exactly which resource caused it.
- **Status presence** — the run fails if any component lacked a `Status` block, catching partial or
  malformed reads rather than scoring them as healthy.

The output is a structured inventory (components keyed by canonical URI and `@odata.type`) plus a
health summary with the worst-severity component named — both reconstructed from observations, both
auditable.

## Constraints, risk, and the guardrail

Risk level: **none**. Every mutation count in this scenario is zero. The guardrail's dry-run →
approval → execute path exists to pause on writes; here the run constrains `methods_allowed` to
`GET`/`HEAD`, so there is no mutating action to gate and every step auto-proceeds. The only
operational hazard is *volume* — **an unpaced recursive crawl has knocked a live BMC offline** — so
`rate_limit_rps` and `max_requests` are the real safety controls, and they cap request rate, not
blast radius. This is precisely why inventory-and-health is the trust-building on-ramp: an operator
can watch IGC drive a controller end-to-end, verify the evaluator's completeness math, and build
confidence in the loop *before* a single mutating use case is ever enabled.

## What transfers / what it learned

The read-only walk is where the RL policy learns the **shape of a machine** cheaply and safely, and
that knowledge is exactly what mutation scenarios reuse. Two angles:

- **HER.** Even a "failed" crawl is a labelled success for the state it *did* reach. If the policy
  stops early having read only the compute subtree, HER relabels that trajectory as "achieve
  inventory-of-Systems" — a goal it did satisfy — so partial walks still yield gradient. Over
  episodes the policy learns a traversal that drains the reachable graph without redundant re-reads,
  guided by the completeness reward rather than a hand-tuned crawl order.
- **Cross-vendor structure.** Every conformant implementation exposes `Chassis`/`Systems`/
  `Managers` and `Status.State`/`Status.Health` — the standard schema is the natural transfer
  surface, and the repo's multi-vendor fixture corpora (Dell iDRAC, Supermicro, HPE iLO, generic
  DMTF) exist to *measure* how far that carries rather than assume it (the recorded evidence lives
  in `uc-06-fleet-remediation-multivendor.md`). The walked tree captured here is also exactly what
  later, riskier use cases draw their legal-method surface from (`rest_api_map.npy`) — read first,
  mutate never before you have.


# Author: Mus mbayramo@stanford.edu
