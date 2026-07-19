# UC-04 — Inventory & health discovery

> Target loop, grounded in `docs/external/architecture/overview.md` and `docs/external/roadmap/decisions.md`. Today the code is a Phase-0 Redfish MDP shell; the read-only crawl, evaluator, and guardrail described here are the target behavior the shell is being built toward.

This is the on-ramp. Before IGC is ever allowed to change a machine, it earns trust by *reading* one completely — every reachable component, every health field — and proving to an evaluator that it saw the whole box and nothing is on fire. Every legal action in this scenario is a `GET` or `HEAD`, so the guardrail auto-proceeds: no approvals, no risk, no writes. It is the safest possible way to watch an RL agent learn to navigate a real Redfish tree.

## The goal (in the operator's words, and the machine-checkable spec)

The operator says: "Walk this machine and give me a full inventory and a health summary — tell me if anything is Critical."

That becomes a `Goal` whose spec is a **completeness + health assertion**, not a free-text report:

```python
Goal(
    instruction="Discover full inventory and roll up component health for this system.",
    spec={
        # completeness: every reachable component class was actually read
        "read_component_classes": [
            "Chassis", "Systems", "Managers",
            "Processors", "Memory",
            "Storage", "Drives",
            "EthernetInterfaces", "NetworkAdapters",
            "Power", "Thermal",
        ],
        "all_reachable_resources_read": True,   # no @odata.id link left unvisited
        # health: rolled up from Status.Health on every resource read
        "max_health_severity": "Warning",       # PASS iff no component reports "Critical"
        "require_status_present": True,          # every component exposes Status.State/Health
    },
    constraints={
        "methods_allowed": ["GET", "HEAD"],      # read-only lane — mutation is out of scope
        "rate_limit_rps": 4,                     # pace the BMC; a naive crawl has knocked one offline
        "max_requests": 2000,
    },
    plan=None,   # discovery is breadth-first over the hypermedia graph, not a fixed script
)
```

The spec is machine-checkable: "all reachable resources read" and "no Critical health" are things the evaluator can *measure* against the crawled tree, not claims the agent gets to assert about itself.

## Why a script or a chatbot struggles here

A hand-written crawler hard-codes today's tree shape — add an NVMe drive, a second NIC, a vendor OEM subtree, and it silently walks past the new `@odata.id` it never learned about. A chatbot will happily *narrate* a healthy-looking summary from the first page it read and call it done, with no guarantee it reached `Storage/Drives` at all. Redfish is a hypermedia graph, not a fixed API: the only correct traversal is "follow every link you actually find," and the only honest health claim is one backed by having read every node. Both the brittle script and the confident chatbot fail the same way — they report completeness they never verified.

## Observation and the legal actions

The agent's observation is a Redfish `GET` result: the resource body, its `@odata.type`, the `@odata.id` links it exposes, and its `Status` block. From each observed resource, the **legal action catalog** is built by construction — the endpoint comes from a link that was actually present in the walked tree, and the method comes from that endpoint's own `Allow` header (`allowed_methods_mapping`). The agent cannot invent a URL or a verb; it can only pick a `(endpoint, method)` pair that Redfish already advertised. In this scenario every advertised legal action on the frontier is read-only:

| Endpoint (from a walked link) | Method | Why it is legal | Lane |
| --- | --- | --- | --- |
| `/redfish/v1/Systems/<id>` | `GET` | `@odata.id` from the `Systems` collection | read-only |
| `/redfish/v1/Systems/<id>/Processors` | `GET` | link in the System body | read-only |
| `/redfish/v1/Systems/<id>/Memory` | `GET` | link in the System body | read-only |
| `/redfish/v1/Systems/<id>/Storage` | `GET` | link in the System body | read-only |
| `/redfish/v1/Chassis/<id>/Thermal` | `GET` | link in the Chassis body | read-only |
| `/redfish/v1/Chassis/<id>/Power` | `GET` | link in the Chassis body | read-only |
| `/redfish/v1/Managers/<id>` | `HEAD` | reachability probe before a full read | read-only |

There is no mutating candidate on the frontier — the goal's `methods_allowed` filters the catalog to `GET`/`HEAD`, so the policy's entire action space is safe reads.

## The trajectory

Observe → choose → (dry-run is trivial for reads) → execute → observe → evaluate, iterated breadth-first until the reachable graph is exhausted.

1. **Observe the root.** `GET /redfish/v1/` → the service root, exposing the `Systems`, `Chassis`, and `Managers` collections as `@odata.id` links.

2. **Choose from the legal catalog.** The policy scores the read-only candidates and picks the `Systems` collection.
   ```http
   GET /redfish/v1/Systems
   ```
   ```json
   {
     "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
     "Members": [ { "@odata.id": "/redfish/v1/Systems/Self" } ]
   }
   ```

3. **Dry-run auto-passes.** The guardrail classifies `GET` as read-only, so the dry-run → approval → execute pipeline collapses to immediate execution — no human pause.
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

4. **Expand the frontier.** Each newly seen `@odata.id` becomes a legal candidate. The agent descends into member resources:
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

5. **Cross to the physical view.** `Chassis` links carry `Power` and `Thermal` (`#Power.Power`, `#Thermal.Thermal`), whose sensor entries also expose `Status.State`/`Status.Health`. These are folded into the same health roll-up.

6. **Terminate on graph exhaustion.** When every reachable `@odata.id` has been read and no unread link remains on the frontier, the episode ends and control passes to the evaluator. Rate limiting (`rate_limit_rps`) paces the whole walk so the BMC is never hammered.

## What "done" means

Done is measured, never self-reported. The `Evaluator` re-derives the answer from the crawled tree and checks it against the spec:

- **Completeness** — every class in `read_component_classes` has at least one resource in the visited set, and `all_reachable_resources_read` holds iff the set of visited `@odata.id`s equals the set of links discovered (the frontier drained to empty). A missed subtree leaves a link unvisited and the assertion fails.
- **Health roll-up** — the evaluator aggregates `Status.Health` across every visited resource and computes the worst severity. `max_health_severity: "Warning"` passes only if no component reported `Critical`. The `Warning` drive above keeps the run PASS-but-flagged; a single `Critical` fan or drive would fail the health assertion and surface exactly which resource caused it.
- **Status presence** — `require_status_present` fails the run if any component lacked a `Status` block, catching partial or malformed reads rather than scoring them as healthy.

The output is a structured inventory (components keyed by canonical URI and `@odata.type`) plus a health summary with the worst-severity component named — both reconstructed from observations, both auditable.

## Constraints, risk, and the guardrail

Risk level: **none**. Every mutation count in this scenario is zero. The guardrail's dry-run → approval → execute path exists to pause on writes; here the goal constrains `methods_allowed` to `GET`/`HEAD`, so the legal catalog contains no mutating action and the guardrail auto-proceeds on every step. The only operational hazard is *volume* — an unpaced recursive crawl has knocked a live BMC offline — so `rate_limit_rps` and `max_requests` are the real safety controls, and they cap request rate, not blast radius. This is precisely why inventory-and-health is the trust-building on-ramp: an operator can watch IGC drive a real controller end-to-end, verify the evaluator's completeness math, and build confidence in the loop *before* a single mutating use case is ever enabled.

## What transfers / what it learned

The read-only crawl is where IGC learns the **shape of a machine** cheaply and safely, and that knowledge is exactly what mutation scenarios reuse. Two angles:

- **HER.** Even a "failed" crawl is a labelled success for the state it *did* reach. If the agent stops early having read only the compute subtree, HER relabels that trajectory as "achieve inventory-of-Systems" — a goal it did satisfy — so partial walks still yield gradient. Over episodes the policy learns the **shortest safe traversal** that drains the reachable graph without redundant re-reads, guided by the completeness reward rather than a hand-tuned crawl order.
- **Cross-vendor generalization.** Because candidates are built from `@odata.type`, containment relation, and the presence of an action target — never from vendor-specific URL tokens (see `docs/external/roadmap/decisions.md`, D-002) — a policy trained to walk Dell iDRAC trees transfers to Supermicro, HPE iLO, and generic DMTF stacks. Every conformant implementation exposes `Chassis`/`Systems`/`Managers` and `Status.State`/`Status.Health`; the standard schema *is* the transfer surface. The inventory learned here becomes the walked tree that later, riskier use cases draw their legal action catalog from — read first, mutate never before you have.


# Author: Mus mbayramo@stanford.edu
