# UC-06 — Fleet remediation across vendors

> Target-loop status: this describes IGC's end-to-end target behavior, grounded in
> `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` (esp. D-002). Today the code is a Phase-0
> Redfish MDP shell — captured-data replay, a legacy one-hot action space, and smoke-only
> DQN/HER metrics. The transfer result below is the design's central bet, and its go/no-go
> experiment is honest (see the last section).

## The goal (in the operator's words, and the machine-checkable spec)

You have a rack of machines from several vendors and one desired state: every node reimages from
the network on its next boot, and every node is powered on. You do not want to write a Dell
script, a Supermicro script, an HPE script, and a "generic" script — you want to state the target
once.

```python
Goal(
    instruction="Bring every node to a reimage-ready state: persistent PXE boot, powered on.",
    spec={
        # machine-checkable, per ComputerSystem member, re-read after acting
        "Boot.BootSourceOverrideTarget":  "Pxe",
        "Boot.BootSourceOverrideEnabled": "Continuous",
        "PowerState":                     "On",
    },
    constraints=[
        "one ComputerSystem member per node (do not touch peers)",
        "no firmware update, no BIOS attribute writes, no virtual media",
        "BootSourceOverrideMode must stay a value the node advertises as allowable",
        "at most one power action per node",
    ],
    plan=None,  # the policy discovers the path per vendor; it is not scripted here
)
```

The spec is the contract. It is not "run these calls" — it is a set of predicates the Evaluator
re-reads from the live tree. The same three predicates apply to a Dell iDRAC node and to a generic
DMTF node whose trees do not look alike.

## Why a script or a chatbot struggles here

A per-vendor script hard-codes the one thing that differs most: the ComputerSystem member id —
`/redfish/v1/Systems/System.Embedded.1` on one box, `/redfish/v1/Systems/1` on another,
`/redfish/v1/Systems/Self` on a generic DMTF service, a bare UUID on a fourth. One new vendor, or
one node that enumerates its systems differently, and the script silently patches nothing or
patches the wrong member. A chatbot will happily emit a `BootSourceOverrideTarget` value or an
`Oem` field the node never advertised — a plausible string that the BMC rejects, or worse,
accepts into a stuck pending state. Neither one *verifies*: both report "done" from having issued
a call, not from re-reading that the state took.

## Observation and the legal actions

The agent's observation is a Redfish GET result, structured (`RedfishStateV0`): resource identity
(`@odata.id`, `@odata.type`, schema version), the `Boot` object, `PowerState`,
`Status.State`/`Status.Health`, the action surface (`Actions[*].target`, allowed methods,
`*@Redfish.AllowableValues`), and pending-vs-current settings.

The action is chosen only from the **legal catalog**: an endpoint that actually exists in the
walked tree, paired with a method that endpoint's own `allowed_methods` permits. There is no free
text — the agent cannot name an endpoint that was never walked or a method the resource forbids.
For a Supermicro node (`/Systems/1`) the catalog around this goal is:

| endpoint (walked)                                  | method | resource_type            | why it is legal                          |
|----------------------------------------------------|--------|--------------------------|------------------------------------------|
| `/redfish/v1/Systems`                              | GET    | `ComputerSystemCollection` | read-only lane, auto-proceeds          |
| `/redfish/v1/Systems/1`                            | GET    | `ComputerSystem`         | read-only lane, auto-proceeds            |
| `/redfish/v1/Systems/1`                            | PATCH  | `ComputerSystem`         | `Boot` is a writable property            |
| `/redfish/v1/Systems/1/Actions/ComputerSystem.Reset` | POST | `ComputerSystem`         | body carries `Actions[*].target`         |

Each candidate is encoded **structurally**, per D-002 v1: `endpoint_path_tokens` (ids normalized
but kept as trailing tokens), `http_method`, `resource_type` from `@odata.type`,
`child_relation_name` (the link that made the endpoint reachable), and `has_action_target` (the
Reset row is the only `True` here — sparse, so discriminative). Nothing in that encoding is the
literal member id `1`. That is the whole point of the next sections.

## The trajectory

Observe → choose → dry-run/approve → execute → observe → evaluate, abbreviated. Read lanes
auto-proceed; the two writes each stop at the guardrail.

**1. Observe the collection (read, auto).**

```
GET /redfish/v1/Systems
→ 200  { "Members": [ { "@odata.id": "/redfish/v1/Systems/1" } ] }
```

**2. Observe the member (read, auto).** The agent reads current state and the allowable set it
must stay inside.

```
GET /redfish/v1/Systems/1
→ 200  {
    "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
    "PowerState": "Off",
    "Boot": {
      "BootSourceOverrideEnabled": "Disabled",
      "BootSourceOverrideTarget":  "None",
      "BootSourceOverrideTarget@Redfish.AllowableValues": ["None","Pxe","Hdd","Cd","BiosSetup"]
    },
    "Actions": { "#ComputerSystem.Reset": {
      "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
      "ResetType@Redfish.AllowableValues": ["On","ForceOff","GracefulRestart"] } }
  }
```

`Pxe` is in the allowable set — the argument stage may legally fill it. Two predicates are unmet
(`Boot` target and `PowerState`), so two actions are needed.

**3. Choose the Boot PATCH → dry-run → approve → execute (mutating).**

```
# dry-run (no call issued): the rendered body the agent WILL send
PATCH /redfish/v1/Systems/1
{ "Boot": { "BootSourceOverrideEnabled": "Continuous", "BootSourceOverrideTarget": "Pxe" } }
--- guardrail: mutation on ComputerSystem, pauses for approval ---
# on approval, execute
→ 200  (settings accepted)
```

**4. Choose the power action → dry-run → approve → execute (mutating).**

```
POST /redfish/v1/Systems/1/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
--- guardrail: mutation, pauses for approval ---
→ 204
```

**5. Re-observe and evaluate.**

```
GET /redfish/v1/Systems/1
→ 200  { "PowerState": "On",
         "Boot": { "BootSourceOverrideEnabled": "Continuous",
                   "BootSourceOverrideTarget": "Pxe" } }
```

Now the second vendor, **same spec, different tree** — a generic DMTF service exposing its system
as `/redfish/v1/Systems/Self`, already powered on:

```
GET  /redfish/v1/Systems           → { "Members":[{"@odata.id":"/redfish/v1/Systems/Self"}] }
GET  /redfish/v1/Systems/Self      → PowerState "On"; Boot target "None"
PATCH /redfish/v1/Systems/Self     { "Boot": { "BootSourceOverrideEnabled":"Continuous",
                                                "BootSourceOverrideTarget":"Pxe" } }   # approve → 200
# PowerState already "On" → no Reset issued; the shorter path is the correct path
GET  /redfish/v1/Systems/Self      → Boot target "Pxe", PowerState "On"
```

Different member id, different starting state, different number of steps — one policy, one spec.
Dell iDRAC (`.../Systems/System.Embedded.1`, `Oem/Dell` present) and HPE iLO (`.../Systems/1`,
`Oem/Hpe` present) resolve the same way; the `Oem` sections are observed but never keyed on
(D-002 deliberately keeps vendor namespace out of the candidate features).

## What "done" means

The Evaluator does not trust that a PATCH returned 200 or that Reset returned 204. It re-reads the
ComputerSystem member and checks each predicate against the fresh body:
`Boot.BootSourceOverrideTarget == "Pxe"`, `Boot.BootSourceOverrideEnabled == "Continuous"`,
`PowerState == "On"`. A node is `done` only when all three hold on the re-read; a node whose PATCH
landed in a *pending* settings resource but has not applied is **not** done — the structured state
separates current from pending exactly so this cannot be self-reported away. Success is a measured
property of the tree, per node, or it did not happen.

## Constraints, risk, and the guardrail

Both writes are mutating and both pause. The `Boot` PATCH is low-to-moderate risk (it changes next
boot behavior, not firmware or BIOS attributes — those are excluded by constraint). The
`ComputerSystem.Reset` is higher risk: it changes the power state of a physical machine. So every
mutation runs the guardrail in order — **dry-run** renders the exact endpoint, method, and body;
**approval** is where the operator sees that body before any call touches the BMC; **execute**
only then. Read lanes (`GET /Systems`, the member GET, the re-read) carry no such pause and
auto-proceed. The `at most one power action per node` constraint bounds blast radius: a policy that
tries to Reset a node twice is illegal before it is ever risky.

## What transfers / what it learned

The transfer payoff is the reason this is RL and not four scripts. Because a candidate is encoded
by its **structure** — `path_tokens`, `method`, `resource_type`, `child_relation`,
`has_action_target` (D-002) — and *not* by a memorized member id, a `PATCH` on an unseen vendor's
`/Systems/<UUID>` lands near the `PATCH` on `/Systems/1` and `/Systems/Self` the policy already
knows: same `resource_type` (`ComputerSystem`), same method, same reachability relation. The
Evaluator's per-node verdict then feeds **HER**: a run that overshot (issued a Reset on a node that
was already On, or PATCHed a mode the node did not advertise) is relabeled by its *achieved* state,
so the policy is rewarded toward the shortest safe path — the `/Systems/Self` node above learns the
one-write solution rather than blindly replaying the two-write one.

Honest bar (D-002): structural similarity alone is **not** enough. The zero-shot go/no-go — a
frozen encoder with no learned projection, ranking each state's true graph neighbors — measured
**k=5 = 0.293 on the 1,499-node Supermicro corpus and 0.754 on the 167-node HPE iLO corpus**, both
under the ≥0.80 top-5 bar (NO-GO), though ~30× over random. The read: on a large host, global text
similarity fills the top-5 with look-alike sibling sensor leaves rather than true transitions. The
conclusion the repo commits to is that the **learned bilinear projection `sᵀ W c` is load-bearing,
not optional** — the transfer above is real only once `W` is behavior-cloning-trained on in-domain
(Supermicro) graph transitions and then re-passes this same harness zero-shot on held-out HPE. The
multi-vendor fixture corpora (`idrac`/`supermicro`/`hpe`/`generic`) shipped in the repo are exactly
the material for that go/no-go, and it is the gate before any RL training spend.


# Author: Mus mbayramo@stanford.edu
