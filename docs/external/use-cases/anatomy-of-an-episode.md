# Anatomy of an episode — one goal, followed through the loop

The other pages tell you *what* IGC does. This one shows *how*, by walking a single goal through the
Markov Decision Process step by step. Read it once and the words "observation", "legal candidate",
"reward", and "guardrail" stop being abstract.

The example goal: **"PXE-boot this server once, then power it on."** It is deliberately small — two
real changes — so the loop is visible without drowning in detail.

> The transcript below is illustrative of the **target** decision loop described in
> [`architecture overview`](../architecture/overview.md). URLs and bodies follow standard Redfish/DMTF schemas;
> exact ids differ per vendor, which is precisely the point of step 1.

## The setup

```
goal = Goal(
    instruction = "PXE-boot this server once, then power it on",
    spec        = { "boot.override_target": "Pxe",
                    "boot.override_enabled": "Once",
                    "power_state": "On" },
    constraints = [ "no firmware changes", "at most one reset" ],
)
```

The `spec` is the contract the **evaluator** will check against reality at the end — three facts that
must be true of the actual hardware for the episode to count as success. The agent never gets to
"decide" it succeeded; the `spec` decides.

## Step 0 — Observe (a real read, not a memory)

The episode starts by reading the service root and the systems collection. The **observation** is
what comes back — here, that the machine's system member is `/redfish/v1/Systems/Self` (not `/1` — a
prompt would have guessed wrong):

```
GET /redfish/v1/Systems
→ 200  { "Members": [ { "@odata.id": "/redfish/v1/Systems/Self" } ] }
GET /redfish/v1/Systems/Self
→ 200  { "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
         "PowerState": "Off",
         "Boot": { "BootSourceOverrideTarget": "None", "BootSourceOverrideEnabled": "Disabled" },
         "Actions": { "#ComputerSystem.Reset": { "target": ".../Actions/ComputerSystem.Reset",
                                                 "ResetType@Redfish.AllowableValues": ["On","ForceOff","GracefulRestart"] } },
         "@odata.id": "/redfish/v1/Systems/Self" }
```

The state encoder turns this into a compact, goal-conditioned latent. Nothing has been changed yet.

## Step 1 — Enumerate the *legal* action catalog

From this observation the environment builds the **legal action catalog** — every (endpoint, method)
the API actually permits from here, discovered from the walked tree and each resource's
`allowed_methods`. Abbreviated:

| # | Endpoint | Method | Why it is legal |
| --- | --- | --- | --- |
| a | `/Systems/Self` | `PATCH` | resource allows PATCH; carries the `Boot` object |
| b | `/Systems/Self/Actions/ComputerSystem.Reset` | `POST` | action target advertised in the body |
| c | `/Systems/Self/BIOS` | `GET` | readable child resource |
| d | `/Managers/Self` | `GET` | reachable via links |
| … | … | … | … (tens to hundreds of entries) |

The policy will choose **only from this table**. There is no row for "invent a new URL"; option (b)'s
`ResetType` is constrained to the `AllowableValues` the resource itself published. This is property #1
from [`why-rl-not-an-llm.md`](why-rl-not-an-llm.md), made literal.

## Step 2 — Score and choose (the learned decision)

The goal-conditioned state latent scores every candidate (a pointer over the encoded candidates, D-001
/ D-002). The goal wants the boot override *set first*, then power on — and the learned value function
has seen that powering on before staging the boot override wastes the "at most one reset" budget. So
it ranks the `PATCH` to stage the boot override above the `Reset`:

```
chosen: (a) PATCH /Systems/Self   Boot={ BootSourceOverrideTarget: "Pxe", BootSourceOverrideEnabled: "Once" }
```

The argument stage fills the typed slots (`Pxe`, `Once`) from the resource's allowable values — again,
not free-text, but a choice among what the schema permits.

## Step 3 — Guardrail: dry-run → approval → execute

Before anything touches the BMC, the chosen action passes the **guardrail**. In dry-run it renders
exactly what would be sent and what it expects to change; depending on the configured mode it either
auto-proceeds (read-only or explicitly trusted) or waits for operator approval on a mutating call:

```
DRY-RUN  PATCH /redfish/v1/Systems/Self
         Boot.BootSourceOverrideTarget: None → Pxe
         Boot.BootSourceOverrideEnabled: Disabled → Once
         risk: low (no power change, reversible)   approve? [y/N]
```

This is the seam that makes IGC safe to point at real hardware: a consequential action is always
inspectable, and can always require a human "yes", before it executes.

## Step 4 — Act, then observe again

On approval the call is sent and the **next observation** is read back — the loop's ground truth:

```
PATCH /redfish/v1/Systems/Self → 200
GET   /redfish/v1/Systems/Self → 200  { "Boot": { "BootSourceOverrideTarget": "Pxe",
                                                   "BootSourceOverrideEnabled": "Once" },
                                        "PowerState": "Off", … }
```

## Step 5 — Evaluate (measured reward, not self-report)

The evaluator checks the new observation against the `spec`. Two of three facts now hold
(`override_target=Pxe`, `override_enabled=Once`); `power_state` is still `Off`. That is real, partial
progress → a shaped reward, goal **not yet** reached, episode continues.

## Steps 6–8 — The second action, the same loop

The catalog is rebuilt from the new state; the policy now ranks the `Reset` action highest (the boot
override is staged, so powering on satisfies the last fact within the one-reset budget); the guardrail
shows a **higher-risk** power change and asks for approval; on approval:

```
POST /Systems/Self/Actions/ComputerSystem.Reset  { "ResetType": "On" } → 204
GET  /Systems/Self → 200  { "PowerState": "On", "Boot": { …"Pxe","Once" } }
```

The evaluator re-checks: all three `spec` facts now hold → **goal reached, verified**. The episode
terminates with success, having used exactly one reset — honoring the constraint.

## What the agent learned from this one episode

- The full trajectory (states, chosen actions, rewards, the achieved goal) goes into the replay
  buffer. If the agent had, say, reset *before* staging the boot override and blown the reset budget,
  **HER** would relabel that trajectory with the goal it *did* achieve ("power on") so the failure
  still teaches something.
- Because the endpoints were encoded structurally, what it learned here ("stage the pending boot
  setting before the reset") transfers to a Dell or HPE machine whose URLs and ids differ — the
  subject of [`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md).

## The loop, in one breath

**Observe → enumerate legal actions → score & choose → dry-run/approve → execute → observe →
evaluate → (repeat until the spec is verifiably satisfied) → learn from the whole trajectory.** Every
scenario in this directory is this same loop with a different goal and a different slice of the
Redfish tree.

Author:
Mus mbayramo@stanford.edu
