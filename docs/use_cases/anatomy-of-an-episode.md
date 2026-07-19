# Anatomy of an episode — one goal, followed through the pipeline

The other pages tell you *what* IGC does. This one shows *how*, by walking a single operator request
through the current pipeline: **text → Phase 2 set → Phase 3 calls → separate encoders → RL
execution → verified success**. Read it once and the contract shapes stop being abstract.

> Everything below is **illustrative**. The machine-readable schema (`configs/contracts/*.yaml`,
> enforced by `igc/ds/rest_goal_contract.py`) is authoritative. The Redfish transcript reflects the
> **current test environment** — captured Redfish JSON replayed by the simulator; Redfish is the
> first proof environment, not a permanent ontology.

## The contract shapes first (generic, k = 1 / 2 / 3)

Phase 2 output is `rest_api_list: list[str]` — an **unordered unique set**. Phase 3 output is
`calls: list[Call]` — **unordered**, each `Call` an object with `rest_api`, an explicit
`http_method`, and an `arguments` object (`{}` for read-only). A single item is **still a length-1
list**; scalar shapes are forbidden. Generic examples:

```jsonc
// k=1 — "set x to 1"
{ "rest_api_list": ["/api/x"] }
{ "calls": [ { "rest_api": "/api/x", "http_method": "PATCH", "arguments": { "x": 1 } } ] }

// k=2 — "set x to 1 and read z"        ([A,B] == [B,A]: the set is UNORDERED)
{ "rest_api_list": ["/api/x", "/api/z"] }
{ "calls": [
    { "rest_api": "/api/x", "http_method": "PATCH", "arguments": { "x": 1 } },
    { "rest_api": "/api/z", "http_method": "GET",   "arguments": {} }
] }

// k=3 — "set x to 1, set y to 2, and read z"
{ "rest_api_list": ["/api/x", "/api/y", "/api/z"] }
{ "calls": [
    { "rest_api": "/api/x", "http_method": "PATCH", "arguments": { "x": 1 } },
    { "rest_api": "/api/y", "http_method": "PATCH", "arguments": { "y": 2 } },
    { "rest_api": "/api/z", "http_method": "GET",   "arguments": {} }
] }
```

**Order is absent on purpose.** If a known-good execution order exists, it is recorded as separate
RL-oracle evidence (`expert_call_order`) for the RL policy — never inside these JSON shapes.

## The example goal

**"PXE-boot this server once, then power it on."** Deliberately small — two real changes — so each
stage is visible.

## Stage 1 — Phase 2: which APIs (unordered set)

The Phase 2 extractor (`goal_extractor`, trained from D1 rows that `model_x` drafted and an
independent judge verified) maps the text to the APIs it names on this interface — a k=2 set:

```jsonc
{ "rest_api_list": [
    "/redfish/v1/Systems/{id}",
    "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset"
] }
```

Nothing here says which comes first. `[A, B] == [B, A]`.

## Stage 2 — Phase 3: method + arguments per API (still unordered)

The Phase 3 extractor (`argument_extractor`) binds each selected API to an explicit HTTP method and
an arguments object. Methods must be legal for the URL per `allowed_methods_mapping` in
`rest_api_map.npy` (the binding contract from `redfish_ctl` discovery); argument values are only
bound where the captured interface evidences them (`AllowableValues`, settings objects):

```jsonc
{ "calls": [
    { "rest_api": "/redfish/v1/Systems/{id}",
      "http_method": "PATCH",
      "arguments": { "Boot": { "BootSourceOverrideTarget": "Pxe",
                               "BootSourceOverrideEnabled": "Once" } } },
    { "rest_api": "/redfish/v1/Systems/{id}/Actions/ComputerSystem.Reset",
      "http_method": "POST",
      "arguments": { "ResetType": "On" } }
] }
```

## Stage 3 — Two separate encoders

`z_rest` encodes the resolved unordered API selection; `z_method` encodes the method/argument
structure. They are **separate latents in v1** — no shared space, no unified encoder — and the exact
argument **values** (`"Pxe"`, `"Once"`, `"On"`) stay **raw, outside both latents**, carried
alongside for execution.

## Stage 4 — The RL policy executes (this is where order lives)

The separate RL policy consumes `z_rest` + `z_method` plus the environment observation and decides
*sequencing*: here, that staging the boot override **before** the reset is the path that reaches the
goal within one reset. That knowledge comes from reward in the JSON simulator and from
`expert_call_order` oracle evidence — not from the Phase 2/3 JSON. The policy may also insert legal
reads, waits, and retries the target set never mentioned.

**Step 0 — Observe (a real read, not a memory).** The episode starts by reading state; the system
member turns out to be `/redfish/v1/Systems/Self` (not `/1` — a prompt would have guessed wrong):

```
GET /redfish/v1/Systems
→ 200  { "Members": [ { "@odata.id": "/redfish/v1/Systems/Self" } ] }
GET /redfish/v1/Systems/Self
→ 200  { "PowerState": "Off",
         "Boot": { "BootSourceOverrideTarget": "None", "BootSourceOverrideEnabled": "Disabled" },
         "Actions": { "#ComputerSystem.Reset": {
             "ResetType@Redfish.AllowableValues": ["On","ForceOff","GracefulRestart"] } } }
```

**Step 1 — Guardrail: dry-run → approval → execute.** Before any write touches a BMC, the chosen
call passes the guardrail. In dry-run it renders exactly what would be sent; a mutating call waits
for operator approval:

```
DRY-RUN  PATCH /redfish/v1/Systems/Self
         Boot.BootSourceOverrideTarget: None → Pxe
         Boot.BootSourceOverrideEnabled: Disabled → Once
         risk: low (no power change)            approve? [y/N]
```

**Step 2 — Act, then observe again.** On approval the call is sent and the next observation is read
back — the loop's ground truth:

```
PATCH /redfish/v1/Systems/Self → 200
GET   /redfish/v1/Systems/Self → 200  { "Boot": { "BootSourceOverrideTarget": "Pxe",
                                                   "BootSourceOverrideEnabled": "Once" },
                                        "PowerState": "Off" }
```

**Step 3 — Evaluate (measured, not self-reported).** The evaluator compares the re-read state to the
goal. Boot override staged; power still `Off`. Partial progress → the episode continues.

**Steps 4–5 — The second call, same loop.** The policy now issues the reset; the guardrail flags the
**higher-risk** power change and pauses again; on approval:

```
POST /Systems/Self/Actions/ComputerSystem.Reset  { "ResetType": "On" } → 204
GET  /Systems/Self → 200  { "PowerState": "On", "Boot": { …"Pxe","Once" } }
```

The evaluator re-checks against the re-read: all target facts hold → **goal reached, verified**. A
`204` on the POST was never the evidence; the re-read is.

## What the policy learned from this one episode

- The full trajectory (states, chosen calls, rewards, the achieved state) goes into the replay
  buffer. If the policy had reset *before* staging the boot override, **HER** would relabel that
  trajectory with the goal it *did* achieve ("power on") so the failure still teaches something.
- The ordering lesson ("stage the pending boot setting before the reset") lives in the RL policy —
  reinforced by reward and by `expert_call_order` evidence — not in the language contract, so the
  same Phase 2/3 outputs remain valid for any vendor's tree while execution adapts per machine.

## The loop, in one breath

**Text → Phase 2 unordered set → Phase 3 unordered calls → `z_rest` + `z_method` → RL policy
(observe → choose → dry-run/approve → execute → re-read → evaluate, repeat) → learn from the whole
trajectory.** Every scenario in this directory is this same pipeline with a different goal and a
different slice of the Redfish tree.

Author:
Mus mbayramo@stanford.edu
