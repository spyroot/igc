# Consumption surfaces — how you actually drive IGC

One pipeline, several front doors. The same stages sit behind all of them — Phase 2 unordered
`rest_api_list` extraction, Phase 3 unordered `calls: list[Call]` binding, the separate RL policy
executing with guarded writes — what changes is who is holding the goal and how much they want to
watch. This page shows each surface with a worked interaction, and is explicit about what exists
**today** versus what is **target**.

> **Where the code is today.** The surfaces that run now are the **training / data-build / eval
> CLIs**: the phase builders and launchers described in [`../phase_2.md`](../phase_2.md),
> [`../phase_3.md`](../phase_3.md), and [`../TRAINING.md`](../TRAINING.md) (entry scripts
> `igc_main.py` and `igc_ctl.py`), plus the offline gate. The *goal-driven* surfaces below
> (interactive terminal, UI, SDK, daemon) are the **target product** built on the pipeline in
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md). Each is labelled so you never mistake a mock-up for a
> shipped feature. All transcripts are illustrative; the machine-readable contract
> (`configs/contracts/*.yaml`) is authoritative.

## Surface 0 — Training, data-build & evaluation CLI  ·  **implemented today**

Before there is an agent to drive, you build the datasets, train the phase models, and gate them:

```bash
# Default offline gate — no GPU, no network, no live host.
pytest -q

# Phase 2 labelled-requests build (see ../phase_2.md for the spec + provider modes).
python scripts/build_phase2_labelled_requests.py --spec configs/phase2_labelled_requests.yaml

# Phase 1/2/3 training launches are spec-driven — see ../TRAINING.md.
```

Everything downstream assumes checkpoints produced here: `model_x` (Phase 1), `goal_extractor`
(Phase 2), `argument_extractor` (Phase 3), and the separate RL policy trained against the JSON
simulator over captured Redfish data.

## Surface 1 — Interactive terminal / CLI  ·  **target**

The operator's default. You type a goal in plain language; the pipeline extracts the unordered API
set and calls; the RL policy runs the observe → choose → dry-run → execute → verify loop from
[`anatomy-of-an-episode.md`](anatomy-of-an-episode.md), pausing for approval on anything that
mutates hardware:

```
$ igc run --host bmc-a --goal "PXE-boot once, then power on" --approve mutating

  ↳ phase 2:  rest_api_list = [ /Systems/{id}, /Systems/{id}/Actions/ComputerSystem.Reset ]  (unordered)
  ↳ phase 3:  calls = [ {PATCH Boot {Pxe, Once}}, {POST Reset {On}} ]                        (unordered)
  ↳ discovered: ComputerSystem = /redfish/v1/Systems/Self  (Supermicro, BMC fw 1.4)
  ↳ RL policy chose execution order: PATCH first (stays within 1 reset)

  step 1/2  DRY-RUN  PATCH /Systems/Self  Boot: None→Pxe, Disabled→Once   risk: low
            apply? [y/N] y
            ✓ applied · verified by re-read: Boot.BootSourceOverrideTarget=Pxe

  step 2/2  DRY-RUN  POST /Systems/Self/Actions/ComputerSystem.Reset {ResetType: On}  risk: HIGH
            apply? [y/N] y
            ✓ applied · verified by re-read: PowerState=On

  GOAL REACHED ✓  (verified by re-read · 1 reset used · 0 out-of-contract calls)
```

Flags that matter: `--approve read-only|mutating|all` sets where the guardrail pauses; `--dry-run`
runs the whole trajectory without executing; `--host` / a hosts file targets one or many
controllers. Credentials come from the environment, never the command line.

## Surface 2 — UI dashboard  ·  **target**

The same run, for an operator who wants to *watch* the fleet and click the approvals. Text
wireframe:

```
┌ IGC ─ goal ───────────────────────────────────────────────┐
│ Goal:  [ PXE-boot once, then power on            ] [ Run ] │
│ Hosts: ◉ bmc-a (Supermicro)  ◉ bmc-b (Dell)  ○ bmc-c (HPE) │
├ extracted contract ────────────────────────────────────────┤
│  rest_api_list (unordered): /Systems/{id} · …/Reset        │
│  calls (unordered): PATCH Boot{Pxe,Once} · POST Reset{On}  │
├ live trajectory ── bmc-a ──────────────────────────────────┤
│  ● observe   /Systems/Self  PowerState: Off                │
│  ● choose    PATCH Boot {Pxe, Once}   (RL-chosen order)    │
│  ⧗ approve   [ Dry-run ▾ ]  risk: low   [ Approve ][ Skip] │
│  ○ execute   …                                             │
│  ○ verify    re-read pending                               │
└────────────────────────────────────────────────────────────┘
```

The UI's job is exactly the three things a chatbot cannot give an operator: **a live, real
trajectory** (not a generated story), **the extracted contract and the RL policy's chosen step**
behind each decision (auditable "why"), and **an approval gate** on every mutation. Success is a
green check because the evaluator re-read the resource, not because a model said so.

## Surface 3 — Programmatic SDK + GitOps / desired-state  ·  **target**

For automation. Embed the agent, or express infrastructure as a desired goal in version control and
let the agent reconcile reality to it:

```python
# Illustrative target API — not shipped.
result = agent.reach(
    host="bmc-a",
    goal="set boot to PXE once and power on",
    approve="mutating",                    # or a callback / auto for trusted lanes
)
assert result.verified                     # measured by re-reading state, not self-reported
print(result.rest_api_list)                # the Phase 2 unordered set that was extracted
print(result.calls)                        # the Phase 3 unordered calls
print(result.trajectory)                   # the RL-executed order, for the audit log
```

```yaml
# desired-state.yaml — checked into git, reconciled by IGC in CI (illustrative)
- host: bmc-a
  goal: "boot PXE once, power on"
```

Because the outcome is **verifiable**, this composes with GitOps the way a chatbot never could: the
pipeline can *gate* on `result.verified`, and a run that could not reach the goal fails loudly
instead of reporting a confident, unchecked "done".

## Surface 4 — Autonomous fleet reconciler  ·  **target**

The end state: a daemon that continuously drives a heterogeneous fleet toward a set of goals,
opening an approval only when a mutation crosses a configured risk threshold, and logging every
verified transition. This is [`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md)
running on a schedule. It is only trustworthy *because* of the properties the other pages establish —
contract-grounded calls, verified success, guarded execution — which is why this surface is the last
to build, not the first.

## The through-line

| Surface | Who holds the goal | What they watch | Status |
| --- | --- | --- | --- |
| Training/data/eval CLI | ML engineer | metrics, the offline gate | **today** |
| Interactive terminal | operator | the live run, approves inline | target |
| UI dashboard | operator / team | contract + trajectory + approvals | target |
| SDK / GitOps | automation | `result.verified`, the audit trail | target |
| Autonomous reconciler | the fleet, on a schedule | risk-gated approvals, logs | target |

Every surface rests on the same guarantee: **a goal in, a *verified* real-world outcome out, with
every consequential step inspectable.** The surface is just how close you want to stand.

Author:
Mus mbayramo@stanford.edu
