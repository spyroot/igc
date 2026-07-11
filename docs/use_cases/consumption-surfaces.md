# Consumption surfaces — how you actually drive IGC

One agent, several front doors. The same goal-conditioned policy sits behind all of them; what
changes is who is holding the goal and how much they want to watch. This page shows each surface with
a worked interaction, and is explicit about what exists **today** versus what is **target**.

> **Where the code is today.** The only surface that runs now is the **training / evaluation CLI**
> (entry scripts `igc_main.py` and `igc_ctl.py`, argument groups in `igc/shared/shared_arg_parser.py`)
> — you train and evaluate the policy offline against the mock REST environment. The *goal-driven*
> surfaces below (interactive terminal, UI, SDK, daemon) are the **target product** built on the
> Phase-0 agent described in [`../ARCHITECTURE.md`](../ARCHITECTURE.md). Each is labelled so you never
> mistake a mock-up for a shipped feature.

## Surface 0 — Training & evaluation CLI  ·  **implemented today**

Before there is an agent to drive, you train and gate it. This is the real, current surface:

```bash
# Train the goal-conditioned policy offline against the mock REST env (CPU smoke path).
python igc_main.py --action_trainer agent --head pointer --model_type gpt2 ...

# Default offline gate — no GPU, no network, no live host.
pytest -q
```

`--action_trainer` selects what is trained (`agent`, `llm`, `all`), `--head` selects the action head
(`pointer` per D-001, or the legacy `onehot`). Everything downstream assumes a policy produced here.

## Surface 1 — Interactive terminal / CLI  ·  **target**

The operator's default. You type a goal in plain language; the agent runs the
observe→choose→dry-run→execute→verify loop from [`anatomy-of-an-episode.md`](anatomy-of-an-episode.md),
pausing for approval on anything that mutates hardware:

```
$ igc run --host bmc-a --goal "PXE-boot once, then power on" --approve mutating

  ↳ discovered: ComputerSystem = /redfish/v1/Systems/Self  (Supermicro, BMC fw 1.4)
  ↳ plan (learned, 2 steps):
      1. PATCH  Boot → {Pxe, Once}         risk: low
      2. POST   ComputerSystem.Reset {On}  risk: HIGH (power change)

  step 1/2  DRY-RUN  PATCH /Systems/Self  Boot: None→Pxe, Disabled→Once
            apply? [y/N] y
            ✓ applied · verified Boot.override_target=Pxe

  step 2/2  DRY-RUN  POST /Systems/Self/Actions/ComputerSystem.Reset {ResetType: On}
            apply? [y/N] y
            ✓ applied · verified PowerState=On

  GOAL REACHED ✓  (spec satisfied: 3/3 · 1 reset used · 2 actions · 0 hallucinated calls)
```

Flags that matter: `--approve read-only|mutating|all` sets where the guardrail pauses; `--dry-run`
runs the whole trajectory without executing; `--explain` prints the candidate scores behind each
choice; `--host` / a hosts file targets one or many controllers. Credentials come from the
environment, never the command line.

## Surface 2 — UI dashboard  ·  **target**

The same run, for an operator who wants to *watch* the fleet and click the approvals. Text wireframe:

```
┌ IGC ─ goal ───────────────────────────────────────────────┐
│ Goal:  [ PXE-boot once, then power on            ] [ Run ] │
│ Hosts: ◉ bmc-a (Supermicro)  ◉ bmc-b (Dell)  ○ bmc-c (HPE) │
├ live trajectory ── bmc-a ──────────────────────────────────┤
│  ● observe   /Systems/Self  PowerState: Off                │
│  ● choose    PATCH Boot {Pxe, Once}      score ▓▓▓▓▓▓░ 0.86 │
│  ⧗ approve   [ Dry-run ▾ ]  risk: low     [ Approve ][ Skip]│
│  ○ execute   …                                             │
│  ○ verify    spec 2/3                                      │
├ why this action ───────────────────────────────────────────┤
│  top candidates:  PATCH Boot 0.86 · GET BIOS 0.41 · Reset  │
│  0.38  — staging the pending boot setting before the reset │
│  keeps the 1-reset budget (learned)                        │
└────────────────────────────────────────────────────────────┘
```

The UI's job is exactly the three things a chatbot cannot give an operator: **a live, real trajectory**
(not a generated story), **the candidate scores** behind each decision (auditable "why"), and **an
approval gate** on every mutation. Success is a green check because the evaluator re-read the resource,
not because a model said so.

## Surface 3 — Programmatic SDK + GitOps / desired-state  ·  **target**

For automation. Embed the agent, or express infrastructure as a desired **goal spec** in version
control and let the agent reconcile reality to it:

```python
from igc import Agent, Goal

agent = Agent.load("policy.ckpt")          # a trained policy
result = agent.reach(
    host="bmc-a",
    goal=Goal(instruction="set boot to PXE once and power on",
              spec={"boot.override_target": "Pxe", "power_state": "On"}),
    approve="mutating",                    # or a callback / auto for trusted lanes
)
assert result.verified                     # measured against the spec, not self-reported
print(result.actions, result.rewards)      # the exact trajectory, for the audit log
```

```yaml
# desired-state.yaml — checked into git, reconciled by IGC in CI
- host: bmc-a
  goal: { instruction: "boot PXE once, power on",
          spec: { boot.override_target: Pxe, power_state: On } }
```

Because the outcome is **verifiable**, this composes with GitOps the way a chatbot never could: the
pipeline can *gate* on `result.verified`, and a run that could not reach the spec fails loudly instead
of reporting a confident, unchecked "done".

## Surface 4 — Autonomous fleet reconciler  ·  **target**

The end state: a daemon that continuously drives a heterogeneous fleet toward a set of goals, opening
an approval only when a mutation crosses a configured risk threshold, and logging every verified
transition. This is [`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md)
running on a schedule. It is only trustworthy *because* of the properties the other pages establish —
grounded actions, verified success, guarded execution — which is why this surface is the last to
build, not the first.

## The through-line

| Surface | Who holds the goal | What they watch | Status |
| --- | --- | --- | --- |
| Training/eval CLI | ML engineer | metrics, the offline gate | **today** |
| Interactive terminal | operator | the live run, approves inline | target |
| UI dashboard | operator / team | trajectories + scores + approvals | target |
| SDK / GitOps | automation | `result.verified`, the audit trail | target |
| Autonomous reconciler | the fleet, on a schedule | risk-gated approvals, logs | target |

Every surface rests on the same guarantee: **a goal in, a *verified* real-world outcome out, with
every consequential step inspectable.** The surface is just how close you want to stand.

Author:
Mus mbayramo@stanford.edu
