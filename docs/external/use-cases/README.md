# IGC use cases — what the agent does, end to end

This directory is the **product north-star for IGC**: a concrete, human-readable picture of what the
Infrastructure Goal-Condition Reinforce Learner *does in the real world* once trained — the goals we
want it to reach, how an operator drives it, and why it is a **reinforcement-learning agent** rather
than a chatbot wrapped around an API.

Read the whole set and you should have one clear mental model: **you hand IGC a goal, and it finds
and executes the shortest safe sequence of real REST operations that provably reaches that goal on
*this specific* piece of hardware — discovering the action space, learning from experience, and
never inventing a call that does not exist.**

> **Status — read this first.** This is the **target** end-to-end behavior. Today the codebase is at
> "Phase 0": a Redfish MDP shell (mock REST env + state encoder + a goal-conditioned DQN/HER trainer)
> plus the module-by-module plan to make it real. The *mechanism* every page relies on — legal
> candidate actions, verified success, HER, multi-vendor transfer — is grounded in the current
> architecture: see [`architecture overview`](../architecture/overview.md) for the current-vs-target map and
> [`decisions`](../roadmap/decisions.md) for the accepted design decisions (D-001 action selection,
> D-002 candidate representation). Where a page describes a surface that does not exist yet (a UI, a
> daemon), it says so explicitly. Nothing here is a claim that a metric was hit — that is what the
> offline gate and [`math checks`](../research/math-checks.md) are for.

## The one-paragraph version

Cloud and datacenter hardware exposes itself over the **Redfish REST API** — a self-describing tree
of resources (systems, BIOS, storage, firmware, power, logs) where each resource advertises the HTTP
methods it legally allows. IGC treats operating that hardware as a **Markov Decision Process**: the
*observation* is what a Redfish `GET` returns, the *actions* are the REST operations the API actually
permits from here, and the *goal* is a machine-checkable target state. A goal-conditioned RL policy
learns, from real captured Redfish data, to pick the action sequence that reaches the goal in the
fewest, safest steps — and, because it is scored against structurally-encoded endpoints rather than
memorized ids, it carries that skill to **vendors and machines it has never seen** (Dell iDRAC,
Supermicro, HPE iLO, generic DMTF).

## Why this matters (the short version — full argument in the next file)

A modern LLM, prompted "power this server on via Redfish," will happily emit a URL and a JSON body.
On real hardware that is a **liability**: it does not know *this* host's actual resource tree, it can
invent an endpoint or method that does not exist, it cannot tell you whether the action *worked*, it
has no notion of the *shortest* or *safest* path, and it does not get better the next time. IGC is
built to remove every one of those failure modes by construction. That is the whole point, and it is
the subject of [`why-rl-not-an-llm.md`](why-rl-not-an-llm.md).

## How to read this directory

Start with the two conceptual pages, then pick the scenarios relevant to you:

| File | What it gives you |
| --- | --- |
| [`why-rl-not-an-llm.md`](why-rl-not-an-llm.md) | The core argument: what a prompted LLM cannot do here, and the five properties IGC guarantees instead. **Start here for the "aha".** |
| [`anatomy-of-an-episode.md`](anatomy-of-an-episode.md) | One goal, followed step by step through the MDP loop — observation, legal candidates, scored choice, guarded execution, verified reward. Makes the mechanism concrete. |
| [`consumption-surfaces.md`](consumption-surfaces.md) | How you actually drive IGC: terminal / CLI, UI dashboard, programmatic SDK + GitOps, and an autonomous fleet reconciler. Each with a worked interaction and an implemented-vs-target label. |
| [`uc-01-power-and-boot.md`](uc-01-power-and-boot.md) | Power on / off and one-shot boot override (e.g. PXE-boot once, then normal). The simplest end-to-end goal. |
| [`uc-02-bios-attribute-convergence.md`](uc-02-bios-attribute-convergence.md) | Drive BIOS/BMC attributes to a target state, handling *pending vs current* settings and apply-at-reboot. |
| [`uc-03-firmware-update.md`](uc-03-firmware-update.md) | A long-running task: push an update, then monitor the job/task phase to real completion. |
| [`uc-04-inventory-and-health.md`](uc-04-inventory-and-health.md) | The safe, read-only lane: crawl to a structured inventory + health roll-up, no mutation. |
| [`uc-05-storage-and-virtual-media.md`](uc-05-storage-and-virtual-media.md) | Multi-step configuration: mount virtual media / create a volume, verifying each precondition. |
| [`uc-06-fleet-remediation-multivendor.md`](uc-06-fleet-remediation-multivendor.md) | The transfer payoff: bring a *heterogeneous* fleet (Dell + Supermicro + HPE) to one desired state with a single policy. |
| [`uc-07-log-and-telemetry-triage.md`](uc-07-log-and-telemetry-triage.md) | Collect/clear service logs and read telemetry — bounded, auditable operational hygiene. |

## Who this is for

- **Operators / SREs** deciding whether to trust an autonomous agent against real BMCs — the guarded
  execution and verified-success model is designed for exactly that scrutiny.
- **ML / RL engineers** who want the problem framing (MDP, goal-conditioning, action space) before
  the module details in `docs/external/architecture/overview.md`.
- **Reviewers and newcomers** who want, in fifteen minutes, a true picture of where IGC is going and
  why the RL framing is load-bearing rather than decorative.

## The vocabulary these pages share

The first time each term appears in a page it is expanded, but here is the shared spine so nothing is
mysterious:

- **Goal** — a natural-language `instruction`, a machine-checkable `spec` (the success condition an
  evaluator can *verify*), optional `constraints`, and an optional ordered `plan`. Defined in
  `igc/core/types.py`.
- **Observation** — the structured result of a Redfish read (`GET`/`HEAD`): the resource body, its
  type, its links, its allowed methods, and freshness metadata.
- **Legal action catalog** — the *dynamic* set of actions available from the current state: an
  endpoint from the walked resource tree paired with an HTTP method that endpoint actually allows
  (its `allowed_methods`), plus optional typed argument slots. The policy only ever chooses from
  this set (D-001).
- **Evaluator** — the component that checks the observation against the goal's `spec` and returns
  success / a reward. Success is *measured*, never self-reported.
- **HER (Hindsight Experience Replay)** — the mechanism that turns a trajectory that missed its goal
  into useful learning by relabeling it with the goal it *did* reach.
- **Guardrail** — the `dry-run → approval → execute` path that stands between a chosen action and a
  real write to hardware.

Author:
Mus mbayramo@stanford.edu
