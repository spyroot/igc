# Why an RL agent, and not "just prompt an LLM"

This is the single most important page in the set. If IGC were something a good prompt could do, it
would not be worth building. It is worth building because the thing operators actually need —
**reach a goal on real hardware, provably, in the fewest safe steps, and get better at it** — is
exactly the thing a language model, used as a language model, cannot give you.

To be clear up front: **IGC contains an LLM.** The backbone that encodes Redfish responses and helps
plan is a language model, and that is on purpose — nothing understands a messy JSON API body like a
model trained on language and code. The argument here is not "LLMs are bad." It is: **an LLM is the
right *prior* and the wrong *decision-maker*.** IGC uses the model for understanding and constrains
every *decision* to a reinforcement-learning agent operating in a real environment.

## The concrete failure: prompt a chatbot to run Redfish

Ask a strong model, with no tools: *"Power on the server at this BMC and set it to PXE-boot once."*
It will confidently produce something like:

```
POST /redfish/v1/Systems/1/Actions/ComputerSystem.Reset
{ "ResetType": "On" }
PATCH /redfish/v1/Systems/1
{ "Boot": { "BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once" } }
```

That looks right. On a specific machine it may be **wrong in ways that matter**, and the model has no
way to know:

- The system member may not be `/Systems/1`. On Supermicro or HPE it is frequently a different id
  (`/Systems/Self`, a UUID); the model **guessed** the URL.
- `ComputerSystem.Reset` may not allow `POST` from the current power state, or the resource may not
  advertise that action at all on this firmware. The model cannot see the host's actual
  `allowed_methods`.
- `BootSourceOverrideEnabled: "Once"` may need to be applied at the next reboot as a *pending*
  setting, not a live one — patching the live resource silently no-ops on some vendors.
- After sending both calls, the model has **no idea whether the server is actually powering on**. It
  will tell you "Done!" whether it worked or not.

None of these are prompt-engineering problems. They are structural: the model is reasoning about a
*generic* Redfish from training data, not about *this* controller's *current* state. Every extra bit
of confidence makes it more dangerous, because a wrong write to a BMC can knock real hardware
offline.

## The three ways to solve it, compared

| Property | Prompt an LLM | Hand-written script / runbook | **IGC (RL agent)** |
| --- | --- | --- | --- |
| Knows *this* host's real resource tree | No — reasons from a generic prior | Only what the author hard-coded | **Yes** — the observation *is* the live `GET` |
| Can invent a non-existent endpoint/method | **Yes, routinely** | No (but brittle if the API differs) | **No** — chooses only from the legal action catalog |
| Handles a new vendor / unseen machine | Guesses, often wrong | Breaks until someone rewrites it | **Yes** — endpoints scored by structure, not memorized ids |
| Finds the *shortest / safest* path | No notion of cost | The author's fixed path, optimal or not | **Yes** — learned value over whole trajectories |
| Knows whether it *succeeded* | Self-reports, unverifiable | Only what the author asserted | **Yes** — an evaluator checks the goal `spec` against the new state |
| Recovers from a failed step | No memory across attempts | Only pre-written error branches | **Yes** — learns ret/recovery from experience (HER) |
| Gets better over time | No | No | **Yes** — every episode is training signal |
| Safe to point at real hardware | No | Depends entirely on the author | **By design** — `dry-run → approval → execute` guardrail |

A hand-written script is honest but frozen: it encodes one operator's path for one vendor, and it
rots the moment the API, the firmware, or the vendor changes. An LLM is flexible but ungrounded. IGC
is built to be **both grounded and adaptive** — which is the combination the job actually requires.

## The five properties IGC guarantees (and why each kills a failure mode)

### 1. A grounded action space — it cannot hallucinate a call
At every state the environment exposes a **dynamic catalog of legal actions**: an endpoint drawn from
the *walked* Redfish resource tree, paired with an HTTP method from *that endpoint's* advertised
`allowed_methods`, plus optional typed argument slots. The policy scores and picks **only from this
set** — it has no way to emit a URL or method that the API did not offer. This is a deliberate design
decision (see [`decisions`](../roadmap/decisions.md), D-001): the alternative of "let the LLM generate
the action" was considered and **rejected**, both because it re-introduces hallucination and because
it breaks the offline TD/HER learning the agent depends on. Hallucinated endpoints are not filtered
out after the fact — they are **impossible to express**.

### 2. Verified success — the reward is measured, not self-reported
A goal carries a machine-checkable `spec` (`igc/core/types.py`). After each action, an **evaluator**
compares the new observation to that spec and returns success/reward. "Did the power state become
`On`?" is answered by *reading the resource*, not by asking the model how it thinks it did. This is
the anti-hallucination property that matters most operationally: **IGC's notion of "done" is a fact
about the hardware, not an opinion about the transcript.**

### 3. An optimal strategy — learned value over trajectories, not a greedy guess
Reaching a goal is usually multi-step (discover the system, check the current boot config, stage the
change, trigger the reset, confirm). IGC learns a **value function over whole trajectories** (DQN-style
targets), so it prefers the sequence that reaches the goal in the fewest, least-risky steps — and it
can tell that a locally-tempting action leads to a dead end. A prompt has no cost model; a script has
the author's fixed path whether or not it is optimal.

### 4. It learns *in the environment*, including from failure
Most attempts on real systems don't go perfectly the first time. **Hindsight Experience Replay**
turns a trajectory that missed its intended goal into training signal for the goal it actually
reached — so failed or partial episodes still make the agent better. This is learning *from the
consequences of real actions*, which is precisely what a stateless prompt cannot do.

### 5. It transfers — one policy, many vendors and machines
Because candidates are encoded by their **structure** (path tokens, HTTP method, resource type,
how the endpoint is reached, whether it carries an action target — see D-002) rather than by a
memorized id, a Supermicro or HPE URL the agent has never seen lands near the Dell URLs it has. IGC's
own go/no-go experiment is *zero-shot on a held-out vendor* over the repo's multi-vendor fixture
corpora. Generalization is the target being measured, not a hope. (That same experiment is honest
about the bar: representation similarity alone is not enough — the *learned* scoring is load-bearing,
which is exactly why this is trained RL and not a clever embedding lookup. See D-002's recorded
result.)

## Where the LLM still earns its place

The model is doing real work — just not the deciding:

- **Understanding** the observation: turning a nested, vendor-specific JSON body into a state the
  policy can use is a language/code problem the backbone is good at.
- **Goal extraction**: mapping a high-level `instruction` into atomic `GoalRef` targets and explicit
  dependency hints is where the model's world knowledge helps; the RL policy still learns execution
  order from reward.
- **Argument values**: choosing enum values for an action's typed slots draws on the model's grasp of
  what the fields mean.

The division of labor is the whole design: **LLM for language, RL for consequential decisions,
the real API for ground truth, an evaluator for the verdict.** Take any one of those away and you are
back to a confident guess. Keep all four and you have an agent an operator can actually turn loose on
hardware.

## The one-sentence takeaway

A prompted LLM answers *"what call would plausibly do this?"*; IGC answers *"what is the shortest,
verifiable, allowed sequence that provably reaches this goal on the machine in front of me — and how
do I do it better next time?"* Those are different questions, and only the second one is safe to
automate.

Author:
Mus mbayramo@stanford.edu
