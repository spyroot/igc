# IGC as a Stochastic Shortest Path problem: existence and convergence of a recovery policy

This note gives the decision-theoretic formulation behind IGC and proves the property that justifies
using reinforcement learning to operate a management controller: **a convergent, Bellman-optimal
recovery policy is guaranteed to exist**, because the controller's own action space is self-healing.
It is written to be self-contained and citable.

## 1. Formulation

We model an agent operating a Redfish-exposed controller as a **Stochastic Shortest Path (SSP)** MDP

$$\mathcal{M} = (\mathcal{S},\ \{\mathcal{A}(s)\},\ P,\ c,\ \mathcal{G}).$$

- **States** $\mathcal{S}$: the (discrete, finite) Redfish resource-tree configurations an operator can
  read. A state is the collection of resource representations returned by `GET`/`HEAD`.
- **Legal actions** $\mathcal{A}(s)$: the *dynamic* catalog of operations the API permits from $s$ —
  an endpoint from the walked resource tree paired with an HTTP method from that endpoint's
  `allowed_methods`, with optional typed arguments. Actions the API does not expose are not in
  $\mathcal{A}(s)$ (the agent cannot emit an operation the controller did not offer).
- **Transition kernel** $P(s' \mid s,a)$: **unknown and stochastic**. The same action can succeed or
  fail with a probability the agent does not know a priori — a reboot that does not take, a request the
  controller rejects, a task that ends in `Exception`. This stochasticity (transient conditions,
  firmware/software faults) is a property of $P$, not of the observation.
- **Cost** $c(s,a) \ge \varepsilon > 0$: each operation has a strictly positive cost (a step budget /
  operational risk).
- **Goal set** $\mathcal{G} \subseteq \mathcal{S}$: the states satisfying the goal's machine-checkable
  specification. Goal states are cost-free absorbing states.

The objective is to reach $\mathcal{G}$ at minimum expected cumulative cost. (Equivalently, the
discounted-return formulation with $\gamma < 1$; see §5.)

Unknown, stochastic $P$ makes this a *model-free* problem — it does **not** make it partially
observed. The agent reads the true configuration; the outcome of an action is realized as a
transition to an observable next state. Residual latency windows (an operation genuinely in flight) are
covered by cheap sensing actions (polling a task) and a small number of history features and do not
change the formulation below.

## 2. Definitions

A stationary policy $\pi$ is **proper** if, from every state $s \in \mathcal{S}$, following $\pi$
reaches $\mathcal{G}$ with probability 1 in finite expected cost; otherwise it is **improper**.

Under the standard SSP assumptions — (i) at least one proper policy exists, and (ii) every improper
policy has infinite cost-to-go for at least one state — the optimal cost-to-go $J^\star$ is the unique
solution of Bellman's equation, value iteration converges to it, and there is an optimal policy that is
proper (Bertsekas & Tsitsiklis, 1991; 1996). With $c(s,a) \ge \varepsilon > 0$ and $\mathcal{G}$
cost-free absorbing, **(ii) holds automatically**: an improper policy never reaches $\mathcal{G}$ and
therefore accumulates unbounded positive cost. Hence the binding condition is **(i): a proper policy
exists.**

## 3. Proposition

> If a proper policy exists, then $J^\star$ is the unique fixed point of the Bellman optimality
> operator, value/Q-iteration (and thus tabular Q-learning / its function-approximation surrogate DQN)
> converges to it, and the greedy policy with respect to $J^\star$ is an optimal, proper — i.e.
> goal-reaching, **recovery** — policy.

By §2 it therefore suffices to establish that a proper policy exists.

## 4. A proper policy exists — proof by contradiction

**Claim.** For any controller that remains responsive to the management API, a proper policy exists.

**Proof.** Suppose not. Then there is a reachable state $s_\dagger$ from which **no** policy reaches
$\mathcal{G}$ with probability 1 — a permanent trap under every policy.

The action space is, however, **self-healing**. It contains not only operations on the managed system
(e.g. `ComputerSystem.Reset`, power control — themselves mediated by the controller) but the reset of
**the management controller itself**: `Manager.Reset` on `/redfish/v1/Managers/{id}`. Executing
`Manager.Reset` returns the controller to a clean, known baseline state $s_0$, from which the goal is
reachable (the original clean configuration from which any operator provisions the target).

Consider the escalating recovery procedure
$$\text{retry the action} \ \rightarrow\ \texttt{ComputerSystem.Reset} \ \rightarrow\ \texttt{Manager.Reset} \ \rightarrow\ \text{re-drive from } s_0 .$$
Each stage is an element of $\mathcal{A}$ for a responsive controller, and the terminal stage reaches
$s_0$ with positive probability; repeating the procedure reaches $s_0$, and thence $\mathcal{G}$, with
probability 1 in the limit. So $s_\dagger$ is **not** a permanent trap — a contradiction.

Therefore a proper policy exists. $\blacksquare$

**Remark (in-band only; no god action).** The recovery procedure above uses only actions in
$\mathcal{A}(s)$ — API calls with their true (stochastic) dynamics. It appeals to **no out-of-MDP
intervention**. A physical power-cycle ("unplug it at the socket") is a *god action*: an omnipotent
external reset of the state, outside $\mathcal{A}(s)$ by construction, because the MDP is defined at the
management-API layer. Admitting god actions would make the existence claim **vacuous** — every system
is trivially "recoverable" if an external agent may reset it to a clean state by fiat. The theorem is
meaningful precisely because recovery is achieved *within* the action space: the controller resets
**itself**.

**Operational corroboration.** Operators do recover controllers in practice. If no in-band recovery
path existed, any controller reaching $s_\dagger$ would be permanently unrecoverable — contradicting
the observed reality that management controllers are routinely returned to service. The self-healing
argument is the mechanism behind that observation: **the controller can reset itself.**

## 5. Boundary conditions (and why they do not weaken the result)

1. **Unresponsive controller.** The one genuinely out-of-band case is a controller so wedged that it
   does not answer the management API at all — `Manager.Reset` cannot be issued. Recovery there would demand a **god action** (a physical
   power-cycle) outside the MDP, which the agent cannot invoke. This is a single absorbing failure state $s_\perp \notin$ (responsive subset); it is often transient (a retry
   restores responsiveness). It has no effect on properness over the responsive subset on which the
   agent operates, and is modelled as an absorbing terminal (or a retry-until-timeout transition).
2. **Redfish-irrecoverable states.** If any state is reachable but not recoverable within the action
   space, the optimal policy's response is **avoidance** — it learns not to take the action that risks
   entering such a state. This is still Bellman-optimal, and it is a *safety* property: it is precisely
   the behaviour that avoids, for example, an operation sequence that drives a live controller offline.
3. **Discounted equivalent.** With $\gamma < 1$ the Bellman optimality operator is a
   $\gamma$-contraction, so $J^\star$ is unique and Q-learning/DQN converge regardless of properness
   (Watkins & Dayan, 1992); properness is then what guarantees the optimum is a *goal-reaching* policy
   rather than a degenerate avoid-everything one.

## 6. Role of HER, and the stochastic caveat

Existence and convergence are consequences of the SSP structure above; they do **not** depend on the
learning algorithm. Hindsight Experience Replay (HER) addresses a different axis — **sample
efficiency**: under sparse goal reward it relabels trajectories with achieved goals so the agent gets
dense signal for the recovery behaviours it would otherwise almost never discover. One caveat under the
injected stochastic failure of §1: naive HER relabeling assumes near-deterministic dynamics and can
credit an action for a goal it only *stochastically* caused (a lucky reboot); bias-corrected/dynamic
HER and seed-reproducible outcomes mitigate this.

## 7. Consequence

The recovery problem is a **well-posed SSP with a guaranteed optimal solution**: a proper, convergent
recovery policy exists because the controller can reset itself. A learned policy therefore converges to
the same kind of recovery strategy a skilled operator uses — retry, escalate, reset the system, reset
the controller, re-drive — **because such a strategy provably exists**. A fixed script cannot adapt to
unknown, stochastic failure, and a language model prompted for the next call has no notion of the
optimal recovery cost-to-go; the SSP formulation is what makes learned, verifiable recovery the right
tool.

## 8. Decidability and generalization

### Decidability — halt states are not the Halting Problem

The Halting Problem — does an arbitrary Turing machine halt on a given input — is undecidable because
the machine's tape, and hence its state space, is **unbounded**. $\mathcal{M}$ is a **finite** MDP:
under a task-relevant state abstraction (the finitely many properties that determine the goal predicate
and reachability) $|\mathcal{S}| < \infty$ and $|\mathcal{A}(s)| < \infty$. For a finite MDP,
reachability of $\mathcal{G}$ from any $s$ — and hence whether a state is a permanent trap, a "halt" —
is **decidable** by finite graph analysis / value iteration. A halt state is an ordinary absorbing
state, and identifying one is a well-posed, decidable computation, categorically unlike the Halting
Problem. What makes the two *look* alike — reachability of a target under dynamics — is exactly what the
finiteness of $\mathcal{S}$ dissolves.

Two consequences, stated plainly:

- **Decidable is not tractable.** $\mathcal{S}$ is finite but combinatorially enormous, so exact value
  iteration is infeasible; the problem is solved by **learning with function approximation**, not exact
  planning. Well-posedness (an optimum exists and is computable *given the model*) is what licenses
  learning to approximate it.
- **The agent has no model.** $P$ is unknown, so the agent does not *compute* reachability a priori — it
  *discovers* it through exploration (model-free). Decidability is a property of the problem, not a
  capability handed to the agent.

### Generalization beyond Redfish

Nothing in Sections 1-7 is Redfish-specific once stated abstractly. The argument transfers to operating
any system exposed as a discoverable API — the target framework in [architecture overview](../architecture/overview.md) —
under three conditions:

1. **Bounded, readable state** — a finite, Markov-sufficient abstraction of the system's configuration
   the agent can observe (not the raw byte space).
2. **Bounded, discoverable action space** — the operations the API exposes from each state.
3. **A proper policy exists** — a within-action-space recovery path to the goal (self-healing such as a
   controller/service reset, or in-band reachability by other means). Where (3) fails for some states,
   those states are decidably identified and the optimal policy is **avoidance**; no god action is assumed.

Under (1)-(3), $\mathcal{M}$ is a well-posed SSP with a unique optimal cost-to-go and a convergent,
Bellman-optimal, **autonomous** recovery policy — decidable in principle, intractable to solve exactly,
and therefore learned. This is the formal license for "one agent, many systems": the guarantees are
properties of the *shape* of the problem — finite state, finite actions, in-band recovery — not of Redfish.

Author:
Mus mbayramo@stanford.edu
