# igc docs

Documentation is split by audience and publication boundary.

## External

Public-safe docs live under [`external/`](external/):

- [`external/architecture/`](external/architecture/) — architecture, MDP framing,
  goal latent design, and Redfish enum spaces.
- [`external/roadmap/`](external/roadmap/) — decisions and performance critical
  sections.
- [`external/phases/`](external/phases/) — Phase 1/2/3 contracts plus public
  training-optimization and RL-scaling plans.
- [`external/research/`](external/research/) — math checks and paper material.
- [`external/use-cases/`](external/use-cases/) — target Redfish operating
  scenarios and episode walkthroughs.
- [`external/diagrams/`](external/diagrams/) — standalone SVG diagrams.

## Internal

Private/operator docs live under gitignored `docs/internal/`. This is where lab
runtime facts, GPU access, Shared Brain/NV72 details, hot trackers, CI/lab
runbooks, node artifact notes, and other operational material belong.

Do not copy internal material into `docs/external/` without a scrub/review pass.
