# How to profile igc

The one-command way to measure every performance-critical path, find the dominant function, and
prove a change is faster (or catch a regression). Runs on **CPU, offline, no GPU** — so it works
on a laptop, on a training node while the GPU is busy, and in CI. The results and the decisions
behind them live in [CRITICAL_SECTIONS.md](CRITICAL_SECTIONS.md).

## The harness

`scripts/bench_hot_paths.py` times every hot stage over a real fixture corpus (data-gen path) or
realistic synthetic tensors (RL path).

```bash
# both sections, timing table only
python scripts/bench_hot_paths.py

# just the RL training path (no corpus needed — pure synthetic tensors)
python scripts/bench_hot_paths.py --section rl

# just the data-gen path over a specific corpus
python scripts/bench_hot_paths.py --section data --corpus tests/hpe_fixtures

# add cProfile: prints the top cumulative-time functions (the critical sections)
python scripts/bench_hot_paths.py --profile
```

Local runs use the project env (`conda activate igc-dev`, or call its Python directly). The
data-gen section reads fixture corpora from the checkout or a materialized `redfish_ctl` dataset; the
RL section needs nothing but the repo.

### Reading the output

A timing table (`stage  seconds`), then with `--profile` a cProfile block sorted by cumulative
time. The **first igc function** in that block with a large `cumtime` is your critical section —
if you cannot name it, the optimization work is not finished (TEAM_GUIDE rule).

## The budget tripwires

`tests/perf/` turns each measured number into a regression test. They are marked `@pytest.mark.perf`
and **excluded from the default gate** (`pytest.ini`), so they never flake a normal `pytest -q`.

```bash
# run all performance budgets
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest -m perf -q
# or
make perf
```

Two kinds of guard:

- **Absolute budgets** — a ceiling 50–100× looser than measured (e.g. all-node neighbors < 0.5 s
  per 1,000 nodes). Machine speed never trips them; an algorithmic regression always does.
- **Ratio tripwires** — where an absolute time is too machine-dependent, the test asserts one
  implementation stays N× faster than another (e.g. the D-002 key cache must stay ≥ 10× faster
  than re-projecting `B×N`). Invariant to runner speed.

## The Make targets

```bash
make gate          # ruff + the offline pytest gate (what CI runs)
make perf          # the budget tripwires
make profile       # bench_hot_paths.py --profile (both sections)
make profile-rl    # RL section only
make docker-test   # build the CPU test image and run pytest inside it
```

## In CI

`.github/workflows/ci.yml` runs on every push and PR to `main`:

- **`gate`** — `ruff check` + the offline `pytest -q` gate.
- **`perf`** — `pytest -m perf`; the budget tripwires above run on every change, so a hot-path
  regression fails the PR automatically, no human profiling required.
- **`docker`** — builds `docker/Dockerfile.test` and runs the gate inside it (and pushes the
  image to Docker Hub on `main` when the registry secrets are configured).

So profiling is not a thing someone remembers to do — the budgets are part of the merge gate.

## Comparing over time (tests, coverage, hot-path timings)

Because the codebase keeps growing, each CI run archives a **per-commit snapshot** so any two
points in history can be diffed instead of re-derived:

- The `gate` job runs the suite **with coverage** (`--cov=igc`, producing `coverage.xml` /
  `coverage.json`), then `scripts/metrics_snapshot.py` writes `metrics.json` —
  `{commit, num_tests, coverage_pct, hot_paths_sec}` — and echoes a table into the run's
  **summary page**. All three are uploaded as the `metrics-<sha>` artifact (90-day retention).
- So "did coverage drop or a hot path get slower since last week?" is: download the two runs'
  `metrics.json` and diff them.

Locally:

```bash
make coverage   # run the gate with coverage (term + coverage.xml + coverage.json)
make metrics    # coverage + write metrics.json and print the summary table
```

## Workflow for a performance change

1. `python scripts/bench_hot_paths.py --profile` → record the before number and the dominant
   function.
2. Make the change.
3. Re-run → record after; for an optimization, add an **output-equivalence check** (the fast and
   slow paths produce identical results on a real corpus — see PR #25 for the pattern).
4. Add/adjust a `tests/perf/` budget or ratio tripwire.
5. Update the table in [CRITICAL_SECTIONS.md](CRITICAL_SECTIONS.md).
6. PR with the before/after table pasted in.
