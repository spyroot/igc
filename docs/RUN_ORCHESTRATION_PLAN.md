# Run orchestration plan

Plan for making igc training runs launchable through one tested, spec-driven path, whether the
target runtime is Docker on a single node or Slurm on a scheduled cluster. This is a public-safe
engineering plan: it intentionally omits private hosts, credentials, operator-only files, and internal
service endpoints.

## Purpose

The repository already has useful launch pieces, including `scripts/gb300_launch.sh` for direct Docker
runs, `scripts/submit_train.sh` and `scripts/train_igc.sbatch` for Slurm, preflight wrappers,
checkpoint publishing, distributed sanity scripts, Bats shell tests, and CI jobs. The missing layer is
a single declarative contract that answers the operational questions before a run starts:

- which backend launches the run;
- which image is used, pulled, built, tagged, or pushed;
- where code, data, scratch, outputs, and checkpoints live;
- which sanity gates must pass before training;
- how dry-run output is rendered and tested;
- how live scheduler checks are kept opt-in and small.

The goal is to avoid ad hoc launcher edits and avoid rebuilding or redeploying a Docker image on every
training attempt. Code changes should be picked up by updating the mounted checkout or pulling a
versioned image according to the run spec.

## Non-goals

This document does not implement the launcher. It defines the desired script, config, and test plan for
future PRs.

Default tests remain offline. A real Slurm submission, Docker push, GPU training run, or live data
collection stays opt-in and must never run from the default unit-test path.

## Existing surfaces

The plan should consolidate these existing entrypoints instead of replacing them all at once:

- `scripts/gb300_launch.sh`, the existing direct-Docker node launcher.
- `scripts/submit_train.sh`, the existing Slurm submit helper.
- `scripts/train_igc.sbatch` and `scripts/train_m1.sbatch`, the existing scheduled training scripts.
- `scripts/preflight_nv72.sh`, the existing fleet-state preflight wrapper; the endpoint comes from
  the runtime environment.
- `scripts/publish_checkpoint.sh`, the existing checkpoint copier and checksum manifest helper.
- `scripts/gb300_sanity_check.sh`, `scripts/dist_sanity.py`, `scripts/dataset_sanity.py`,
  `scripts/rl_env_sanity.py`, and `scripts/nccl_smoke.py`, the current GPU and distributed sanity
  tools.
- `tests/bats/`, the shell-test suite that mocks command-line tools.
- `.github/workflows/ci.yml`, the current shell, Python, performance, and Docker CI split.

## Run spec

`IGC_RUN_SPEC`, a planned environment variable, points to a YAML file that defines one runnable job.
The schema should live at `configs/run/schema.yaml`, and committed examples should live under
`configs/run/examples/`. The examples must use placeholder paths and public-safe values.

Planned top-level keys:

| Key | Meaning |
| --- | --- |
| `backend` | `docker` or `slurm`; selected explicitly by the spec. |
| `image` | Image repository, tag or digest, pull policy, build context, push policy, and cache key. |
| `runtime` | Command, training profile, environment allowlist, and dry-run rendering options. |
| `paths` | Code checkout, data root, scratch root, output root, and artifact root. |
| `resources` | GPUs, nodes, CPUs, memory, wall time, and distributed mode. |
| `docker` | Docker-only mounts, user, IPC, ulimits, network mode, and env-file references. |
| `slurm` | Slurm-only partition, account, constraints, exclude/include rules, and `sbatch` options. |
| `data` | Data staging mode, source, target, read-only policy, expected files, and checksum policy. |
| `checkpoint` | Save interval, retention, resume source, publish destination, and manifest requirement. |
| `sanity` | Required preflight, dry-run, dataset, distributed, RL, and scheduler checks. |
| `observability` | Log directory, metrics backend, run name, and post-run bundle path. |

The schema validator should reject unknown keys by default. Exceptions must be deliberate, documented,
and covered by tests.

## Configuration rules

No private host, token, username, absolute operator path, or endpoint should be committed in a spec.
Private values come from the runtime environment or a gitignored operator file, and launch rendering
must print placeholders rather than secret values.

Named variables should have one source of truth:

- `IGC_RUN_SPEC` points to the YAML run spec.
- `IGC_SHARED_ROOT` names the shared artifact filesystem when a run has one.
- `IGC_NODE_SCRATCH` names node-local scratch space.
- `IGC_IMAGE_REF` names the resolved image tag or digest after validation.
- `IGC_RUN_ID` names the logical run and should be stable across curriculum stages.
- `IGC_OUTPUT_DIR` names the output directory created by the launcher.
- `IGC_CHECKPOINT_DIR` names the checkpoint directory created or resumed by training.

These names are planned launcher concepts unless a current script already defines them.

## Script roadmap

The implementation should land in small PRs. Each script below is planned unless it already exists.

| Phase | Script or file | Purpose |
| --- | --- | --- |
| 1 | `igc/shared/run_spec.py` | Parse, validate, normalize, and redact run specs with no shell side effects. |
| 1 | `configs/run/schema.yaml` | Machine-checkable contract for Docker and Slurm runs. |
| 1 | `configs/run/examples/docker-smoke.yaml` | Public-safe direct-Docker smoke spec. |
| 1 | `configs/run/examples/slurm-smoke.yaml` | Public-safe scheduled smoke spec. |
| 2 | `scripts/render_run.py` | Render the exact command plan for review without executing it. |
| 2 | `scripts/launch_run.py` | Dispatcher that reads `IGC_RUN_SPEC`, chooses `backend`, and calls the backend adapter. |
| 3 | `scripts/docker_run_from_spec.sh` | Docker adapter: pull or reuse image, mount paths, pass allowlisted env, run command. |
| 3 | `scripts/docker_image_sync.sh` | Image lifecycle: pull by digest/tag, build only when requested, tag, push, and report cache state. |
| 4 | `scripts/slurm_submit_from_spec.sh` | Slurm adapter: render `sbatch` options and submit from the validated spec. |
| 4 | `scripts/slurm_sanity_from_spec.sh` | Opt-in scheduler sanity: submit a tiny job, poll it, verify logs, and clean up. |
| 5 | `scripts/stage_data_from_spec.py` | Validate and stage datasets with checksums and symlink-safe path handling. |
| 5 | `scripts/checkpoint_manifest.py` | Write and verify checkpoint manifests before publishing or resuming. |
| 6 | `scripts/run_sanity_ladder.py` | Drive dry-run, preflight, dataset, distributed, RL, and scheduler sanity gates from `sanity`. |
| 6 | `scripts/collect_run_bundle.py` | Collect logs, spec, resolved command, metrics summary, and manifests into a review bundle. |

The first implementation PR should stop after Phase 2. Backend mutation should wait until the spec and
dry-run output are stable and tested.

## Docker plan

Docker runs should not rebuild the training image for every code change. The code checkout is mounted
into the container, while the image supplies the CUDA, PyTorch, and system dependency baseline.

Image policy should be explicit in the spec:

- `pull`: always attempt to pull the configured tag or digest before launch.
- `if_missing`: pull or build only when the image is absent locally.
- `build`: build from the configured Dockerfile and context.
- `push`: push only when explicitly requested and authenticated outside the spec.
- `locked_digest`: require the resolved digest to match the spec.

`docker_image_sync.sh` should compute and print an image cache key from the Dockerfile, requirements
files, base image, and build args. A normal training rerun should be able to reuse the image and update
only the mounted checkout.

Docker dry-runs must render mounts, environment variables, image ref, user, IPC, ulimits, working
directory, and command. Secret values must be redacted.

## Slurm plan

Slurm launches should be rendered from the same run spec. `slurm_submit_from_spec.sh` should not carry
hardcoded partitions, accounts, nodes, excludes, wall times, images, or output paths. The wrapper can
still enforce safety checks, but the actual choices come from the spec or a private operator overlay.

The Slurm adapter should render:

- `sbatch` resource options;
- environment exports;
- container image options when the scheduler provides container support;
- output and error log paths;
- resume and checkpoint publish settings;
- a stable run id and stage name.

Every scheduled run should write a launch bundle containing the redacted spec, resolved command,
scheduler job id, log paths, and expected artifact paths.

## Slurm sanity checker

`scripts/slurm_sanity_from_spec.sh`, a planned opt-in tool, should prove that scheduling works before a
long training job is submitted.

Offline behavior:

- `--dry-run` renders the exact `sbatch`, polling, log-check, and cleanup commands.
- Bats tests replace `sbatch`, `squeue`, `sacct`, `scontrol`, and `scancel` with fakes on `PATH`.
- Tests assert that the script handles `COMPLETED`, `FAILED`, timeout, missing log, and cleanup paths.

Live behavior:

- The job is deliberately tiny, for example a shell command that prints a sentinel line and exits.
- The script polls with a timeout, verifies the final scheduler state, checks the log for the sentinel,
  and removes only artifacts it created.
- Live mode requires an explicit flag such as `--live` and a spec value like `sanity.slurm.enabled:
  true`.
- The default CI job must never run live Slurm. A manual or operator-triggered workflow may run it with
  private credentials supplied by the CI environment.

## Storage and checkpoints

The spec should distinguish node-local scratch from shared artifact storage. Training reads should
prefer local staged data when the run is I/O-bound, while durable checkpoints and launch bundles should
publish to a shared artifact root or a dedicated weights repository.

Checkpoint handling should include:

- free-space and write-permission checks before training;
- save and retention policy;
- resume source validation;
- checksum manifest generation;
- manifest verification after publish;
- a clear "not published" status when the publish destination is unset.

No checkpoint, tokenizer, dataset, tarball, captured response, or private run bundle should be staged
into the igc repository.

## Data staging

`stage_data_from_spec.py` should validate data before a run starts. It should support copy, symlink, or
read-only mount modes, but each mode must be explicit in the spec.

The validator should check:

- source exists and is readable;
- target is inside an allowed scratch or artifact root;
- expected files are present;
- optional checksums match;
- multiple ranks do not build the same dataset concurrently;
- dry-run output lists planned actions without moving data.

## Sanity ladder

`scripts/run_sanity_ladder.py` should make pre-run checks composable. A run can require only the cheap
checks, or a heavier opt-in ladder before a large GPU job.

Recommended ladder:

1. Spec validation and secret scan.
2. Dry-run render and redaction check.
3. Shell syntax and Bats mocked launch tests.
4. Local offline Python gate for changed launcher modules.
5. Docker image presence or pull policy check.
6. Data staging dry-run and manifest check.
7. One-process CPU or GPU smoke, depending on the profile.
8. Distributed sanity using existing DDP/FSDP scripts when GPUs are explicitly requested.
9. RL environment sanity when the stage uses rollouts.
10. Slurm sanity submission when `backend: slurm` and live mode is explicitly approved.

## Testing plan

Default tests must be offline and deterministic.

Planned pytest coverage:

- spec parsing, defaults, unknown-key rejection, and validation errors;
- backend selection for `docker` and `slurm`;
- image policy decisions without calling Docker;
- path normalization and symlink safety;
- redaction of secret-like values;
- dry-run command rendering;
- checkpoint manifest generation and verification;
- sanity ladder ordering and skip reasons.

Planned Bats coverage:

- `render_run.py` exits cleanly for committed examples;
- Docker adapter calls fake `docker pull`, `docker image inspect`, `docker build`, and `docker run`
  in the expected order;
- Slurm adapter calls fake `sbatch` with expected resource flags;
- Slurm sanity handles completed, failed, timed-out, and missing-log jobs;
- image sync avoids build when the image is already present;
- checkpoint publish refuses unset or unsafe destinations;
- dry-run output redacts private environment values.

Planned CI gates:

- shell syntax for every shell and sbatch wrapper;
- `shellcheck` for shell entrypoints;
- `bats tests/bats`;
- pytest for launcher pure logic;
- `git diff --check`;
- Docker build smoke only in the existing Docker job;
- live Slurm sanity only in a manual or operator-approved workflow.

## Execution order

1. Add the schema, examples, and pure Python spec loader.
2. Add dry-run rendering and golden-output tests.
3. Add Docker adapter parity behind dry-run and mocked Bats tests.
4. Add image sync policy and image cache reporting.
5. Add Slurm adapter dry-run and mocked Bats tests.
6. Add the opt-in Slurm sanity checker with mocked tests first, then one live scheduled smoke.
7. Add data staging and checkpoint manifest tools.
8. Fold the sanity ladder into training runbooks.
9. Convert existing launchers to call the spec-driven path once parity is proven.

## Review checklist

Before any launcher PR is marked ready:

- no private hosts, usernames, credentials, captured payload paths, or private endpoints in committed
  specs, logs, docs, tests, or golden files;
- no default test submits to Slurm, runs Docker with GPUs, pushes images, downloads models, or performs
  live data collection;
- every shell entrypoint has a dry-run or mocked test path;
- every command that can mutate remote state requires an explicit flag and a validated spec;
- dry-run output is useful enough to paste into a review without exposing secrets;
- run bundles contain the spec and evidence needed to reproduce the run.

## Relationship to current docs

Keep [TRAINING.md](TRAINING.md) as the runnable guide for current launchers. Keep
[DISTRIBUTED_PLAN.md](DISTRIBUTED_PLAN.md) focused on distributed-training strategy and collective
invariants. This document is the bridge between those two: it defines the future automation layer that
turns launch decisions into tested scripts.
