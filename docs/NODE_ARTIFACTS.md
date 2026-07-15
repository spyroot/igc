# Node artifacts: LFS push from a slot, and Docker Hub images

How to move large artifacts, including captured corpora, built datasets, model adapters, tarballs,
and images, without dragging multi-GB data across the shared VPN. The rule: data that is generated
on a training node is pushed *from* that node over its own datacenter uplink — never pulled to a laptop
and pushed from there (a multi-GB transfer over the VPN has saturated the shared tunnel and blocked
SSH for everyone).

## Push large LFS artifacts from a slot

`scripts/lfs_push_from_slot.sh` commits explicitly supplied artifact paths on a `data/<name>` branch
and uploads their git-LFS objects to the remote from the node. Run it **on the slot**, inside the repo
whose `.gitattributes` tracks that file type (the `igc` checkout, or the data-collection submodule
for capture corpora).

```bash
# on the slot, in the repo checkout
scripts/lfs_push_from_slot.sh ~/.json_responses/<host>          # push a captured host walk
scripts/lfs_push_from_slot.sh datasets/raw/processed_dataset.pt # push a built dataset
scripts/lfs_push_from_slot.sh models/model_x/adapter.safetensors # push a reviewed adapter
```

It:

1. verifies the artifact is matched by an LFS filter (warns if `.gitattributes` won't catch it —
   add the pattern via a PR first, so the file lands as an LFS object, not a huge git blob);
2. creates a `data/<basename>-<UTC>` branch, stages **only** the given paths (never `git add -A`),
   commits, `git lfs push`es the objects, and pushes the branch;
3. prints the **PR URL** — integration is PR-only, so the artifact branch is reviewed and merged
   like any other change; the script never pushes `main`.

### git-lfs on the slot — no OS modification

By default the script fails if `git-lfs` is missing rather than touch the host. Two
non-destructive opt-ins:

```bash
IGC_LFS_INSTALL=apt    scripts/lfs_push_from_slot.sh <path>   # minimal: apt-get install
                                                              # --no-install-recommends git-lfs
                                                              # (adds one package, no OS upgrade)
IGC_LFS_INSTALL=docker scripts/lfs_push_from_slot.sh <path>   # git-lfs from a container; the
                                                              # host OS is never modified
```

The `docker` path runs git-lfs in a container with the repo and your git config/credentials mounted
read-only, so nothing is installed on the node.

### Credentials (one-time, operator)

The push uses the node's existing git auth — set up once, then reused:

- **SSH deploy key** (recommended): generate a key on the node (`ssh-keygen -t ed25519`), add the
  public key as a repo deploy key with write access, and use the `git@github.com:` remote.
- **Personal access token**: a fine-grained PAT with `contents:write`, stored via
  `git config --global credential.helper store` on the node, with the `https://` remote.

Never commit or print a key/token; keep node-specific host/key details in the gitignored team
guide, not in this file.

## Docker Hub images

The CI `docker` job (`.github/workflows/ci.yml`) builds `docker/Dockerfile.test`, smoke-runs the
gate inside it, and — on pushes to `main` — pushes the image to Docker Hub when the repo secrets
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (a Docker Hub access token) are configured. Without the
secrets the job still builds and tests; only the push is skipped.

Locally or from a node:

```bash
DOCKER_REPO=youruser/igc-test make docker-push   # build docker/Dockerfile.test and push
```
