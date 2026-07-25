#!/usr/bin/env bash
# Safe NV72 Docker image operations for IGC.
#
# This script intentionally keeps the SSH-keyed image private to the NV72 lab
# nodes. It may build igc-train-ssh:local on those nodes from .internal/docker,
# but it must never save, push, upload, or publish that image or its layers.
set -euo pipefail

ACTION="${1:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-igc-train}"
TAG="${TAG:-ngc26.03-py3}"
KEYLESS_REF="${KEYLESS_REF:-${IMAGE}:${TAG}}"
SSH_REF="${SSH_REF:-igc-train-ssh:local}"
CONTAINER="${CONTAINER:-igc-phase1-pretrain}"
NV72_USER="${NV72_USER:-nvidia}"
IGC_CODE_DIR="${IGC_CODE_DIR:-/home/nvidia/igc}"
IGC_DATA_DIR="${IGC_DATA_DIR:-/home/nvidia/.json_responses}"
IGC_MODELS_DIR="${IGC_MODELS_DIR:-/models}"
IGC_BRANCH="${IGC_BRANCH:-$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" branch --show-current 2>/dev/null || printf main)}"
IGC_INTERNAL_ROOT="${IGC_INTERNAL_ROOT:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"
RECREATE="${RECREATE:-0}"
IGC_DRY_RUN="${IGC_DRY_RUN:-0}"
IGC_DEV_NAME="${IGC_DEV_NAME:-${DEV_NAME:-${USER:-igcdev}}}"
IGC_DEV_CONTAINER="${IGC_DEV_CONTAINER:-}"
IGC_DEV_CODE_DIR="${IGC_DEV_CODE_DIR:-}"
IGC_DEV_RECREATE="${IGC_DEV_RECREATE:-${DEV_RECREATE:-0}}"
IGC_DEV_CMD="${IGC_DEV_CMD:-${DEV_CMD:-}}"
IGC_HOST_MAP_FILE="${IGC_HOST_MAP_FILE:-${REPO_ROOT}/.internal/nv72_hosts}"
IGC_NODES_FILE="${IGC_NODES_FILE:-${REPO_ROOT}/.internal/gb300_nodes}"

IGC_INTERNAL_ROOT="${IGC_INTERNAL_ROOT:-${REPO_ROOT}}"
TRACKED_CONTEXT_FILES=(
    "docker/Dockerfile.train"
    "docker/requirements-train.txt"
)
PRIVATE_CONTEXT_FILES=(
    ".internal/docker/Dockerfile.test"
    ".internal/docker/Dockerfile.train"
    ".internal/docker/id_rsa"
)
DEPLOY_CONTEXT_FILES=(
    ".internal/docker/Dockerfile.test"
    ".internal/docker/Dockerfile.train"
    ".internal/docker/id_rsa"
    ".internal/hf.token"
    ".internal/wandb.env"
)

usage() {
    cat <<EOF
Usage:
  $0 upload
  HOST=slot2 $0 deploy
  HOST=slot11 $0 stop
  HOST=slot2 DEV_NAME=worker-a IGC_BRANCH=feature/my-branch $0 dev-create
  HOST=slot2 DEV_NAME=worker-a IGC_BRANCH=feature/my-branch DEV_CMD='python -m pytest -q tests/...' $0 dev-run
  HOST=slot2 DEV_NAME=worker-a $0 dev-stop

Environment:
  IGC_NODES     Space-separated upload targets. Defaults to IGC_NODES_FILE.
  IGC_NODES_FILE
                Private file containing one host per line for the upload fleet.
  IGC_HOST_MAP_FILE
                Private map containing '<slot> <host>' rows for slot aliases.
  HOST          Deploy/stop target: slot2, n02, 2, or a host name/IP.
  RECREATE=1    Allow deploy to replace only ${CONTAINER} on HOST.
  DEV_NAME       Short per-agent name for CPU dev containers. Defaults to local USER.
  DEV_RECREATE=1 Replace only the resolved CPU dev container.
  DEV_CMD        Bounded command to run inside the CPU dev container on HOST.
  IGC_BRANCH     Branch to fetch/pull inside the container before DEV_CMD.
  IGC_INTERNAL_ROOT
                Local checkout containing gitignored .internal/ secrets. Defaults to this repo.
  IGC_DRY_RUN=1 Print commands without touching hosts.
EOF
}

blocker() {
    echo "BLOCKER: $*" >&2
    exit 2
}

log() {
    echo "=== $* ==="
}

sanitize_dev_name() {
    local name="$1"
    case "$name" in
        ''|*[!a-zA-Z0-9_.-]*)
            blocker "DEV_NAME must contain only letters, digits, dot, underscore, or dash"
            ;;
    esac
    case "$name" in
        [a-zA-Z0-9]*) ;;
        *) blocker "DEV_NAME must start with a letter or digit" ;;
    esac
    printf '%s\n' "$name"
}

dev_defaults() {
    local name
    name="$(sanitize_dev_name "$IGC_DEV_NAME")"
    IGC_DEV_CONTAINER="${IGC_DEV_CONTAINER:-igc-dev-${name}}"
    IGC_DEV_CODE_DIR="${IGC_DEV_CODE_DIR:-/models/igc/dev/${name}/igc}"
}

all_nodes() {
    [ -r "$IGC_NODES_FILE" ] ||
        blocker "set IGC_NODES or provide IGC_NODES_FILE=${IGC_NODES_FILE}"
    awk 'NF && $1 !~ /^#/ {print $1}' "$IGC_NODES_FILE"
}

nodes_for_upload() {
    if [ -n "${IGC_NODES:-}" ]; then
        # shellcheck disable=SC2086 # intentional word split for space-separated node overrides.
        printf '%s\n' ${IGC_NODES}
    else
        all_nodes
    fi
}

resolve_host() {
    local host="${1:-}"
    local slot=""
    local mapped=""

    case "$host" in
        "") return 1 ;;
        slot*) slot="${host#slot}" ;;
        n*) slot="${host#n}" ;;
        [0-9]*) slot="$host" ;;
        *)
            if [[ "$host" =~ ^[[:alnum:].:-]+$ ]]; then
                printf '%s\n' "$host"
                return 0
            fi
            return 1
            ;;
    esac

    case "$slot" in
        ''|*[!0-9]*) return 1 ;;
    esac

    # 10# handles n02 without octal interpretation.
    slot=$((10#$slot))
    if [ "$slot" -lt 0 ]; then
        return 1
    fi
    [ -r "$IGC_HOST_MAP_FILE" ] || return 1
    mapped="$(awk -v slot="$slot" \
        '$1 == slot || $1 == ("slot" slot) || $1 == ("n" sprintf("%02d", slot)) {print $2; exit}' \
        "$IGC_HOST_MAP_FILE")"
    [ -n "$mapped" ] || return 1
    printf '%s\n' "$mapped"
}

require_internal_context() {
    local path
    for path in "${TRACKED_CONTEXT_FILES[@]}"; do
        [ -f "${REPO_ROOT}/${path}" ] || blocker "missing required Docker context file: ${path}"
    done
    for path in "${PRIVATE_CONTEXT_FILES[@]}"; do
        [ -f "${IGC_INTERNAL_ROOT}/${path}" ] || blocker "missing required private Docker context file: ${path}"
    done
}

require_deploy_context() {
    local path
    for path in "${DEPLOY_CONTEXT_FILES[@]}"; do
        [ -f "${IGC_INTERNAL_ROOT}/${path}" ] || blocker "missing required deploy context file: ${path}"
    done
}

ssh_dest() {
    printf '%s@%s' "$NV72_USER" "$1"
}

run_ssh() {
    local ip="$1"
    shift
    # shellcheck disable=SC2086 # SSH_OPTS is intentionally a shell-style option string.
    # shellcheck disable=SC2029 # Remote command strings are intentionally rendered locally.
    ssh ${SSH_OPTS} "$(ssh_dest "$ip")" "$@"
}

remote_upload_script() {
    cat <<'REMOTE'
set -euo pipefail
KEYLESS_REF="$1"
SSH_REF="$2"
REMOTE_TMP="$3"
tmp="${REMOTE_TMP}/context"
cleanup() { rm -rf "$REMOTE_TMP"; }
trap cleanup EXIT

mkdir -p "$tmp"
tar -C "$tmp" -xf "${REMOTE_TMP}/context.tar"
chmod 700 "$tmp/.internal" "$tmp/.internal/docker"
chmod 600 "$tmp/.internal/docker/id_rsa"
cd "$tmp"

docker build \
    -f docker/Dockerfile.train \
    -t "$KEYLESS_REF" \
    .

docker build \
    --build-arg "BASE=$KEYLESS_REF" \
    -f .internal/docker/Dockerfile.train \
    -t "$SSH_REF" \
    .

docker run --rm "$KEYLESS_REF" \
    bash -lc 'command -v conda && conda --version && command -v git-lfs && git lfs version'

docker run --rm "$SSH_REF" \
    bash -lc 'command -v conda && conda --version && command -v git-lfs && test -f /root/.ssh/id_rsa'

docker image inspect "$KEYLESS_REF" >/dev/null
docker image inspect "$SSH_REF" >/dev/null
echo "image_gate=ok keyless=$KEYLESS_REF private_ssh=$SSH_REF"
REMOTE
}

upload_one() {
    local raw="$1"
    local ip
    ip="$(resolve_host "$raw")" || blocker "invalid upload node: ${raw}"

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN upload node $(ssh_dest "$ip")"
        echo "  tar minimal Docker context | ssh $(ssh_dest "$ip") 'docker build -f docker/Dockerfile.train -t ${KEYLESS_REF} .'"
        echo "  ssh $(ssh_dest "$ip") 'docker build -f .internal/docker/Dockerfile.train --build-arg BASE=${KEYLESS_REF} -t ${SSH_REF} .'"
        echo "  ssh $(ssh_dest "$ip") 'docker run --rm ${KEYLESS_REF} bash -lc \"command -v conda && conda --version && command -v git-lfs\"'"
        echo "  ssh $(ssh_dest "$ip") 'docker run --rm ${SSH_REF} bash -lc \"command -v conda && conda --version && command -v git-lfs && test -f /root/.ssh/id_rsa\"'"
        echo "  ${SSH_REF} is a private NV72 node image; never docker save or docker push it."
        return 0
    fi

    log "upload/build ${KEYLESS_REF} + ${SSH_REF} on $(ssh_dest "$ip")"
    local remote_tmp
    remote_tmp="$(run_ssh "$ip" "umask 077; mktemp -d /tmp/igc-docker-build.XXXXXX")"
    (
        tar -cf - \
            -C "$REPO_ROOT" "${TRACKED_CONTEXT_FILES[@]}" \
            -C "$IGC_INTERNAL_ROOT" "${PRIVATE_CONTEXT_FILES[@]}"
    ) | run_ssh "$ip" "cat > '${remote_tmp}/context.tar'"
    run_ssh "$ip" "bash -s" -- "$KEYLESS_REF" "$SSH_REF" "$remote_tmp" <<<"$(remote_upload_script)"
}

upload() {
    require_internal_context
    local node
    local ok=0
    local fail=0

    while IFS= read -r node; do
        [ -n "$node" ] || continue
        if upload_one "$node"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done < <(nodes_for_upload)

    log "docker upload gate ready on ${ok} node(s), ${fail} failed"
    [ "$fail" -eq 0 ]
}

deploy_remote_script() {
    cat <<'REMOTE'
set -euo pipefail
CONTAINER="$1"
SSH_REF="$2"
IGC_CODE_DIR="$3"
IGC_DATA_DIR="$4"
IGC_MODELS_DIR="$5"
RECREATE="$6"
IGC_BRANCH="$7"
REMOTE_TMP="$8"
ctx="${REMOTE_TMP}/context"
cleanup() { rm -rf "$REMOTE_TMP"; }
trap cleanup EXIT

docker image inspect "$SSH_REF" >/dev/null

mkdir -p "$ctx"
tar -C "$ctx" -xf "${REMOTE_TMP}/deploy-context.tar"
host_uid="$(id -u)"
host_gid="$(id -g)"
checkout_parent="$(dirname "$IGC_CODE_DIR")"
checkout_base="$(basename "$IGC_CODE_DIR")"

if [ -d "$IGC_CODE_DIR/.git" ]; then
    :
elif [ ! -e "$IGC_CODE_DIR" ] || [ -z "$(find "$IGC_CODE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    mkdir -p "$checkout_parent"
    rm -rf "$IGC_CODE_DIR"
    docker run --rm \
        -v "${checkout_parent}:/host_parent" \
        "$SSH_REF" \
        bash -lc 'cd /host_parent && git clone git@github.com:spyroot/igc.git "$1"' _ "$checkout_base"
else
    echo "BLOCKER: ${IGC_CODE_DIR} exists but is not an igc git checkout; refusing to overwrite it" >&2
    exit 2
fi

docker run --rm \
    -v "${checkout_parent}:/host_parent" \
    "$SSH_REF" \
    bash -lc 'chown -R "$1:$2" "/host_parent/$3"' _ "$host_uid" "$host_gid" "$checkout_base"

mkdir -p "${IGC_CODE_DIR}/.internal"
tar -C "${ctx}/.internal" -cf - docker | tar -C "${IGC_CODE_DIR}/.internal" -xf -
install -m 600 "${ctx}/.internal/wandb.env" "${IGC_CODE_DIR}/.internal/wandb.env"
token="$(tr -d '\r\n' < "${ctx}/.internal/hf.token")"
{
    printf '%s\n' 'export HF_HOME=/models/hf-cache'
    printf '%s\n' 'export HF_HUB_CACHE=/models/hf-cache/hub'
    printf 'export HF_TOKEN=%s\n' "$token"
    printf 'export HUGGINGFACE_TOKEN=%s\n' "$token"
} > "${IGC_CODE_DIR}/.internal/hf.env"
chmod 700 "${IGC_CODE_DIR}/.internal"
chmod 600 \
    "${IGC_CODE_DIR}/.internal/hf.env" \
    "${IGC_CODE_DIR}/.internal/wandb.env" \
    "${IGC_CODE_DIR}/.internal/docker/id_rsa"
mkdir -p /models/hf-cache/hub
chmod 700 /models/hf-cache /models/hf-cache/hub 2>/dev/null || true

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    if [ "$RECREATE" != "1" ]; then
        echo "BLOCKER: ${CONTAINER} already exists on $(hostname); set RECREATE=1 to replace only this container" >&2
        exit 2
    fi
    docker rm -f "$CONTAINER" >/dev/null
fi

docker run -d \
    --name "$CONTAINER" \
    --gpus all \
    --ipc=host \
    --shm-size=32g \
    -v "${IGC_CODE_DIR}:/workspace/igc" \
    -v "${IGC_DATA_DIR}:/root/.json_responses" \
    -v "${IGC_MODELS_DIR}:/models" \
    -w /workspace/igc \
    "$SSH_REF" \
    sleep infinity >/dev/null

docker exec "$CONTAINER" bash -lc '
    set -euo pipefail
    git config --global --add safe.directory /workspace/igc 2>/dev/null || true
    cd /workspace/igc
    git remote set-url origin git@github.com:spyroot/igc.git
    git config core.sshCommand "ssh -i /root/.ssh/id_rsa -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    git fetch origin "'"${IGC_BRANCH}"'"
    if git show-ref --verify --quiet "refs/heads/'"${IGC_BRANCH}"'"; then
        git checkout "'"${IGC_BRANCH}"'"
    else
        git checkout -B "'"${IGC_BRANCH}"'" "origin/'"${IGC_BRANCH}"'"
    fi
    git pull --ff-only origin "'"${IGC_BRANCH}"'"
    git lfs install --local
    git lfs locks >/tmp/igc_lfs_locks.out 2>/tmp/igc_lfs_locks.err || true
    command -v conda
    conda --version
    command -v git-lfs
    test -f .internal/hf.env
    test -f .internal/wandb.env
    test -f /root/.ssh/id_rsa
'
echo "deploy=ok container=${CONTAINER} image=${SSH_REF}"
REMOTE
}

deploy() {
    [ -n "${HOST:-}" ] || blocker "set HOST to a single target, e.g. HOST=slot2"
    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN deploy $(ssh_dest "$ip")"
        echo "  ssh $(ssh_dest "$ip") 'docker inspect ${CONTAINER} >/dev/null 2>&1 || true'"
        if [ "$RECREATE" = "1" ]; then
            echo "  ssh $(ssh_dest "$ip") 'docker rm -f ${CONTAINER}'"
        else
            echo "  existing ${CONTAINER} blocks deploy unless RECREATE=1"
        fi
        echo "  ssh $(ssh_dest "$ip") 'docker run -d --name ${CONTAINER} --gpus all ... ${SSH_REF} sleep infinity'"
        return 0
    fi

    require_deploy_context
    log "deploy ${CONTAINER} on $(ssh_dest "$ip")"
    local remote_tmp
    remote_tmp="$(run_ssh "$ip" "umask 077; mktemp -d /tmp/igc-deploy.XXXXXX")"
    (
        cd "$IGC_INTERNAL_ROOT"
        tar -cf - "${DEPLOY_CONTEXT_FILES[@]}"
    ) | run_ssh "$ip" "cat > '${remote_tmp}/deploy-context.tar'"
    run_ssh "$ip" "bash -s" -- \
        "$CONTAINER" "$SSH_REF" "$IGC_CODE_DIR" "$IGC_DATA_DIR" "$IGC_MODELS_DIR" "$RECREATE" \
        "$IGC_BRANCH" "$remote_tmp" \
        <<<"$(deploy_remote_script)"
}

stop_remote_script() {
    cat <<'REMOTE'
set -euo pipefail
CONTAINER="$1"
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker stop "$CONTAINER" >/dev/null
    echo "stop=ok container=${CONTAINER}"
else
    echo "stop=missing container=${CONTAINER}"
fi
REMOTE
}

stop_one() {
    local ip="$1"
    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN stop $(ssh_dest "$ip")"
        echo "  ssh $(ssh_dest "$ip") 'docker stop ${CONTAINER}'"
        return 0
    fi
    log "stop ${CONTAINER} on $(ssh_dest "$ip")"
    run_ssh "$ip" "bash -s" -- "$CONTAINER" <<<"$(stop_remote_script)"
}

stop() {
    [ -n "${HOST:-}" ] || blocker "set HOST to a target, e.g. HOST=slot11"

    if [ "$HOST" = "all" ]; then
        local node
        while IFS= read -r node; do
            stop_one "$node"
        done < <(all_nodes)
        return 0
    fi

    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"
    stop_one "$ip"
}

dev_create_remote_script() {
    cat <<'REMOTE'
set -euo pipefail
IGC_DEV_CONTAINER="$1"
SSH_REF="$2"
IGC_DEV_CODE_DIR="$3"
IGC_MODELS_DIR="$4"
IGC_DEV_RECREATE="$5"
IGC_BRANCH="$6"
REMOTE_TMP="$7"
ctx="${REMOTE_TMP}/context"
cleanup() { rm -rf "$REMOTE_TMP"; }
trap cleanup EXIT

docker image inspect "$SSH_REF" >/dev/null

mkdir -p "$ctx"
tar -C "$ctx" -xf "${REMOTE_TMP}/dev-context.tar"
host_uid="$(id -u)"
host_gid="$(id -g)"
checkout_parent="$(dirname "$IGC_DEV_CODE_DIR")"
checkout_base="$(basename "$IGC_DEV_CODE_DIR")"

if [ -d "$IGC_DEV_CODE_DIR/.git" ]; then
    :
elif [ ! -e "$IGC_DEV_CODE_DIR" ] || [ -z "$(find "$IGC_DEV_CODE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    mkdir -p "$checkout_parent"
    rm -rf "$IGC_DEV_CODE_DIR"
    docker run --rm \
        -v "${checkout_parent}:/host_parent" \
        "$SSH_REF" \
        bash -lc 'cd /host_parent && git clone git@github.com:spyroot/igc.git "$1"' _ "$checkout_base"
else
    echo "BLOCKER: ${IGC_DEV_CODE_DIR} exists but is not an igc git checkout; refusing to overwrite it" >&2
    exit 2
fi

docker run --rm \
    -v "${checkout_parent}:/host_parent" \
    "$SSH_REF" \
    bash -lc 'chown -R "$1:$2" "/host_parent/$3"' _ "$host_uid" "$host_gid" "$checkout_base"

mkdir -p "${IGC_DEV_CODE_DIR}/.internal"
tar -C "${ctx}/.internal" -cf - docker | tar -C "${IGC_DEV_CODE_DIR}/.internal" -xf -
install -m 600 "${ctx}/.internal/wandb.env" "${IGC_DEV_CODE_DIR}/.internal/wandb.env"
token="$(tr -d '\r\n' < "${ctx}/.internal/hf.token")"
{
    printf '%s\n' 'export HF_HOME=/models/hf-cache'
    printf '%s\n' 'export HF_HUB_CACHE=/models/hf-cache/hub'
    printf 'export HF_TOKEN=%s\n' "$token"
    printf 'export HUGGINGFACE_TOKEN=%s\n' "$token"
} > "${IGC_DEV_CODE_DIR}/.internal/hf.env"
chmod 700 "${IGC_DEV_CODE_DIR}/.internal"
chmod 600 \
    "${IGC_DEV_CODE_DIR}/.internal/hf.env" \
    "${IGC_DEV_CODE_DIR}/.internal/wandb.env" \
    "${IGC_DEV_CODE_DIR}/.internal/docker/id_rsa"
mkdir -p "${IGC_MODELS_DIR}/hf-cache/hub"
chmod 700 "${IGC_MODELS_DIR}/hf-cache" "${IGC_MODELS_DIR}/hf-cache/hub" 2>/dev/null || true

if docker inspect "$IGC_DEV_CONTAINER" >/dev/null 2>&1; then
    if [ "$IGC_DEV_RECREATE" = "1" ]; then
        docker rm -f "$IGC_DEV_CONTAINER" >/dev/null
    else
        running="$(docker inspect -f '{{.State.Running}}' "$IGC_DEV_CONTAINER")"
        if [ "$running" != "true" ]; then
            docker start "$IGC_DEV_CONTAINER" >/dev/null
        fi
    fi
fi

if ! docker inspect "$IGC_DEV_CONTAINER" >/dev/null 2>&1; then
    docker run -d \
        --name "$IGC_DEV_CONTAINER" \
        --label igc.role=cpu-dev \
        --ipc=host \
        --shm-size=16g \
        -v "${IGC_DEV_CODE_DIR}:/workspace/igc" \
        -v "${IGC_MODELS_DIR}:/models" \
        -w /workspace/igc \
        "$SSH_REF" \
        sleep infinity >/dev/null
fi

docker exec "$IGC_DEV_CONTAINER" bash -lc '
    set -euo pipefail
    git config --global --add safe.directory /workspace/igc 2>/dev/null || true
    cd /workspace/igc
    git remote set-url origin git@github.com:spyroot/igc.git
    git config core.sshCommand "ssh -i /root/.ssh/id_rsa -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    if [ -n "'"${IGC_BRANCH}"'" ]; then
        git fetch origin "'"${IGC_BRANCH}"'"
        if git show-ref --verify --quiet "refs/heads/'"${IGC_BRANCH}"'"; then
            git checkout "'"${IGC_BRANCH}"'"
        else
            git checkout -B "'"${IGC_BRANCH}"'" "origin/'"${IGC_BRANCH}"'"
        fi
        git pull --ff-only origin "'"${IGC_BRANCH}"'"
    fi
    git lfs install --local
    git lfs locks >/tmp/igc_lfs_locks.out 2>/tmp/igc_lfs_locks.err || true

    set -a
    . .internal/hf.env
    . .internal/wandb.env
    set +a

    command -v conda
    conda --version
    command -v git-lfs
    test -f .internal/hf.env
    test -f .internal/wandb.env
    test -f /root/.ssh/id_rsa
'
echo "dev_create=ok container=${IGC_DEV_CONTAINER} image=${SSH_REF} code=${IGC_DEV_CODE_DIR} gpu=none"
REMOTE
}

dev_create() {
    [ -n "${HOST:-}" ] || blocker "set HOST to a single target, e.g. HOST=slot2"
    dev_defaults

    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN dev-create $(ssh_dest "$ip")"
        echo "  ssh $(ssh_dest "$ip") 'docker image inspect ${SSH_REF}'"
        if [ "$IGC_DEV_RECREATE" = "1" ]; then
            echo "  ssh $(ssh_dest "$ip") 'docker rm -f ${IGC_DEV_CONTAINER}'"
        fi
        echo "  ssh $(ssh_dest "$ip") 'clone/pull ${IGC_BRANCH} into ${IGC_DEV_CODE_DIR}'"
        echo "  ssh $(ssh_dest "$ip") 'docker run -d --name ${IGC_DEV_CONTAINER} --label igc.role=cpu-dev --shm-size=16g -v ${IGC_DEV_CODE_DIR}:/workspace/igc -v ${IGC_MODELS_DIR}:/models ${SSH_REF} sleep infinity'"
        echo "  CPU-only dev container: no --gpus flag, no laptop docker, no laptop pytest."
        return 0
    fi

    require_deploy_context
    log "dev-create ${IGC_DEV_CONTAINER} on $(ssh_dest "$ip") branch=${IGC_BRANCH}"
    local remote_tmp
    remote_tmp="$(run_ssh "$ip" "umask 077; mktemp -d /tmp/igc-dev.XXXXXX")"
    (
        cd "$IGC_INTERNAL_ROOT"
        tar -cf - "${DEPLOY_CONTEXT_FILES[@]}"
    ) | run_ssh "$ip" "cat > '${remote_tmp}/dev-context.tar'"
    run_ssh "$ip" "bash -s" -- \
        "$IGC_DEV_CONTAINER" "$SSH_REF" "$IGC_DEV_CODE_DIR" "$IGC_MODELS_DIR" "$IGC_DEV_RECREATE" \
        "$IGC_BRANCH" "$remote_tmp" \
        <<<"$(dev_create_remote_script)"
}

dev_remote_script() {
    cat <<'REMOTE'
set -euo pipefail
CONTAINER="$1"
IGC_BRANCH="$2"
IGC_DEV_CMD="$3"

docker inspect "$CONTAINER" >/dev/null
docker exec \
    -e IGC_BRANCH="$IGC_BRANCH" \
    -e IGC_DEV_CMD="$IGC_DEV_CMD" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail
        cd /workspace/igc

        git config --global --add safe.directory /workspace/igc 2>/dev/null || true
        git remote set-url origin git@github.com:spyroot/igc.git
        git config core.sshCommand \
            "ssh -i /root/.ssh/id_rsa -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

        if [ -n "$IGC_BRANCH" ]; then
            git fetch origin "$IGC_BRANCH"
            if git show-ref --verify --quiet "refs/heads/${IGC_BRANCH}"; then
                git checkout "$IGC_BRANCH"
            else
                git checkout -B "$IGC_BRANCH" "origin/${IGC_BRANCH}"
            fi
            git pull --ff-only origin "$IGC_BRANCH"
        fi

        git lfs install --local
        git lfs locks >/tmp/igc_lfs_locks.out 2>/tmp/igc_lfs_locks.err || true

        set -a
        [ -f .internal/hf.env ] && . .internal/hf.env
        [ -f .internal/wandb.env ] && . .internal/wandb.env
        set +a

        command -v conda >/dev/null
        conda --version >/tmp/igc_conda_version.out
        command -v git-lfs >/dev/null
        test -f /root/.ssh/id_rsa
        test -f .internal/hf.env
        test -f .internal/wandb.env

        echo "remote_dev_start host=$(hostname) branch=$(git branch --show-current) container_ready=1"
        bash -lc "$IGC_DEV_CMD"
        echo "remote_dev_ok"
    '
REMOTE
}

dev_run() {
    [ -n "${HOST:-}" ] || blocker "set HOST to a single target, e.g. HOST=slot2"
    dev_defaults
    [ -n "$IGC_DEV_CMD" ] || blocker "set DEV_CMD to the bounded command to run inside ${IGC_DEV_CONTAINER}"

    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN dev-run $(ssh_dest "$ip")"
        echo "  ssh $(ssh_dest "$ip") 'docker inspect ${IGC_DEV_CONTAINER} >/dev/null'"
        echo "  ssh $(ssh_dest "$ip") 'docker exec ${IGC_DEV_CONTAINER} bash -lc <git fetch/pull ${IGC_BRANCH}; DEV_CMD>'"
        echo "  command runs remotely in ${IGC_DEV_CONTAINER}; no laptop docker or laptop pytest is used."
        return 0
    fi

    log "dev-run in ${IGC_DEV_CONTAINER} on $(ssh_dest "$ip") branch=${IGC_BRANCH}"
    run_ssh "$ip" "bash -s" -- "$IGC_DEV_CONTAINER" "$IGC_BRANCH" "$IGC_DEV_CMD" \
        <<<"$(dev_remote_script)"
}

dev_stop() {
    [ -n "${HOST:-}" ] || blocker "set HOST to a single target, e.g. HOST=slot2"
    dev_defaults

    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN dev-stop $(ssh_dest "$ip")"
        echo "  ssh $(ssh_dest "$ip") 'docker stop ${IGC_DEV_CONTAINER}'"
        return 0
    fi

    log "dev-stop ${IGC_DEV_CONTAINER} on $(ssh_dest "$ip")"
    run_ssh "$ip" "bash -s" -- "$IGC_DEV_CONTAINER" <<<"$(stop_remote_script)"
}

case "$ACTION" in
    upload) upload ;;
    deploy) deploy ;;
    stop) stop ;;
    dev-create) dev_create ;;
    dev-run|dev-test) dev_run ;;
    dev-stop) dev_stop ;;
    -h|--help|help|"") usage; [ -n "$ACTION" ] ;;
    *) usage >&2; blocker "unknown action: ${ACTION}" ;;
esac
