#!/usr/bin/env bash
# Build the full GoalExtractor / GoalEncoder dataset inside the NV72 Docker lab.
#
# This wrapper is intentionally thin. It prepares the redfish_ctl submodule,
# pulls its Git LFS objects, verifies JSON captures exist, and then delegates to
# scripts/build_goal_dataset.py. The Mac/local path should use unit tests and tiny
# fixture smoke runs only; the full X-generation pass belongs in the lab Docker
# environment where the Redfish corpus and local model endpoint are close.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${IGC_REDFISH_CTL_DIR:-}" ]]; then
	IGC_REDFISH_CTL_DIR="${REPO_ROOT}/redfish_ctl"
fi
IGC_REDFISH_FULL_CORPUS_DIR="${IGC_REDFISH_FULL_CORPUS_DIR:-${IGC_REDFISH_CTL_DIR}/full_corpus}"
IGC_GOAL_DATASET_OUT="${IGC_GOAL_DATASET_OUT:-}"
IGC_GOAL_DATASET_WORK_DIR="${IGC_GOAL_DATASET_WORK_DIR:-}"
IGC_GOAL_DATASET_ENV_FILE="${IGC_GOAL_DATASET_ENV_FILE:-${REPO_ROOT}/.internal/goal_dataset.env}"
IGC_GOAL_DATASET_PARAPHRASE_MODE="${IGC_GOAL_DATASET_PARAPHRASE_MODE:-openai}"
IGC_GOAL_DATASET_GLOB="${IGC_GOAL_DATASET_GLOB:-**/*.json}"
IGC_GOAL_PARAPHRASE_COUNT="${IGC_GOAL_PARAPHRASE_COUNT:-8}"
IGC_LFS_PULL="${IGC_LFS_PULL:-1}"
IGC_LAB_DRY_RUN="${IGC_LAB_DRY_RUN:-0}"
IGC_REQUIRE_VENDOR_CORPUS="${IGC_REQUIRE_VENDOR_CORPUS:-1}"
REQUIRED_FULL_CORPORA=(
	"dell_xr8620t_full_corpus.tar.gz"
	"hpe_dl360_full_corpus.tar.gz"
	"supermicro_gb300_full_corpus.tar.gz"
	"supermicro_x10_full_corpus.tar.gz"
)

say() {
	printf '%s\n' "$*"
}

blocker() {
	printf 'BLOCKER: %s\n' "$*" >&2
	exit 1
}

quote_cmd() {
	local quoted=()
	local arg
	for arg in "$@"; do
		quoted+=("$(printf '%q' "$arg")")
	done
	printf '%s\n' "${quoted[*]}"
}

run_or_print() {
	if [[ "${IGC_LAB_DRY_RUN}" = "1" ]]; then
		quote_cmd "$@"
	else
		"$@"
	fi
}

require_command() {
	command -v "$1" >/dev/null 2>&1 || blocker "$1 not found in PATH"
}

source_private_env() {
	if [[ -f "${IGC_GOAL_DATASET_ENV_FILE}" ]]; then
		set -a
		# shellcheck disable=SC1090
		. "${IGC_GOAL_DATASET_ENV_FILE}"
		set +a
		say "loaded private goal-dataset env file"
	fi
}

append_root_if_present() {
	local root="$1"
	if [[ -d "${root}" ]]; then
		CAPTURE_ROOTS+=("${root}")
	fi
}

split_roots() {
	local roots="$1"
	local old_ifs="${IFS}"
	local root
	IFS=':'
	for root in ${roots}; do
		if [[ -n "${root}" ]]; then
			CAPTURE_ROOTS+=("${root}")
		fi
	done
	IFS="${old_ifs}"
}

count_json_files() {
	local root="$1"
	find "${root}" -type f -name '*.json' 2>/dev/null | wc -l | tr -dc '0-9'
}

is_lfs_pointer() {
	local archive="$1"
	LC_ALL=C head -c 96 "${archive}" 2>/dev/null |
		grep -q 'version https://git-lfs.github.com/spec/v1'
}

tar_contains() {
	local archive="$1"
	local pattern="$2"
	tar -tzf "${archive}" | grep -Eq "${pattern}"
}

validate_tar_members() {
	local archive="$1"
	local member
	while IFS= read -r member; do
		case "${member}" in
		"" | /* | ../* | */../* | */.. | . | ..)
			blocker "unsafe path in redfish_ctl full corpus archive: ${archive}: ${member}"
			;;
		esac
	done < <(tar -tzf "${archive}")

	local listing
	while IFS= read -r listing; do
		case "${listing:0:1}" in
		l | h)
			blocker "unsafe link entry in redfish_ctl full corpus archive: ${archive}"
			;;
		esac
	done < <(tar -tvzf "${archive}")
}

tar_json_count() {
	local archive="$1"
	tar -tzf "${archive}" |
		awk -F/ '$NF ~ /[.]json$/ && $NF != "corpus_manifest.json" {count++} END {print count + 0}'
}

archive_host_roots() {
	local archive="$1"
	tar -tzf "${archive}" |
		awk -F/ 'NF == 2 && $2 == "rest_api_map.npy" {print $1}' |
		sort -u
}

require_archive_host_roots() {
	local archive="$1"
	local roots
	roots="$(archive_host_roots "${archive}")"
	[[ -n "${roots}" ]] ||
		blocker "redfish_ctl full corpus archive lacks top-level host rest_api_map.npy: ${archive}"
}

require_full_corpus_archive() {
	local archive_name="$1"
	local archive="${IGC_REDFISH_FULL_CORPUS_DIR}/${archive_name}"
	[[ -f "${archive}" ]] ||
		blocker "missing required redfish_ctl full corpus archive: ${archive}"
	if is_lfs_pointer "${archive}"; then
		blocker "redfish_ctl full corpus archive is still an LFS pointer: ${archive}"
	fi
	tar -tzf "${archive}" >/dev/null ||
		blocker "redfish_ctl full corpus archive is not readable tar.gz: ${archive}"
	validate_tar_members "${archive}"
	tar_contains "${archive}" '(^|/)rest_api_map[.]npy$' ||
		blocker "redfish_ctl full corpus archive lacks rest_api_map.npy: ${archive}"
	require_archive_host_roots "${archive}"
	tar_contains "${archive}" '(^|/)corpus_manifest[.]json$' ||
		blocker "redfish_ctl full corpus archive lacks corpus_manifest.json: ${archive}"

	local count
	count="$(tar_json_count "${archive}")"
	((count > 0)) ||
		blocker "redfish_ctl full corpus archive contains no resource JSON: ${archive}"
	say "full corpus archive: ${archive_name} json_files=${count}"
}

prepare_redfish_ctl() {
	require_command git
	if [[ ! -d "${IGC_REDFISH_CTL_DIR}/.git" && ! -f "${IGC_REDFISH_CTL_DIR}/.git" ]]; then
		say "initializing redfish_ctl submodule"
		run_or_print git -C "${REPO_ROOT}" submodule update --init --recursive redfish_ctl
	fi

	if [[ "${IGC_LFS_PULL}" = "1" ]]; then
		git lfs version >/dev/null 2>&1 || blocker "git lfs is required inside the lab container"
		say "pulling redfish_ctl Git LFS objects"
		run_or_print git -C "${IGC_REDFISH_CTL_DIR}" lfs install --local
		run_or_print git -C "${IGC_REDFISH_CTL_DIR}" lfs pull
	fi
}

require_vendor_corpora() {
	if [[ "${IGC_REQUIRE_VENDOR_CORPUS}" != "1" ]]; then
		return
	fi

	local archive_name
	for archive_name in "${REQUIRED_FULL_CORPORA[@]}"; do
		require_full_corpus_archive "${archive_name}"
	done
}

extract_full_corpora() {
	local extract_root="${IGC_GOAL_DATASET_WORK_DIR}/full_corpus"
	local archive_name
	local archive
	local dest
	local host_root
	mkdir -p "${extract_root}"

	for archive_name in "${REQUIRED_FULL_CORPORA[@]}"; do
		archive="${IGC_REDFISH_FULL_CORPUS_DIR}/${archive_name}"
		dest="${extract_root}/${archive_name%.tar.gz}"
		run_or_print mkdir -p "${dest}"
		run_or_print tar -xzf "${archive}" -C "${dest}"

		while IFS= read -r host_root; do
			[[ -n "${host_root}" ]] || continue
			CAPTURE_ROOTS+=("${dest}/${host_root}")
		done < <(archive_host_roots "${archive}")
	done
}

discover_capture_roots() {
	CAPTURE_ROOTS=()
	if [[ -n "${IGC_CAPTURE_ROOTS:-}" ]]; then
		split_roots "${IGC_CAPTURE_ROOTS}"
	else
		extract_full_corpora
	fi
	if [[ -n "${IGC_EXTRA_CAPTURE_ROOTS:-}" ]]; then
		split_roots "${IGC_EXTRA_CAPTURE_ROOTS}"
	fi

	((${#CAPTURE_ROOTS[@]} > 0)) || blocker "no capture roots found; set IGC_CAPTURE_ROOTS"

	local total=0
	local root
	local count
	for root in "${CAPTURE_ROOTS[@]}"; do
		if [[ "${IGC_LAB_DRY_RUN}" = "1" && ! -d "${root}" ]]; then
			say "capture root: ${root} json_files=dry-run"
			continue
		fi
		[[ -d "${root}" ]] || blocker "capture root does not exist: ${root}"
		count="$(count_json_files "${root}")"
		say "capture root: ${root} json_files=${count}"
		total=$((total + count))
	done
	if [[ "${IGC_LAB_DRY_RUN}" != "1" ]]; then
		((total > 0)) || blocker "capture roots contain no JSON files after git lfs pull"
	fi
}

validate_generation_env() {
	[[ -n "${IGC_GOAL_DATASET_OUT}" ]] || blocker "set IGC_GOAL_DATASET_OUT to a private dataset output directory"
	if [[ -z "${IGC_GOAL_DATASET_WORK_DIR}" ]]; then
		IGC_GOAL_DATASET_WORK_DIR="${IGC_GOAL_DATASET_OUT}/_build"
	fi
	case "${IGC_GOAL_DATASET_PARAPHRASE_MODE}" in
	none | template | static | openai) ;;
	*) blocker "IGC_GOAL_DATASET_PARAPHRASE_MODE must be none, template, static, or openai" ;;
	esac
	if [[ "${IGC_GOAL_DATASET_PARAPHRASE_MODE}" = "openai" ]]; then
		[[ -n "${GOAL_PARAPHRASE_BASE_URL:-}" ]] || blocker "GOAL_PARAPHRASE_BASE_URL is required for openai paraphrases"
		[[ -n "${GOAL_PARAPHRASE_MODEL:-}" ]] || blocker "GOAL_PARAPHRASE_MODEL is required for openai paraphrases"
		say "paraphrase backend env is present"
	fi
}

build_dataset() {
	require_command "${PYTHON_BIN}"
	mkdir -p "${IGC_GOAL_DATASET_OUT}"

	local surfaces_out="${IGC_GOAL_DATASET_OUT}/goal_surfaces.jsonl"
	local text_out="${IGC_GOAL_DATASET_OUT}/goal_text_examples.jsonl"
	local manifest_out="${IGC_GOAL_DATASET_OUT}/goal_dataset_manifest.json"
	local args=(
		"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_goal_dataset.py"
		--source full_redfish_corpus
		--glob-pattern "${IGC_GOAL_DATASET_GLOB}"
		--surfaces-out "${surfaces_out}"
		--manifest-out "${manifest_out}"
	)
	local root
	for root in "${CAPTURE_ROOTS[@]}"; do
		args+=(--capture-root "${root}")
	done

	if [[ "${IGC_GOAL_DATASET_PARAPHRASE_MODE}" != "none" ]]; then
		args+=(
			--text-out "${text_out}"
			--paraphrase-mode "${IGC_GOAL_DATASET_PARAPHRASE_MODE}"
			--generate-all-goals
			--count "${IGC_GOAL_PARAPHRASE_COUNT}"
		)
	fi

	say "building goal dataset"
	run_or_print "${args[@]}"
	say "dataset outputs: ${IGC_GOAL_DATASET_OUT}"
}

main() {
	source_private_env
	prepare_redfish_ctl
	require_vendor_corpora
	validate_generation_env
	discover_capture_roots
	build_dataset
}

main "$@"
