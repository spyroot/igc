#!/usr/bin/env bats

setup() {
	REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
	export REPO_ROOT

	mkdir -p "${BATS_TEST_TMPDIR}/bin"
	export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"
}

install_lab_fakes() {
	cat >"${BATS_TEST_TMPDIR}/bin/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_GIT_LOG}"
if [[ "${1:-}" == "lfs" && "${2:-}" == "version" ]]; then
    printf 'git-lfs/3.0.0\n'
fi
EOF
	chmod +x "${BATS_TEST_TMPDIR}/bin/git"

	cat >"${BATS_TEST_TMPDIR}/bin/python" <<'EOF'
#!/usr/bin/env bash
printf 'python-stub should not execute in dry-run\n' >&2
exit 2
EOF
	chmod +x "${BATS_TEST_TMPDIR}/bin/python"
}

make_redfish_ctl_corpus() {
	local root="$1"
	mkdir -p "${root}/full_corpus"
	touch "${root}/.git"
	make_full_corpus_archive "${root}" "dell_xr8620t_full_corpus.tar.gz" "10.0.0.1" \
		'{"@odata.id":"/redfish/v1/Systems/1","@odata.type":"#ComputerSystem.v1_20_0.ComputerSystem","PowerState":"Off"}'
	make_full_corpus_archive "${root}" "supermicro_gb300_full_corpus.tar.gz" "10.0.0.2" \
		'{"@odata.id":"/redfish/v1/Managers/1/NetworkProtocol","@odata.type":"#ManagerNetworkProtocol.v1_10_0.ManagerNetworkProtocol","NTP":{"ProtocolEnabled":false}}'
	make_full_corpus_archive "${root}" "hpe_dl360_full_corpus.tar.gz" "10.0.0.3" \
		'{"@odata.id":"/redfish/v1/Systems/1/Bios","@odata.type":"#Bios.v1_2_0.Bios","Attributes":{"BootMode":"Uefi"}}'
	make_full_corpus_archive "${root}" "supermicro_x10_full_corpus.tar.gz" "10.0.0.4" \
		'{"@odata.id":"/redfish/v1/Chassis/1","@odata.type":"#Chassis.v1_0_0.Chassis","PowerState":"On"}'
}

make_full_corpus_archive() {
	local root="$1"
	local archive="$2"
	local host="$3"
	local body="$4"
	local build="${BATS_TEST_TMPDIR}/${archive%.tar.gz}"
	rm -rf "${build}"
	mkdir -p "${build}/${host}"
	printf '%s\n' "${body}" >"${build}/${host}/resource.json"
	printf '{"schema_version":"1","artifact_type":"full_training","json_file_count":1}\n' \
		>"${build}/${host}/corpus_manifest.json"
	printf 'npy-stub\n' >"${build}/${host}/rest_api_map.npy"
	tar -czf "${root}/full_corpus/${archive}" -C "${build}" "${host}"
}

make_unsafe_link_archive() {
	local root="$1"
	local archive="$2"
	local build="${BATS_TEST_TMPDIR}/unsafe-${archive%.tar.gz}"
	rm -rf "${build}"
	mkdir -p "${build}/10.0.0.9"
	printf '{"@odata.id":"/redfish/v1/Systems/1"}\n' \
		>"${build}/10.0.0.9/resource.json"
	printf '{"schema_version":"1","artifact_type":"full_training","json_file_count":1}\n' \
		>"${build}/10.0.0.9/corpus_manifest.json"
	printf 'npy-stub\n' >"${build}/10.0.0.9/rest_api_map.npy"
	ln -s /tmp "${build}/10.0.0.9/unsafe-link"
	tar -czf "${root}/full_corpus/${archive}" -C "${build}" "10.0.0.9"
}

make_nested_map_archive() {
	local root="$1"
	local archive="$2"
	local build="${BATS_TEST_TMPDIR}/nested-${archive%.tar.gz}"
	rm -rf "${build}"
	mkdir -p "${build}/nested/10.0.0.9"
	printf '{"@odata.id":"/redfish/v1/Systems/1"}\n' \
		>"${build}/nested/10.0.0.9/resource.json"
	printf '{"schema_version":"1","artifact_type":"full_training","json_file_count":1}\n' \
		>"${build}/nested/10.0.0.9/corpus_manifest.json"
	printf 'npy-stub\n' >"${build}/nested/10.0.0.9/rest_api_map.npy"
	tar -czf "${root}/full_corpus/${archive}" -C "${build}" "nested"
}

@test "lab wrapper requires a private output directory" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	make_redfish_ctl_corpus "${corpus}"

	run env \
		FAKE_GIT_LOG="${BATS_TEST_TMPDIR}/git.log" \
		HOME="${BATS_TEST_TMPDIR}/home" \
		IGC_LAB_DRY_RUN=1 \
		IGC_REDFISH_CTL_DIR="${corpus}" \
		GOAL_PARAPHRASE_BASE_URL=http://example.invalid \
		GOAL_PARAPHRASE_MODEL=local-pro \
		bash "${REPO_ROOT}/scripts/build_goal_dataset_lab.sh"

	[ "$status" -eq 1 ]
	[[ "$output" == *"IGC_GOAL_DATASET_OUT"* ]]
}

@test "lab wrapper pulls LFS and renders full-corpus generation command" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	make_redfish_ctl_corpus "${corpus}"

	run env \
		FAKE_GIT_LOG="${BATS_TEST_TMPDIR}/git.log" \
		HOME="${BATS_TEST_TMPDIR}/home" \
		IGC_LAB_DRY_RUN=1 \
		IGC_REDFISH_CTL_DIR="${corpus}" \
		IGC_GOAL_DATASET_OUT="${BATS_TEST_TMPDIR}/out" \
		GOAL_PARAPHRASE_BASE_URL=http://example.invalid \
		GOAL_PARAPHRASE_MODEL=local-pro \
		bash "${REPO_ROOT}/scripts/build_goal_dataset_lab.sh"

	[ "$status" -eq 0 ]
	[[ "$output" == *"pulling redfish_ctl Git LFS objects"* ]]
	[[ "$output" == *"full corpus archive: dell_xr8620t_full_corpus.tar.gz"* ]]
	[[ "$output" == *"full corpus archive: supermicro_gb300_full_corpus.tar.gz"* ]]
	[[ "$output" == *"full corpus archive: hpe_dl360_full_corpus.tar.gz"* ]]
	[[ "$output" == *"full corpus archive: supermicro_x10_full_corpus.tar.gz"* ]]
	[[ "$output" == *"--generate-all-goals"* ]]
	[[ "$output" == *"--capture-root ${BATS_TEST_TMPDIR}/out/_build/full_corpus/dell_xr8620t_full_corpus/10.0.0.1"* ]]
	[[ "$output" == *"--capture-root ${BATS_TEST_TMPDIR}/out/_build/full_corpus/supermicro_gb300_full_corpus/10.0.0.2"* ]]
	[[ "$output" == *"git -C ${corpus} lfs pull"* ]]
}

@test "lab wrapper blocks when a required full corpus archive is missing" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	mkdir -p "${corpus}/full_corpus"
	touch "${corpus}/.git"

	run env \
		FAKE_GIT_LOG="${BATS_TEST_TMPDIR}/git.log" \
		HOME="${BATS_TEST_TMPDIR}/home" \
		IGC_LAB_DRY_RUN=1 \
		IGC_REDFISH_CTL_DIR="${corpus}" \
		IGC_GOAL_DATASET_OUT="${BATS_TEST_TMPDIR}/out" \
		GOAL_PARAPHRASE_BASE_URL=http://example.invalid \
		GOAL_PARAPHRASE_MODEL=local-pro \
		bash "${REPO_ROOT}/scripts/build_goal_dataset_lab.sh"

	[ "$status" -eq 1 ]
	[[ "$output" == *"missing required redfish_ctl full corpus archive"* ]]
}

@test "lab wrapper rejects unsafe link entries before extracting full corpora" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	make_redfish_ctl_corpus "${corpus}"
	make_unsafe_link_archive "${corpus}" "dell_xr8620t_full_corpus.tar.gz"

	run env \
		FAKE_GIT_LOG="${BATS_TEST_TMPDIR}/git.log" \
		HOME="${BATS_TEST_TMPDIR}/home" \
		IGC_LAB_DRY_RUN=1 \
		IGC_REDFISH_CTL_DIR="${corpus}" \
		IGC_GOAL_DATASET_OUT="${BATS_TEST_TMPDIR}/out" \
		GOAL_PARAPHRASE_BASE_URL=http://example.invalid \
		GOAL_PARAPHRASE_MODEL=local-pro \
		bash "${REPO_ROOT}/scripts/build_goal_dataset_lab.sh"

	[ "$status" -eq 1 ]
	[[ "$output" == *"unsafe link entry in redfish_ctl full corpus archive"* ]]
}

@test "lab wrapper rejects full corpora without top-level host map" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	make_redfish_ctl_corpus "${corpus}"
	make_nested_map_archive "${corpus}" "dell_xr8620t_full_corpus.tar.gz"

	run env \
		FAKE_GIT_LOG="${BATS_TEST_TMPDIR}/git.log" \
		HOME="${BATS_TEST_TMPDIR}/home" \
		IGC_LAB_DRY_RUN=1 \
		IGC_REDFISH_CTL_DIR="${corpus}" \
		IGC_GOAL_DATASET_OUT="${BATS_TEST_TMPDIR}/out" \
		GOAL_PARAPHRASE_BASE_URL=http://example.invalid \
		GOAL_PARAPHRASE_MODEL=local-pro \
		bash "${REPO_ROOT}/scripts/build_goal_dataset_lab.sh"

	[ "$status" -eq 1 ]
	[[ "$output" == *"lacks top-level host rest_api_map.npy"* ]]
}
