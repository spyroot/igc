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
	mkdir -p \
		"${root}/tests/idrac_fixtures" \
		"${root}/tests/supermicro_fixtures" \
		"${root}/tests/hpe_fixtures" \
		"${root}/tests/generic_fixtures"
	touch "${root}/.git"
	printf '{"@odata.id":"/redfish/v1/Systems/1","@odata.type":"#ComputerSystem.v1_20_0.ComputerSystem","PowerState":"Off"}\n' \
		>"${root}/tests/idrac_fixtures/system.json"
	printf '{"@odata.id":"/redfish/v1/Managers/1/NetworkProtocol","@odata.type":"#ManagerNetworkProtocol.v1_10_0.ManagerNetworkProtocol","NTP":{"ProtocolEnabled":false}}\n' \
		>"${root}/tests/supermicro_fixtures/ntp.json"
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
	[[ "$output" == *"--generate-all-goals"* ]]
	[[ "$output" == *"--capture-root ${corpus}/tests/idrac_fixtures"* ]]
	[[ "$output" == *"--capture-root ${corpus}/tests/supermicro_fixtures"* ]]
	[[ "$output" == *"git -C ${corpus} lfs pull"* ]]
}

@test "lab wrapper blocks when LFS leaves no JSON captures" {
	install_lab_fakes
	corpus="${BATS_TEST_TMPDIR}/redfish_ctl"
	mkdir -p "${corpus}/tests/idrac_fixtures"
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
	[[ "$output" == *"no JSON files after git lfs pull"* ]]
}
