"""Static checks for the Phase 2 validation guard wrappers.

These tests prove the wrapper routes through the guarded entrypoint. The actual
pytest and ruff commands still run only on the approved CI/Kubernetes or remote
CPU validation surface.

Author:
Mus mbayramo@stanford.edu
"""
from pathlib import Path


def test_phase2_validation_script_routes_through_guarded_check() -> None:
    """The Phase 2 wrapper must not invoke pytest directly."""
    text = Path("scripts/validate_phase2_labelled_requests.sh").read_text(encoding="utf-8")

    assert "scripts/check.sh --profile phase2_labelled_requests --category unit" in text
    assert "pytest " not in text
    assert "python -m pytest" not in text


def test_check_script_refuses_laptop_execution_by_default() -> None:
    """The shared check wrapper names the only accepted validation surfaces."""
    text = Path("scripts/check.sh").read_text(encoding="utf-8")

    assert "KUBERNETES_SERVICE_HOST" in text
    assert "IGC_REMOTE_VALIDATION" in text
    assert '$(pwd -P)" = "/workspace/igc"' in text
    assert "refusing laptop execution" in text
    assert "BLOCKER:" in text


def test_check_script_uses_python_module_pytest_for_phase2_unit_gate() -> None:
    """The guarded unit command uses the project interpreter module form."""
    text = Path("scripts/check.sh").read_text(encoding="utf-8")

    assert "python -m pytest -q -ra" in text
    assert "tests/ds/test_phase2_labelled_requests.py" in text
    assert "tests/scripts/test_phase2_labelled_requests_cli.py" in text
    assert "tests/modules/test_phase2_labelled_request_metric_keys.py" in text
