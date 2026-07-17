"""Offline tests for the NV72 fleet preflight decision logic.

TEAM_GUIDE mandates that unit tests never call the live dashboard: the
``/api/v1/state`` payload is mocked as small fixture dicts and the pre-flight
verdict logic is asserted against them. Pins the blocker rules — RoCE
degradation, ``/models`` mount degradation, a missing required model endpoint,
and a malformed/empty state all block; a healthy fleet passes.

Author:
Mus mbayramo@stanford.edu
"""

from igc.shared.nv72_preflight import evaluate_state


def _healthy_state(**overrides):
    """A minimal healthy /api/v1/state fixture."""
    state = {
        "summaries": {
            "roce": {"rdma_active": 18, "nodes": 18},
            "models_mount": {"mounted": 18, "nodes": 18},
        },
        "model_endpoints": {
            "flash": [{"model": "m", "url": "u", "role": "fast_worker"}],
            "pro": [],
        },
    }
    state.update(overrides)
    return state


def test_healthy_fleet_passes():
    """All-green summaries with no endpoint requirement pass."""
    ok, reasons = evaluate_state(_healthy_state())
    assert ok and reasons == []


def test_roce_degradation_blocks():
    """rdma_active < nodes is a blocker naming the counts."""
    state = _healthy_state()
    state["summaries"]["roce"] = {"rdma_active": 16, "nodes": 18}
    ok, reasons = evaluate_state(state)
    assert not ok
    assert any("RoCE" in r and "16/18" in r for r in reasons)


def test_models_mount_degradation_blocks():
    """An unmounted /models node is a blocker."""
    state = _healthy_state()
    state["summaries"]["models_mount"] = {"mounted": 17, "nodes": 18}
    ok, reasons = evaluate_state(state)
    assert not ok
    assert any("/models" in r for r in reasons)


def test_required_endpoint_absent_blocks():
    """Requiring a scaled-down endpoint group blocks with its name."""
    ok, reasons = evaluate_state(_healthy_state(), require_endpoint="pro")
    assert not ok
    assert any("'pro'" in r for r in reasons)


def test_required_endpoint_present_passes():
    """Requiring a serving group passes."""
    ok, reasons = evaluate_state(_healthy_state(), require_endpoint="flash")
    assert ok


def test_empty_state_blocks_on_missing_sections():
    """A degraded dashboard payload (no summaries) is itself a blocker."""
    ok, reasons = evaluate_state({})
    assert not ok
    assert len(reasons) == 2  # roce + models_mount sections missing


def test_zero_node_summaries_block():
    """Equal zero counts are not a healthy fleet."""
    ok, reasons = evaluate_state({
        "summaries": {
            "roce": {"rdma_active": 0, "nodes": 0},
            "models_mount": {"mounted": 0, "nodes": 0},
        }
    })
    assert not ok
    assert any("summaries.roce" in reason and "nodes" in reason for reason in reasons)
    assert any("summaries.models_mount" in reason and "nodes" in reason for reason in reasons)


def test_malformed_summary_counts_block():
    """Dashboard count fields must be real integer counts."""
    ok, reasons = evaluate_state({
        "summaries": {
            "roce": {"rdma_active": "18", "nodes": 18},
            "models_mount": {"mounted": True, "nodes": 18},
        }
    })
    assert not ok
    assert any("rdma_active" in reason for reason in reasons)
    assert any("mounted" in reason for reason in reasons)


# Author: Mus mbayramo@stanford.edu
