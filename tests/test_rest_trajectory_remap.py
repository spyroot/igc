"""Offline tests for RestTrajectory.remap_respond_location path portability.

The discovery ``.npy`` mapping stores absolute response paths rooted at the
capture machine's raw dir. When the dataset is rebuilt on a different machine
those paths must be re-rooted to a host-relative ``/<host>/<file>`` suffix so the
downstream ``_default_original_dir + value`` lookups resolve. These tests pin
that remap and guard against the regression where a cross-machine prefix left
every value untouched and produced an empty dataset.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.ds.ds_rest_trajectories import RestTrajectory


def _make_trajectory(tmp_path: Path, host: str) -> RestTrajectory:
    """Build a RestTrajectory over existing tmp dirs (no discovery/IO)."""
    raw = tmp_path / ".json_responses"
    (raw / host).mkdir(parents=True)
    orig = tmp_path / "datasets" / "orig"
    orig.mkdir(parents=True)
    return RestTrajectory(raw_json_dir=str(raw), rest_new_prefix=str(orig))


def test_remap_cross_machine_absolute_paths(tmp_path: Path) -> None:
    """Capture-machine absolute paths are re-rooted to /<host>/<file> here."""
    host = "192.0.2.10"
    rt = _make_trajectory(tmp_path, host)
    rt._rest_map_data[host] = {
        "rest_api_map": {
            "/redfish/v1": f"/Users/capture/.json_responses/{host}/_redfish_v1.json",
            "/redfish/v1/Systems": f"/Users/capture/.json_responses/{host}/_redfish_v1_Systems.json",
        }
    }

    rt.remap_respond_location(host)

    data = rt._rest_map_data[host]["rest_api_map"]
    assert data["/redfish/v1"] == f"/{host}/_redfish_v1.json"
    assert data["/redfish/v1/Systems"] == f"/{host}/_redfish_v1_Systems.json"


def test_remap_same_machine_strips_raw_dir(tmp_path: Path) -> None:
    """Same-machine absolute paths still yield the host-relative suffix."""
    host = "h1"
    rt = _make_trajectory(tmp_path, host)
    raw_resolved = rt.raw_json_dir
    rt._rest_map_data[host] = {
        "rest_api_map": {"/a": f"{raw_resolved}/{host}/_a.json"}
    }

    rt.remap_respond_location(host)

    assert rt._rest_map_data[host]["rest_api_map"]["/a"] == f"/{host}/_a.json"


def test_remap_is_idempotent_on_relative_values(tmp_path: Path) -> None:
    """Re-running on already host-relative values is a no-op."""
    host = "h1"
    rt = _make_trajectory(tmp_path, host)
    rt._rest_map_data[host] = {"rest_api_map": {"/a": f"/{host}/_a.json"}}

    rt.remap_respond_location(host)

    assert rt._rest_map_data[host]["rest_api_map"]["/a"] == f"/{host}/_a.json"


# Author: Mus mbayramo@stanford.edu
