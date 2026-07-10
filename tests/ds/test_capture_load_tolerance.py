"""Offline regression: capture loading tolerates non-host subdirectories.

``RestTrajectory.load`` iterates every subdirectory of the capture root, but
real capture roots also hold ``orig/``/``post/``/``pre/`` (nested copies, no
top-level ``.npy``) and hosts whose crawl produced no map — the first such
directory previously raised and killed every on-node dataset rebuild. Loading
must skip them with a warning, keep the hosts that do carry maps, and only
fail when NO host has a map at all. Uses tiny synthetic ``.npy`` maps.

Author:
Mus mbayramo@stanford.edu
"""

import json

import numpy as np
import pytest

from igc.ds.ds_rest_trajectories import RestTrajectory


def _host_dir(root, name):
    """A minimal host capture dir: one response file + its .npy map."""
    host = root / name
    host.mkdir(parents=True)
    response = host / "redfish_v1.json"
    response.write_text(json.dumps({"@odata.id": "/redfish/v1"}))
    np.save(
        host / "rest_api_map.npy",
        {
            "url_file_mapping": {"/redfish/v1": str(response)},
            "allowed_methods_mapping": {"/redfish/v1": ["GET"]},
        },
    )
    return host


def test_non_host_subdirs_are_skipped(tmp_path):
    """orig/-style nested dirs and empty dirs don't abort the load."""
    _host_dir(tmp_path, "172.0.0.1")
    (tmp_path / "orig" / "172.0.0.1").mkdir(parents=True)  # nested, no flat .npy
    (tmp_path / "post").mkdir()  # empty

    # ctor contract: load() iterates rest_new_prefix (the SECOND arg — the
    # datasets/orig-style map root); raw_json_dir only anchors response remapping.
    trajectory = RestTrajectory(str(tmp_path), str(tmp_path))
    trajectory.load()

    mapping, methods = trajectory.merged_view()
    assert "/redfish/v1" in mapping
    assert methods["/redfish/v1"] == ["GET"]


def test_all_subdirs_mapless_raises(tmp_path):
    """A capture root with no usable host at all still fails loudly."""
    (tmp_path / "orig").mkdir()
    (tmp_path / "post").mkdir()

    trajectory = RestTrajectory(str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError, match="No host capture directory"):
        trajectory.load()


# Author: Mus mbayramo@stanford.edu
