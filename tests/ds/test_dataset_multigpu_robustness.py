"""Offline regressions for the multi-GPU on-node dataset build.

The first FSDP2 run on a fresh GB300 node exposed two build-path crashes that the
local gate never hit (prebuilt caches bypass the build): a strict-UTF-8 read of a
captured response carrying a latin-1 byte, and a check-then-act ``copytree`` race
where every rank passed the ``os.path.exists`` guard before any rank created the
directory, so all-but-one died with ``FileExistsError``. Both are driven directly
here with class doubles over the real modules.

Author:
Mus mbayramo@stanford.edu
"""

import os

from igc.ds import redfish_dataset
from igc.ds.igc_json_pipeline import JsonPipeline
from igc.ds.redfish_dataset import JSONDataset


def test_load_json_files_tolerates_non_utf8_bytes(tmp_path):
    """A captured .json with a latin-1 byte (0xa3) loads without UnicodeDecodeError."""
    (tmp_path / "sensor.json").write_bytes(
        b'{"Name": "Inlet \xa3 sensor", "ReadingCelsius": 42}'
    )

    pipeline = JsonPipeline.__new__(JsonPipeline)
    pipeline._json_directory_path = str(tmp_path)
    # extract_recursive touches these when it meets action/target keys; the payload
    # above has none, but initialise them so the walk cannot AttributeError.
    pipeline._json_target = {}
    pipeline._action_to_rest = {}
    pipeline._target_names = set()

    # Must not raise; the stray byte is replaced, the JSON still parses.
    pipeline.load_json_files()


def test_load_json_files_skips_empty_and_malformed(tmp_path):
    """Empty (204) and non-JSON captures are skipped; valid ones still process."""
    (tmp_path / "empty.json").write_text("")            # 204/empty body
    (tmp_path / "truncated.json").write_text('{"Name": ')  # truncated write
    (tmp_path / "errorpage.json").write_text("<html>Gateway Timeout</html>")
    (tmp_path / "good.json").write_text('{"Name": "ok"}')

    pipeline = JsonPipeline.__new__(JsonPipeline)
    pipeline._json_directory_path = str(tmp_path)
    pipeline._json_target = {}
    pipeline._action_to_rest = {}
    pipeline._target_names = set()

    # Must not raise despite three unparseable files in the directory.
    pipeline.load_json_files()


def test_copy_json_responses_survives_concurrent_dir_creation(tmp_path, monkeypatch):
    """Simulate the multi-rank TOCTOU: the exist-guard says 'absent' but a peer rank
    already created the destination before copytree — must merge, not FileExistsError.
    """
    src = tmp_path / "unprocessed"
    src.mkdir()
    (src / "a.json").write_text('{"x": 1}')
    dst = tmp_path / "orig"
    dst.mkdir()  # a peer rank already made it

    ds = JSONDataset.__new__(JSONDataset)
    ds._unprocessed = str(src)
    ds._json_directory_path = str(dst)
    ds.logger = redfish_dataset.AbstractLogger.create_logger(__name__)
    ds._json_files = []

    # Make ONLY the destination look absent to the guard (as every rank sees it),
    # while it physically exists — the exact window that produced FileExistsError.
    real_exists = os.path.exists
    dst_key = str(dst).rstrip("/")
    monkeypatch.setattr(
        "os.path.exists",
        lambda p: False if str(p).rstrip("/") == dst_key else real_exists(p),
    )

    ds._copy_json_responses()  # must not raise

    assert (dst / "a.json").exists()


# Author: Mus mbayramo@stanford.edu
