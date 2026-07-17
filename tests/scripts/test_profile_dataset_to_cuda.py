import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from scripts.profile_dataset_to_cuda import _sample_corpus_tar


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _rest_map_bytes(payload: dict) -> bytes:
    out = io.BytesIO()
    np.save(out, payload, allow_pickle=True)
    return out.getvalue()


def test_sample_corpus_tar_warns_on_corrupt_rest_map(tmp_path, capsys):
    corpus_tar = tmp_path / "corrupt-map.tar.gz"
    with tarfile.open(corpus_tar, "w:gz") as tar:
        _add_bytes(tar, "host/rest_api_map.npy", b"not a numpy payload")
        _add_bytes(
            tar,
            "host/_redfish_v1.json",
            json.dumps({"@odata.id": "/redfish/v1/"}).encode("utf-8"),
        )

    with pytest.raises(RuntimeError, match="no usable Redfish JSON rows"):
        _sample_corpus_tar(
            corpus_tar,
            tmp_path / "out",
            sample_rows=1,
            trust_npy_pickle=True,
        )

    captured = capsys.readouterr()
    assert "WARNING: could not load host/rest_api_map.npy" in captured.err


def test_sample_corpus_tar_resolves_historical_absolute_map_values(tmp_path):
    corpus_tar = tmp_path / "absolute-map.tar.gz"
    resource_name = "_redfish_v1_Systems_1.json"
    rest_map = {
        "url_file_mapping": {
            "/redfish/v1/Systems/1": f"/historic/export/{resource_name}",
        },
        "allowed_methods_mapping": {
            "/redfish/v1/Systems/1": ["GET", "PATCH"],
        },
    }
    with tarfile.open(corpus_tar, "w:gz") as tar:
        _add_bytes(tar, "host/rest_api_map.npy", _rest_map_bytes(rest_map))
        _add_bytes(
            tar,
            f"host/{resource_name}",
            json.dumps({"@odata.id": "/redfish/v1/Systems/1"}).encode("utf-8"),
        )

    corpus_dir = _sample_corpus_tar(
        corpus_tar,
        tmp_path / "out",
        sample_rows=1,
        trust_npy_pickle=True,
    )

    row = json.loads(Path(corpus_dir, "examples.jsonl").read_text().splitlines()[0])
    assert row["request_or_action"] == {
        "method": "GET",
        "url": "/redfish/v1/Systems/1",
    }
    assert row["allowed_methods"] == ["GET", "PATCH"]
    assert row["source"]["member"] == f"host/{resource_name}"
