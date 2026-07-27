"""Offline tests for the Phase 1 Redfish source registry materializer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pytest
import torch

import igc.ds.source_registry as source_registry
from igc.ds.phase1_chunking import Phase1ChunkingPolicy, reassemble_phase1_rows
from igc.ds.source_registry import (
    load_source_registry,
    materialize_phase1_registry_corpus,
)
from igc.ds.sources.base import SourceRecord, TrustLevel


class _FakeRedfishAdapter:
    """Small source adapter whose records follow the mutated source label."""

    def __init__(
        self,
        *,
        source: str,
        trust_level: TrustLevel,
        urls: Iterable[str],
        vendor: str | None,
        payload_size: int = 0,
    ) -> None:
        self.source = source
        self.trust_level = trust_level
        self._urls = list(urls)
        self._vendor = vendor
        self._payload_size = payload_size

    def iter_records(self):
        for url in self._urls:
            response = {"@odata.id": url, "source": self.source}
            if self._payload_size:
                response["Payload"] = "x" * self._payload_size
            yield SourceRecord(
                url=url,
                response=response,
                source=self.source,
                trust_level=self.trust_level,
                vendor=self._vendor,
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _row_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_registry(
    path: Path,
    *,
    real_weight: float = 1.0,
    replay_weight: float = 1.0,
    require_all: bool = True,
    min_heldout_rows_per_source: object = 1,
) -> Path:
    path.write_text(
        f"""
version: 1
artifact_contract:
  producer: redfish_ctl_discovery
  json_glob: "*.json"
  api_map:
    accepted_names: ["rest_api_map.v1.json", "rest_api_map.npy"]
    required_keys: ["url_file_mapping", "allowed_methods_mapping"]
  semantics_bundle: semantics.json
sources:
  real_dell:
    origin: lab capture
    trust_level: REAL
    root_env: IGC_REAL_ROOT
    manifest_env: IGC_REAL_MANIFEST
    training_weight: {real_weight}
    evaluation_group: real_vendor
  dsp_replay:
    origin: DSP2043 replay
    trust_level: REPLAY
    root_env: IGC_REPLAY_ROOT
    manifest_env: IGC_REPLAY_MANIFEST
    training_weight: {replay_weight}
    evaluation_group: spec_replay
evaluation:
  report_by: ["evaluation_group", "trust_level"]
  real_observation_anchor: real_dell
  require_all_configured_sources: {str(require_all).lower()}
  min_heldout_rows_per_source: {min_heldout_rows_per_source}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _install_manifest_factory(
    monkeypatch,
    real_manifest: Path,
    replay_manifest: Path,
    *,
    real_urls: Iterable[str] = ("/redfish/v1/Systems/1",),
    replay_urls: Iterable[str] = ("/redfish/v1/Systems/1",),
    real_payload_size: int = 0,
    replay_payload_size: int = 0,
) -> list[dict]:
    import igc.ds.sources.redfish_fixture_source as fixture_source

    calls: list[dict] = []

    def fake_from_redfish_ctl_manifest(
        manifest_path: str,
        root: str,
        *,
        trust_level: TrustLevel,
        kind: str,
        require_api_map: bool = False,
    ):
        calls.append({
            "manifest_path": manifest_path,
            "root": root,
            "trust_level": trust_level,
            "kind": kind,
            "require_api_map": require_api_map,
        })
        if Path(manifest_path) == real_manifest.resolve():
            return [
                _FakeRedfishAdapter(
                    source="systems",
                    trust_level=trust_level,
                    urls=real_urls,
                    vendor="dell",
                    payload_size=real_payload_size,
                )
            ]
        if Path(manifest_path) == replay_manifest.resolve():
            return [
                _FakeRedfishAdapter(
                    source="mockup",
                    trust_level=trust_level,
                    urls=replay_urls,
                    vendor=None,
                    payload_size=replay_payload_size,
                )
            ]
        raise AssertionError(f"unexpected manifest path: {manifest_path}")

    monkeypatch.setattr(
        fixture_source.RedfishFixtureSource,
        "from_redfish_ctl_manifest",
        staticmethod(fake_from_redfish_ctl_manifest),
    )
    return calls


class _CharacterTokenizer:
    """Tokenizer stub whose token count equals rendered character count."""

    def __call__(
        self,
        text,
        *,
        padding=False,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        del padding, truncation, return_tensors, add_special_tokens
        ids = torch.arange(1, len(text) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _registry_env(
    tmp_path: Path,
    monkeypatch,
    *,
    real_urls: Iterable[str] = ("/redfish/v1/Systems/1",),
    replay_urls: Iterable[str] = ("/redfish/v1/Systems/1",),
) -> tuple[Path, dict[str, str]]:
    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text('{"source":"real_dell"}\n', encoding="utf-8")
    replay_manifest.write_text('{"source":"dsp_replay"}\n', encoding="utf-8")
    real_root = tmp_path / "real-root"
    replay_root = tmp_path / "replay-root"
    real_root.mkdir()
    replay_root.mkdir()
    _install_manifest_factory(
        monkeypatch,
        real_manifest,
        replay_manifest,
        real_urls=real_urls,
        replay_urls=replay_urls,
    )
    return registry, {
        "IGC_REAL_ROOT": str(real_root),
        "IGC_REAL_MANIFEST": str(real_manifest),
        "IGC_REPLAY_ROOT": str(replay_root),
        "IGC_REPLAY_MANIFEST": str(replay_manifest),
    }


def _write_redfish_ctl_manifest(path: Path, *, corpus_id: str, vendor: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpora": [
                    {
                        "id": corpus_id,
                        "kind": "dataset",
                        "vendor": vendor,
                        "model": "unit",
                        "archive": f"corpora/dataset/{corpus_id}.tar.gz",
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_materialized_corpus(
    materialized_root: Path,
    *,
    corpus_id: str,
    map_payload: object | None,
) -> None:
    corpus_root = materialized_root / "dataset" / corpus_id
    _write_json_resource(
        corpus_root / "json_responses" / "_redfish_v1_Systems_1.json",
        "/redfish/v1/Systems/1",
    )
    if map_payload is not None:
        (corpus_root / "rest_api_map.v1.json").write_text(
            json.dumps(map_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_json_resource(path: Path, rest_api: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"@odata.id": rest_api, "Name": path.stem}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_materialize_phase1_registry_corpus_records_registry_and_upstream_shas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """All env-named sources are resolved, written, and stamped with exact lineage."""
    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text('{"source":"real_dell"}\n', encoding="utf-8")
    replay_manifest.write_text('{"source":"dsp_replay"}\n', encoding="utf-8")
    real_root = tmp_path / "real-root"
    replay_root = tmp_path / "replay-root"
    real_root.mkdir()
    replay_root.mkdir()
    calls = _install_manifest_factory(monkeypatch, real_manifest, replay_manifest)

    summary = materialize_phase1_registry_corpus(
        registry_path=registry,
        output_root=tmp_path / "corpus",
        env={
            "IGC_REAL_ROOT": str(real_root),
            "IGC_REAL_MANIFEST": str(real_manifest),
            "IGC_REPLAY_ROOT": str(replay_root),
            "IGC_REPLAY_MANIFEST": str(replay_manifest),
        },
        eval_fraction=1.0,
        seed=3,
    )

    train_examples = tmp_path / "corpus" / "train" / "examples.jsonl"
    heldout_examples = tmp_path / "corpus" / "heldout" / "examples.jsonl"
    train_manifest = tmp_path / "corpus" / "train" / "manifest.json"
    heldout_manifest = tmp_path / "corpus" / "heldout" / "manifest.json"
    manifest_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
    train_rows = _read_jsonl(train_examples)
    heldout_rows = _read_jsonl(heldout_examples)
    rows_by_source = {
        row["metadata"]["source_corpus"]: row for row in [*train_rows, *heldout_rows]
    }

    assert [
        (
            Path(call["manifest_path"]),
            call["root"],
            call["trust_level"],
            call["kind"],
            call["require_api_map"],
        )
        for call in calls
    ] == [
        (real_manifest.resolve(), str(real_root), TrustLevel.REAL, "dataset", True),
        (replay_manifest.resolve(), str(replay_root), TrustLevel.REPLAY, "dataset", True),
    ]
    assert summary["source_registry_sha"] == _sha256(registry)
    assert summary["source_manifest_shas"] == {
        "dsp_replay": _sha256(replay_manifest),
        "real_dell": _sha256(real_manifest),
    }
    assert summary["train_artifact_sha"] == _sha256(train_examples)
    assert summary["heldout_artifact_sha"] == _sha256(heldout_examples)
    assert summary["written_manifest_sha"] == _sha256(train_manifest)
    assert summary["train_rows"] == 1
    assert summary["heldout_rows"] == 1
    assert len(train_rows) == 1
    assert len(heldout_rows) == 1
    assert set(rows_by_source) == {
        "real_vendor:real_dell:systems",
        "spec_replay:dsp_replay:mockup",
    }
    for source, row in rows_by_source.items():
        rest_api = row["x"]["rest_api"]
        assert set(row) == {"phase", "dataset", "task", "x", "y_true", "metadata"}
        assert row["phase"] == 1
        assert row["dataset"] == "D0"
        assert row["task"] == "redfish_json_reconstruction"
        assert row["x"] == {
            "rest_api": "/redfish/v1/Systems/1",
            "allowed_methods": [],
            "json": row["y_true"]["json"],
        }
        assert row["x"]["json"]["source"] == source
        assert row["metadata"]["row_id"] == _row_id(source, rest_api)
        assert row["metadata"]["source_corpus"] == source
        assert row["metadata"]["trust_level"] in {"REAL", "REPLAY"}
    assert rows_by_source["real_vendor:real_dell:systems"]["metadata"] == {
        "row_id": _row_id("real_vendor:real_dell:systems", "/redfish/v1/Systems/1"),
        "source_corpus": "real_vendor:real_dell:systems",
        "trust_level": "REAL",
        "vendor": "dell",
    }
    assert rows_by_source["spec_replay:dsp_replay:mockup"]["metadata"] == {
        "row_id": _row_id("spec_replay:dsp_replay:mockup", "/redfish/v1/Systems/1"),
        "source_corpus": "spec_replay:dsp_replay:mockup",
        "trust_level": "REPLAY",
        "vendor": None,
    }
    assert heldout_manifest.read_text(encoding="utf-8") == train_manifest.read_text(
        encoding="utf-8"
    )
    assert manifest_payload["source_registry_sha"] == _sha256(registry)
    assert manifest_payload["source_manifest_shas"] == summary["source_manifest_shas"]
    assert manifest_payload["by_trust"] == {"REAL": 1, "REPLAY": 1}
    assert manifest_payload["sources"] == [
        "real_vendor:real_dell:systems",
        "spec_replay:dsp_replay:mockup",
    ]
    assert manifest_payload["required_heldout_sources"] == [
        "real_vendor:real_dell:systems"
    ]
    assert manifest_payload["heldout_by_source"] == {
        "real_vendor:real_dell:systems": 1
    }
    assert manifest_payload["min_eval_per_source"] == 1


def test_materialize_phase1_registry_corpus_chunks_after_split_and_records_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Lossless chunks stay in one split and replace manifest row IDs deterministically."""

    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text('{"source":"real_dell"}\n', encoding="utf-8")
    replay_manifest.write_text('{"source":"dsp_replay"}\n', encoding="utf-8")
    real_root = tmp_path / "real-root"
    replay_root = tmp_path / "replay-root"
    real_root.mkdir()
    replay_root.mkdir()
    _install_manifest_factory(
        monkeypatch,
        real_manifest,
        replay_manifest,
        real_payload_size=420,
        replay_payload_size=420,
    )
    policy = Phase1ChunkingPolicy(
        transform="phase1.lossless-json-chunk.v1",
        tokenizer_sha="sha256:" + "1" * 64,
        max_tokens=320,
        telemetry_rest_api_markers=("/TelemetryService",),
    )
    output = tmp_path / "corpus"

    summary = materialize_phase1_registry_corpus(
        registry_path=registry,
        output_root=output,
        env={
            "IGC_REAL_ROOT": str(real_root),
            "IGC_REAL_MANIFEST": str(real_manifest),
            "IGC_REPLAY_ROOT": str(replay_root),
            "IGC_REPLAY_MANIFEST": str(replay_manifest),
        },
        eval_fraction=1.0,
        seed=3,
        tokenizer=_CharacterTokenizer(),
        chunking_policy=policy,
    )

    train_rows = _read_jsonl(output / "train" / "examples.jsonl")
    heldout_rows = _read_jsonl(output / "heldout" / "examples.jsonl")
    manifest = json.loads((output / "train" / "manifest.json").read_text())
    groups: dict[str, list[dict]] = {}
    for split_rows in (train_rows, heldout_rows):
        split_original_ids = {
            row["metadata"]["chunk"]["original_row_id"] for row in split_rows
        }
        assert len(split_original_ids) == 1
        for row in split_rows:
            original_id = row["metadata"]["chunk"]["original_row_id"]
            groups.setdefault(original_id, []).append(row)
    assert set(row["metadata"]["row_id"] for row in train_rows) == set(
        manifest["train_row_ids"]
    )
    assert set(row["metadata"]["row_id"] for row in heldout_rows) == set(
        manifest["heldout_row_ids"]
    )
    assert set(manifest["train_row_ids"]).isdisjoint(manifest["heldout_row_ids"])
    assert all(
        reassemble_phase1_rows(group)["Payload"] == "x" * 420
        for group in groups.values()
    )
    assert summary["train_resources"] == 1
    assert summary["heldout_resources"] == 1
    assert summary["train_rows"] > summary["train_resources"]
    assert summary["heldout_rows"] > summary["heldout_resources"]
    assert manifest["phase1_transform"]["exact_reassembly_verified"] is True
    assert manifest["phase1_transform"]["original_token_lengths"][
        "non_telemetry_resources"
    ]["count"] == 2


def test_materialize_phase1_registry_corpus_publishes_release_dir_without_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A successful materialization publishes train/heldout once and removes pending."""
    registry, env = _registry_env(tmp_path, monkeypatch)
    output = tmp_path / "corpus-release"
    pending = output.with_name(f"{output.name}.pending")

    summary = materialize_phase1_registry_corpus(
        registry_path=registry,
        output_root=output,
        env=env,
        eval_fraction=1.0,
        seed=3,
    )

    assert output.is_dir()
    assert (output / "train" / "examples.jsonl").is_file()
    assert (output / "train" / "manifest.json").is_file()
    assert (output / "heldout" / "examples.jsonl").is_file()
    assert (output / "heldout" / "manifest.json").is_file()
    assert not pending.exists()
    assert summary["train_rows"] == 1
    assert summary["heldout_rows"] == 1


def test_materialize_phase1_registry_corpus_rejects_existing_final_release_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An immutable canonical release directory is never overwritten."""
    registry, env = _registry_env(tmp_path, monkeypatch)
    output = tmp_path / "corpus-release"
    pending = output.with_name(f"{output.name}.pending")
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("canonical bytes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release already exists"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=output,
            env=env,
            eval_fraction=1.0,
            seed=3,
        )

    assert sentinel.read_text(encoding="utf-8") == "canonical bytes\n"
    assert not pending.exists()


def test_materialize_phase1_registry_corpus_rejects_stale_pending_release_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A stale pending directory blocks publication and is left for inspection."""
    registry, env = _registry_env(tmp_path, monkeypatch)
    output = tmp_path / "corpus-release"
    pending = output.with_name(f"{output.name}.pending")
    pending.mkdir()
    sentinel = pending / "sentinel.txt"
    sentinel.write_text("pending bytes\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stale Phase 1 corpus pending release"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=output,
            env=env,
            eval_fraction=1.0,
            seed=3,
        )

    assert not output.exists()
    assert pending.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "pending bytes\n"


def test_materialize_phase1_registry_corpus_cleans_pending_on_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The only canonical publication is pending-dir to final-dir os.replace."""
    registry, env = _registry_env(tmp_path, monkeypatch)
    output = (tmp_path / "corpus-release").resolve()
    pending = output.with_name(f"{output.name}.pending")
    replace_calls: list[tuple[Path, Path]] = []

    def fail_replace(src: Path, dst: Path) -> None:
        replace_calls.append((Path(src), Path(dst)))
        raise OSError("simulated source-registry publish failure")

    monkeypatch.setattr(source_registry.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated source-registry publish failure"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=output,
            env=env,
            eval_fraction=1.0,
            seed=3,
        )

    assert replace_calls == [(pending, output)]
    assert not output.exists()
    assert not pending.exists()


@pytest.mark.parametrize(
    ("map_payload", "error_type", "message"),
    [
        (None, FileNotFoundError, "REST API map is missing"),
        (["not", "an", "object"], ValueError, "REST API map must be an object"),
        (
            {"allowed_methods_mapping": {"/redfish/v1/Systems/1": ["GET"]}},
            ValueError,
            "requires url_file_mapping",
        ),
        (
            {"url_file_mapping": {"/redfish/v1/Systems/1": "json_responses/a.json"}},
            ValueError,
            "requires url_file_mapping",
        ),
        (
            {"url_file_mapping": {}, "allowed_methods_mapping": {}},
            ValueError,
            "REST API map cannot be empty",
        ),
    ],
)
def test_materialize_phase1_registry_corpus_requires_complete_api_maps_before_publish(
    tmp_path: Path,
    map_payload: object | None,
    error_type: type[Exception],
    message: str,
) -> None:
    """Canonical registry materialization fails before publishing without API maps."""
    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = _write_redfish_ctl_manifest(
        tmp_path / "real.manifest.json",
        corpus_id="real-dell",
        vendor="dell",
    )
    replay_manifest = _write_redfish_ctl_manifest(
        tmp_path / "replay.manifest.json",
        corpus_id="dsp-replay",
        vendor="dmtf",
    )
    real_root = tmp_path / "real-root"
    replay_root = tmp_path / "replay-root"
    _write_materialized_corpus(
        real_root,
        corpus_id="real-dell",
        map_payload=map_payload,
    )
    _write_materialized_corpus(
        replay_root,
        corpus_id="dsp-replay",
        map_payload={
            "url_file_mapping": {
                "/redfish/v1/Systems/1": "json_responses/_redfish_v1_Systems_1.json",
            },
            "allowed_methods_mapping": {"/redfish/v1/Systems/1": ["GET"]},
        },
    )
    output = tmp_path / "corpus"

    with pytest.raises(error_type, match=message):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=output,
            env={
                "IGC_REAL_ROOT": str(real_root),
                "IGC_REAL_MANIFEST": str(real_manifest),
                "IGC_REPLAY_ROOT": str(replay_root),
                "IGC_REPLAY_MANIFEST": str(replay_manifest),
            },
            eval_fraction=1.0,
            seed=3,
        )

    assert not output.exists()
    assert not output.with_name(f"{output.name}.pending").exists()


@pytest.mark.parametrize(
    "floor",
    [
        0,
        -1,
        "false",
        "true",
    ],
)
def test_source_registry_rejects_non_positive_or_bool_heldout_floor(
    tmp_path: Path,
    floor: object,
) -> None:
    """The spec-driven held-out floor must be a positive integer."""
    registry = _write_registry(
        tmp_path / "redfish_sources.yaml",
        min_heldout_rows_per_source=floor,
    )

    with pytest.raises(ValueError, match="min_heldout_rows_per_source"):
        load_source_registry(registry)


def test_materialize_phase1_registry_corpus_passes_configured_floor_to_mixer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Registry materialization applies the configured REAL-source held-out floor."""
    registry = _write_registry(
        tmp_path / "redfish_sources.yaml",
        min_heldout_rows_per_source=2,
    )
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text('{"source":"real_dell"}\n', encoding="utf-8")
    replay_manifest.write_text('{"source":"dsp_replay"}\n', encoding="utf-8")
    real_root = tmp_path / "real-root"
    replay_root = tmp_path / "replay-root"
    real_root.mkdir()
    replay_root.mkdir()
    _install_manifest_factory(
        monkeypatch,
        real_manifest,
        replay_manifest,
        real_urls=(
            "/redfish/v1/Systems/1",
            "/redfish/v1/Systems/2",
            "/redfish/v1/Systems/3",
        ),
        replay_urls=("/redfish/v1/Systems/1",),
    )

    summary = materialize_phase1_registry_corpus(
        registry_path=registry,
        output_root=tmp_path / "corpus",
        env={
            "IGC_REAL_ROOT": str(real_root),
            "IGC_REAL_MANIFEST": str(real_manifest),
            "IGC_REPLAY_ROOT": str(replay_root),
            "IGC_REPLAY_MANIFEST": str(replay_manifest),
        },
        eval_fraction=0.0,
        seed=3,
    )

    train_rows = _read_jsonl(tmp_path / "corpus" / "train" / "examples.jsonl")
    heldout_rows = _read_jsonl(tmp_path / "corpus" / "heldout" / "examples.jsonl")
    manifest_payload = json.loads(
        (tmp_path / "corpus" / "train" / "manifest.json").read_text(encoding="utf-8")
    )

    assert summary["train_rows"] == 2
    assert summary["heldout_rows"] == 2
    assert manifest_payload["min_eval_per_source"] == 2
    assert manifest_payload["heldout_by_source"] == {
        "real_vendor:real_dell:systems": 2
    }
    assert sum(
        row["metadata"]["source_corpus"] == "real_vendor:real_dell:systems"
        for row in heldout_rows
    ) == 2
    assert all(
        row["metadata"]["source_corpus"] != "spec_replay:dsp_replay:mockup"
        for row in heldout_rows
    )
    assert any(
        row["metadata"]["source_corpus"] == "spec_replay:dsp_replay:mockup"
        for row in train_rows
    )


def test_materialize_phase1_registry_corpus_requires_all_configured_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A required source missing its root/manifest env blocks materialization."""
    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text("real\n", encoding="utf-8")
    replay_manifest.write_text("replay\n", encoding="utf-8")
    _install_manifest_factory(monkeypatch, real_manifest, replay_manifest)

    with pytest.raises(ValueError, match="required source 'dsp_replay'"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=tmp_path / "corpus",
            env={
                "IGC_REAL_ROOT": str(tmp_path / "real-root"),
                "IGC_REAL_MANIFEST": str(real_manifest),
            },
        )


def test_materialize_phase1_registry_corpus_rejects_unimplemented_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The registry corpus path is strict until weighted sampling is implemented."""
    registry = _write_registry(tmp_path / "redfish_sources.yaml", real_weight=0.5)
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text("real\n", encoding="utf-8")
    replay_manifest.write_text("replay\n", encoding="utf-8")
    _install_manifest_factory(monkeypatch, real_manifest, replay_manifest)

    with pytest.raises(ValueError, match="training_weight is not implemented"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=tmp_path / "corpus",
            env={
                "IGC_REAL_ROOT": str(tmp_path / "real-root"),
                "IGC_REAL_MANIFEST": str(real_manifest),
                "IGC_REPLAY_ROOT": str(tmp_path / "replay-root"),
                "IGC_REPLAY_MANIFEST": str(replay_manifest),
            },
        )


def test_materialize_phase1_registry_corpus_rejects_sources_with_no_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A configured manifest that emits no corpus adapters cannot be silently skipped."""
    import igc.ds.sources.redfish_fixture_source as fixture_source

    registry = _write_registry(tmp_path / "redfish_sources.yaml")
    real_manifest = tmp_path / "real.manifest.json"
    replay_manifest = tmp_path / "replay.manifest.json"
    real_manifest.write_text("real\n", encoding="utf-8")
    replay_manifest.write_text("replay\n", encoding="utf-8")
    monkeypatch.setattr(
        fixture_source.RedfishFixtureSource,
        "from_redfish_ctl_manifest",
        staticmethod(lambda *_args, **_kwargs: []),
    )

    with pytest.raises(ValueError, match="selected no 'dataset' corpora"):
        materialize_phase1_registry_corpus(
            registry_path=registry,
            output_root=tmp_path / "corpus",
            env={
                "IGC_REAL_ROOT": str(tmp_path / "real-root"),
                "IGC_REAL_MANIFEST": str(real_manifest),
                "IGC_REPLAY_ROOT": str(tmp_path / "replay-root"),
                "IGC_REPLAY_MANIFEST": str(replay_manifest),
            },
        )
