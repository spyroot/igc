"""Strict source registry for real and DSP2043-derived Redfish corpora."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SOURCE_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "data"
    / "redfish_sources.yaml"
)
_TRUST_LEVELS = {"REAL", "REPLAY", "SIM_VENDOR", "SIM_GENERIC", "SIM_DRIFT"}


@dataclass(frozen=True)
class RedfishArtifactContract:
    """Artifact names and semantic evidence emitted around one discovery crawl."""

    producer: str
    json_glob: str
    api_map_names: tuple[str, ...]
    api_map_keys: tuple[str, ...]
    semantics_bundle: str


@dataclass(frozen=True)
class RedfishSourceSpec:
    """One provenance and sampling entry in the shared Redfish source registry."""

    name: str
    origin: str
    trust_level: str
    root_env: str
    manifest_env: str
    training_weight: float
    evaluation_group: str
    spec_sha256: str


@dataclass(frozen=True)
class RedfishSourceRegistry:
    """Resolved discovery artifact contract and its configured source populations."""

    artifact_contract: RedfishArtifactContract
    sources: Mapping[str, RedfishSourceSpec]
    report_by: tuple[str, ...]
    real_observation_anchor: str
    require_all_configured_sources: bool
    min_heldout_rows_per_source: int
    spec_sha256: str


def load_source_registry(
    path: str | Path = SOURCE_SPEC_PATH,
) -> RedfishSourceRegistry:
    """Load and strictly validate the common redfish_ctl corpus contract."""
    spec_path = Path(path)
    raw_bytes = spec_path.read_bytes()
    try:
        payload = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse source registry {spec_path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("Redfish source registry version must be 1")
    digest = f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"

    artifact = _mapping(payload, "artifact_contract")
    api_map = _mapping(artifact, "api_map")
    contract = RedfishArtifactContract(
        producer=_required_text(artifact, "producer"),
        json_glob=_required_text(artifact, "json_glob"),
        api_map_names=_string_tuple(api_map, "accepted_names"),
        api_map_keys=_string_tuple(api_map, "required_keys"),
        semantics_bundle=_required_text(artifact, "semantics_bundle"),
    )


    if set(contract.api_map_keys) != {
        "url_file_mapping",
        "allowed_methods_mapping",
    }:
        raise ValueError("api_map.required_keys must contain the redfish_ctl map keys")

    raw_sources = _mapping(payload, "sources")
    sources: dict[str, RedfishSourceSpec] = {}
    required_source_fields = {
        "origin",
        "trust_level",
        "root_env",
        "manifest_env",
        "training_weight",
        "evaluation_group",
    }
    for name, raw in raw_sources.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("source names must map to objects")
        if set(raw) != required_source_fields:
            raise ValueError(f"source {name!r} has missing or unknown fields")
        trust_level = _required_text(raw, "trust_level")
        if trust_level not in _TRUST_LEVELS:
            raise ValueError(f"source {name!r} has unknown trust_level {trust_level!r}")
        weight = float(raw["training_weight"])
        if weight <= 0:
            raise ValueError(f"source {name!r} training_weight must be positive")
        sources[name] = RedfishSourceSpec(
            name=name,
            origin=_required_text(raw, "origin"),
            trust_level=trust_level,
            root_env=_required_text(raw, "root_env"),
            manifest_env=_required_text(raw, "manifest_env"),
            training_weight=weight,
            evaluation_group=_required_text(raw, "evaluation_group"),
            spec_sha256=digest,
        )
    if not sources:
        raise ValueError("source registry must contain at least one source")

    evaluation = _mapping(payload, "evaluation")
    anchor = _required_text(evaluation, "real_observation_anchor")
    if anchor not in sources or sources[anchor].trust_level != "REAL":
        raise ValueError("real_observation_anchor must name a REAL source")
    min_heldout_rows = evaluation.get("min_heldout_rows_per_source")
    if (
        not isinstance(min_heldout_rows, int)
        or isinstance(min_heldout_rows, bool)
        or min_heldout_rows < 1
    ):
        raise ValueError("min_heldout_rows_per_source must be a positive integer")
    return RedfishSourceRegistry(
        artifact_contract=contract,
        sources=sources,
        report_by=_string_tuple(evaluation, "report_by"),
        real_observation_anchor=anchor,
        require_all_configured_sources=bool(
            evaluation.get("require_all_configured_sources", False)
        ),
        min_heldout_rows_per_source=min_heldout_rows,
        spec_sha256=digest,
    )


def materialize_phase1_registry_corpus(
    *,
    registry_path: str | Path,
    output_root: str | Path,
    env: Mapping[str, str] | None = None,
    corpus_kind: str = "dataset",
    eval_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    """Build immutable train/held-out Phase 1 corpora from every registry source."""
    from igc.ds.phase1_render import build_phase1_source_row
    from igc.ds.sources.base import TrustLevel
    from igc.ds.sources.corpus_io import write_corpus
    from igc.ds.sources.mixer import SourceMix
    from igc.ds.sources.redfish_fixture_source import RedfishFixtureSource

    registry = load_source_registry(registry_path)
    environment = os.environ if env is None else env
    adapters = []
    source_manifest_shas: dict[str, str] = {}
    for source_name, source_spec in registry.sources.items():
        if source_spec.training_weight != 1.0:
            raise ValueError(
                f"source {source_name!r} training_weight is not implemented; "
                "use 1.0 until a weighted sampler is part of the corpus contract"
            )
        root = str(environment.get(source_spec.root_env, "")).strip()
        manifest = str(environment.get(source_spec.manifest_env, "")).strip()
        if not root or not manifest:
            if registry.require_all_configured_sources:
                missing = source_spec.root_env if not root else source_spec.manifest_env
                raise ValueError(
                    f"required source {source_name!r} is missing environment variable {missing}"
                )
            continue
        manifest_path = Path(manifest).expanduser().resolve()
        source_adapters = RedfishFixtureSource.from_redfish_ctl_manifest(
            str(manifest_path),
            root,
            trust_level=TrustLevel[source_spec.trust_level],
            kind=corpus_kind,
            require_api_map=True,
        )
        if not source_adapters:
            raise ValueError(
                f"source {source_name!r} manifest selected no {corpus_kind!r} corpora"
            )
        source_manifest_shas[source_name] = _sha256_file(manifest_path)
        for adapter in source_adapters:
            adapter.source = ":".join((
                source_spec.evaluation_group,
                source_name,
                adapter.source,
            ))
        adapters.extend(source_adapters)

    if not adapters:
        raise ValueError("source registry resolved no corpus adapters")
    mix = SourceMix(
        adapters,
        eval_fraction=eval_fraction,
        eval_trust_floor=TrustLevel.REAL,
        min_eval_per_source=registry.min_heldout_rows_per_source,
        seed=seed,
    )
    train_records, heldout_records = mix.split()
    if not train_records or not heldout_records:
        raise ValueError("registry corpus split must contain train and held-out rows")
    observed_registry_sources = {
        record.source.split(":", 2)[1]
        for record in mix.records()
        if record.source.count(":") >= 2
    }
    if (
        registry.require_all_configured_sources
        and observed_registry_sources != set(registry.sources)
    ):
        missing = sorted(set(registry.sources) - observed_registry_sources)
        raise ValueError(f"configured registry sources emitted no records: {missing}")

    manifest = mix.manifest()
    manifest.source_registry_sha = registry.spec_sha256
    manifest.source_manifest_shas = dict(sorted(source_manifest_shas.items()))
    output = Path(output_root).expanduser().resolve()
    pending = output.with_name(f"{output.name}.pending")
    release_lock = output.with_name(f"{output.name}.release.lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with release_lock.open("xb"):
            pass
    except FileExistsError as exc:
        raise ValueError(
            "Phase 1 corpus release is locked; an active or stale publisher "
            "requires inspection"
        ) from exc
    try:
        if output.exists():
            raise ValueError(
                f"immutable Phase 1 corpus release already exists: {output}"
            )
        if pending.exists():
            raise ValueError(
                f"stale Phase 1 corpus pending release requires inspection: {pending}"
            )
        pending.mkdir()
        try:
            train_paths = write_corpus(
                (build_phase1_source_row(record) for record in train_records),
                manifest,
                str(pending / "train"),
            )
            heldout_paths = write_corpus(
                (build_phase1_source_row(record) for record in heldout_records),
                manifest,
                str(pending / "heldout"),
            )
            train_artifact_sha = _sha256_file(Path(train_paths["examples"]))
            heldout_artifact_sha = _sha256_file(Path(heldout_paths["examples"]))
            written_manifest_sha = _sha256_file(Path(train_paths["manifest"]))
            os.replace(pending, output)
        except Exception:
            if pending.exists():
                shutil.rmtree(pending)
            raise
    finally:
        release_lock.unlink(missing_ok=True)
    return {
        "source_registry_sha": registry.spec_sha256,
        "source_manifest_shas": dict(sorted(source_manifest_shas.items())),
        "train_artifact_sha": train_artifact_sha,
        "heldout_artifact_sha": heldout_artifact_sha,
        "written_manifest_sha": written_manifest_sha,
        "manifest_sha": manifest.content_hash(),
        "train_rows": len(train_records),
        "heldout_rows": len(heldout_records),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _required_text(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _string_tuple(source: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = source.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{key} must be a non-empty list[str]")
    if len(value) != len(set(value)):
        raise ValueError(f"{key} must not contain duplicates")
    return tuple(value)
