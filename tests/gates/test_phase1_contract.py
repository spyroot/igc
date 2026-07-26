"""Regression coverage for the Phase 1 D0 project contract gate."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


GATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gates"
    / "phase1_contract.py"
)


def _gate_module():
    spec = importlib.util.spec_from_file_location("phase1_contract_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _isolated_machine_contract(module, tmp_path: Path, monkeypatch):
    contract = yaml.safe_load(module.CONTRACT_PATH.read_text(encoding="utf-8"))
    source_registry = yaml.safe_load(
        module.SOURCE_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    promotion = yaml.safe_load(module.PROMOTION_PATH.read_text(encoding="utf-8"))
    contract_path = _write_yaml(tmp_path / "phase1.yaml", contract)
    source_path = _write_yaml(tmp_path / "redfish_sources.yaml", source_registry)
    promotion_path = _write_yaml(tmp_path / "phase1_golden_acceptance.yaml", promotion)
    monkeypatch.setattr(module, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(module, "SOURCE_REGISTRY_PATH", source_path)
    monkeypatch.setattr(module, "PROMOTION_PATH", promotion_path)
    return (
        contract,
        source_registry,
        promotion,
        contract_path,
        source_path,
        promotion_path,
    )


def _machine_contract_failures(module) -> list[str]:
    failures: list[str] = []
    module._check_machine_contract(failures)
    return failures


def test_phase1_contract_gate_has_no_violations() -> None:
    """Every malformed Phase 1 invariant fixture remains rejected."""
    assert _gate_module().run_gate() == []


@pytest.mark.parametrize(
    "mutate_materialization",
    [
        pytest.param(
            lambda materialization: materialization.__setitem__(
                "canonical_unit",
                "files",
            ),
            id="not-directory-unit",
        ),
        pytest.param(
            lambda materialization: materialization["members"].remove(
                "heldout/manifest.json",
            ),
            id="missing-heldout-manifest",
        ),
        pytest.param(
            lambda materialization: materialization.__setitem__(
                "single_atomic_directory_rename",
                False,
            ),
            id="not-single-directory-rename",
        ),
        pytest.param(
            lambda materialization: materialization.__setitem__("immutable", False),
            id="not-immutable",
        ),
        pytest.param(
            lambda materialization: materialization.__setitem__(
                "complete",
                True,
            ),
            id="extra-field",
        ),
    ],
)
def test_phase1_contract_gate_requires_exact_immutable_directory_materialization(
    tmp_path: Path,
    monkeypatch,
    mutate_materialization,
) -> None:
    """The machine gate pins Phase 1 to one immutable train/heldout directory."""
    module = _gate_module()
    contract, _source, _promotion, contract_path, _source_path, _promotion_path = (
        _isolated_machine_contract(module, tmp_path, monkeypatch)
    )

    assert _machine_contract_failures(module) == []

    mutated = deepcopy(contract)
    mutate_materialization(mutated["materialization"])
    _write_yaml(contract_path, mutated)

    assert (
        "Phase 1 materialization must be one immutable directory release"
        in _machine_contract_failures(module)
    )


def test_phase1_contract_gate_requires_source_and_promotion_heldout_floor_parity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The registry held-out floor and Phase 1 promotion floor are one contract."""
    module = _gate_module()
    _contract, source, promotion, _contract_path, source_path, promotion_path = (
        _isolated_machine_contract(module, tmp_path, monkeypatch)
    )

    assert _machine_contract_failures(module) == []

    mismatched_source = deepcopy(source)
    mismatched_source["evaluation"]["min_heldout_rows_per_source"] = (
        promotion["thresholds"]["min_heldout_rows_per_corpus"] + 1
    )
    _write_yaml(source_path, mismatched_source)
    assert (
        "Phase 1 source and promotion per-corpus held-out floors disagree"
        in _machine_contract_failures(module)
    )

    _write_yaml(source_path, source)
    bad_policy = deepcopy(promotion)
    bad_policy["thresholds"]["small_corpus_policy"] = "floor_only"
    _write_yaml(promotion_path, bad_policy)
    assert (
        "Phase 1 small-corpus policy must require all available rows"
        in _machine_contract_failures(module)
    )


@pytest.mark.parametrize(
    ("mutate_source_registry", "failure"),
    [
        pytest.param(
            lambda source: source["artifact_contract"].__setitem__(
                "producer",
                "redfish_ctl",
            ),
            "Phase 1 source producer must be redfish_ctl discovery",
            id="producer",
        ),
        pytest.param(
            lambda source: source["artifact_contract"]["api_map"].__setitem__(
                "accepted_names",
                ["rest_api_map.npy"],
            ),
            "Phase 1 source registry has invalid REST API map names",
            id="accepted-map-names",
        ),
        pytest.param(
            lambda source: source["artifact_contract"]["api_map"].__setitem__(
                "required_keys",
                ["allowed_methods_mapping", "url_file_mapping"],
            ),
            "Phase 1 source registry has invalid REST API map keys",
            id="required-map-keys",
        ),
        pytest.param(
            lambda source: source["artifact_contract"].__setitem__(
                "semantics_bundle",
                "",
            ),
            "Phase 1 source registry requires a semantics bundle",
            id="semantics-bundle",
        ),
    ],
)
def test_phase1_contract_gate_requires_exact_source_registry_artifact_contract(
    tmp_path: Path,
    monkeypatch,
    mutate_source_registry,
    failure: str,
) -> None:
    """The Phase 1 machine gate pins redfish_ctl discovery API-map lineage."""
    module = _gate_module()
    _contract, source, _promotion, _contract_path, source_path, _promotion_path = (
        _isolated_machine_contract(module, tmp_path, monkeypatch)
    )
    artifact_contract = source["artifact_contract"]

    assert artifact_contract["producer"] == "redfish_ctl_discovery"
    assert artifact_contract["api_map"]["accepted_names"] == [
        "rest_api_map.v1.json",
        "rest_api_map.npy",
    ]
    assert artifact_contract["api_map"]["required_keys"] == [
        "url_file_mapping",
        "allowed_methods_mapping",
    ]
    assert artifact_contract["semantics_bundle"]
    assert _machine_contract_failures(module) == []

    mutated = deepcopy(source)
    mutate_source_registry(mutated)
    _write_yaml(source_path, mutated)

    assert failure in _machine_contract_failures(module)
