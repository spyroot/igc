"""Parser tests for promotion gate required evidence inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(module_name: str, relpath: str):
    """Import a gate script without executing its main()."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _without_flag(argv: list[str], flag: str) -> list[str]:
    """Drop one required flag and its value from a parser argv fixture."""
    index = argv.index(flag)
    return [*argv[:index], *argv[index + 2 :]]


def test_phase2_promotion_parser_requires_manifests_split_release_and_run_report() -> None:
    """Phase2 promotion CLI requires source-full, train, heldout, split, and run report."""
    script = _load_script("phase2_promotion_gate_args", "scripts/gates/phase2_promotion.py")

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--heldout-jsonl",
                "heldout.jsonl",
                "--artifact-evidence",
                "artifact.json",
                "--output-json",
                "out.json",
            ]
        )
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--source-full-jsonl",
                "source-full.jsonl",
                "--source-full-manifest",
                "source-full.manifest.json",
                "--train-jsonl",
                "train.jsonl",
                "--train-manifest",
                "train.manifest.json",
                "--heldout-jsonl",
                "heldout.jsonl",
                "--heldout-manifest",
                "heldout.manifest.json",
                "--artifact-evidence",
                "artifact.json",
                "--run-report",
                "report.json",
                "--output-json",
                "out.json",
            ]
        )
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--source-full-jsonl",
                "source-full.jsonl",
                "--source-full-manifest",
                "source-full.manifest.json",
                "--train-jsonl",
                "train.jsonl",
                "--train-manifest",
                "train.manifest.json",
                "--heldout-jsonl",
                "heldout.jsonl",
                "--heldout-manifest",
                "heldout.manifest.json",
                "--split-release-manifest",
                "split.release.json",
                "--artifact-evidence",
                "artifact.json",
                "--output-json",
                "out.json",
            ]
        )

    args = script.parse_args(
        [
            "--source-full-jsonl",
            "source-full.jsonl",
            "--source-full-manifest",
            "source-full.manifest.json",
            "--train-jsonl",
            "train.jsonl",
            "--train-manifest",
            "train.manifest.json",
            "--heldout-jsonl",
            "heldout.jsonl",
            "--heldout-manifest",
            "heldout.manifest.json",
            "--split-release-manifest",
            "split.release.json",
            "--artifact-evidence",
            "artifact.json",
            "--run-report",
            "report.json",
            "--output-json",
            "out.json",
        ]
    )

    assert args.source_full_jsonl == "source-full.jsonl"
    assert args.source_full_manifest == "source-full.manifest.json"
    assert args.train_jsonl == "train.jsonl"
    assert args.train_manifest == "train.manifest.json"
    assert args.heldout_manifest == "heldout.manifest.json"
    assert args.split_release_manifest == "split.release.json"
    assert args.run_report == "report.json"


@pytest.mark.parametrize("missing_flag", ["--source-full-jsonl", "--train-jsonl"])
def test_phase2_promotion_parser_requires_source_and_train_jsonl(
    missing_flag: str,
) -> None:
    """Phase2 promotion fails closed if source-full or train JSONL is absent."""
    script = _load_script(
        "phase2_promotion_gate_jsonl_args",
        "scripts/gates/phase2_promotion.py",
    )
    argv = [
        "--source-full-jsonl",
        "source-full.jsonl",
        "--source-full-manifest",
        "source-full.manifest.json",
        "--train-jsonl",
        "train.jsonl",
        "--train-manifest",
        "train.manifest.json",
        "--heldout-jsonl",
        "heldout.jsonl",
        "--heldout-manifest",
        "heldout.manifest.json",
        "--split-release-manifest",
        "split.release.json",
        "--artifact-evidence",
        "artifact.json",
        "--run-report",
        "report.json",
        "--output-json",
        "out.json",
    ]

    with pytest.raises(SystemExit):
        script.parse_args(_without_flag(argv, missing_flag))


def test_phase3_promotion_parser_requires_manifests_split_views_and_run_report() -> None:
    """Phase3 promotion CLI requires source-full, train, heldout, split, views, and run report."""
    script = _load_script("phase3_promotion_gate_args", "scripts/gates/phase3_promotion.py")

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--heldout-jsonl",
                "heldout.jsonl",
                "--view-pairs-jsonl",
                "views.jsonl",
                "--artifact-evidence",
                "artifact.json",
                "--output-json",
                "out.json",
            ]
        )
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--source-full-jsonl",
                "source-full.jsonl",
                "--source-full-manifest",
                "source-full.manifest.json",
                "--train-jsonl",
                "train.jsonl",
                "--train-manifest",
                "train.manifest.json",
                "--heldout-jsonl",
                "heldout.jsonl",
                "--heldout-manifest",
                "heldout.manifest.json",
                "--view-pairs-jsonl",
                "views.jsonl",
                "--artifact-evidence",
                "artifact.json",
                "--run-report",
                "report.json",
                "--output-json",
                "out.json",
            ]
        )
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--source-full-jsonl",
                "source-full.jsonl",
                "--source-full-manifest",
                "source-full.manifest.json",
                "--train-jsonl",
                "train.jsonl",
                "--train-manifest",
                "train.manifest.json",
                "--heldout-jsonl",
                "heldout.jsonl",
                "--heldout-manifest",
                "heldout.manifest.json",
                "--split-release-manifest",
                "split.release.json",
                "--view-pairs-jsonl",
                "views.jsonl",
                "--artifact-evidence",
                "artifact.json",
                "--output-json",
                "out.json",
            ]
        )

    args = script.parse_args(
        [
            "--source-full-jsonl",
            "source-full.jsonl",
            "--source-full-manifest",
            "source-full.manifest.json",
            "--train-jsonl",
            "train.jsonl",
            "--train-manifest",
            "train.manifest.json",
            "--heldout-jsonl",
            "heldout.jsonl",
            "--heldout-manifest",
            "heldout.manifest.json",
            "--split-release-manifest",
            "split.release.json",
            "--view-pairs-jsonl",
            "views.jsonl",
            "--artifact-evidence",
            "artifact.json",
            "--run-report",
            "report.json",
            "--output-json",
            "out.json",
        ]
    )

    assert args.source_full_jsonl == "source-full.jsonl"
    assert args.source_full_manifest == "source-full.manifest.json"
    assert args.train_jsonl == "train.jsonl"
    assert args.train_manifest == "train.manifest.json"
    assert args.heldout_manifest == "heldout.manifest.json"
    assert args.split_release_manifest == "split.release.json"
    assert args.run_report == "report.json"


@pytest.mark.parametrize("missing_flag", ["--source-full-jsonl", "--train-jsonl"])
def test_phase3_promotion_parser_requires_source_and_train_jsonl(
    missing_flag: str,
) -> None:
    """Phase3 promotion fails closed if source-full or train JSONL is absent."""
    script = _load_script(
        "phase3_promotion_gate_jsonl_args",
        "scripts/gates/phase3_promotion.py",
    )
    argv = [
        "--source-full-jsonl",
        "source-full.jsonl",
        "--source-full-manifest",
        "source-full.manifest.json",
        "--train-jsonl",
        "train.jsonl",
        "--train-manifest",
        "train.manifest.json",
        "--heldout-jsonl",
        "heldout.jsonl",
        "--heldout-manifest",
        "heldout.manifest.json",
        "--split-release-manifest",
        "split.release.json",
        "--view-pairs-jsonl",
        "views.jsonl",
        "--artifact-evidence",
        "artifact.json",
        "--run-report",
        "report.json",
        "--output-json",
        "out.json",
    ]

    with pytest.raises(SystemExit):
        script.parse_args(_without_flag(argv, missing_flag))


def test_d1_promotion_parser_requires_release_heldout_manifest_and_jsonl() -> None:
    """D1 promotion CLI requires release, heldout manifest, and heldout JSONL inputs."""
    script = _load_script("d1_promotion_gate_args", "scripts/gates/d1_promotion.py")

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--dataset-jsonl",
                "d1.jsonl",
                "--release-manifest",
                "d1.manifest.json",
                "--build-metrics",
                "metrics.json",
                "--judge-calibration-metrics",
                "calibration.json",
                "--artifact-evidence",
                "artifact.json",
                "--output-json",
                "out.json",
            ]
        )

    args = script.parse_args(
        [
            "--dataset-jsonl",
            "d1.jsonl",
            "--release-manifest",
            "d1.manifest.json",
            "--heldout-jsonl",
            "heldout.jsonl",
            "--heldout-manifest",
            "heldout.manifest.json",
            "--build-metrics",
            "metrics.json",
            "--judge-calibration-metrics",
            "calibration.json",
            "--artifact-evidence",
            "artifact.json",
            "--output-json",
            "out.json",
        ]
    )

    assert args.release_manifest == "d1.manifest.json"
    assert args.heldout_jsonl == "heldout.jsonl"
    assert args.heldout_manifest == "heldout.manifest.json"
