"""CLI tests for ``scripts/phase1_inference_gate.py``.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric
from igc.modules.train.phase1_promotion import evaluate_phase1_promotion

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "phase1_inference_gate.py"
SHA_1 = "sha256:" + "1" * 64
SHA_2 = "sha256:" + "2" * 64
SHA_3 = "sha256:" + "3" * 64
SHA_4 = "sha256:" + "4" * 64
SHA_5 = "sha256:" + "5" * 64
SHA_6 = "sha256:" + "6" * 64
SHA_7 = "sha256:" + "7" * 64
COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"
ROW_A = "sha256:" + "a" * 64
ROW_B = "sha256:" + "b" * 64


def _load_script():
    """Import the script as a module without requiring PYTHONPATH setup."""
    spec = importlib.util.spec_from_file_location("phase1_inference_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_row(
    row_id: str,
    source_corpus: str,
    *,
    exact: bool,
    rest_api: str | None = None,
) -> dict:
    rest_api = rest_api or f"/redfish/v1/Systems/{row_id[-1]}"
    target = {"@odata.id": rest_api, "Name": f"System {source_corpus}"}
    prediction = target if exact else {"@odata.id": rest_api, "Name": "Wrong"}
    return {
        "row_id": row_id,
        "source_corpus": source_corpus,
        "x": {"rest_api": rest_api, "json": target},
        "y_true": {"json": target},
        "y_pred": {"json": prediction},
        "target_tokens": 4,
        "max_new_tokens": 16,
        "generated_tokens": 10,
        "sequence_length": 40,
        "latency_sec": 1.0,
        "confidence": 0.9,
    }


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, *, exact: bool) -> Path:
    row = _prediction_row(ROW_A, "fixture_corpus", exact=exact)
    _write_rows(path, [row])
    return path


def _write_spec(path: Path, *, exact_delta: float = 0.0) -> Path:
    path.write_text(
        "\n".join(
            [
                "schema_version: phase1_golden_acceptance.v1",
                "metrics:",
                "  ece_bins: 10",
                "thresholds:",
                "  min_rows: 1",
                "  max_missing_target_rows: 0",
                "  max_missing_prediction_rows: 0",
                "  min_model_json_parse_rate: 1.0",
                "  min_model_odata_id_match_rate: 1.0",
                f"  min_exact_match_delta_vs_baseline: {exact_delta}",
            ]
        )
        + "\n"
    )
    return path


def test_phase1_inference_gate_writes_metrics_and_evidence(tmp_path: Path, capsys) -> None:
    """The CLI writes compact JSON outputs and returns 0 for passing thresholds."""
    script = _load_script()
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", exact=False)
    model = _write_jsonl(tmp_path / "model_x.jsonl", exact=True)
    spec = _write_spec(tmp_path / "spec.yaml", exact_delta=1.0)
    metrics_out = tmp_path / "metrics.json"
    evidence_out = tmp_path / "evidence.json"

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(baseline),
            "--model-jsonl",
            str(model),
            "--metrics-out",
            str(metrics_out),
            "--evidence-out",
            str(evidence_out),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    metrics = json.loads(metrics_out.read_text())
    evidence = json.loads(evidence_out.read_text())
    assert metrics["metrics"]["model_x"][
        phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    ] == 1.0
    assert evidence["acceptance"]["status"] == "pass"
    assert evidence["role"] == "model_x"
    assert evidence["artifact_sha"] == (
        "sha256:" + evidence["artifacts"]["model_x"]["sha256"]
    )
    assert evidence["artifacts"]["baseline"]["name"] == "baseline.jsonl"


def test_build_outputs_preserves_full_evidence_by_corpus_for_promotion(
    tmp_path: Path,
) -> None:
    """build_outputs evidence can feed Phase1 promotion without lossy reshaping."""
    script = _load_script()
    baseline = _write_rows(
        tmp_path / "baseline.jsonl",
        [
            _prediction_row(
                ROW_A,
                "corpus_a",
                exact=True,
                rest_api="/redfish/v1/Systems/A",
            ),
            _prediction_row(
                ROW_B,
                "corpus_b",
                exact=True,
                rest_api="/redfish/v1/Systems/B",
            ),
        ],
    )
    model = _write_rows(
        tmp_path / "model_x.jsonl",
        [
            _prediction_row(
                ROW_A,
                "corpus_a",
                exact=True,
                rest_api="/redfish/v1/Systems/A",
            ),
            _prediction_row(
                ROW_B,
                "corpus_b",
                exact=True,
                rest_api="/redfish/v1/Systems/B",
            ),
        ],
    )
    spec = _write_spec(tmp_path / "spec.yaml", exact_delta=0.0)

    metrics, evidence = script.build_outputs(
        spec_path=spec,
        baseline_jsonl=baseline,
        model_jsonl=model,
    )

    assert evidence["baseline"]["by_corpus"] == {
        "corpus_a": {
            "rows": 1,
            "json_parse_rate": 1.0,
            "json_exact_match_rate": 1.0,
            "resource_identity_match_rate": 1.0,
        },
        "corpus_b": {
            "rows": 1,
            "json_parse_rate": 1.0,
            "json_exact_match_rate": 1.0,
            "resource_identity_match_rate": 1.0,
        },
    }
    assert evidence["model_x"]["by_corpus"] == evidence["baseline"]["by_corpus"]
    assert evidence["counts"]["model_x"]["target_token_rows"] == 2
    assert evidence["counts"]["model_x"]["generation_budget_rows"] == 2
    assert evidence["role"] == "model_x"
    assert evidence["artifact_sha"] == (
        "sha256:" + evidence["artifacts"]["model_x"]["sha256"]
    )

    promotion = evaluate_phase1_promotion(
        thresholds={
            "min_model_json_parse_rate": 1.0,
            "min_model_json_exact_match_rate": 1.0,
            "min_model_resource_identity_match_rate": 1.0,
            "min_exact_match_delta_vs_foundation": 0.0,
            "max_instruction_judge_accept_rate_drop": 0.0,
            "deterministic_golden_json_parse_rate": 1.0,
            "deterministic_golden_resource_identity_match_rate": 1.0,
            "min_heldout_rows_per_corpus": 1,
            "small_corpus_policy": "require_all_available_rows",
            "generation_target_token_margin": 0,
        },
        full_metrics=metrics,
        full_evidence=evidence,
        golden_metrics=metrics,
        golden_evidence=evidence,
        retention_evidence={"comparison": {"delta": {"judge_acceptance_rate": 0.0}}},
        run_report={
            "manifest": {
                "phase": "phase1_finetune",
                "task": "redfish_json_reconstruction",
                "parent_role": "foundation_instruct",
                "output_role": "model_x",
                "data_manifest": SHA_1,
                "eval_split": SHA_2,
                "train_data_sha": SHA_3,
                "eval_data_sha": SHA_2,
                "source_manifest_sha": SHA_4,
                "source_registry_sha": SHA_5,
                "source_artifact_manifest_shas": {"corpus_a": SHA_6},
                "foundation_model_sha": SHA_6,
                "tokenizer_sha": SHA_7,
                "git_commit": COMMIT_SHA,
                "promotion_source": "best_checkpoint",
                "checkpoint_path": "/checkpoints/model_x_epoch_best.pt",
                "promoted_artifact_path": "/models/model_x",
                "training": {
                    "optimizer_steps": 1,
                    "train_loss": 0.1,
                },
            },
            "metrics": {"eval_loss": 0.1},
        },
        heldout_manifest={
            "approved_heldout_rows": 2,
            "required_corpora": ["corpus_a", "corpus_b"],
            "rows_by_corpus": {"corpus_a": 1, "corpus_b": 1},
            "full_rows_by_corpus": {"corpus_a": 1, "corpus_b": 1},
            "row_ids": [ROW_A, ROW_B],
            "artifact_sha256": SHA_2,
            "full_corpus_manifest_sha256": SHA_4,
            "source_registry_sha256": SHA_5,
            "source_manifest_shas": {"corpus_a": SHA_6},
        },
        load_evidence={
            "status": "pass",
            "role": evidence["role"],
            "artifact_sha": evidence["artifact_sha"],
            "adapter_dir": "/models/model_x",
        },
    )

    assert promotion["status"] == "pass"
    assert promotion["role"] == "model_x"
    assert promotion["artifact_sha"] == evidence["artifact_sha"]


def test_phase1_inference_gate_returns_one_on_acceptance_failure(tmp_path: Path) -> None:
    """Failed acceptance still writes evidence, then exits with code 1."""
    script = _load_script()
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", exact=True)
    model = _write_jsonl(tmp_path / "model_x.jsonl", exact=True)
    spec = _write_spec(tmp_path / "spec.yaml", exact_delta=0.5)
    evidence_out = tmp_path / "evidence.json"

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(baseline),
            "--model-jsonl",
            str(model),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--evidence-out",
            str(evidence_out),
        ]
    )

    assert code == 1
    assert json.loads(evidence_out.read_text())["acceptance"]["status"] == "fail"


def test_phase1_inference_gate_returns_two_for_missing_input(tmp_path: Path) -> None:
    """Missing JSONL input returns the documented error code."""
    script = _load_script()
    spec = _write_spec(tmp_path / "spec.yaml")

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(tmp_path / "missing-baseline.jsonl"),
            "--model-jsonl",
            str(tmp_path / "missing-model.jsonl"),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--evidence-out",
            str(tmp_path / "evidence.json"),
        ]
    )

    assert code == 2


# Author: Mus mbayramo@stanford.edu
