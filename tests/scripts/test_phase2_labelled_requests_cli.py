"""Script tests for offline ``phase2_labelled_requests`` generation.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)

SCRIPT = Path("scripts/build_phase2_labelled_requests.py")
MODEL_X_ARTIFACT_SHA = "sha256:" + "9" * 64


def _load_script() -> ModuleType:
    """Load the script module for direct ``main(argv)`` testing."""
    spec = importlib.util.spec_from_file_location("build_phase2_labelled_requests", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_records(path: Path, count: int = 8) -> Path:
    """Write tiny REST API record fixtures."""
    rows = []
    for index in range(count):
        rows.append({
            "rest_api": f"/redfish/v1/Systems/{index}",
            "allowed_methods": ["get", "HEAD"],
            "json": {
                "@odata.id": f"/redfish/v1/Systems/{index}",
                "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
                "Name": f"System {index}",
            },
            "vendor": "fixture_vendor",
            "source_corpus": "fixture_corpus",
        })
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    """Read all non-blank JSONL rows."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _release_rows(release_dir: Path) -> list[dict]:
    """Read rows from a canonical D1 release directory."""
    return _read_jsonl(release_dir / "data.jsonl")


def _release_manifest(release_dir: Path) -> dict:
    """Read the manifest from a canonical D1 release directory."""
    return json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))


def _judge_json(
    rest_api_list: list[str],
    *,
    accepted: bool = True,
    natural: bool = True,
    nonsense: bool = False,
    ambiguous: bool = False,
    duplicate_intent: bool = False,
    extra_intents: bool = False,
    method_semantics_valid: bool = True,
) -> str:
    """Return a strict Phase 2 judge verdict fixture."""
    return json.dumps({
        "accepted": accepted,
        "natural": natural,
        "nonsense": nonsense,
        "ambiguous": ambiguous,
        "duplicate_intent": duplicate_intent,
        "extra_intents": extra_intents,
        "method_semantics_valid": method_semantics_valid,
        "covered_api_set": rest_api_list,
        "reason": "fixture",
    })


def _base_args(tmp_path: Path, *, sample_width: int = 2, count: int = 2) -> list[str]:
    """Return common CLI args for fixture tests."""
    return [
        "--records-jsonl",
        str(_write_records(tmp_path / "records.jsonl")),
        "--output-release-dir",
        str(tmp_path / "out" / "phase2_labelled_requests"),
        "--metrics-out",
        str(tmp_path / "out" / "metrics.json"),
        "--sample-width",
        str(sample_width),
        "--count",
        str(count),
        "--seed",
        "11",
    ]


def _write_small_all_width_spec(tmp_path: Path) -> Path:
    """Write a tiny all-width spec so tests do not build 1000 empty-set rows."""
    spec_text = Path("configs/phase2_labelled_requests.yaml").read_text(encoding="utf-8")
    spec_text = spec_text.replace(
        "  empty_set_candidates: 1000\n",
        "  empty_set_candidates: 2\n",
    )
    spec_text = spec_text.replace(
        "  min_empty_set_accepted_rows: 100\n",
        "  min_empty_set_accepted_rows: 2\n",
    )
    spec_path = tmp_path / "phase2-small-all-widths.yaml"
    spec_path.write_text(spec_text, encoding="utf-8")
    return spec_path


def _metric(group: str, name: str | None = None) -> str:
    """Return a Phase 2 labelled-request metric key."""
    return phase_metric(PHASE2_LABELLED_REQUESTS, group, name)


def _sampled_apis(*, sample_width: int, seed: int = 11, count: int = 8) -> list[str]:
    """Mirror the CLI's deterministic sample for file-provider fixtures."""
    return [
        f"/redfish/v1/Systems/{index}"
        for index in random.Random(seed).sample(range(count), sample_width)
    ]


def _set_live_identity_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_sha: str | None = MODEL_X_ARTIFACT_SHA,
) -> None:
    """Set resolved live identities without exposing real provider credentials."""
    monkeypatch.setenv("PHASE1_MODEL_X_MODEL_ID", "restored-model-x")
    if artifact_sha is not None:
        monkeypatch.setenv("PHASE1_MODEL_X_ARTIFACT_SHA", artifact_sha)
    monkeypatch.setenv("PHASE2_JUDGE_ROUTE", "private-pro-route")
    monkeypatch.setenv("PHASE2_JUDGE_MODEL_ID", "private-pro")
    monkeypatch.setenv("PHASE2_JUDGE_PROFILE", "think-max")


def _set_live_transport_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake endpoint variables for the OpenAI-compatible provider path."""
    monkeypatch.setenv("PHASE2_MODEL_X_BASE_URL", "http://model-x.invalid")
    monkeypatch.setenv("PHASE2_JUDGE_BASE_URL", "http://judge.invalid")
    monkeypatch.setenv("PHASE2_MODEL_X_API_KEY", "draft-token")
    monkeypatch.setenv("PHASE2_JUDGE_API_KEY", "judge-token")


class _FakeWandbRun:
    """Small W&B run double that records log, summary, and finish calls."""

    def __init__(self, *, fail_log: bool = False) -> None:
        self.fail_log = fail_log
        self.logs: list[tuple[int, dict]] = []
        self.summary: dict[str, object] = {}
        self.finish_calls: list[dict] = []

    def log(self, payload: dict, *, step: int) -> None:
        """Record one W&B log payload or fail like the real client."""
        if self.fail_log:
            raise RuntimeError("fake wandb log failed")
        self.logs.append((step, dict(payload)))

    def finish(self, **kwargs) -> None:
        """Record run cleanup."""
        self.finish_calls.append(dict(kwargs))


class _FakeWandbModule(ModuleType):
    """Importable fake ``wandb`` module for metric-report tests."""

    def __init__(
        self,
        run: _FakeWandbRun | None = None,
        *,
        fail_init: bool = False,
    ) -> None:
        super().__init__("wandb")
        self.run = run
        self.fail_init = fail_init
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        """Record init config or fail before returning a run."""
        self.init_calls.append(dict(kwargs))
        if self.fail_init:
            raise RuntimeError("fake wandb init failed")
        return self.run


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run: _FakeWandbRun | None = None,
    fail_init: bool = False,
) -> _FakeWandbModule:
    """Install a fake wandb module for the script's import-time lookup."""
    fake = _FakeWandbModule(run=run, fail_init=fail_init)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_mock_judge_provider_emits_strict_verdict_schema() -> None:
    """The built-in mock judge follows the same strict schema as live judges."""
    script = _load_script()
    expected = ["/redfish/v1/Systems/1"]

    verdict = json.loads(
        script._mock_judge_provider({"expected_rest_api_list": expected}),
    )

    assert set(verdict) == {
        "accepted",
        "natural",
        "nonsense",
        "ambiguous",
        "duplicate_intent",
        "extra_intents",
        "method_semantics_valid",
        "covered_api_set",
        "reason",
    }
    assert verdict["accepted"] is True
    assert verdict["covered_api_set"] == expected


def test_cli_mock_mode_writes_accepted_jsonl_and_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mock providers produce accepted Phase 2 rows and registered metrics."""
    script = _load_script()
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=3, count=2))
    stdout = capsys.readouterr().out

    assert code == 0
    assert "dataset=D1" in stdout
    assert "accepted=2" in stdout
    assert f"output_release_dir={release_dir}" in stdout
    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["phase"] == 2
    assert rows[0]["dataset"] == "D1"
    assert rows[0]["source_dataset"] == "D0"
    assert rows[0]["task"] == "text_to_rest_api_list"
    assert rows[0]["x"]["text"].startswith("fixture request covering 3")
    assert "rest_api_list" not in rows[0]["x"]
    assert len(rows[0]["x"]["api_context"]) == 7
    target_apis = set(rows[0]["y_true"]["rest_api_list"])
    context_apis = {context["rest_api"] for context in rows[0]["x"]["api_context"]}
    assert len(rows[0]["y_true"]["rest_api_list"]) == 3
    assert target_apis <= context_apis
    assert len(context_apis - target_apis) >= 4
    assert all("selected" not in context for context in rows[0]["x"]["api_context"])
    assert rows[0]["validation"]["set_coverage_preserved"] is True
    assert rows[0]["validation"]["review_judged"] is True
    assert "calls" not in json.dumps(rows)
    assert '"method":' not in json.dumps(rows)
    assert '"arguments":' not in json.dumps(rows)
    assert set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS) <= set(metrics)
    assert metrics["dataset"] == "D1"
    assert metrics[_metric("draft_total")] == 2
    assert metrics[_metric("accepted_total")] == 2
    assert metrics[_metric("rejected_total")] == 0
    assert metrics[_metric("natural_command_rate")] == 1.0
    assert metrics[_metric("ambiguous_rate")] == 0.0
    assert metrics[_metric("duplicate_intent_rate")] == 0.0
    assert metrics[_metric("extra_intent_rate")] == 0.0
    assert metrics[_metric("method_semantics_valid_rate")] == 1.0
    assert metrics[_metric("sample_width", "k")] == 3
    assert metrics[_metric("vendor", "source_corpus")] == "fixture_vendor:fixture_corpus"
    assert metrics["thresholds_pass"] is True
    assert all(
        not key.startswith("phase2_goal_extraction/")
        for key in metrics
    )


def test_cli_all_sample_widths_releases_balanced_canonical_d1(
    tmp_path: Path,
) -> None:
    """Canonical D1 CLI mode releases one balanced artifact across widths 1, 2, and 3."""
    script = _load_script()
    records = _write_records(tmp_path / "records.jsonl", count=10)
    spec_path = _write_small_all_width_spec(tmp_path)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        [
            "--records-jsonl",
            str(records),
            "--spec",
            str(spec_path),
            "--output-release-dir",
            str(release_dir),
            "--metrics-out",
            str(metrics_path),
            "--all-sample-widths",
            "--count",
            "1",
            "--seed",
            "11",
        ],
    )

    assert code == 0
    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = _release_manifest(release_dir)
    assert [row["metadata"]["sample_width_k"] for row in rows] == [1, 2, 3, 0, 0]
    assert {row["dataset"] for row in rows} == {"D1"}
    assert {row["task"] for row in rows} == {"text_to_rest_api_list"}
    assert metrics["sample_widths"] == [1, 2, 3, 0]
    assert set(metrics["by_sample_width"]) == {"0", "1", "2", "3"}
    assert metrics["accepted_rows"] == 5
    assert metrics["thresholds_pass"] is True
    assert metrics["sampling_budget"]["limits"] == {
        "max_accepted_rows": 100000,
        "max_candidates": 300000,
        "max_accepted_per_combination": 8,
        "max_attempts_per_combination": 24,
        "max_accepted_per_api": 200,
        "max_empty_set_candidates": 2,
    }
    assert metrics["sampling_budget"]["observed"]["attempts_total"] == 5
    assert metrics["sampling_budget"]["observed"]["accepted_total"] == 5
    assert metrics["sampling_budget"]["observed"]["unique_combinations_attempted"] == 3
    assert manifest["sample_width_counts"] == {"0": 2, "1": 1, "2": 1, "3": 1}
    assert manifest["sampling_budget"] == metrics["sampling_budget"]
    rendered_manifest = json.dumps(manifest, sort_keys=True)
    assert "draft-token" not in rendered_manifest
    assert "judge-token" not in rendered_manifest
    assert "secret" not in rendered_manifest
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_cli_metric_report_wandb_logs_registered_numeric_metrics_by_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W&B reporting logs per-width numeric Phase 2 metrics without secrets."""
    script = _load_script()
    run = _FakeWandbRun()
    fake_wandb = _install_fake_wandb(monkeypatch, run=run)
    monkeypatch.setenv("PHASE2_MODEL_X_API_KEY", "secret-draft-token")
    monkeypatch.setenv("PHASE2_JUDGE_API_KEY", "secret-judge-token")
    monkeypatch.setenv("PHASE2_JUDGE_ROUTE", "secret-route-value")
    records = _write_records(tmp_path / "records.jsonl", count=10)
    spec_path = _write_small_all_width_spec(tmp_path)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        [
            "--records-jsonl",
            str(records),
            "--spec",
            str(spec_path),
            "--output-release-dir",
            str(release_dir),
            "--metrics-out",
            str(metrics_path),
            "--metric-report",
            "wandb",
            "--all-sample-widths",
            "--count",
            "1",
            "--seed",
            "11",
        ],
    )

    assert code == 0
    assert [step for step, _payload in run.logs] == [0, 1, 2, 3]
    registered = set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)
    for _step, payload in run.logs:
        assert payload
        assert set(payload) <= registered
        assert all(key.startswith("phase2_labelled_requests/") for key in payload)
        assert all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in payload.values()
        )
    config = fake_wandb.init_calls[0]["config"]
    assert config["dataset"] == "D1"
    assert config["model_x_model"] == "${PHASE1_MODEL_X_MODEL_ID}"
    assert config["model_x_artifact_sha"] == "${PHASE1_MODEL_X_ARTIFACT_SHA}"
    assert config["judge_model"] == "${PHASE2_JUDGE_MODEL_ID}"
    assert config["judge_profile"] == "${PHASE2_JUDGE_PROFILE}"
    assert config["sample_widths"] == [0, 1, 2, 3]
    assert config["seed"] == 11
    assert config["sampling_limits"] == {
        "max_accepted_rows": 100000,
        "max_candidates": 300000,
        "max_accepted_per_combination": 8,
        "max_attempts_per_combination": 24,
        "max_accepted_per_api": 200,
        "max_empty_set_candidates": 2,
    }
    rendered_config = json.dumps(config, sort_keys=True)
    assert "judge_route" not in config
    assert "secret-draft-token" not in rendered_config
    assert "secret-judge-token" not in rendered_config
    assert "secret-route-value" not in rendered_config
    assert run.summary == {
        "requested_candidates": 5,
        "accepted_rows": 5,
        "thresholds_pass": True,
    }
    assert run.finish_calls == [{}]
    assert (release_dir / "data.jsonl").exists()
    assert (release_dir / "manifest.json").exists()


def test_balanced_release_rows_trims_positive_widths_to_common_count() -> None:
    """Unequal accepted k1/k2/k3 rows are deterministically prefix-trimmed."""
    script = _load_script()
    rows_by_width = {
        1: [{"id": "k1-a"}, {"id": "k1-b"}, {"id": "k1-c"}],
        2: [{"id": "k2-a"}, {"id": "k2-b"}],
        3: [{"id": "k3-a"}, {"id": "k3-b"}, {"id": "k3-c"}, {"id": "k3-d"}],
    }

    balanced = script._balanced_release_rows(rows_by_width, (1, 2, 3))

    assert [row["id"] for row in balanced] == [
        "k1-a",
        "k1-b",
        "k2-a",
        "k2-b",
        "k3-a",
        "k3-b",
    ]


def test_balanced_release_rows_preserves_all_k0_rows_after_positive_balance() -> None:
    """Empty-set negatives are carried on their own bound after positive rows."""
    script = _load_script()
    rows_by_width = {
        0: [{"id": "k0-a"}, {"id": "k0-b"}, {"id": "k0-c"}],
        1: [{"id": "k1-a"}, {"id": "k1-b"}],
        2: [{"id": "k2-a"}],
        3: [{"id": "k3-a"}, {"id": "k3-b"}],
    }

    balanced = script._balanced_release_rows(rows_by_width, (1, 2, 3))

    assert [row["id"] for row in balanced] == [
        "k1-a",
        "k2-a",
        "k3-a",
        "k0-a",
        "k0-b",
        "k0-c",
    ]


@pytest.mark.parametrize(
    "rows_by_width",
    [
        {1: [{"id": "k1"}], 3: [{"id": "k3"}]},
        {1: [{"id": "k1"}], 2: [], 3: [{"id": "k3"}]},
    ],
)
def test_balanced_release_rows_fails_closed_when_positive_width_missing(
    rows_by_width: dict[int, list[dict[str, str]]],
) -> None:
    """A canonical D1 release cannot be balanced without every positive width."""
    script = _load_script()

    with pytest.raises(SystemExit, match="accepted positive widths are missing"):
        script._balanced_release_rows(rows_by_width, (1, 2, 3))


def test_cli_rejects_requested_candidates_above_yaml_budget_before_release(
    tmp_path: Path,
) -> None:
    """sampling.max_candidates is a hard CLI ceiling before generation/release."""
    script = _load_script()
    records = _write_records(tmp_path / "records.jsonl", count=10)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(SystemExit, match="sampling.max_candidates"):
        script.main(
            [
                "--records-jsonl",
                str(records),
                "--output-release-dir",
                str(release_dir),
                "--metrics-out",
                str(metrics_path),
                "--sample-width",
                "1",
                "--count",
                "300001",
                "--seed",
                "11",
            ],
        )

    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    assert not metrics_path.exists()


def test_cli_metric_report_wandb_init_failure_prevents_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A W&B initialization failure aborts before releasing canonical D1."""
    script = _load_script()
    _install_fake_wandb(monkeypatch, fail_init=True)
    records = _write_records(tmp_path / "records.jsonl", count=10)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(RuntimeError, match="required D1 W&B metric logging failed"):
        script.main(
            [
                "--records-jsonl",
                str(records),
                "--output-release-dir",
                str(release_dir),
                "--metrics-out",
                str(metrics_path),
                "--metric-report",
                "wandb",
                "--all-sample-widths",
                "--count",
                "1",
                "--seed",
                "11",
            ],
        )

    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_cli_metric_report_wandb_log_failure_finishes_and_prevents_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A W&B log failure performs run cleanup and aborts before D1 release."""
    script = _load_script()
    run = _FakeWandbRun(fail_log=True)
    _install_fake_wandb(monkeypatch, run=run)
    records = _write_records(tmp_path / "records.jsonl", count=10)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(RuntimeError, match="required D1 W&B metric logging failed"):
        script.main(
            [
                "--records-jsonl",
                str(records),
                "--output-release-dir",
                str(release_dir),
                "--metrics-out",
                str(metrics_path),
                "--metric-report",
                "wandb",
                "--all-sample-widths",
                "--count",
                "1",
                "--seed",
                "11",
            ],
        )

    assert run.finish_calls == [{"exit_code": 1}]
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


@pytest.mark.parametrize("sample_width", (1, 2, 3))
def test_cli_mock_mode_accepts_all_configured_sample_widths(
    tmp_path: Path,
    sample_width: int,
) -> None:
    """Mock mode accepts every configured sample width on a successful path."""
    script = _load_script()
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=sample_width, count=1))

    assert code == 0
    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert len(rows[0]["y_true"]["rest_api_list"]) == sample_width
    assert len(rows[0]["x"]["api_context"]) == sample_width + 4
    assert metrics[_metric("sample_width", "k")] == sample_width


def test_cli_rejects_insufficient_positive_source_pool_before_provider_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive k needs k plus configured distractors before draft/judge calls."""
    script = _load_script()
    calls = {"draft": 0, "judge": 0}

    def fail_draft(_request):
        calls["draft"] += 1
        raise AssertionError("draft provider must not be invoked")

    def fail_judge(_request):
        calls["judge"] += 1
        raise AssertionError("judge provider must not be invoked")

    monkeypatch.setattr(script, "_mock_draft_provider", fail_draft)
    monkeypatch.setattr(script, "_mock_judge_provider", fail_judge)
    records = _write_records(tmp_path / "records.jsonl", count=6)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(ValueError, match="targets plus distractors"):
        script.main(
            [
                "--records-jsonl",
                str(records),
                "--output-release-dir",
                str(release_dir),
                "--metrics-out",
                str(metrics_path),
                "--sample-width",
                "3",
                "--count",
                "1",
                "--seed",
                "11",
            ],
        )

    assert calls == {"draft": 0, "judge": 0}
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    assert not metrics_path.exists()


def test_cli_rejects_duplicate_rest_api_records_before_provider_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate source rows fail during input load before provider construction."""
    script = _load_script()
    provider_setup_calls: list[dict] = []

    def fail_providers(*args, **kwargs):
        provider_setup_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("providers must not be constructed for duplicate input")

    monkeypatch.setattr(script, "_providers", fail_providers)
    duplicate = {
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["GET"],
        "json": {"@odata.id": "/redfish/v1/Systems/1"},
    }
    records = tmp_path / "duplicate-records.jsonl"
    records.write_text(
        json.dumps(duplicate, sort_keys=True)
        + "\n"
        + json.dumps(duplicate, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    with pytest.raises(SystemExit, match="duplicate rest_api"):
        script.main(
            [
                "--records-jsonl",
                str(records),
                "--output-release-dir",
                str(release_dir),
                "--metrics-out",
                str(metrics_path),
                "--sample-width",
                "1",
                "--count",
                "1",
            ],
        )

    assert provider_setup_calls == []
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    assert not metrics_path.exists()


def test_cli_file_providers_are_used_without_network(tmp_path: Path) -> None:
    """Local provider fixture files control draft text and judge acceptance."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("show the sampled fixture systems\n", encoding="utf-8")
    expected = _sampled_apis(sample_width=2)
    judges.write_text(
        _judge_json(expected)
        + "\n",
        encoding="utf-8",
    )

    code = script.main(
        _base_args(tmp_path, sample_width=2, count=1)
        + [
            "--provider-mode",
            "file",
            "--drafts-jsonl",
            str(drafts),
            "--judges-jsonl",
            str(judges),
        ],
    )

    assert code == 0
    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert rows[0]["x"]["text"] == "show the sampled fixture systems"
    assert set(rows[0]["y_true"]["rest_api_list"]) == set(expected)
    assert metrics[_metric("accepted_total")] == 1


def test_cli_config_provider_mode_uses_yaml_mock_adapters(tmp_path: Path) -> None:
    """Default config mode uses the checked-in mock adapters without fixture files."""
    script = _load_script()
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=1, count=1))

    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert rows[0]["x"]["text"].startswith("fixture request covering 1")
    assert metrics[_metric("accepted_total")] == 1


def test_cli_split_provider_adapter_overrides(tmp_path: Path) -> None:
    """Draft and judge adapters can be overridden independently from YAML config."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("inspect the selected system\n", encoding="utf-8")

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=1)
        + [
            "--draft-provider-adapter",
            "file",
            "--drafts-jsonl",
            str(drafts),
            "--judge-provider-adapter",
            "mock",
        ],
    )

    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert rows[0]["x"]["text"] == "inspect the selected system"
    assert metrics[_metric("accepted_total")] == 1


def test_cli_openai_compatible_provider_uses_fake_http_and_env_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live provider mode is testable through injected fake HTTP transport."""
    script = _load_script()
    calls: list[dict] = []

    _set_live_identity_env(monkeypatch)
    _set_live_transport_env(monkeypatch)

    def fake_transport(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        """Capture request shape and return OpenAI-compatible fixture JSON."""
        calls.append({
            "url": url,
            "payload": payload,
            "headers": headers,
            "timeout": timeout,
        })
        if payload["model"] == "restored-model-x":
            return {"choices": [{"message": {"content": "show the sampled fixture system"}}]}
        assert payload["model"] == "private-pro"
        assert payload["route"] == "private-pro-route"
        assert payload["profile"] == "think-max"
        prompt = payload["messages"][0]["content"]
        sampled = [
            f"/redfish/v1/Systems/{index}"
            for index in range(8)
            if f"/redfish/v1/Systems/{index}" in prompt
        ]
        return {
            "choices": [
                {
                    "message": {
                        "content": _judge_json(sampled),
                    },
                },
            ],
        }

    monkeypatch.setattr(script, "_urlopen_json_transport", fake_transport)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=1)
        + [
            "--provider-mode",
            "openai-compatible",
        ],
    )

    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = _release_manifest(release_dir)
    assert code == 0
    assert len(calls) == 2
    assert calls[0]["url"] == "http://model-x.invalid/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer draft-token"
    assert calls[0]["payload"]["model"] == "restored-model-x"
    assert calls[0]["payload"]["max_new_tokens"] == 96
    assert calls[1]["url"] == "http://judge.invalid/v1/chat/completions"
    assert calls[1]["headers"]["Authorization"] == "Bearer judge-token"
    assert rows[0]["x"]["text"] == "show the sampled fixture system"
    assert "rest_api_list" not in rows[0]["x"]
    assert rows[0]["y_true"]["rest_api_list"]
    assert metrics[_metric("accepted_total")] == 1
    assert metrics[_metric("model_x", "artifact_sha")] == MODEL_X_ARTIFACT_SHA
    assert manifest["draft_provider_adapter"] == "openai-compatible"
    assert manifest["judge_provider_adapter"] == "openai-compatible"
    assert manifest["judge_route"] == "private-pro-route"
    assert manifest["judge_model"] == "private-pro"
    assert manifest["judge_profile"] == "think-max"
    assert manifest["model_x_artifact_sha"] == MODEL_X_ARTIFACT_SHA


def test_resolve_live_identities_resolves_model_x_and_judge_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live identity placeholders resolve before providers are constructed or called."""
    script = _load_script()
    _set_live_identity_env(monkeypatch)
    spec = script.load_phase2_labelled_requests_spec(
        "configs/phase2_labelled_requests.yaml",
    )

    resolved = script._resolve_live_identities(
        spec,
        draft_adapter="openai-compatible",
        judge_adapter="openai-compatible",
        env=dict(os.environ),
    )

    assert resolved.model_x.model_id == "restored-model-x"
    assert resolved.model_x.artifact_sha == MODEL_X_ARTIFACT_SHA
    assert resolved.judge.route == "private-pro-route"
    assert resolved.judge.model_id == "private-pro"
    assert resolved.judge.profile == "think-max"
    assert spec.model_x.artifact_sha == "${PHASE1_MODEL_X_ARTIFACT_SHA}"
    assert spec.judge.route == "${PHASE2_JUDGE_ROUTE}"


@pytest.mark.parametrize(
    ("artifact_sha", "message"),
    [
        (None, "PHASE1_MODEL_X_ARTIFACT_SHA"),
        ("not-a-sha256", "model_x.artifact_sha must resolve"),
    ],
)
def test_cli_live_model_x_artifact_sha_fails_before_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_sha: str | None,
    message: str,
) -> None:
    """Live D1 generation requires a canonical model_x artifact SHA before HTTP."""
    script = _load_script()
    calls: list[dict] = []
    _set_live_identity_env(monkeypatch, artifact_sha=artifact_sha)
    _set_live_transport_env(monkeypatch)
    monkeypatch.setattr(
        script,
        "_urlopen_json_transport",
        lambda *args: calls.append({"args": args}) or {},
    )

    with pytest.raises(SystemExit, match=message):
        script.main(
            _base_args(tmp_path, sample_width=1, count=1)
            + [
                "--provider-mode",
                "openai-compatible",
            ],
        )

    assert calls == []


def test_cli_live_provider_gate_flag_allows_larger_fake_http_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit live gate flag allows above-cap fake HTTP validation."""
    script = _load_script()
    _set_live_identity_env(monkeypatch)
    _set_live_transport_env(monkeypatch)

    def fake_transport(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        """Return valid provider JSON without opening the network."""
        _ = (url, headers, timeout)
        if payload["model"] == "restored-model-x":
            return {"choices": [{"message": {"content": "show the sampled fixture system"}}]}
        prompt = payload["messages"][0]["content"]
        sampled = [
            f"/redfish/v1/Systems/{index}"
            for index in range(8)
            if f"/redfish/v1/Systems/{index}" in prompt
        ]
        return {
            "choices": [
                {
                    "message": {
                        "content": _judge_json(sampled),
                    },
                },
            ],
        }

    monkeypatch.setattr(script, "_urlopen_json_transport", fake_transport)
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=4)
        + [
            "--provider-mode",
            "openai-compatible",
            "--live-provider-gate-passed",
        ],
    )

    rows = _release_rows(release_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert len(rows) == 4
    assert metrics[_metric("accepted_total")] == 4


def test_cli_live_provider_blocks_dataset_scale_without_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live provider mode refuses larger runs until the explicit gate flag is passed."""
    script = _load_script()
    _set_live_identity_env(monkeypatch)

    with pytest.raises(SystemExit, match="live provider runs"):
        script.main(
            _base_args(tmp_path, sample_width=1, count=4)
            + [
                "--provider-mode",
                "openai-compatible",
            ],
        )


@pytest.mark.parametrize(
    "adapter_args",
    (
        ["--draft-provider-adapter", "openai-compatible"],
        ["--judge-provider-adapter", "openai-compatible"],
    ),
)
def test_cli_live_provider_override_blocks_dataset_scale_without_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_args: list[str],
) -> None:
    """One-sided live adapter overrides also require the explicit live gate."""
    script = _load_script()
    _set_live_identity_env(monkeypatch)
    spec = script.load_phase2_labelled_requests_spec(
        "configs/phase2_labelled_requests.yaml",
    )
    gated_count = spec.live_without_gate_max_candidates + 1

    with pytest.raises(SystemExit, match="live provider runs"):
        script.main(
            _base_args(tmp_path, sample_width=1, count=gated_count)
            + adapter_args,
        )


def test_cli_live_override_requires_live_provider_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI live overrides fail before HTTP when the YAML lacks live routing fields."""
    script = _load_script()
    _set_live_identity_env(monkeypatch)
    spec_path = tmp_path / "phase2-without-provider-fields.yaml"
    spec_text = Path("configs/phase2_labelled_requests.yaml").read_text(encoding="utf-8")
    spec_path.write_text(
        spec_text
        .replace("    base_url_env: PHASE2_MODEL_X_BASE_URL\n", "", 1)
        .replace("    endpoint_path: /v1/chat/completions\n", "", 1),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="providers.draft.base_url_env"):
        script.main(
            _base_args(tmp_path, sample_width=1, count=1)
            + [
                "--spec",
                str(spec_path),
                "--provider-mode",
                "openai-compatible",
            ],
        )


def test_openai_compatible_provider_requires_env_placeholders() -> None:
    """Live providers fail closed instead of falling back to hardcoded models."""
    script = _load_script()
    config = script.ProviderAdapterSpec(
        adapter="openai-compatible",
        base_url_env="PHASE2_MODEL_X_BASE_URL",
        endpoint_path="/v1/chat/completions",
        response_text_path="choices.0.message.content",
    )
    provider = script._OpenAICompatibleChatProvider(
        config,
        label="draft",
        env={},
        transport=lambda *_args: {},
    )

    with pytest.raises(SystemExit, match="PHASE2_MODEL_X_BASE_URL"):
        provider({
            "prompt": "offline prompt",
            "model_id": "${PHASE1_MODEL_X_MODEL_ID}",
            "generation": {},
        })

    provider = script._OpenAICompatibleChatProvider(
        config,
        label="draft",
        env={"PHASE2_MODEL_X_BASE_URL": "http://model.invalid"},
        transport=lambda *_args: {},
    )
    with pytest.raises(SystemExit, match="PHASE1_MODEL_X_MODEL_ID"):
        provider({
            "prompt": "offline prompt",
            "model_id": "${PHASE1_MODEL_X_MODEL_ID}",
            "generation": {},
        })


def test_cli_rejects_mismatch_nonsense_and_invalid_judge_json(tmp_path: Path) -> None:
    """Rejected candidates update metrics but do not release canonical D1."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("mismatch\nnonsense\ninvalid\n", encoding="utf-8")
    judges.write_text(
        "\n".join([
            _judge_json(["/redfish/v1/not-sampled"]),
            _judge_json(
                ["/redfish/v1/Systems/3"],
                accepted=False,
                natural=False,
                nonsense=True,
            ),
            "not-json",
        ])
        + "\n",
        encoding="utf-8",
    )

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=3)
        + [
            "--provider-mode",
            "file",
            "--drafts-jsonl",
            str(drafts),
            "--judges-jsonl",
            str(judges),
            "--allow-threshold-failure",
        ],
    )

    assert code == 0
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics[_metric("draft_total")] == 3
    assert metrics[_metric("accepted_total")] == 0
    assert metrics[_metric("rejected_total")] == 3
    assert metrics[_metric("nonsense_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("invalid_json_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("pro_accept_rate")] == 0.0
    assert metrics[_metric("rest_api_set_match_rate")] == 0.0
    assert metrics[_metric("natural_command_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("ambiguous_rate")] == 0.0
    assert metrics[_metric("duplicate_intent_rate")] == 0.0
    assert metrics[_metric("extra_intent_rate")] == 0.0
    assert metrics[_metric("method_semantics_valid_rate")] == pytest.approx(2 / 3)
    assert metrics["thresholds_pass"] is False


def test_cli_fixture_jsonl_validation_reports_line_numbers(tmp_path: Path) -> None:
    """Bad input rows fail with line-numbered validation messages."""
    script = _load_script()
    records = tmp_path / "bad-records.jsonl"
    records.write_text('{"rest_api": "/redfish/v1/A"}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match=r"bad-records\.jsonl:1: allowed_methods"):
        script.main([
            "--records-jsonl",
            str(records),
            "--output-release-dir",
            str(tmp_path / "out-release"),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--sample-width",
            "1",
        ])

    records.write_text(
        '{"rest_api": "/redfish/v1/A", "allowed_methods": [], "json": {}}\nnot-json\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=r"bad-records\.jsonl:2: invalid JSON"):
        script.load_rest_api_records(records)


def test_cli_seed_and_sample_width_are_deterministic(tmp_path: Path) -> None:
    """The same seed and width produce byte-identical output."""
    script = _load_script()
    first = tmp_path / "first" / "phase2_labelled_requests"
    second = tmp_path / "second" / "phase2_labelled_requests"
    first_metrics = tmp_path / "first" / "metrics.json"
    second_metrics = tmp_path / "second" / "metrics.json"
    records = _write_records(tmp_path / "records.jsonl")

    common = [
        "--records-jsonl",
        str(records),
        "--sample-width",
        "2",
        "--count",
        "3",
        "--seed",
        "19",
    ]
    assert script.main(
        common + ["--output-release-dir", str(first), "--metrics-out", str(first_metrics)],
    ) == 0
    assert script.main(
        common + ["--output-release-dir", str(second), "--metrics-out", str(second_metrics)],
    ) == 0

    assert (first / "data.jsonl").read_text(encoding="utf-8") == (
        second / "data.jsonl"
    ).read_text(encoding="utf-8")
    assert (first / "manifest.json").read_text(encoding="utf-8") == (
        second / "manifest.json"
    ).read_text(encoding="utf-8")
    assert first_metrics.read_text(encoding="utf-8") == second_metrics.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="sample-width"):
        script.main(
            common
            + [
                "--sample-width",
                "4",
                "--output-release-dir",
                str(first),
                "--metrics-out",
                str(first_metrics),
            ],
        )


def test_cli_threshold_failure_does_not_release_final_directory(tmp_path: Path) -> None:
    """A failed validation leaves the final release directory unpublished."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    release_dir = tmp_path / "out" / "phase2_labelled_requests"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("mismatch\n", encoding="utf-8")
    judges.write_text(
        _judge_json(["/redfish/v1/not-sampled"])
        + "\n",
        encoding="utf-8",
    )

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=1)
        + [
            "--provider-mode",
            "file",
            "--drafts-jsonl",
            str(drafts),
            "--judges-jsonl",
            str(judges),
        ],
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 2
    assert metrics["thresholds_pass"] is False
    assert metrics[_metric("accepted_total")] == 0
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()


# Author: Mus mbayramo@stanford.edu
