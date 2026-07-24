"""Script tests for offline ``phase2_labelled_requests`` generation.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)

SCRIPT = Path("scripts/build_phase2_labelled_requests.py")


def _load_script() -> ModuleType:
    """Load the script module for direct ``main(argv)`` testing."""
    spec = importlib.util.spec_from_file_location("build_phase2_labelled_requests", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_records(path: Path, count: int = 4) -> Path:
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


def _base_args(tmp_path: Path, *, sample_width: int = 2, count: int = 2) -> list[str]:
    """Return common CLI args for fixture tests."""
    return [
        "--records-jsonl",
        str(_write_records(tmp_path / "records.jsonl")),
        "--output-jsonl",
        str(tmp_path / "out" / "phase2_labelled_requests.jsonl"),
        "--metrics-out",
        str(tmp_path / "out" / "metrics.json"),
        "--sample-width",
        str(sample_width),
        "--count",
        str(count),
        "--seed",
        "11",
    ]


def _metric(group: str, name: str | None = None) -> str:
    """Return a Phase 2 labelled-request metric key."""
    return phase_metric(PHASE2_LABELLED_REQUESTS, group, name)


def test_cli_mock_mode_writes_accepted_jsonl_and_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mock providers produce accepted Phase 2 rows and registered metrics."""
    script = _load_script()
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=3, count=2))
    stdout = capsys.readouterr().out

    assert code == 0
    assert "dataset=phase2_labelled_requests" in stdout
    assert "accepted=2" in stdout
    assert f"output_jsonl={output}" in stdout
    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["phase"] == 2
    assert rows[0]["dataset"] == PHASE2_LABELLED_REQUESTS
    assert rows[0]["task"] == "text_to_rest_api_list"
    assert rows[0]["x"]["text"].startswith("fixture request covering 3")
    assert len(rows[0]["x"]["json"]) == 3
    assert len(rows[0]["x"]["rest_api_list"]) == 3
    assert rows[0]["x"]["allowed_methods"][rows[0]["x"]["rest_api_list"][0]][0] == "GET"
    assert set(rows[0]["y_true"]["rest_api_list"]) == set(rows[0]["x"]["rest_api_list"])
    assert rows[0]["validation"]["set_coverage_preserved"] is True
    assert rows[0]["validation"]["review_judged"] is True
    assert "calls" not in json.dumps(rows)
    assert '"method":' not in json.dumps(rows)
    assert '"arguments":' not in json.dumps(rows)
    assert set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS) <= set(metrics)
    assert metrics["dataset"] == PHASE2_LABELLED_REQUESTS
    assert metrics[_metric("draft_total")] == 2
    assert metrics[_metric("accepted_total")] == 2
    assert metrics[_metric("rejected_total")] == 0
    assert metrics[_metric("sample_width", "k")] == 3
    assert metrics[_metric("vendor", "source_corpus")] == "fixture_vendor:fixture_corpus"
    assert metrics["thresholds_pass"] is True
    assert all(
        not key.startswith("phase2_goal_extraction/")
        for key in metrics
    )


@pytest.mark.parametrize("sample_width", (1, 2, 3))
def test_cli_mock_mode_accepts_all_configured_sample_widths(
    tmp_path: Path,
    sample_width: int,
) -> None:
    """Mock mode accepts every configured sample width on a successful path."""
    script = _load_script()
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=sample_width, count=1))

    assert code == 0
    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert len(rows[0]["x"]["json"]) == sample_width
    assert len(rows[0]["x"]["rest_api_list"]) == sample_width
    assert metrics[_metric("sample_width", "k")] == sample_width


def test_cli_file_providers_are_used_without_network(tmp_path: Path) -> None:
    """Local provider fixture files control draft text and judge acceptance."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("show the sampled fixture systems\n", encoding="utf-8")
    judges.write_text(
        json.dumps({
            "accepted": True,
            "rest_api_list": ["/redfish/v1/Systems/3", "/redfish/v1/Systems/2"],
            "nonsense": False,
            "reason": "fixture",
            "order_evidence": "none",
        })
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
    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert rows[0]["x"]["text"] == "show the sampled fixture systems"
    assert set(rows[0]["y_true"]["rest_api_list"]) == {
        "/redfish/v1/Systems/2",
        "/redfish/v1/Systems/3",
    }
    assert metrics[_metric("accepted_total")] == 1


def test_cli_records_accept_nested_x_with_top_level_source_metadata(tmp_path: Path) -> None:
    """Fixture rows may carry Phase-shaped x records plus top-level provenance."""
    script = _load_script()
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({
            "x": {
                "rest_api": "/redfish/v1/Systems/1",
                "allowed_methods": ["get", "head"],
                "json": {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "Name": "System 1",
                },
            },
            "vendor": "top_vendor",
            "source_corpus": "top_corpus",
        })
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main([
        "--records-jsonl",
        str(records),
        "--output-jsonl",
        str(output),
        "--metrics-out",
        str(metrics_path),
        "--sample-width",
        "1",
        "--count",
        "1",
    ])

    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert rows[0]["x"]["rest_api_list"] == ["/redfish/v1/Systems/1"]
    assert rows[0]["x"]["allowed_methods"]["/redfish/v1/Systems/1"] == ["GET", "HEAD"]
    assert rows[0]["metadata"]["vendor"] == ["top_vendor"]
    assert rows[0]["metadata"]["source_corpus"] == ["top_corpus"]
    assert metrics[_metric("vendor", "source_corpus")] == "top_vendor:top_corpus"


def test_cli_config_provider_mode_uses_yaml_mock_adapters(tmp_path: Path) -> None:
    """Default config mode uses the checked-in mock adapters without fixture files."""
    script = _load_script()
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path, sample_width=1, count=1))

    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert rows[0]["x"]["text"].startswith("fixture request covering 1")
    assert metrics[_metric("accepted_total")] == 1


def test_cli_split_provider_adapter_overrides(tmp_path: Path) -> None:
    """Draft and judge adapters can be overridden independently from YAML config."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
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

    rows = _read_jsonl(output)
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

    monkeypatch.setenv("PHASE1_MODEL_X_MODEL_ID", "restored-model-x")
    monkeypatch.setenv("PHASE2_JUDGE_MODEL_ID", "private-pro")
    monkeypatch.setenv("PHASE2_JUDGE_ROUTE", "private-pro-route")
    monkeypatch.setenv("PHASE2_JUDGE_PROFILE", "think-max")
    monkeypatch.setenv("PHASE2_MODEL_X_BASE_URL", "http://model-x.invalid")
    monkeypatch.setenv("PHASE2_JUDGE_BASE_URL", "http://judge.invalid")
    monkeypatch.setenv("PHASE2_MODEL_X_API_KEY", "draft-token")
    monkeypatch.setenv("PHASE2_JUDGE_API_KEY", "judge-token")

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
            for index in range(4)
            if f"/redfish/v1/Systems/{index}" in prompt
        ]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "accepted": True,
                            "rest_api_list": sampled,
                            "nonsense": False,
                            "reason": "fixture",
                            "order_evidence": "none",
                        }),
                    },
                },
            ],
        }

    monkeypatch.setattr(script, "_urlopen_json_transport", fake_transport)
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=1)
        + [
            "--provider-mode",
            "openai-compatible",
        ],
    )

    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert len(calls) == 2
    assert calls[0]["url"] == "http://model-x.invalid/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer draft-token"
    assert calls[0]["payload"]["model"] == "restored-model-x"
    assert calls[0]["payload"]["max_new_tokens"] == 96
    assert calls[1]["url"] == "http://judge.invalid/v1/chat/completions"
    assert calls[1]["headers"]["Authorization"] == "Bearer judge-token"
    assert rows[0]["x"]["text"] == "show the sampled fixture system"
    assert rows[0]["y_true"]["rest_api_list"] == rows[0]["x"]["rest_api_list"]
    assert metrics[_metric("accepted_total")] == 1


def test_cli_live_provider_gate_flag_allows_larger_fake_http_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit live gate flag allows above-cap fake HTTP validation."""
    script = _load_script()
    monkeypatch.setenv("PHASE1_MODEL_X_MODEL_ID", "restored-model-x")
    monkeypatch.setenv("PHASE2_JUDGE_MODEL_ID", "private-pro")
    monkeypatch.setenv("PHASE2_JUDGE_ROUTE", "private-pro-route")
    monkeypatch.setenv("PHASE2_JUDGE_PROFILE", "think-max")
    monkeypatch.setenv("PHASE2_MODEL_X_BASE_URL", "http://model-x.invalid")
    monkeypatch.setenv("PHASE2_JUDGE_BASE_URL", "http://judge.invalid")

    def fake_transport(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        """Return valid provider JSON without opening the network."""
        _ = (url, headers, timeout)
        if payload["model"] == "restored-model-x":
            return {"choices": [{"message": {"content": "show the sampled fixture system"}}]}
        prompt = payload["messages"][0]["content"]
        sampled = [
            f"/redfish/v1/Systems/{index}"
            for index in range(4)
            if f"/redfish/v1/Systems/{index}" in prompt
        ]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "accepted": True,
                            "rest_api_list": sampled,
                            "nonsense": False,
                        }),
                    },
                },
            ],
        }

    monkeypatch.setattr(script, "_urlopen_json_transport", fake_transport)
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path, sample_width=1, count=4)
        + [
            "--provider-mode",
            "openai-compatible",
            "--live-provider-gate-passed",
        ],
    )

    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert len(rows) == 4
    assert metrics[_metric("accepted_total")] == 4


def test_cli_live_provider_blocks_dataset_scale_without_gate(tmp_path: Path) -> None:
    """Live provider mode refuses larger runs until the explicit gate flag is passed."""
    script = _load_script()

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
    adapter_args: list[str],
) -> None:
    """One-sided live adapter overrides also require the explicit live gate."""
    script = _load_script()
    spec = script.load_phase2_labelled_requests_spec(
        "configs/phase2_labelled_requests.yaml",
    )
    gated_count = spec.live_without_gate_max_candidates + 1

    with pytest.raises(SystemExit, match="live provider runs"):
        script.main(
            _base_args(tmp_path, sample_width=1, count=gated_count)
            + adapter_args,
        )


def test_cli_live_override_requires_live_provider_config(tmp_path: Path) -> None:
    """CLI live overrides fail before HTTP when the YAML lacks live routing fields."""
    script = _load_script()
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


@pytest.mark.parametrize(
    ("response", "message"),
    (
        ({}, "missing choices.0.message.content"),
        ({"choices": []}, "out of range"),
        ({"choices": [{"message": {"content": 17}}]}, "did not contain text"),
        ({"choices": [{"message": {"content": "   "}}]}, "did not contain text"),
    ),
)
def test_openai_compatible_provider_rejects_bad_response_text_paths(
    response: dict,
    message: str,
) -> None:
    """Provider response-path errors fail closed before a row can be accepted."""
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
        env={
            "PHASE2_MODEL_X_BASE_URL": "http://model.invalid",
            "PHASE1_MODEL_X_MODEL_ID": "configured-model-x",
        },
        transport=lambda *_args: response,
    )

    with pytest.raises(SystemExit, match=message):
        provider({
            "prompt": "offline prompt",
            "model_id": "${PHASE1_MODEL_X_MODEL_ID}",
            "generation": {},
        })


def test_cli_rejects_mismatch_nonsense_and_invalid_judge_json(tmp_path: Path) -> None:
    """Rejected candidates are omitted while aggregate counters still update."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    output = tmp_path / "out" / "phase2_labelled_requests.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("mismatch\nnonsense\ninvalid\n", encoding="utf-8")
    judges.write_text(
        "\n".join([
            json.dumps({
                "accepted": True,
                "rest_api_list": ["/redfish/v1/not-sampled"],
                "nonsense": False,
            }),
            json.dumps({
                "accepted": False,
                "rest_api_list": ["/redfish/v1/Systems/3"],
                "nonsense": True,
            }),
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
    assert _read_jsonl(output) == []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics[_metric("draft_total")] == 3
    assert metrics[_metric("accepted_total")] == 0
    assert metrics[_metric("rejected_total")] == 3
    assert metrics[_metric("nonsense_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("invalid_json_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("pro_accept_rate")] == pytest.approx(1 / 3)
    assert metrics[_metric("rest_api_set_match_rate")] == pytest.approx(1 / 3)
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
            "--output-jsonl",
            str(tmp_path / "out.jsonl"),
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
    first = tmp_path / "first" / "phase2_labelled_requests.jsonl"
    second = tmp_path / "second" / "phase2_labelled_requests.jsonl"
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
        common + ["--output-jsonl", str(first), "--metrics-out", str(first_metrics)],
    ) == 0
    assert script.main(
        common + ["--output-jsonl", str(second), "--metrics-out", str(second_metrics)],
    ) == 0

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert first_metrics.read_text(encoding="utf-8") == second_metrics.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="sample-width"):
        script.main(
            common
            + [
                "--sample-width",
                "4",
                "--output-jsonl",
                str(first),
                "--metrics-out",
                str(first_metrics),
            ],
        )


def test_cli_threshold_failure_returns_nonzero_after_writing_metrics(tmp_path: Path) -> None:
    """Threshold failures are explicit and do not prevent artifact inspection."""
    script = _load_script()
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"
    drafts.write_text("mismatch\n", encoding="utf-8")
    judges.write_text(
        json.dumps({
            "accepted": True,
            "rest_api_list": ["/redfish/v1/not-sampled"],
            "nonsense": False,
        })
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


# Author: Mus mbayramo@stanford.edu
