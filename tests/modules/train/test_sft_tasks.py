"""Focused tests for the shared SFT task registry cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from igc.modules.train import sft_tasks


@pytest.fixture(autouse=True)
def _clear_sft_task_cache():
    sft_tasks._resolve_sft_task_cached.cache_clear()
    yield
    sft_tasks._resolve_sft_task_cached.cache_clear()


def _task(name: str, *, spec_sha256: str) -> sft_tasks.SFTTaskSpec:
    return sft_tasks.SFTTaskSpec(
        name=name,
        phase=1,
        contract_version="unit/v1",
        parent_role="foundation_instruct",
        output_role="model_x",
        dataset_format="jsonl",
        metric_namespace="phase1",
        renderer="unit_renderer",
        order_sensitive=False,
        prompt_template="${text}",
        sources=("redfish_ctl_full_corpus",),
        input_policy={"unit": True},
        spec_sha256=spec_sha256,
    )


def test_resolve_sft_task_caches_by_resolved_spec_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same resolved path returns one frozen object; distinct spec paths are isolated."""
    calls: list[str] = []

    def fake_load_sft_tasks(path: str | Path = sft_tasks.TASK_SPEC_PATH):
        resolved = str(Path(path).expanduser().resolve())
        calls.append(resolved)
        return {
            "phase1_pretrain": _task(
                "phase1_pretrain",
                spec_sha256="sha256:" + str(len(calls)) * 64,
            ),
        }

    monkeypatch.setattr(sft_tasks, "load_sft_tasks", fake_load_sft_tasks)
    spec_a = tmp_path / "sft_tasks.yaml"
    spec_a_same = tmp_path / "." / "sft_tasks.yaml"
    spec_b = tmp_path / "other" / "sft_tasks.yaml"

    first = sft_tasks.resolve_sft_task("phase1_pretrain", spec_a)
    second = sft_tasks.resolve_sft_task("phase1_pretrain", spec_a_same)
    third = sft_tasks.resolve_sft_task("phase1_pretrain", spec_b)

    assert first is second
    assert third is not first
    assert calls == [
        str(spec_a.resolve()),
        str(spec_b.resolve()),
    ]
