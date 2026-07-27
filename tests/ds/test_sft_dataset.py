"""Unit tests for shared PromptCompletion SFT tokenization."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from igc.ds.sft_dataset import PromptCompletionJSONLDataset, tokenize_prompt_completion


class _CharTokenizer:
    """Minimal tokenizer that returns one id per character."""

    pad_token_id = 0

    def __call__(
        self,
        text,
        padding=None,
        truncation=None,
        return_tensors=None,
        add_special_tokens=None,
    ):
        _ = (padding, truncation, return_tensors, add_special_tokens)
        ids = [ord(char) % 1000 + 1 for char in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_prompt_completion_overflow_raises_instead_of_truncating() -> None:
    """Completion-only SFT rows fail closed when prompt+completion exceeds max_len."""
    with pytest.raises(ValueError, match="exceeds max_len"):
        tokenize_prompt_completion(
            _CharTokenizer(),
            prompt="prompt:",
            completion="completion",
            max_len=8,
        )


def test_prompt_completion_masks_prompt_and_padding_when_it_fits() -> None:
    """The non-overflow path still pads and labels only completion tokens."""
    item = tokenize_prompt_completion(
        _CharTokenizer(),
        prompt="ab",
        completion="cd",
        max_len=6,
    )

    assert item["input_ids"].shape == (6,)
    assert item["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]
    assert item["labels"][:2].tolist() == [-100, -100]
    assert item["labels"][2:4].tolist() == item["input_ids"][2:4].tolist()
    assert item["labels"][4:].tolist() == [-100, -100]


def _write_jsonl(path, rows):
    """Write deterministic JSONL bytes and return the expected sha256 identity."""
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _write_release_manifest(
    path,
    *,
    artifact_sha,
    rows,
    dataset="D1",
    immutable=True,
    complete=True,
):
    """Write a deterministic release manifest and return its exact sha256."""
    manifest = {
        "schema_version": "d1_release.v1",
        "dataset": dataset,
        "immutable": immutable,
        "complete": complete,
        "artifact_sha256": artifact_sha,
        "rows": rows,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _render_prompt_completion(row):
    """Renderer stub for tests that keeps raw JSONL validation out of scope."""
    return row["prompt"], row["completion"]


def test_prompt_completion_dataset_verifies_release_manifest_identity(tmp_path) -> None:
    """A required D1 manifest must match exact JSONL bytes, rows, and dataset."""
    train_path = tmp_path / "train.jsonl"
    train_sha = _write_jsonl(
        train_path,
        [{"prompt": "phase2 prompt", "completion": "phase2 completion"}],
    )
    manifest_sha = _write_release_manifest(
        tmp_path / "train.manifest.json",
        artifact_sha=train_sha,
        rows=1,
    )

    train = PromptCompletionJSONLDataset(
        train_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase2_goal_extraction",
        max_len=64,
        tokenizer=_CharTokenizer(),
        manifest_path=tmp_path / "train.manifest.json",
        require_manifest=True,
        expected_dataset="D1",
    )

    assert train.data_sha256 == train_sha
    assert train.manifest_sha256 == manifest_sha
    assert train.run_manifest_fields() == {
        "data_manifest": manifest_sha,
        "eval_split": "",
        "train_data_sha": train_sha,
        "eval_data_sha": "",
        "source_manifest_sha": manifest_sha,
        "eval_manifest_sha": "",
    }


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"artifact_sha": "sha256:" + "9" * 64}, "artifact SHA"),
        ({"rows": 2}, "row count"),
        ({"dataset": "phase2_labelled_requests"}, "dataset"),
        ({"immutable": False}, "immutable=true"),
        ({"complete": False}, "immutable=true"),
    ],
)
def test_prompt_completion_dataset_rejects_bad_release_manifests(
    tmp_path,
    manifest_overrides,
    message,
) -> None:
    """Missing, mismatched, mutable, or wrong-dataset manifests fail closed."""
    train_path = tmp_path / "train.jsonl"
    train_sha = _write_jsonl(train_path, [{"prompt": "p", "completion": "c"}])
    manifest_args = {
        "artifact_sha": train_sha,
        "rows": 1,
        "dataset": "D1",
        "immutable": True,
        "complete": True,
    }
    manifest_args.update(manifest_overrides)
    manifest_path = tmp_path / "train.manifest.json"
    _write_release_manifest(manifest_path, **manifest_args)

    with pytest.raises(ValueError, match=message):
        PromptCompletionJSONLDataset(
            train_path,
            renderer=_render_prompt_completion,
            metric_namespace="phase2_goal_extraction",
            max_len=64,
            tokenizer=_CharTokenizer(),
            manifest_path=manifest_path,
            require_manifest=True,
            expected_dataset="D1",
        )


def test_prompt_completion_dataset_rejects_missing_required_manifest(tmp_path) -> None:
    """A required manifest is not optional for Phase 2/3 SFT release inputs."""
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, [{"prompt": "p", "completion": "c"}])

    with pytest.raises(FileNotFoundError, match="release manifest"):
        PromptCompletionJSONLDataset(
            train_path,
            renderer=_render_prompt_completion,
            metric_namespace="phase2_goal_extraction",
            max_len=64,
            tokenizer=_CharTokenizer(),
            manifest_path=tmp_path / "missing.manifest.json",
            require_manifest=True,
            expected_dataset="D1",
        )


def test_prompt_completion_dataset_bind_eval_dataset_records_data_and_manifest_sha(
    tmp_path,
) -> None:
    """Run manifest fields carry exact train/eval JSONL and release-manifest SHAs."""
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_sha = _write_jsonl(
        train_path,
        [{"prompt": "phase2 prompt", "completion": "phase2 completion"}],
    )
    eval_sha = _write_jsonl(
        eval_path,
        [{"prompt": "phase2 heldout", "completion": "phase2 target"}],
    )
    train_manifest_sha = _write_release_manifest(
        tmp_path / "train.manifest.json",
        artifact_sha=train_sha,
        rows=1,
    )
    eval_manifest_sha = _write_release_manifest(
        tmp_path / "eval.manifest.json",
        artifact_sha=eval_sha,
        rows=1,
    )
    train = PromptCompletionJSONLDataset(
        train_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase2_goal_extraction",
        max_len=64,
        tokenizer=_CharTokenizer(),
        manifest_path=tmp_path / "train.manifest.json",
        require_manifest=True,
        expected_dataset="D1",
    )
    heldout = PromptCompletionJSONLDataset(
        eval_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase2_goal_extraction",
        max_len=64,
        tokenizer=train.tokenizer,
        manifest_path=tmp_path / "eval.manifest.json",
        require_manifest=True,
        expected_dataset="D1",
    )

    assert train.data_sha256 == train_sha
    assert train.manifest_sha256 == train_manifest_sha
    assert heldout.data_sha256 == eval_sha
    assert heldout.manifest_sha256 == eval_manifest_sha
    assert train.run_manifest_fields() == {
        "data_manifest": train_manifest_sha,
        "eval_split": "",
        "train_data_sha": train_sha,
        "eval_data_sha": "",
        "source_manifest_sha": train_manifest_sha,
        "eval_manifest_sha": "",
    }

    train.bind_eval_dataset(heldout)

    assert train.run_manifest_fields() == {
        "data_manifest": train_manifest_sha,
        "eval_split": eval_sha,
        "train_data_sha": train_sha,
        "eval_data_sha": eval_sha,
        "source_manifest_sha": train_manifest_sha,
        "eval_manifest_sha": eval_manifest_sha,
    }


def test_prompt_completion_dataset_bind_eval_requires_verified_manifest(tmp_path) -> None:
    """Eval datasets without verified release manifests cannot enter run lineage."""
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_sha = _write_jsonl(train_path, [{"prompt": "p", "completion": "c"}])
    _write_jsonl(eval_path, [{"prompt": "e", "completion": "v"}])
    _write_release_manifest(
        tmp_path / "train.manifest.json",
        artifact_sha=train_sha,
        rows=1,
    )
    train = PromptCompletionJSONLDataset(
        train_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase3_argument_extraction",
        max_len=16,
        tokenizer=_CharTokenizer(),
        manifest_path=tmp_path / "train.manifest.json",
        require_manifest=True,
        expected_dataset="D1",
    )
    heldout = PromptCompletionJSONLDataset(
        eval_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase3_argument_extraction",
        max_len=16,
        tokenizer=train.tokenizer,
    )

    with pytest.raises(ValueError, match="verified release manifest"):
        train.bind_eval_dataset(heldout)


@pytest.mark.parametrize("bad", ["", "sha256:not-hex", "sha256:" + "a" * 63])
def test_prompt_completion_dataset_rejects_invalid_eval_sha(tmp_path, bad) -> None:
    """Eval split identities must be canonical sha256 digests."""
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, [{"prompt": "p", "completion": "c"}])
    train = PromptCompletionJSONLDataset(
        train_path,
        renderer=_render_prompt_completion,
        metric_namespace="phase3_argument_extraction",
        max_len=16,
        tokenizer=_CharTokenizer(),
    )

    with pytest.raises(ValueError, match="sha256 digest"):
        train.set_eval_split_sha256(bad)
