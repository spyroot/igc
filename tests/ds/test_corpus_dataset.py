"""
Offline tests for the corpus tokenizer bridge (CorpusJSONLDataset).

Pins that a corpus written by write_corpus loads into fixed-length
input_ids/attention_mask items the trainer's collate stacks, that the trainer-facing
shared-SFT surface (tokenizer property, labels, and run-manifest fields) is present,
that run_manifest_fields round-trips the mixer's data_manifest/eval_split ids, and that
a missing corpus raises. Uses a fake tokenizer — no downloads, no network.

Author:
Mus mbayramo@stanford.edu
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from igc.ds.corpus_dataset import CorpusJSONLDataset
from igc.ds.phase1_render import (
    PHASE1_DATASET,
    PHASE1_TASK,
    build_phase1_row,
    render_phase1_completion,
    render_phase1_prompt,
    validate_phase1_row,
)
from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.corpus_io import write_corpus
from igc.ds.sources.mixer import DataManifest, SourceMix
from igc.ds.sources.training_object import normalize


class _FakeTokenizer:
    """Minimal HF-like tokenizer: char-code ids, pads/truncates to max_length."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=None,
                 return_offsets_mapping=False):
        ids = [ord(c) % 1000 + 1 for c in text]
        if add_special_tokens:
            ids = [999] + ids + [998]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            while len(ids) < max_length:
                ids.append(0)
                mask.append(0)
        result = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index in range(len(text))]]
            )
        return result


class _EmptyCompletionTokenizer(_FakeTokenizer):
    """Tokenizer stub that exposes a degenerate zero-token completion."""

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=None,
                 return_offsets_mapping=False):
        if text == '{\n  "empty": true\n}\n':
            empty = torch.empty((1, 0), dtype=torch.long)
            result = {"input_ids": empty, "attention_mask": empty}
            if return_offsets_mapping:
                result["offset_mapping"] = torch.empty((1, 0, 2), dtype=torch.long)
            return result
        return super().__call__(
            text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )


class _NoAddSpecialCallTokenizer(_FakeTokenizer):
    """Tokenizer where only encode() can suppress special tokens."""

    def __init__(self) -> None:
        self.encode_add_special_tokens = None

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=None):
        if add_special_tokens is not None:
            raise TypeError("legacy tokenizer does not accept add_special_tokens")
        ids = [777] + [ord(c) % 1000 + 1 for c in text] + [778]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}

    def encode(self, text, add_special_tokens=True):
        self.encode_add_special_tokens = add_special_tokens
        ids = [ord(c) % 1000 + 1 for c in text]
        if add_special_tokens:
            ids = [777] + ids + [778]
        return ids


class _FakeSource:
    """Fixed-record source for building a small corpus."""

    def __init__(self, records):
        self.source = records[0].source if records else "real"
        self.trust_level = TrustLevel.REAL
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _corpus_dir(tmp_path: Path, n=4) -> str:
    """Write a small normalized corpus (examples.jsonl + manifest.json)."""
    recs = [SourceRecord(url=f"/redfish/v1/S/{i}",
                         response={"@odata.id": f"/redfish/v1/S/{i}", "Id": str(i)},
                         source="real_dell", trust_level=TrustLevel.REAL,
                         allowed_methods=["GET"], vendor="dell") for i in range(n)]
    mix = SourceMix([_FakeSource(recs)], eval_fraction=0.25, seed=0)
    train, _ = mix.split()
    out = tmp_path / "corpus"
    write_corpus(normalize(train), mix.manifest(), str(out))
    return str(out)


def _phase1_row_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _phase1_metadata(
    source: str = "unit-fixture",
    rest_api: str = "/redfish/v1/Systems/1",
) -> dict[str, object]:
    return {
        "row_id": _phase1_row_id(source, rest_api),
        "source_corpus": source,
        "trust_level": "REAL",
        "vendor": "unit",
    }


def _explicit_phase1_corpus_dir(tmp_path: Path, body: dict, target: dict | None = None) -> str:
    """Write one explicit canonical D0 Phase 1 row without invoking the mixer."""
    out = tmp_path / "explicit_corpus"
    out.mkdir()
    rest_api = "/redfish/v1/Systems/1"
    row = build_phase1_row(
        rest_api=rest_api,
        allowed_methods=["GET", "PATCH"],
        input_json=body,
        target_json=target or body,
        metadata=_phase1_metadata(rest_api=rest_api),
    )
    (out / "examples.jsonl").write_text(json.dumps(row) + "\n")
    return str(out)


def _first_example(corpus_dir: str) -> dict:
    """Return the first JSONL row from a tiny test corpus."""

    examples = Path(corpus_dir) / "examples.jsonl"
    return json.loads(examples.read_text(encoding="utf-8").splitlines()[0])


def _file_sha(path: Path) -> str:
    """Return the canonical sha256 identity for an exact file."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_items_are_fixed_length_tensor_dicts(tmp_path: Path):
    """Every item is a {input_ids, attention_mask} pair of length max_len."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=64, tokenizer=_FakeTokenizer())
    assert len(ds) > 0
    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask"}
    assert item["input_ids"].shape == (64,) and item["attention_mask"].shape == (64,)


def test_phase1_items_mask_prompt_and_padding_labels(tmp_path: Path):
    """Phase 1 rows train only on the completion JSON, never prompt or padding tokens."""
    ds = CorpusJSONLDataset(
        _explicit_phase1_corpus_dir(
            tmp_path,
            {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
        ),
        max_len=256,
        tokenizer=_FakeTokenizer(),
        objective="phase1_pretrain",
    )

    item = ds[0]

    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (256,)
    assert item["labels"].shape == (256,)
    active = item["labels"].ne(-100).nonzero(as_tuple=False).flatten()
    assert active.numel() > 0
    assert active[0].item() > 0
    assert torch.equal(item["labels"][active], item["input_ids"][active])
    assert item["labels"][item["attention_mask"].eq(0)].eq(-100).all()
    assert ds.metric_namespace == "phase1_finetune"


def test_phase1_structural_loss_changes_train_view_and_active_labels(tmp_path: Path):
    """Epochs rotate structural masks while preserving canonical target bytes."""
    tokenizer = _FakeTokenizer()
    ds = CorpusJSONLDataset(
        _explicit_phase1_corpus_dir(
            tmp_path,
            {
                "@odata.id": "/redfish/v1/Systems/1",
                "Id": "1",
                "Actions": {
                    "#ComputerSystem.Reset": {
                        "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                        "ResetType@Redfish.AllowableValues": ["On", "ForceOff"],
                    }
                },
            },
        ),
        max_len=2048,
        tokenizer=tokenizer,
        objective="phase1_pretrain",
        phase1_structural_loss_profile="historical_structural_mask_v1",
        phase1_structural_loss_mode="train",
        phase1_structural_loss_seed=31,
    )

    items = []
    for epoch in range(6):
        ds.set_epoch(epoch)
        items.append(ds[0])

    assert len({tuple(item["input_ids"].tolist()) for item in items}) > 1
    active_labels = [
        tuple(item["labels"][item["labels"].ne(-100)].tolist())
        for item in items
    ]
    assert len(set(active_labels)) > 1
    canonical_completion = render_phase1_completion(
        _first_example(ds._corpus_dir)["y_true"]["json"]
    )
    assert all(0 < len(labels) < len(canonical_completion) for labels in active_labels)
    assert ds.phase1_structural_loss_profile == "historical_structural_mask_v1"
    assert ds.phase1_structural_loss_spec_sha.startswith("sha256:")


def test_phase1_structural_loss_heldout_view_ignores_epoch_changes(tmp_path: Path):
    """Held-out structural masking is row-fixed for stable promotion evidence."""
    ds = CorpusJSONLDataset(
        _explicit_phase1_corpus_dir(
            tmp_path,
            {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
        ),
        max_len=1024,
        tokenizer=_FakeTokenizer(),
        objective="phase1_pretrain",
        phase1_structural_loss_profile="historical_structural_mask_v1",
        phase1_structural_loss_mode="evaluation",
        phase1_structural_loss_seed=31,
    )

    before = ds[0]
    ds.set_epoch(99)
    after = ds[0]

    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(before[key], after[key])
    active = before["labels"].ne(-100).sum().item()
    attended = before["attention_mask"].sum().item()
    assert 0 < active < attended


def test_phase1_tiny_sequence_raises_instead_of_truncating_sft(tmp_path: Path):
    """Phase 1 SFT overflow fails closed instead of silently truncating JSON."""
    with pytest.raises(ValueError, match="exceeds max_len"):
        CorpusJSONLDataset(
            _explicit_phase1_corpus_dir(
                tmp_path,
                {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
            ),
            max_len=8,
            tokenizer=_FakeTokenizer(),
            objective="phase1_pretrain",
        )


def test_phase1_rejects_max_len_too_small(tmp_path: Path):
    """A one-token Phase 1 sequence cannot hold prompt context and completion."""
    with pytest.raises(ValueError, match="max_len >= 2"):
        CorpusJSONLDataset(
            _explicit_phase1_corpus_dir(
                tmp_path,
                {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
            ),
            max_len=1,
            tokenizer=_FakeTokenizer(),
            objective="phase1_pretrain",
        )


def test_phase1_rejects_empty_completion_tokens(tmp_path: Path):
    """A tokenizer bug or odd target must not produce an all-ignored training row."""
    with pytest.raises(ValueError, match="completion tokenized to zero tokens"):
        CorpusJSONLDataset(
            _explicit_phase1_corpus_dir(
                tmp_path,
                {"@odata.id": "/redfish/v1/Systems/1"},
                target={"empty": True},
            ),
            max_len=32,
            tokenizer=_EmptyCompletionTokenizer(),
            objective="phase1_pretrain",
        )


def test_phase1_long_prompt_overflow_raises_without_truncating(tmp_path: Path):
    """Long Phase 1 SFT rows are rejected until a truncation-safe contract exists."""
    with pytest.raises(ValueError, match="exceeds max_len"):
        CorpusJSONLDataset(
            _explicit_phase1_corpus_dir(
                tmp_path,
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "Description": "A" * 500,
                },
                target={"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
            ),
            max_len=96,
            tokenizer=_FakeTokenizer(),
            objective="phase1_pretrain",
        )


def test_phase1_renderer_matches_existing_token_stream(tmp_path: Path):
    """The shared renderer preserves the original prompt/completion tokens."""

    body = {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"}
    target = {"@odata.id": "/redfish/v1/Systems/1", "Id": "1", "PowerState": "On"}
    corpus_dir = _explicit_phase1_corpus_dir(tmp_path, body, target=target)
    tok = _FakeTokenizer()
    ds = CorpusJSONLDataset(
        corpus_dir,
        max_len=512,
        tokenizer=tok,
        objective="phase1_pretrain",
    )
    example = _first_example(corpus_dir)

    assert example["phase"] == 1
    assert example["dataset"] == PHASE1_DATASET
    assert example["task"] == PHASE1_TASK
    prompt, target_json = render_phase1_prompt(example)
    completion = render_phase1_completion(target_json)
    expected_prompt = (
        "### REST API\n"
        "/redfish/v1/Systems/1\n\n"
        "### Allowed Methods\n"
        "GET, PATCH\n\n"
        "### Redfish JSON Input\n"
        "{\n"
        "  \"@odata.id\": \"/redfish/v1/Systems/1\",\n"
        "  \"PowerState\": \"On\"\n"
        "}\n\n"
        "### Complete Redfish JSON\n"
    )
    expected_completion = (
        "{\n"
        "  \"@odata.id\": \"/redfish/v1/Systems/1\",\n"
        "  \"Id\": \"1\",\n"
        "  \"PowerState\": \"On\"\n"
        "}\n"
    )

    assert prompt == expected_prompt
    assert target_json == target
    assert completion == expected_completion

    item = ds[0]
    expected = ds._tokenize_prompt_completion(tok, expected_prompt, expected_completion)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(item[key], expected[key])


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        pytest.param(
            {
                "row_id": "sha256:" + "1" * 64,
                "source_corpus": "unit-fixture",
                "trust_level": "REAL",
            },
            "metadata must contain source lineage",
            id="missing-vendor",
        ),
        pytest.param(
            {
                "row_id": "sha256:" + "1" * 64,
                "source_corpus": "unit-fixture",
                "trust_level": "REAL",
                "vendor": "unit",
                "extra": "forbidden",
            },
            "metadata must contain source lineage",
            id="extra-field",
        ),
        pytest.param(
            {
                "row_id": "not-a-sha",
                "source_corpus": "unit-fixture",
                "trust_level": "REAL",
                "vendor": "unit",
            },
            "metadata.row_id",
            id="bad-row-id",
        ),
        pytest.param(
            {
                "row_id": "sha256:" + "1" * 64,
                "source_corpus": "",
                "trust_level": "REAL",
                "vendor": "unit",
            },
            "metadata.source_corpus",
            id="empty-source",
        ),
        pytest.param(
            {
                "row_id": "sha256:" + "1" * 64,
                "source_corpus": "unit-fixture",
                "trust_level": "",
                "vendor": "unit",
            },
            "metadata.trust_level",
            id="empty-trust",
        ),
        pytest.param(
            {
                "row_id": "sha256:" + "1" * 64,
                "source_corpus": "unit-fixture",
                "trust_level": "REAL",
                "vendor": ["unit"],
            },
            "metadata.vendor",
            id="bad-vendor",
        ),
    ],
)
def test_phase1_row_rejects_invalid_metadata_contract(
    metadata: dict[str, object],
    message: str,
) -> None:
    """Explicit D0 rows validate source-lineage metadata without defaults."""
    row = build_phase1_row(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET"],
        input_json={"@odata.id": "/redfish/v1/Systems/1"},
        target_json={"@odata.id": "/redfish/v1/Systems/1"},
    )
    row["metadata"] = metadata

    with pytest.raises(ValueError, match=message):
        validate_phase1_row(row)


def test_phase1_pretrain_rejects_normalized_legacy_rows(tmp_path: Path):
    """Phase 1 pretraining accepts canonical D0 rows only."""
    corpus_dir = _corpus_dir(tmp_path, n=4)

    with pytest.raises(ValueError, match="Phase 1 row phase must equal 1"):
        CorpusJSONLDataset(
            corpus_dir,
            max_len=256,
            tokenizer=_FakeTokenizer(),
            objective="phase1_pretrain",
        )


def test_legacy_objective_accepts_normalized_rows(tmp_path: Path):
    """Legacy objective compatibility remains explicit and separate from Phase 1."""
    ds = CorpusJSONLDataset(
        _corpus_dir(tmp_path, n=4),
        max_len=64,
        tokenizer=_FakeTokenizer(),
        objective="legacy",
    )

    assert len(ds) > 0
    assert ds.metric_namespace == ""
    assert set(ds[0]) == {"input_ids", "attention_mask"}


def test_items_stack_like_the_trainer_collate(tmp_path: Path):
    """torch.stack over items works — the exact contract of custom_collate_fn."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=32, tokenizer=_FakeTokenizer())
    batch = {
        k: torch.stack([ds[i][k] for i in range(len(ds))])
        for k in ("input_ids", "attention_mask")
    }
    assert batch["input_ids"].shape == (len(ds), 32)
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["input_ids"].dtype == torch.int64


def test_shared_sft_dataset_surface(tmp_path: Path):
    """CorpusJSONLDataset exposes the shared SFT tensor and lineage surface."""
    ds = CorpusJSONLDataset(
        _explicit_phase1_corpus_dir(
            tmp_path,
            {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
        ),
        max_len=512,
        tokenizer=_FakeTokenizer(),
        objective="phase1_pretrain",
    )

    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (512,)
    assert item["attention_mask"].shape == (512,)
    assert item["labels"].shape == (512,)
    assert ds.tokenizer is not None
    ds.load_tokenizer()  # idempotent
    fields = ds.run_manifest_fields()
    assert fields == {
        "data_manifest": "",
        "eval_split": "",
        "train_data_sha": ds.data_sha256,
        "eval_data_sha": "",
        "source_manifest_sha": "",
        "source_registry_sha": "",
        "source_artifact_manifest_shas": {},
        "phase1_structural_loss_profile": ds.phase1_structural_loss_profile,
        "phase1_structural_loss_spec_sha": ds.phase1_structural_loss_spec_sha,
    }


def test_run_manifest_fields_round_trip_exact_file_lineage(tmp_path: Path):
    """Run manifest fields carry semantic manifest plus exact train/eval file SHAs."""
    train_dir = _corpus_dir(tmp_path / "train")
    eval_dir = _corpus_dir(tmp_path / "eval")
    ds = CorpusJSONLDataset(train_dir, max_len=16, tokenizer=_FakeTokenizer())
    heldout = CorpusJSONLDataset(eval_dir, max_len=16, tokenizer=ds.tokenizer)
    train_examples_sha = _file_sha(Path(train_dir) / "examples.jsonl")
    train_manifest_sha = _file_sha(Path(train_dir) / "manifest.json")
    eval_examples_sha = _file_sha(Path(eval_dir) / "examples.jsonl")
    fields = ds.run_manifest_fields()
    expected_manifest = DataManifest(**ds.manifest).content_hash()

    assert ds.data_sha256 == train_examples_sha
    assert ds.manifest_sha256 == train_manifest_sha
    assert heldout.data_sha256 == eval_examples_sha
    assert fields == {
        "data_manifest": expected_manifest,
        "eval_split": "",
        "train_data_sha": train_examples_sha,
        "eval_data_sha": "",
        "source_manifest_sha": train_manifest_sha,
        "source_registry_sha": "",
        "source_artifact_manifest_shas": {},
        "phase1_structural_loss_profile": ds.phase1_structural_loss_profile,
        "phase1_structural_loss_spec_sha": ds.phase1_structural_loss_spec_sha,
    }

    ds.set_eval_split_sha256(heldout.data_sha256)

    fields = ds.run_manifest_fields()
    assert fields == {
        "data_manifest": expected_manifest,
        "eval_split": eval_examples_sha,
        "train_data_sha": train_examples_sha,
        "eval_data_sha": eval_examples_sha,
        "source_manifest_sha": train_manifest_sha,
        "source_registry_sha": "",
        "source_artifact_manifest_shas": {},
        "phase1_structural_loss_profile": ds.phase1_structural_loss_profile,
        "phase1_structural_loss_spec_sha": ds.phase1_structural_loss_spec_sha,
    }
    assert fields["data_manifest"].startswith("sha256:")
    assert fields["train_data_sha"].startswith("sha256:")
    assert fields["eval_data_sha"].startswith("sha256:")
    assert fields["source_manifest_sha"].startswith("sha256:")
    digest_fields = [
        "data_manifest",
        "eval_split",
        "train_data_sha",
        "eval_data_sha",
        "source_manifest_sha",
    ]
    assert all(len(fields[name]) == 71 for name in digest_fields)


def test_missing_corpus_raises(tmp_path: Path):
    """A directory without examples.jsonl fails fast, not at first batch."""
    with pytest.raises(FileNotFoundError):
        CorpusJSONLDataset(str(tmp_path / "nope"), tokenizer=_FakeTokenizer())


def test_unknown_objective_rejected(tmp_path: Path):
    """The corpus objective is explicit; typos fail before tokenization."""
    with pytest.raises(ValueError, match="unknown corpus objective"):
        CorpusJSONLDataset(_corpus_dir(tmp_path), tokenizer=_FakeTokenizer(), objective="phase9")


def test_token_ids_fallback_keeps_special_tokens_disabled():
    """Legacy tokenizer fallback must preserve the prompt/completion boundary."""
    tok = _NoAddSpecialCallTokenizer()

    ids = CorpusJSONLDataset._token_ids(tok, "abc")

    assert ids.tolist() == [98, 99, 100]
    assert tok.encode_add_special_tokens is False


def _contains_subsequence(values: list[int], needle: list[int]) -> bool:
    """Whether ``needle`` appears contiguously in ``values``."""
    return any(values[i:i + len(needle)] == needle for i in range(len(values) - len(needle) + 1))


# Author: Mus mbayramo@stanford.edu
