"""
Offline tests for the corpus tokenizer bridge (CorpusJSONLDataset).

Pins that a corpus written by write_corpus loads into fixed-length
input_ids/attention_mask items the trainer's collate stacks, that the trainer-facing
duck-type surface (tokenizer property, load_tokenizer, the masking no-ops) is present,
that run_manifest_fields round-trips the mixer's data_manifest/eval_split ids, and that
a missing corpus raises. Uses a fake tokenizer — no downloads, no network.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

import pytest
import torch

from igc.ds.corpus_dataset import CorpusJSONLDataset
from igc.ds.phase1_render import render_phase1_completion, render_phase1_prompt
from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.corpus_io import write_corpus
from igc.ds.sources.mixer import SourceMix
from igc.ds.sources.training_object import normalize


class _FakeTokenizer:
    """Minimal HF-like tokenizer: char-code ids, pads/truncates to max_length."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=None):
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
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class _EmptyCompletionTokenizer(_FakeTokenizer):
    """Tokenizer stub that exposes a degenerate zero-token completion."""

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=None):
        if text == '{\n  "empty": true\n}\n':
            empty = torch.empty((1, 0), dtype=torch.long)
            return {"input_ids": empty, "attention_mask": empty}
        return super().__call__(
            text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
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


def _explicit_phase1_corpus_dir(tmp_path: Path, body: dict, target: dict | None = None) -> str:
    """Write an explicit x/y_true Phase 1 row without invoking the corpus mixer."""
    out = tmp_path / "explicit_corpus"
    out.mkdir()
    row = {
        "x": {
            "rest_api": "/redfish/v1/Systems/1",
            "allowed_methods": ["GET", "PATCH"],
            "json": body,
        },
        "y_true": {"json": target or body},
    }
    (out / "examples.jsonl").write_text(json.dumps(row) + "\n")
    return str(out)


def _first_example(corpus_dir: str) -> dict:
    """Return the first JSONL row from a tiny test corpus."""

    examples = Path(corpus_dir) / "examples.jsonl"
    return json.loads(examples.read_text(encoding="utf-8").splitlines()[0])


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
        _corpus_dir(tmp_path),
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


def test_phase1_tiny_sequence_keeps_prompt_and_completion(tmp_path: Path):
    """Even when truncated hard, Phase 1 keeps context before active completion labels."""
    ds = CorpusJSONLDataset(
        _corpus_dir(tmp_path),
        max_len=8,
        tokenizer=_FakeTokenizer(),
        objective="phase1_pretrain",
    )

    item = ds[0]
    active = item["labels"].ne(-100).nonzero(as_tuple=False).flatten()

    assert active.numel() > 0
    assert active[0].item() > 0


def test_phase1_rejects_max_len_too_small(tmp_path: Path):
    """A one-token Phase 1 sequence cannot hold prompt context and completion."""
    with pytest.raises(ValueError, match="max_len >= 2"):
        CorpusJSONLDataset(
            _corpus_dir(tmp_path),
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


def test_phase1_long_prompt_keeps_completion_marker(tmp_path: Path):
    """Long resources are left-truncated so the prompt tail still names the task."""
    tok = _FakeTokenizer()
    ds = CorpusJSONLDataset(
        _explicit_phase1_corpus_dir(
            tmp_path,
            {
                "@odata.id": "/redfish/v1/Systems/1",
                "Description": "A" * 500,
            },
            target={"@odata.id": "/redfish/v1/Systems/1", "Id": "1"},
        ),
        max_len=96,
        tokenizer=tok,
        objective="phase1_pretrain",
    )

    item = ds[0]
    active = item["labels"].ne(-100).nonzero(as_tuple=False).flatten()
    prompt_ids = item["input_ids"][:active[0].item()].tolist()
    marker_ids = tok("### Complete Redfish JSON\n", return_tensors="pt",
                     add_special_tokens=False)["input_ids"].squeeze(0).tolist()

    assert _contains_subsequence(prompt_ids, marker_ids)


def test_phase1_renderer_matches_existing_token_stream(tmp_path: Path):
    """The shared renderer preserves the original prompt/completion tokens."""

    body = {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"}
    target = {"@odata.id": "/redfish/v1/Systems/1", "Id": "1", "PowerState": "On"}
    corpus_dir = _explicit_phase1_corpus_dir(tmp_path, body, target=target)
    tok = _FakeTokenizer()
    ds = CorpusJSONLDataset(
        corpus_dir,
        max_len=256,
        tokenizer=tok,
        objective="phase1_pretrain",
    )
    example = _first_example(corpus_dir)

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


def test_phase1_renderer_supports_normalized_legacy_rows(tmp_path: Path):
    """Normalized corpus rows still render with URL, methods, and response target."""

    corpus_dir = _corpus_dir(tmp_path, n=4)
    example = _first_example(corpus_dir)

    prompt, target_json = render_phase1_prompt(example)
    completion = render_phase1_completion(target_json)

    assert prompt.startswith("### REST API\n/redfish/v1/S/")
    assert "\n### Allowed Methods\nGET\n\n" in prompt
    assert prompt.endswith("### Complete Redfish JSON\n")
    assert target_json == example["response"]
    assert completion.endswith("}\n")


def test_items_stack_like_the_trainer_collate(tmp_path: Path):
    """torch.stack over items works — the exact contract of custom_collate_fn."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=32, tokenizer=_FakeTokenizer())
    batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ("input_ids", "attention_mask")}
    assert batch["input_ids"].shape == (len(ds), 32)
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["input_ids"].dtype == torch.int64


def test_trainer_duck_type_surface(tmp_path: Path):
    """The masking hooks and tokenizer surface the trainer touches all exist."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=16, tokenizer=_FakeTokenizer())
    for hook in ("disable_masking", "enable_masking", "mask_section", "mask_new_tokens",
                 "mask_targets", "mask_allowed_value", "mask_odata_id", "mask_targets_key",
                 "mask_objects", "mask_arrays", "mask_api_prefix"):
        getattr(ds, hook)()  # must not raise
    assert ds.tokenizer is not None
    ds.load_tokenizer()  # idempotent


def test_run_manifest_fields_round_trip(tmp_path: Path):
    """data_manifest/eval_split ids match what the mixer computed for this corpus."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=16, tokenizer=_FakeTokenizer())
    fields = ds.run_manifest_fields()
    assert set(fields) == {"data_manifest", "eval_split"}
    assert len(fields["data_manifest"]) == 16  # blake2b hex from DataManifest.content_hash
    assert fields["eval_split"] == "floor=REAL:frac=0.25:seed=0"


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
