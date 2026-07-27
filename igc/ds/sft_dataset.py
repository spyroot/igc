"""Shared prompt-completion dataset primitives for all IGC SFT phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol, Sequence

import torch
from torch.utils.data import Dataset


class SFTExampleRenderer(Protocol):
    """Convert one raw row into the exact SFT prompt and completion."""

    def __call__(self, row: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(prompt, completion)`` for one validated row."""


def token_ids(tokenizer: Any, text: str) -> torch.Tensor:
    """Return unpadded token IDs shaped ``[tokens]`` for ``text``."""
    try:
        encoded = tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except TypeError:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = torch.as_tensor(
                encode(text, add_special_tokens=False),
                dtype=torch.long,
            )
            return ids.squeeze(0) if ids.dim() > 1 else ids
        encoded = tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
    return encoded["input_ids"].squeeze(0).long()


def tokenize_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_len: int,
    *,
    completion_label_spans: Sequence[tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize one SFT example and mask prompt/padding loss positions.

    The returned tensors all have shape ``[max_len]``. ``labels`` is aligned
    with ``input_ids`` because Hugging Face causal LMs shift labels internally;
    prompt and padding positions are ``-100``. When
    ``completion_label_spans`` is supplied, only completion tokens whose
    character offsets overlap those spans receive labels. Overflow raises
    instead of silently truncating a full-document target.
    """
    if max_len < 2:
        raise ValueError("SFT examples require max_len >= 2")

    prompt_ids = token_ids(tokenizer, prompt)
    completion_ids = token_ids(tokenizer, completion)
    if completion_ids.numel() == 0:
        raise ValueError("SFT completion tokenized to zero tokens")

    total_tokens = int(prompt_ids.numel() + completion_ids.numel())
    if total_tokens > max_len:
        raise ValueError(
            "SFT example exceeds max_len without a truncation-safe contract: "
            f"prompt_tokens={prompt_ids.numel()} "
            f"completion_tokens={completion_ids.numel()} "
            f"max_len={max_len}"
        )

    input_ids = torch.cat((prompt_ids, completion_ids)).long()
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    completion_labels = _completion_labels(
        tokenizer,
        completion,
        completion_ids,
        completion_label_spans,
    )
    labels[prompt_ids.numel():] = completion_labels

    if input_ids.numel() < max_len:
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        pad_len = max_len - input_ids.numel()
        input_ids = torch.cat(
            (input_ids, torch.full((pad_len,), pad_id, dtype=torch.long))
        )
        attention_mask = torch.cat(
            (attention_mask, torch.zeros(pad_len, dtype=torch.long))
        )
        labels = torch.cat(
            (labels, torch.full((pad_len,), -100, dtype=torch.long))
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _completion_labels(
    tokenizer: Any,
    completion: str,
    completion_ids: torch.Tensor,
    spans: Sequence[tuple[int, int]] | None,
) -> torch.Tensor:
    """Return full or character-span-selective completion labels."""

    if spans is None:
        return completion_ids.clone()
    normalized = tuple(_validate_label_span(span, len(completion)) for span in spans)
    if not normalized:
        raise ValueError("completion_label_spans must not be empty")
    try:
        encoded = tokenizer(
            completion,
            padding=False,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError) as exc:
        raise TypeError(
            "selective completion labels require a fast tokenizer with "
            "return_offsets_mapping support"
        ) from exc
    offsets = encoded.get("offset_mapping")
    encoded_ids = encoded.get("input_ids")
    if offsets is None or encoded_ids is None:
        raise TypeError(
            "selective completion labels require input_ids and offset_mapping"
        )
    encoded_ids = torch.as_tensor(encoded_ids).squeeze(0).long()
    offsets = torch.as_tensor(offsets).squeeze(0).long()
    if encoded_ids.shape != completion_ids.shape or not torch.equal(
        encoded_ids,
        completion_ids,
    ):
        raise ValueError("offset tokenization does not match completion tokenization")
    if offsets.ndim != 2 or offsets.shape != (completion_ids.numel(), 2):
        raise ValueError("completion offset mapping must have shape [tokens, 2]")

    labels = torch.full_like(completion_ids, -100)
    for token_index, (token_start, token_end) in enumerate(offsets.tolist()):
        if token_start == token_end:
            continue
        if any(token_start < span_end and token_end > span_start
               for span_start, span_end in normalized):
            labels[token_index] = completion_ids[token_index]
    if not torch.any(labels != -100):
        raise ValueError("completion label spans selected zero tokens")
    return labels


def _validate_label_span(
    span: tuple[int, int],
    completion_length: int,
) -> tuple[int, int]:
    """Validate one half-open completion character span."""

    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
    ):
        raise TypeError("completion label spans must be (int, int) tuples")
    start, end = span
    if not 0 <= start < end <= completion_length:
        raise ValueError(
            "completion label span must be within the completion: "
            f"span={span} completion_length={completion_length}"
        )
    return start, end


class PromptCompletionJSONLDataset(Dataset):
    """Tokenized JSONL dataset driven by one phase-specific renderer.

    :param path: JSON Lines file containing raw phase rows.
    :param renderer: validates a row and returns its prompt/completion strings.
    :param metric_namespace: phase metric prefix used by :class:`SFTTrainer`.
    :param max_len: fixed token length; output tensors are ``[max_len]``.
    :param tokenizer: initialized tokenizer, or ``None`` to load ``tokenizer_name``.
    :param tokenizer_name: Hugging Face tokenizer id/path used when needed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        renderer: SFTExampleRenderer,
        metric_namespace: str,
        max_len: int,
        tokenizer: Any | None = None,
        tokenizer_name: str | None = None,
        manifest_path: str | Path | None = None,
        require_manifest: bool = False,
        expected_dataset: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"SFT JSONL file does not exist: {self.path}")
        self.renderer = renderer
        self.metric_namespace = metric_namespace
        self._max_len = int(max_len)
        self._tokenizer = tokenizer
        self._tokenizer_name = tokenizer_name
        self._offsets: list[tuple[int, int]] = []
        self._handle: BinaryIO | None = None
        self._manifest: dict[str, Any] = {}
        self._manifest_sha256 = ""
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if line.strip():
                    self._offsets.append((offset, line_number))
        self._data_sha256 = digest.hexdigest()
        self._eval_split_sha256 = ""
        self._eval_manifest_sha256 = ""

        if not self._offsets:
            raise ValueError(f"SFT JSONL file has no examples: {self.path}")
        if manifest_path is not None or require_manifest:
            resolved_manifest = (
                Path(manifest_path).expanduser().resolve()
                if manifest_path is not None
                else Path(f"{self.path}.manifest.json")
            )
            self._load_release_manifest(
                resolved_manifest,
                expected_dataset=expected_dataset,
            )

    @property
    def tokenizer(self) -> Any:
        """Return the configured tokenizer, loading it once on demand."""
        if self._tokenizer is None:
            self.load_tokenizer()
        return self._tokenizer

    def load_tokenizer(self) -> None:
        """Load ``tokenizer_name`` and configure EOS as padding when needed."""
        if self._tokenizer is None:
            if not self._tokenizer_name:
                raise ValueError("tokenizer_name is required when tokenizer is not supplied")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def run_manifest_fields(self) -> dict[str, str]:
        """Return exact dataset, release-manifest, and held-out identities."""
        return {
            "data_manifest": self.manifest_sha256 or self.data_sha256,
            "eval_split": self._eval_split_sha256,
            "train_data_sha": self.data_sha256,
            "eval_data_sha": self._eval_split_sha256,
            "source_manifest_sha": self.source_manifest_sha256,
            "eval_manifest_sha": self._eval_manifest_sha256,
        }

    @property
    def data_sha256(self) -> str:
        """Return the canonical digest of the exact JSONL bytes."""
        return f"sha256:{self._data_sha256}"

    @property
    def manifest_sha256(self) -> str:
        """Return the digest of the verified release manifest, when required."""
        return self._manifest_sha256

    @property
    def source_manifest_sha256(self) -> str:
        """Return the complete source release SHA named by a split manifest."""
        value = self._manifest.get("source_full_manifest_sha256")
        if value is None:
            return self.manifest_sha256
        _validate_sha256(value, label="source full manifest identity")
        return str(value)

    def bind_eval_dataset(self, dataset: "PromptCompletionJSONLDataset") -> None:
        """Bind exact held-out data and manifest identities into run lineage."""
        if not isinstance(dataset, PromptCompletionJSONLDataset):
            raise TypeError("eval dataset must be PromptCompletionJSONLDataset")
        if not dataset.manifest_sha256:
            raise ValueError("eval dataset requires a verified release manifest")
        self.set_eval_split_sha256(dataset.data_sha256)
        self.set_eval_manifest_sha256(dataset.manifest_sha256)

    def set_eval_split_sha256(self, value: str) -> None:
        """Bind the immutable held-out JSONL digest into run lineage."""
        _validate_sha256(value, label="eval split identity")
        self._eval_split_sha256 = value

    def set_eval_manifest_sha256(self, value: str) -> None:
        """Bind the exact held-out release-manifest identity into run lineage."""
        _validate_sha256(value, label="eval manifest identity")
        self._eval_manifest_sha256 = value

    def _load_release_manifest(
        self,
        path: Path,
        *,
        expected_dataset: str | None,
    ) -> None:
        """Verify a release manifest against the exact JSONL bytes and rows."""
        if not path.is_file():
            raise FileNotFoundError(f"SFT release manifest does not exist: {path}")
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid SFT release manifest {path}: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"SFT release manifest must be an object: {path}")
        if manifest.get("immutable") is not True or manifest.get("complete") is not True:
            raise ValueError("SFT release manifest must declare immutable=true and complete=true")
        if manifest.get("artifact_sha256") != self.data_sha256:
            raise ValueError("SFT release manifest artifact SHA does not match JSONL bytes")
        rows = manifest.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows != len(self._offsets):
            raise ValueError("SFT release manifest row count does not match JSONL rows")
        if expected_dataset is not None and manifest.get("dataset") != expected_dataset:
            raise ValueError(
                f"SFT release manifest dataset must be {expected_dataset!r}"
            )
        self._manifest = dict(manifest)
        self._manifest_sha256 = f"sha256:{hashlib.sha256(raw).hexdigest()}"

    def __len__(self) -> int:
        """Return the number of tokenized examples."""
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one ``input_ids``/``attention_mask``/``labels`` example."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset, line_number = self._offsets[index]
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        self._handle.seek(offset)
        line = self._handle.readline()
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid JSON at {self.path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"row at {self.path}:{line_number} must be an object")
        prompt, completion = self.renderer(row)
        return tokenize_prompt_completion(
            self.tokenizer,
            prompt,
            completion,
            self._max_len,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Drop the per-worker file handle when pickling the dataset."""
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        """Close the lazy JSONL handle owned by this worker."""
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()


def _validate_sha256(value: Any, *, label: str) -> None:
    digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest.lower())
    ):
        raise ValueError(f"{label} must be a sha256 digest")
