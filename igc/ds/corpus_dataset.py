"""
Torch dataset over a written training corpus (the tokenizer bridge).

Feeds the provenance-tagged corpus produced by the ``igc.ds.sources`` pipeline
(``write_corpus`` -> ``examples.jsonl`` + ``manifest.json``) into the shared SFT
engine. Phase 1 examples use the same completion-only label mask as Phases 2 and 3.

Used by ``IgcMain.dataset`` (``igc/modules/igc_main.py``): when the ``--corpus_dir`` CLI flag
is set, a run loads this class instead of rebuilding ``MaskedJSONDataset`` from
``~/.json_responses``, so it trains on the pre-written trust-tier corpus. The trainer also
reads ``run_manifest_fields`` to stamp the corpus hash / eval split into the run report.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset

from igc.ds.phase1_structural_loss import (
    Phase1StructuralLossResult,
    build_phase1_structural_loss_view,
    load_phase1_structural_loss_profile,
)
from igc.ds.phase1_render import render_phase1_completion, render_phase1_prompt
from igc.ds.sft_dataset import token_ids, tokenize_prompt_completion
from igc.ds.sources.corpus_io import iter_examples, read_manifest
from igc.ds.sources.mixer import DataManifest
from igc.modules.base.metric_keys import PHASE1_FINETUNE, PHASE1_OBJECTIVE_PRETRAIN


LEGACY_OBJECTIVE = "legacy"
PHASE1_PRETRAIN_OBJECTIVE = PHASE1_OBJECTIVE_PRETRAIN
CORPUS_OBJECTIVES = (LEGACY_OBJECTIVE, PHASE1_PRETRAIN_OBJECTIVE)


class CorpusJSONLDataset(Dataset):
    """Fixed-length tokenized dataset over a ``write_corpus`` output directory.

    :param corpus_dir: directory holding ``examples.jsonl`` (and ``manifest.json``).
    :param default_tokenize: HF tokenizer id/path used when ``tokenizer`` is not given.
    :param max_len: fixed sequence length every item is padded/truncated to.
    :param tokenizer: pre-built tokenizer (tests inject one; training resolves from
        ``default_tokenize`` lazily so importing this module stays offline-safe).
    :raises FileNotFoundError: when ``corpus_dir`` has no ``examples.jsonl``.
    """

    def __init__(self,
                 corpus_dir: str,
                 default_tokenize: Optional[str] = "gpt2",
                 max_len: Optional[int] = 1024,
                 tokenizer: Optional[Any] = None,
                 objective: str = LEGACY_OBJECTIVE,
                 phase1_structural_loss_profile: str = "none",
                 phase1_structural_loss_mode: str = "train",
                 phase1_structural_loss_seed: int = 42):
        self._corpus_dir = os.path.abspath(os.path.expanduser(corpus_dir))
        self._default_tokenize = default_tokenize
        self._max_len = max_len
        self._tokenizer = tokenizer
        self.objective = objective
        self._epoch = 0
        self._phase1_structural_loss_seed = int(phase1_structural_loss_seed)
        self._phase1_structural_loss_mode = phase1_structural_loss_mode
        self._phase1_structural_loss = load_phase1_structural_loss_profile(
            phase1_structural_loss_profile
        )
        if objective not in CORPUS_OBJECTIVES:
            raise ValueError(
                f"unknown corpus objective {objective!r}; choose from {CORPUS_OBJECTIVES}")
        if phase1_structural_loss_mode not in {"train", "evaluation"}:
            raise ValueError(
                "phase1_structural_loss_mode must be 'train' or 'evaluation'"
            )
        if self._phase1_structural_loss.enabled and objective != PHASE1_PRETRAIN_OBJECTIVE:
            raise ValueError(
                "Phase 1 structural loss requires objective='phase1_pretrain'"
            )
        self.metric_namespace = PHASE1_FINETUNE if objective == PHASE1_PRETRAIN_OBJECTIVE else ""

        examples_path = os.path.join(self._corpus_dir, "examples.jsonl")
        if not os.path.isfile(examples_path):
            raise FileNotFoundError(f"no examples.jsonl under {self._corpus_dir}")

        manifest_path = os.path.join(self._corpus_dir, "manifest.json")
        self.manifest: Optional[Dict] = (
            read_manifest(manifest_path) if os.path.isfile(manifest_path) else None)
        self._data_sha256 = _sha256_file(examples_path)
        self._manifest_sha256 = (
            _sha256_file(manifest_path) if os.path.isfile(manifest_path) else ""
        )
        self._eval_data_sha256 = ""

        examples = list(iter_examples(examples_path))
        self._dynamic_phase1_structural_loss = (
            objective == PHASE1_PRETRAIN_OBJECTIVE
            and self._phase1_structural_loss.enabled
            and phase1_structural_loss_mode == "train"
        )
        self._examples: List[Mapping[str, Any]] = (
            examples if self._dynamic_phase1_structural_loss else []
        )
        self._data: List[Dict[str, torch.Tensor]] = []
        tok = self.tokenizer
        if self._dynamic_phase1_structural_loss:
            return
        for row_index, example in enumerate(examples):
            if objective == PHASE1_PRETRAIN_OBJECTIVE:
                self._data.append(
                    self._phase1_item(tok, example, row_index=row_index)
                )
            else:
                self._data.append(self._legacy_item(tok, example))

    # --- tokenizer surface (mirrors JSONDataset) ---------------------------------

    @property
    def tokenizer(self):
        """The dataset tokenizer, resolved from ``default_tokenize`` on first use."""
        if self._tokenizer is None:
            self.load_tokenizer()
        return self._tokenizer

    def load_tokenizer(self) -> None:
        """Build the tokenizer from ``default_tokenize`` (pad falls back to eos)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._default_tokenize)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # --- render/tokenize objectives ------------------------------------------------

    def _legacy_item(self, tok: Any, example: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Render the historical whole-text objective for backward compatibility."""
        action = example.get("request_or_action", {}) or {}
        text = (f"{action.get('method', 'GET')} {action.get('url', '')}\n"
                f"{json.dumps(example.get('response', {}), sort_keys=True)}")
        out = tok(text, padding="max_length", max_length=self._max_len,
                  truncation=True, return_tensors="pt")
        return {
            "input_ids": out["input_ids"].squeeze(0).long(),
            "attention_mask": out["attention_mask"].squeeze(0).long(),
        }

    def _phase1_item(
        self,
        tok: Any,
        example: Mapping[str, Any],
        *,
        row_index: int,
    ) -> Dict[str, torch.Tensor]:
        """Render Phase 1 as prompt context plus JSON completion labels."""
        view = self._phase1_view(example, row_index=row_index)
        prompt, target_json = render_phase1_prompt(view.row)
        completion = render_phase1_completion(target_json)
        completion_label_spans = (
            tuple(view.completion_spans) if self._phase1_structural_loss.enabled else None
        )
        return self._tokenize_prompt_completion(
            tok,
            prompt,
            completion,
            completion_label_spans=completion_label_spans,
        )

    def _phase1_view(
        self,
        example: Mapping[str, Any],
        *,
        row_index: int,
    ) -> Phase1StructuralLossResult:
        """Return the current deterministic train or fixed held-out view."""

        return build_phase1_structural_loss_view(
            example,
            profile=self._phase1_structural_loss,
            mode=self._phase1_structural_loss_mode,
            run_seed=self._phase1_structural_loss_seed,
            epoch=self._epoch,
            row_index=row_index,
        )

    def _tokenize_prompt_completion(
        self,
        tok: Any,
        prompt: str,
        completion: str,
        *,
        completion_label_spans: tuple[tuple[int, int], ...] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompt/completion and mask loss over prompt + padding."""
        prompt_ids = token_ids(tok, prompt)
        completion_ids = token_ids(tok, completion)
        max_len = int(self._max_len or (prompt_ids.numel() + completion_ids.numel()))
        return tokenize_prompt_completion(
            tok,
            prompt,
            completion,
            max_len,
            completion_label_spans=completion_label_spans,
        )

    @staticmethod
    def _token_ids(tok: Any, text: str) -> torch.Tensor:
        """Return unpadded token ids for ``text``."""
        return token_ids(tok, text)

    # --- provenance for run reports -----------------------------------------------

    def run_manifest_fields(self) -> Dict[str, str]:
        """Return semantic and exact-byte corpus identities for the run report.

        Reconstructs the :class:`DataManifest` written next to the corpus so the run
        report carries the same content hash / split id the mixer computed.

        :return: semantic manifest identity plus exact train/eval/manifest digests.
        """
        if not self.manifest:
            return {
                "data_manifest": "",
                "eval_split": self._eval_data_sha256,
                "train_data_sha": self.data_sha256,
                "eval_data_sha": self._eval_data_sha256,
                "source_manifest_sha": "",
                "source_registry_sha": "",
                "source_artifact_manifest_shas": {},
                "phase1_structural_loss_profile": self.phase1_structural_loss_profile,
                "phase1_structural_loss_spec_sha": self.phase1_structural_loss_spec_sha,
            }
        manifest = DataManifest(**self.manifest)
        fields = manifest.to_run_manifest_fields()
        fields.update({
            "eval_split": self._eval_data_sha256,
            "train_data_sha": self.data_sha256,
            "eval_data_sha": self._eval_data_sha256,
            "source_manifest_sha": self.manifest_sha256,
            "source_registry_sha": manifest.source_registry_sha,
            "source_artifact_manifest_shas": dict(manifest.source_manifest_shas),
            "phase1_structural_loss_profile": self.phase1_structural_loss_profile,
            "phase1_structural_loss_spec_sha": self.phase1_structural_loss_spec_sha,
        })
        return fields

    @property
    def phase1_structural_loss_profile(self) -> str:
        """Resolved runtime structural-loss profile name."""

        return self._phase1_structural_loss.name

    @property
    def phase1_structural_loss_spec_sha(self) -> str:
        """Exact SHA-256 identity of the structural-loss profile registry."""

        return self._phase1_structural_loss.spec_sha256

    def set_epoch(self, epoch: int) -> None:
        """Select the reproducible per-epoch train view before iteration starts."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._epoch = epoch

    @property
    def data_sha256(self) -> str:
        """Return the SHA-256 identity of the exact ``examples.jsonl`` bytes."""
        return f"sha256:{self._data_sha256}"

    @property
    def manifest_sha256(self) -> str:
        """Return the SHA-256 identity of the exact ``manifest.json`` bytes."""
        return f"sha256:{self._manifest_sha256}" if self._manifest_sha256 else ""

    def set_eval_split_sha256(self, value: str) -> None:
        """Bind the exact held-out JSONL identity used by the trainer."""
        digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
        if (
            not isinstance(value, str)
            or not value.startswith("sha256:")
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            raise ValueError("eval split identity must be a sha256 digest")
        self._eval_data_sha256 = value

    # --- torch Dataset -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of tokenized examples."""
        return (
            len(self._examples)
            if self._dynamic_phase1_structural_loss
            else len(self._data)
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """The fixed-length ``input_ids``/``attention_mask`` item at ``idx``."""
        if self._dynamic_phase1_structural_loss:
            return self._phase1_item(
                self.tokenizer,
                self._examples[idx],
                row_index=idx,
            )
        return self._data[idx]


def _sha256_file(path: str) -> str:
    """Return the lowercase SHA-256 hex digest for one local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Author: Mus mbayramo@stanford.edu
