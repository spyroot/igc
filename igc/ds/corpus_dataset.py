"""
Torch dataset over a written training corpus (the tokenizer bridge).

Feeds the provenance-tagged corpus produced by the ``igc.ds.sources`` pipeline
(``write_corpus`` -> ``examples.jsonl`` + ``manifest.json``) into the existing M1
state-encoder training loop. Each example is rendered as its request line plus the
response JSON, tokenized once at construction to the same fixed-length
``input_ids``/``attention_mask`` items the trainer's collate function stacks — so the
trainer consumes a trust-tier-split corpus without any loop changes.

The class duck-types the surface the trainer touches on ``MaskedJSONDataset``:
``tokenizer``/``load_tokenizer`` plus the masking-method hooks, which are no-ops here
because a written corpus trains the plain causal-LM (NO_MASK) objective.

Used by ``IgcMain.dataset`` (``igc/modules/igc_main.py``): when the ``--corpus_dir`` CLI flag
is set, a run loads this class instead of rebuilding ``MaskedJSONDataset`` from
``~/.json_responses``, so it trains on the pre-written trust-tier corpus. The trainer also
reads ``run_manifest_fields`` to stamp the corpus hash / eval split into the run report.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from igc.ds.sources.corpus_io import iter_examples, read_manifest
from igc.ds.sources.mixer import DataManifest


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
                 tokenizer: Optional[Any] = None):
        self._corpus_dir = os.path.abspath(os.path.expanduser(corpus_dir))
        self._default_tokenize = default_tokenize
        self._max_len = max_len
        self._tokenizer = tokenizer

        examples_path = os.path.join(self._corpus_dir, "examples.jsonl")
        if not os.path.isfile(examples_path):
            raise FileNotFoundError(f"no examples.jsonl under {self._corpus_dir}")

        manifest_path = os.path.join(self._corpus_dir, "manifest.json")
        self.manifest: Optional[Dict] = (
            read_manifest(manifest_path) if os.path.isfile(manifest_path) else None)

        self._data: List[Dict[str, torch.Tensor]] = []
        tok = self.tokenizer
        for example in iter_examples(examples_path):
            action = example.get("request_or_action", {}) or {}
            text = (f"{action.get('method', 'GET')} {action.get('url', '')}\n"
                    f"{json.dumps(example.get('response', {}), sort_keys=True)}")
            out = tok(text, padding="max_length", max_length=self._max_len,
                      truncation=True, return_tensors="pt")
            self._data.append({
                "input_ids": out["input_ids"].squeeze(0),
                "attention_mask": out["attention_mask"].squeeze(0),
            })

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

    # --- provenance for run reports -----------------------------------------------

    def run_manifest_fields(self) -> Dict[str, str]:
        """``{data_manifest, eval_split}`` for the run report, or empty strings.

        Reconstructs the :class:`DataManifest` written next to the corpus so the run
        report carries the same content hash / split id the mixer computed.

        :return: dict with ``data_manifest`` and ``eval_split`` keys.
        """
        if not self.manifest:
            return {"data_manifest": "", "eval_split": ""}
        manifest = DataManifest(**self.manifest)
        return manifest.to_run_manifest_fields()

    # --- masking surface the trainer may touch (NO_MASK corpus: all no-ops) --------

    def disable_masking(self) -> None:
        """No-op: a written corpus trains the plain causal-LM objective."""

    def enable_masking(self, *args, **kwargs) -> None:
        """No-op: span masking is not defined for the packed corpus."""

    def mask_section(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_new_tokens(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_targets(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_allowed_value(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_odata_id(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_targets_key(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_objects(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    def mask_arrays(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    # --- torch Dataset -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of tokenized examples."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """The fixed-length ``input_ids``/``attention_mask`` item at ``idx``."""
        return self._data[idx]


# Author: Mus mbayramo@stanford.edu
