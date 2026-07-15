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
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

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
                 objective: str = LEGACY_OBJECTIVE):
        self._corpus_dir = os.path.abspath(os.path.expanduser(corpus_dir))
        self._default_tokenize = default_tokenize
        self._max_len = max_len
        self._tokenizer = tokenizer
        self.objective = objective
        if objective not in CORPUS_OBJECTIVES:
            raise ValueError(
                f"unknown corpus objective {objective!r}; choose from {CORPUS_OBJECTIVES}")
        self.metric_namespace = PHASE1_FINETUNE if objective == PHASE1_PRETRAIN_OBJECTIVE else ""

        examples_path = os.path.join(self._corpus_dir, "examples.jsonl")
        if not os.path.isfile(examples_path):
            raise FileNotFoundError(f"no examples.jsonl under {self._corpus_dir}")

        manifest_path = os.path.join(self._corpus_dir, "manifest.json")
        self.manifest: Optional[Dict] = (
            read_manifest(manifest_path) if os.path.isfile(manifest_path) else None)

        self._data: List[Dict[str, torch.Tensor]] = []
        tok = self.tokenizer
        for example in iter_examples(examples_path):
            if objective == PHASE1_PRETRAIN_OBJECTIVE:
                self._data.append(self._phase1_item(tok, example))
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

    def _phase1_item(self, tok: Any, example: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Render Phase 1 as prompt context plus JSON completion labels."""
        rest_api, allowed_methods, input_json, target_json = self._phase1_fields(example)
        allowed = ", ".join(allowed_methods) if allowed_methods else "UNKNOWN"
        prompt = (
            "### REST API\n"
            f"{rest_api}\n\n"
            "### Allowed Methods\n"
            f"{allowed}\n\n"
            "### Redfish JSON Input\n"
            f"{self._json_dumps(input_json)}\n\n"
            "### Complete Redfish JSON\n"
        )
        completion = f"{self._json_dumps(target_json)}\n"
        return self._tokenize_prompt_completion(tok, prompt, completion)

    def _tokenize_prompt_completion(
            self, tok: Any, prompt: str, completion: str) -> Dict[str, torch.Tensor]:
        """Tokenize prompt/completion and mask loss over prompt + padding."""
        prompt_ids = self._token_ids(tok, prompt)
        completion_ids = self._token_ids(tok, completion)
        if completion_ids.numel() == 0:
            raise ValueError("phase1_pretrain completion tokenized to zero tokens")
        max_len = int(self._max_len or (prompt_ids.numel() + completion_ids.numel()))

        if max_len < 2:
            raise ValueError("phase1_pretrain requires max_len >= 2")

        # Keep at least one prompt token when there is prompt context, and at least
        # one completion token. All-ignored rows can produce NaN CausalLM loss, and
        # prompt-free truncation would turn the objective into unconditional JSON LM.
        max_completion = max_len - 1 if prompt_ids.numel() > 0 else max_len
        completion_budget = min(max(1, completion_ids.numel()), max_completion)
        prompt_budget = max_len - completion_budget
        # Preserve the prompt tail because it carries the "complete JSON" marker
        # immediately before the completion. Keeping the head would leave the model
        # with context but no clear generation boundary on long Redfish resources.
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else prompt_ids[:0]
        completion_ids = completion_ids[:completion_budget]

        input_ids = torch.cat((prompt_ids, completion_ids)).long()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        # Labels stay unshifted: Hugging Face CausalLM shifts logits/labels
        # internally, so completion-token positions carry their own ids here.
        labels[prompt_ids.numel():] = completion_ids

        if input_ids.numel() < max_len:
            pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
            pad_len = max_len - input_ids.numel()
            input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad_len, dtype=torch.long)))
            labels = torch.cat((labels, torch.full((pad_len,), -100, dtype=torch.long)))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def _token_ids(tok: Any, text: str) -> torch.Tensor:
        """Return unpadded token ids for ``text``."""
        try:
            out = tok(text, padding=False, truncation=False, return_tensors="pt",
                      add_special_tokens=False)
        except TypeError:
            encode = getattr(tok, "encode", None)
            if callable(encode):
                ids = encode(text, add_special_tokens=False)
                ids_tensor = torch.as_tensor(ids, dtype=torch.long)
                return ids_tensor.squeeze(0) if ids_tensor.dim() > 1 else ids_tensor
            out = tok(text, padding=False, truncation=False, return_tensors="pt")
        return out["input_ids"].squeeze(0).long()

    @staticmethod
    def _phase1_fields(
            example: Mapping[str, Any]) -> tuple[str, List[str], Dict[str, Any], Dict[str, Any]]:
        """Extract Phase 1 ``x``/``y_true`` from explicit or normalized corpus rows."""
        x = example.get("x")
        y_true = example.get("y_true")
        if isinstance(x, Mapping) and isinstance(y_true, Mapping):
            input_json = x.get("json", {})
            target_json = y_true.get("json", input_json)
            rest_api = str(x.get("rest_api") or CorpusJSONLDataset._odata_id(input_json))
            methods = CorpusJSONLDataset._methods(x.get("allowed_methods", []))
            return rest_api, methods, dict(input_json), dict(target_json)

        action = example.get("request_or_action", {}) or {}
        response = example.get("response", {}) or {}
        rest_api = str(action.get("url") or CorpusJSONLDataset._odata_id(response))
        methods = CorpusJSONLDataset._methods(example.get("allowed_methods", []))
        return rest_api, methods, dict(response), dict(response)

    @staticmethod
    def _methods(value: Any) -> List[str]:
        """Normalize an allowed-method field to uppercase strings."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            return []
        return [str(method).upper() for method in value]

    @staticmethod
    def _odata_id(body: Any) -> str:
        """Best-effort ``@odata.id`` extraction from a JSON body."""
        return str(body.get("@odata.id", "")) if isinstance(body, Mapping) else ""

    @staticmethod
    def _json_dumps(value: Any) -> str:
        """Stable pretty JSON rendering used by Phase 1/2/3 renderers."""
        return json.dumps(value or {}, indent=2, sort_keys=True)

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

    def mask_api_prefix(self, *args, **kwargs) -> None:
        """No-op masking hook (see :meth:`enable_masking`)."""

    # --- torch Dataset -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of tokenized examples."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """The fixed-length ``input_ids``/``attention_mask`` item at ``idx``."""
        return self._data[idx]


# Author: Mus mbayramo@stanford.edu
