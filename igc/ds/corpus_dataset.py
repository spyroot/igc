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

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from igc.ds.sources.corpus_io import iter_examples, read_manifest
from igc.ds.sources.mixer import DataManifest
from igc.ds.state_graph import build_state_record, candidate_feature_tensor_payload


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
                 max_candidates: int = 16):
        self._corpus_dir = os.path.abspath(os.path.expanduser(corpus_dir))
        self._default_tokenize = default_tokenize
        self._max_len = max_len
        self._max_candidates = max_candidates
        self._tokenizer = tokenizer

        examples_path = os.path.join(self._corpus_dir, "examples.jsonl")
        if not os.path.isfile(examples_path):
            raise FileNotFoundError(f"no examples.jsonl under {self._corpus_dir}")

        manifest_path = os.path.join(self._corpus_dir, "manifest.json")
        self.manifest: Optional[Dict] = (
            read_manifest(manifest_path) if os.path.isfile(manifest_path) else None)

        self._data: List[Dict[str, torch.Tensor]] = []
        self._state_records: List[Dict[str, Any]] = []
        tok = self.tokenizer
        for example in iter_examples(examples_path):
            state = build_state_record(example)
            state_dict = state.to_dict()
            text = state.resource_text
            out = tok(text, padding="max_length", max_length=self._max_len,
                      truncation=True, return_tensors="pt")
            graph = state.resource_graph
            affordances = state.action_affordances
            candidates = affordances["templates"]
            candidate_tensors = _candidate_tensors(
                candidates, max_candidates=self._max_candidates)
            scope_tensors = _scope_tensors(state_dict, max_candidates=self._max_candidates)
            self._data.append({
                "input_ids": out["input_ids"].squeeze(0),
                "attention_mask": out["attention_mask"].squeeze(0),
                "graph_node_count": torch.tensor(graph["node_count"], dtype=torch.long),
                "graph_edge_count": torch.tensor(graph["edge_count"], dtype=torch.long),
                "action_candidate_count": torch.tensor(
                    affordances["candidate_count"], dtype=torch.long),
                **candidate_tensors,
                **scope_tensors,
                "state_fingerprint": state.state_fingerprint,
                "state_id": state.state_id,
            })
            self._state_records.append(state_dict)

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

    def state_record(self, idx: int) -> Dict[str, Any]:
        """Structured state record backing item ``idx`` for debugging/eval."""
        return self._state_records[idx]


def render_training_example(example: Dict[str, Any]) -> str:
    """Render one ``TrainingExample`` dict into the M1/M2 state text contract.

    The active M1/M2 trainers consume token ids, not Python dicts. This renderer
    is therefore the deliberate bridge from the normalized data contract into the
    model path: raw response text, compact before/after state graph, legal
    methods, semantics, and provenance all become stable text that the tokenizer
    sees. M3 remains compatible because the request/response surface is still
    present, while planned M6 can recover legal-action and semantics signals from
    the same corpus.
    """
    return build_state_record(example).resource_text


def _candidate_tensors(candidates, max_candidates: int) -> Dict[str, torch.Tensor]:
    payload = candidate_feature_tensor_payload(list(candidates))
    count = min(len(candidates), max_candidates)
    out = {
        "candidate_mask": torch.zeros(max_candidates, dtype=torch.bool),
        "candidate_resource_type_id": torch.zeros(max_candidates, dtype=torch.long),
        "candidate_parent_type_id": torch.zeros(max_candidates, dtype=torch.long),
        "candidate_relation_name_id": torch.zeros(max_candidates, dtype=torch.long),
        "candidate_depth_bucket": torch.zeros(max_candidates, dtype=torch.long),
        "candidate_method_id": torch.zeros(max_candidates, dtype=torch.long),
        "candidate_has_action_target": torch.zeros(max_candidates, dtype=torch.float32),
        "candidate_is_collection": torch.zeros(max_candidates, dtype=torch.float32),
        "candidate_is_oem": torch.zeros(max_candidates, dtype=torch.float32),
        "candidate_path_segment_hashes": torch.zeros(max_candidates, 8, dtype=torch.long),
        "candidate_allowed_method_mask": torch.zeros(max_candidates, 6, dtype=torch.float32),
        "candidate_local_state_summary": torch.zeros(max_candidates, 4, dtype=torch.long),
    }
    if count == 0:
        return out

    out["candidate_mask"][:count] = True
    scalar_map = {
        "candidate_resource_type_id": "resource_type_id",
        "candidate_parent_type_id": "parent_type_id",
        "candidate_relation_name_id": "relation_name_id",
        "candidate_depth_bucket": "depth_bucket",
        "candidate_method_id": "method_id",
        "candidate_has_action_target": "has_action_target",
        "candidate_is_collection": "is_collection",
        "candidate_is_oem": "is_oem",
    }
    for target, source in scalar_map.items():
        dtype = out[target].dtype
        out[target][:count] = torch.tensor(payload[source][:count], dtype=dtype)
    out["candidate_path_segment_hashes"][:count] = torch.tensor(
        payload["path_segment_hashes"][:count], dtype=torch.long)
    out["candidate_allowed_method_mask"][:count] = torch.tensor(
        payload["allowed_method_mask"][:count], dtype=torch.float32)
    out["candidate_local_state_summary"][:count] = torch.tensor(
        payload["local_state_summary"][:count], dtype=torch.long)
    return out


def _scope_tensors(state: Dict[str, Any], max_candidates: int) -> Dict[str, torch.Tensor]:
    """Build the v1 observation scope tensor set: current resource node only."""
    node = state["resource_node"]
    attrs = node.get("attributes", {})
    templates = state["action_affordances"].get("templates", [])
    first = templates[0] if templates else {}
    return {
        "scope_mask": torch.tensor([True], dtype=torch.bool),
        "scope_resource_type_id": torch.tensor([attrs.get("resource_type_id", 0)], dtype=torch.long),
        "scope_parent_type_id": torch.tensor([first.get("parent_type_id", 0)], dtype=torch.long),
        "scope_relation_name_id": torch.tensor([first.get("relation_name_id", 0)], dtype=torch.long),
        "scope_depth_bucket": torch.tensor([first.get("depth_bucket", 0)], dtype=torch.long),
        "scope_method_id": torch.tensor([0], dtype=torch.long),
        "scope_has_action_target": torch.tensor([first.get("has_action_target", 0)], dtype=torch.float32),
        "scope_is_collection": torch.tensor([attrs.get("is_collection", 0)], dtype=torch.float32),
        "scope_is_oem": torch.tensor([attrs.get("is_oem", 0)], dtype=torch.float32),
        "scope_path_segment_hashes": torch.tensor([
            first.get("path_segment_hashes", [0] * 8)
        ], dtype=torch.long),
        "scope_allowed_method_mask": torch.tensor([
            first.get("allowed_method_mask", [0] * 6)
        ], dtype=torch.float32),
        "scope_local_state_summary": torch.tensor([
            first.get("local_state_summary", [0] * 4)
        ], dtype=torch.long),
        "candidate_endpoint_scope_index": torch.zeros(max_candidates, dtype=torch.long),
    }


# Author: Mus mbayramo@stanford.edu
