"""Pin the dataset-to-model contract for Phase 1/2/3 across the real builders.

PR #108 slipped a parallel contract past green per-module tests; these tests lock
the *data shape* the three staged trainers actually consume, using the real
canonical code paths (no mocks of the builders themselves):

* Phase 1 D0: :class:`igc.ds.corpus_dataset.CorpusJSONLDataset` with the
  ``phase1_pretrain`` objective — the example keys it reads, the prompt/target
  boundary (loss masked over prompt + padding), and the fixed-length tokenized
  tensor keys/shapes.
* Phase 2 D1: :func:`igc.ds.rest_goal_contract.build_d1_rest_api_list_row` — the
  ``x``/``y_true`` field names and the empty-set (hard-negative) allowance.
* Phase 3: :func:`igc.ds.rest_goal_contract.build_call_row` — the ``x``
  fields and the per-call ``y_true`` field names.

The tests run offline on CI / an approved remote container: a tiny char-level
in-test tokenizer stands in for the real HF tokenizer, so nothing is downloaded
and no GPU is required. Real-model generation/decoding stays out of scope here
(GB300 acceptance gate), so there is deliberately no live-model stage in this
file.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os
from typing import Any, List

import torch

from igc.ds.corpus_dataset import CorpusJSONLDataset, PHASE1_PRETRAIN_OBJECTIVE
from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
    build_call_row,
)


class _TinyCharTokenizer:
    """Offline char-level stand-in for an HF tokenizer (no download, no GPU).

    ``CorpusJSONLDataset._token_ids`` first tries ``tok(text, ...)`` and falls back
    to ``tok.encode`` on ``TypeError``; this tokenizer raises ``TypeError`` from
    ``__call__`` so the ``encode`` path is exercised, and reserves id ``0`` for
    padding so completion ids never collide with the pad id.
    """

    pad_token_id = 0  # reserved pad id; encode() never emits 0 so padding is detectable.

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Force ``CorpusJSONLDataset._token_ids`` onto its ``encode`` fallback."""
        raise TypeError("tiny tokenizer only supports encode()")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Return one nonzero id per character (0 stays reserved for padding)."""
        return [(ord(char) % 255) + 1 for char in text]


def _write_corpus(tmp_path, examples: List[dict]) -> str:
    """Write ``examples.jsonl`` under a corpus dir and return the dir path."""
    corpus_dir = os.path.join(str(tmp_path), "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    with open(os.path.join(corpus_dir, "examples.jsonl"), "w") as handle:
        for example in examples:
            handle.write(json.dumps(example))
            handle.write("\n")
    return corpus_dir


# --- Phase 1 D0: CorpusJSONLDataset phase1_pretrain render/tokenize -----------------


def test_phase1_fields_consume_x_and_y_true_keys() -> None:
    """Phase 1 D0 reads x.rest_api / x.allowed_methods / x.json and y_true.json."""
    example = {
        "x": {
            "rest_api": "/redfish/v1/Systems/1",
            "allowed_methods": ["get", "patch"],
            "json": {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"},
        },
        "y_true": {
            "json": {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "Off"},
        },
    }
    rest_api, methods, input_json, target_json = CorpusJSONLDataset._phase1_fields(example)
    assert rest_api == "/redfish/v1/Systems/1"
    assert methods == ["GET", "PATCH"]  # allowed_methods normalized to upper-case.
    assert input_json == example["x"]["json"]
    assert target_json == example["y_true"]["json"]  # y_true.json is the completion target.


def test_phase1_pretrain_item_keys_shapes_and_prompt_masking(tmp_path) -> None:
    """phase1_pretrain emits fixed-length input_ids/attention_mask/labels with loss masked over prompt+padding."""
    max_len = 128
    example = {
        "x": {
            "rest_api": "/redfish/v1/Chassis/1",
            "allowed_methods": ["GET"],
            "json": {"@odata.id": "/redfish/v1/Chassis/1", "Name": "Chassis"},
        },
        "y_true": {
            "json": {"@odata.id": "/redfish/v1/Chassis/1", "Name": "Chassis"},
        },
    }
    corpus_dir = _write_corpus(tmp_path, [example])
    dataset = CorpusJSONLDataset(
        corpus_dir,
        max_len=max_len,
        tokenizer=_TinyCharTokenizer(),
        objective=PHASE1_PRETRAIN_OBJECTIVE,
    )
    assert len(dataset) == 1
    item = dataset[0]

    # Tokenized item tensor keys the trainer's collate stacks.
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    for key in ("input_ids", "attention_mask", "labels"):
        assert item[key].shape == (max_len,), key  # fixed-length padded tensors.
        assert item[key].dtype == torch.long, key

    input_ids = item["input_ids"]
    attention_mask = item["attention_mask"]
    labels = item["labels"]

    supervised = labels != -100
    # Some tokens are supervised (the completion) and some are masked (the prompt).
    assert supervised.any()
    assert (~supervised).any()
    assert int(supervised.sum()) < max_len  # prompt + padding are masked, never the whole row.
    assert labels[0].item() == -100  # the prompt prefix is masked out of the loss.

    # Padding is masked in both attention and loss; supervised tokens are real (attended).
    pad = attention_mask == 0
    assert torch.all(labels[pad] == -100)
    assert torch.all(input_ids[pad] == _TinyCharTokenizer.pad_token_id)
    assert torch.all(attention_mask[supervised] == 1)

    # Supervised labels carry the completion token ids (unshifted; HF shifts internally).
    assert torch.all(labels[supervised] == input_ids[supervised])


# --- Phase 2 D1: build_d1_rest_api_list_row ----------------------------------------


def test_phase2_labelled_row_field_names() -> None:
    """Phase 2 D1 row is TEXT-ONLY x with the unordered unique rest_api_list label."""
    context = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET", "PATCH"],
        json={"@odata.id": "/redfish/v1/Systems/1"},
    )
    row = build_d1_rest_api_list_row(
        text="power off system 1",
        contexts=[context],
        rest_api_list=["/redfish/v1/Systems/1"],
    )
    x = row["x"]
    assert x == {"text": "power off system 1"}  # json/allowed_methods here = input leakage.
    assert row["target_semantics"] == "unordered_unique_set"
    assert row["y_true"] == {"rest_api_list": ["/redfish/v1/Systems/1"]}  # the API-set label.
    assert set(row["validation"]) == {
        "text_source",
        "review_judged",
        "natural",
        "exact_api_coverage",
        "extra_intent",
        "duplicate_intent",
        "ambiguous",
        "nonsense",
        "method_semantics_valid",
    }


def test_phase2_empty_rest_api_list_is_allowed_hard_negative() -> None:
    """An empty y_true.rest_api_list ([] == no-action) is a legal hard-negative row."""
    context = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET"],
        json={"@odata.id": "/redfish/v1/Systems/1"},
    )
    row = build_d1_rest_api_list_row(
        text="what is the weather today",
        contexts=[context],
        rest_api_list=[],
    )
    assert row["y_true"]["rest_api_list"] == []  # empty set is retained, not rejected.


# --- Phase 3: build_call_row --------------------------------------------------------


def test_phase3_call_row_field_names() -> None:
    """Phase 3 row carries x.text/x.rest_api_list/x.json/x.allowed_methods and per-call y_true keys."""
    read_ctx = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET"],
        json={"@odata.id": "/redfish/v1/Systems/1"},
    )
    write_ctx = RedfishContext(
        rest_api="/redfish/v1/Managers/1/EthernetInterfaces/1",
        allowed_methods=["GET", "PATCH"],
        json={"@odata.id": "/redfish/v1/Managers/1/EthernetInterfaces/1"},
    )
    rest_api_list = [read_ctx.rest_api, write_ctx.rest_api]
    row = build_call_row(
        text="read system 1 and set the manager NIC address",
        contexts=[read_ctx, write_ctx],
        rest_api_list=rest_api_list,
        method_by_api={read_ctx.rest_api: "GET", write_ctx.rest_api: "PATCH"},
        arguments_by_api={write_ctx.rest_api: {"Address": "192.168.1.1"}},
    )

    x = row["x"]
    assert x["text"] == "read system 1 and set the manager NIC address"
    # Canonical unique set (sorted identity — never execution order).
    assert x["rest_api_list"] == sorted(rest_api_list)
    assert x["json"] == [dict(read_ctx.json), dict(write_ctx.json)]  # current resource bodies.
    assert x["allowed_methods"] == {
        read_ctx.rest_api: ["GET"],
        write_ctx.rest_api: ["GET", "PATCH"],
    }  # per-API method legality map.
    assert row["target_semantics"] == "unordered_call_set"

    calls = row["y_true"]["calls"]
    assert sorted(call["rest_api"] for call in calls) == sorted(rest_api_list)
    for call in calls:
        # A Call is exactly these four fields; allowed_methods stays in x.
        assert set(call.keys()) == {"rest_api", "http_method", "operation_name", "arguments"}
    by_api = {call["rest_api"]: call for call in calls}
    read_call = by_api[read_ctx.rest_api]
    write_call = by_api[write_ctx.rest_api]
    assert read_call["http_method"] == "GET"
    assert read_call["arguments"] == {}  # read-only calls carry no body args.
    assert write_call["http_method"] == "PATCH"
    assert write_call["arguments"] == {"Address": "192.168.1.1"}  # explicit body binding.


# Author: Mus mbayramo@stanford.edu
