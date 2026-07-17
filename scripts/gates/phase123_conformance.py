"""Deterministic Phase 1 -> D1 -> Phase 2 -> Phase 3 contract conformance gate.

This gate uses tiny fixtures and the real render/parse/eval functions to prove
that keys, tensor shapes, and JSON envelopes line up end-to-end. It does not
claim model quality: the model/judge responses are deterministic fixture
responses that stand in for a perfect model and a lightweight schema judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from igc.ds.corpus_dataset import CorpusJSONLDataset, PHASE1_PRETRAIN_OBJECTIVE  # noqa: E402
from igc.ds.phase1_render import render_phase1_completion, render_phase1_prompt  # noqa: E402
from igc.ds.rest_goal_contract import (  # noqa: E402
    RedfishContext,
    build_d1_rest_api_list_row,
    build_ordered_call_row,
    evaluate_ordered_calls_y_pred,
    evaluate_rest_api_set,
    inference_target_calls_json,
    parse_ordered_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_ordered_call_example,
    render_rest_api_list_example,
)


class TinyCharTokenizer:
    """Offline tokenizer for deterministic tensor-shape checks."""

    pad_token_id = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Force the dataset to use the ``encode`` fallback."""

        raise TypeError("TinyCharTokenizer supports encode() only")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return one nonzero token id per character."""

        return [(ord(char) % 255) + 1 for char in text]


def _write_phase1_corpus(tmp_path: Path, example: dict[str, Any]) -> Path:
    """Write a one-row Phase 1 corpus fixture."""

    corpus = tmp_path / "phase1_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "examples.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    return corpus


def _tensor_report(item: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Return tensor key/shape/dtype metadata for a tokenized row."""

    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "supervised_tokens": int(value.ne(-100).sum().item()) if key == "labels" else None,
        }
        for key, value in item.items()
    }


def lightweight_judge(text: str, target: list[str], response: dict[str, Any]) -> dict[str, Any]:
    """Deterministic schema judge for D1 conformance tests."""

    parsed = parse_rest_api_list_y_pred(response)
    return {
        "accepted": bool(text.strip()) and set(parsed) == set(target),
        "rest_api_set_match": set(parsed) == set(target),
        "nonsense": not bool(text.strip()),
    }


def build_report(tmp_path: Path, *, max_len: int = 192) -> dict[str, Any]:
    """Build the deterministic end-to-end conformance report."""

    initial = {
        "x": {
            "rest_api": "/redfish/v1/Systems/1",
            "allowed_methods": ["GET", "PATCH"],
            "json": {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "Off"},
        },
        "y_true": {
            "json": {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "Off"},
        },
    }
    prompt, target = render_phase1_prompt(initial)
    completion = render_phase1_completion(target)
    dataset = CorpusJSONLDataset(
        str(_write_phase1_corpus(tmp_path, initial)),
        max_len=max_len,
        tokenizer=TinyCharTokenizer(),
        objective=PHASE1_PRETRAIN_OBJECTIVE,
    )
    phase1_item = dataset[0]

    context = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=("GET", "PATCH"),
        json={"@odata.id": "/redfish/v1/Systems/1", "PowerState": "Off"},
    )
    d1_row = build_d1_rest_api_list_row(
        text="power on system 1",
        contexts=(context,),
        rest_api_list=(context.rest_api,),
    )
    phase2_rendered = render_rest_api_list_example(d1_row)
    phase2_response = json.loads(phase2_rendered.target_json)
    phase2_parsed = parse_rest_api_list_y_pred(phase2_response)
    phase2_metrics = evaluate_rest_api_set(phase2_parsed, d1_row["y_true"])
    judge = lightweight_judge(
        d1_row["x"]["text"],
        list(d1_row["y_true"]["rest_api_list"]),
        phase2_response,
    )

    phase3_row = build_ordered_call_row(
        text=d1_row["x"]["text"],
        contexts=(context,),
        rest_api_list=phase2_parsed,
        method_by_api={context.rest_api: "PATCH"},
        arguments_by_api={context.rest_api: {"PowerState": "On"}},
    )
    phase3_rendered = render_ordered_call_example(phase3_row)
    phase3_parsed = parse_ordered_calls_y_pred(phase3_rendered.target_json)
    phase3_eval = evaluate_ordered_calls_y_pred(phase3_row, phase3_rendered.target_json)
    handoff = inference_target_calls_json(phase3_row)

    return {
        "schema": "igc.phase123_conformance.v1",
        "phase1": {
            "input_keys": sorted(initial["x"].keys()),
            "target_keys": sorted(initial["y_true"].keys()),
            "prompt_len": len(prompt),
            "completion_len": len(completion),
            "tensor_keys": sorted(phase1_item.keys()),
            "tensors": _tensor_report(phase1_item),
        },
        "d1": {
            "row_keys": sorted(d1_row.keys()),
            "x_keys": sorted(d1_row["x"].keys()),
            "y_true_keys": sorted(d1_row["y_true"].keys()),
            "judge": judge,
        },
        "phase2": {
            "prompt_len": len(phase2_rendered.prompt),
            "parsed_rest_api_list": phase2_parsed,
            "set_match": phase2_metrics.set_match,
            "empty_set_match_example": evaluate_rest_api_set([], []).empty_set_match,
        },
        "phase3": {
            "prompt_len": len(phase3_rendered.prompt),
            "parsed_call_count": len(phase3_parsed),
            "call_ordered_exact_match": phase3_eval["call_ordered_exact_match"],
            "target_calls_keys": sorted(handoff.keys()),
            "target_call_fields": sorted(handoff["target_calls"][0].keys()),
        },
    }


def assert_report(report: dict[str, Any]) -> None:
    """Raise ``AssertionError`` when the deterministic contract is broken."""

    assert report["phase1"]["tensor_keys"] == ["attention_mask", "input_ids", "labels"]
    assert report["phase1"]["tensors"]["input_ids"]["shape"] == [192]
    assert report["phase1"]["tensors"]["labels"]["supervised_tokens"] > 0
    assert report["d1"]["x_keys"] == ["allowed_methods", "json", "text"]
    assert report["d1"]["y_true_keys"] == ["order_evidence", "rest_api_list"]
    assert report["d1"]["judge"]["accepted"] is True
    assert report["phase2"]["set_match"] is True
    assert report["phase2"]["empty_set_match_example"] is True
    assert report["phase3"]["call_ordered_exact_match"] is True
    assert report["phase3"]["target_calls_keys"] == ["target_calls", "text"]
    assert report["phase3"]["target_call_fields"] == [
        "allowed_methods",
        "arguments",
        "method",
        "rest_api",
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--report", default="reports/gate-report-phase123-conformance.json")
    args = parser.parse_args(argv)

    tmp_path = Path(args.work_dir) if args.work_dir else Path(".phase123-conformance-tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = build_report(tmp_path)
    try:
        assert_report(report)
    except AssertionError as exc:
        print(f"PHASE123_CONFORMANCE_FAIL {exc}")
        return 1

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PHASE123_CONFORMANCE_PASS report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
