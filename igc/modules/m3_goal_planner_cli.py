"""Command-line entry point for M3 goal planning and SFT.

Examples:

```
python -m igc.modules.m3_goal_planner_cli infer \
  --instruction "boot server with ubuntu2204 and adjust bios for fast boot" \
  --context-json '{"approved_images":[{"image_id":"approved_ubuntu_22_04_iso","os":"ubuntu","version":"22.04","approved":true}]}'

python -m igc.modules.m3_goal_planner_cli train \
  --model-type gpt2 \
  --dataset-jsonl data/m3_goal_plans.jsonl \
  --output-dir experiments/m3_goal_planner

python -m igc.modules.m3_goal_planner_cli build-redfish-ctl-dataset \
  --json-root ~/.json_responses \
  --output-jsonl data/m3_goal_plans.jsonl
```

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

from igc.ds.m3_goal_plan_dataset import (
    M3GoalPlanJsonlDataset,
    load_m3_goal_plan_dataset,
)
from igc.ds.m3_redfish_goal_plan_builder import M3RedfishCtlGoalPlanDatasetBuilder
from igc.modules.m3_goal_planner import (
    M3DeterministicGoalPlanner,
    M3GoalPlannerContext,
    M3ModelGoalPlanner,
)
from igc.modules.m3_goal_planner_train import (
    M3GoalPlannerSFTConfig,
    M3GoalPlannerSFTTrainer,
)


def main(argv: list[str] | None = None) -> int:
    """Run the M3 CLI."""
    parser = argparse.ArgumentParser(description="M3 goal planner: infer, train, or export data.")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="Plan from an NLP instruction.")
    infer.add_argument("--instruction", required=True)
    infer.add_argument("--context-json", default="{}")
    infer.add_argument("--model-path", default=None)
    infer.add_argument("--max-new-tokens", type=int, default=1024)

    train = sub.add_parser("train", help="Supervised fine-tune an M3 goal planner.")
    train.add_argument("--model-type", default="gpt2")
    train.add_argument("--dataset-jsonl", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--max-records", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--trust-remote-code", action="store_true")

    export_actions = sub.add_parser(
        "export-redfish-actions",
        help="Write M3 training JSONL from a Redfish action catalog JSON file.",
    )
    export_actions.add_argument("--action-catalog-json", required=True)
    export_actions.add_argument("--output-jsonl", required=True)
    export_actions.add_argument("--templates-per-action", type=int, default=3)

    build_redfish = sub.add_parser(
        "build-redfish-dataset",
        help="Build M3 JSONL from captured Redfish JSON and rest_api_map.npy.",
    )
    build_redfish.add_argument("--json-root", required=True)
    build_redfish.add_argument("--rest-api-map-dir", default=None)
    build_redfish.add_argument("--output-jsonl", required=True)
    build_redfish.add_argument("--templates-per-action", type=int, default=3)
    build_redfish.add_argument("--source", default="redfish_ctl_capture")
    build_redfish.add_argument("--vendor", default=None)

    build_redfish_ctl = sub.add_parser(
        "build-redfish-ctl-dataset",
        help="Build M3 JSONL directly from the redfish_ctl ~/.json_responses corpus.",
    )
    build_redfish_ctl.add_argument("--json-root", required=True)
    build_redfish_ctl.add_argument("--rest-api-map-dir", default=None)
    build_redfish_ctl.add_argument("--output-jsonl", required=True)
    build_redfish_ctl.add_argument("--templates-per-action", type=int, default=3)
    build_redfish_ctl.add_argument("--source", default="redfish_ctl_capture")
    build_redfish_ctl.add_argument("--vendor", default=None)

    args = parser.parse_args(argv)
    if args.command == "infer":
        return _infer(args)
    if args.command == "train":
        return _train(args)
    if args.command == "export-redfish-actions":
        return _export_redfish_actions(args)
    if args.command in {"build-redfish-dataset", "build-redfish-ctl-dataset"}:
        return _build_redfish_dataset(args)
    raise ValueError(args.command)


def _infer(args: argparse.Namespace) -> int:
    """Run deterministic or model-backed inference."""
    context = M3GoalPlannerContext.from_mapping(_load_json_arg(args.context_json))
    if args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        planner = M3ModelGoalPlanner(model, tokenizer, fallback=M3DeterministicGoalPlanner())
        plan = planner.plan(args.instruction, context, max_new_tokens=args.max_new_tokens)
    else:
        plan = M3DeterministicGoalPlanner().plan(args.instruction, context)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _train(args: argparse.Namespace) -> int:
    """Train a model from JSONL records."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_m3_goal_plan_dataset(args.dataset_jsonl)
    if args.max_records is not None:
        dataset = M3GoalPlanJsonlDataset(dataset.records[:args.max_records])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_type,
        trust_remote_code=args.trust_remote_code,
    )
    trainer = M3GoalPlannerSFTTrainer(
        model,
        tokenizer,
        dataset,
        M3GoalPlannerSFTConfig(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_steps=args.max_steps,
            device=args.device,
        ),
    )
    history = trainer.train()
    trainer.save()
    print(json.dumps({"history": history, "output_dir": args.output_dir}, indent=2))
    return 0


def _export_redfish_actions(args: argparse.Namespace) -> int:
    """Export training records for every Redfish action in a catalog file."""
    action_catalog = _load_json_arg(f"@{args.action_catalog_json}")
    if isinstance(action_catalog, dict):
        action_catalog = action_catalog.get("actions") or action_catalog.get("tool_catalog") or []
    dataset = M3GoalPlanJsonlDataset.from_redfish_action_catalog(
        action_catalog,
        templates_per_action=args.templates_per_action,
    )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_jsonl(output)
    print(json.dumps({"output_jsonl": str(output), "records": len(dataset)}))
    return 0


def _build_redfish_dataset(args: argparse.Namespace) -> int:
    """Build M3 training records from offline Redfish captures."""
    builder = M3RedfishCtlGoalPlanDatasetBuilder(
        json_root=args.json_root,
        rest_api_map_dir=args.rest_api_map_dir,
        source=args.source,
        vendor=args.vendor,
    )
    dataset = builder.build_dataset(templates_per_action=args.templates_per_action)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_jsonl(output)
    print(json.dumps({
        "output_jsonl": str(output),
        "records": len(dataset),
        "actions": len(builder.action_catalog()),
    }))
    return 0


def _load_json_arg(value: str) -> dict[str, Any]:
    """Load JSON from a literal string or ``@path``."""
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
