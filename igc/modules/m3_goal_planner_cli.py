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
import os
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
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.modules.m3_goal_planner_specs import (
    DEFAULT_M3_GOAL_PLANNER_SPEC,
    load_m3_goal_planner_profile,
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
    train.add_argument("--spec-file", default=str(DEFAULT_M3_GOAL_PLANNER_SPEC))
    train.add_argument("--profile", default=None)
    train.add_argument("--model-type", default=None)
    train.add_argument("--dataset-jsonl", default=None)
    train.add_argument("--output-dir", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=None)
    train.add_argument("--max-length", type=int, default=None)
    train.add_argument("--max-records", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--dataloader-num-workers", type=int, default=None)
    train.add_argument("--metric-report", default=None)
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
    build_redfish.add_argument("--json-root", action="append", required=True)
    build_redfish.add_argument("--rest-api-map-dir", action="append", default=None)
    build_redfish.add_argument("--output-jsonl", required=True)
    build_redfish.add_argument("--templates-per-action", type=int, default=3)
    build_redfish.add_argument("--source", default="redfish_ctl_capture")
    build_redfish.add_argument("--vendor", default=None)

    build_redfish_ctl = sub.add_parser(
        "build-redfish-ctl-dataset",
        help="Build M3 JSONL directly from the redfish_ctl ~/.json_responses corpus.",
    )
    build_redfish_ctl.add_argument("--json-root", action="append", required=True)
    build_redfish_ctl.add_argument("--rest-api-map-dir", action="append", default=None)
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

    profile = load_m3_goal_planner_profile(args.profile, args.spec_file)
    dataset_cfg = profile.get("dataset") or {}
    model_cfg = profile.get("model") or {}
    optimizer_cfg = profile.get("optimizer") or {}
    trainer_cfg = profile.get("trainer") or {}
    tracking_cfg = profile.get("tracking") or {}
    metric_names = (tracking_cfg.get("metrics") or {}).get("plot_names") or {}

    dataset_jsonl = args.dataset_jsonl or dataset_cfg.get("jsonl")
    if not dataset_jsonl:
        raise ValueError("M3 training requires --dataset-jsonl or dataset.jsonl in the profile")
    dataset = load_m3_goal_plan_dataset(dataset_jsonl)
    max_records = _coalesce(args.max_records, dataset_cfg.get("max_records"))
    if max_records is not None:
        dataset = M3GoalPlanJsonlDataset(dataset.records[:int(max_records)])

    model_type = args.model_type or model_cfg.get("model_type") or "gpt2"
    trust_remote_code = bool(args.trust_remote_code or model_cfg.get("trust_remote_code", False))
    torch_dtype = _torch_dtype(model_cfg.get("torch_dtype"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_type,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_type,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model = _apply_peft_if_requested(model, profile.get("peft"))

    output_dir = args.output_dir or trainer_cfg.get("output_dir") or "experiments/m3_goal_planner"
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    metric_logger = _build_metric_logger(
        args,
        profile,
        tracking_cfg,
        model_type,
        output_dir,
        len(dataset),
    )
    if metric_logger is not None:
        _log_dataset_metrics(metric_logger, metric_names, len(dataset))
    trainer = M3GoalPlannerSFTTrainer(
        model,
        tokenizer,
        dataset,
        M3GoalPlannerSFTConfig(
            output_dir=output_dir,
            epochs=int(_coalesce(args.epochs, trainer_cfg.get("epochs"), 1)),
            batch_size=int(_coalesce(args.batch_size, trainer_cfg.get("batch_size"), 1)),
            gradient_accumulation_steps=int(_coalesce(
                args.gradient_accumulation_steps,
                trainer_cfg.get("gradient_accumulation_steps"),
                1,
            )),
            learning_rate=float(_coalesce(
                args.learning_rate,
                optimizer_cfg.get("learning_rate"),
                5e-5,
            )),
            weight_decay=float(_coalesce(args.weight_decay, optimizer_cfg.get("weight_decay"), 0.0)),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(optimizer_cfg.get("eps", 1e-8)),
            optimizer_name=str(optimizer_cfg.get("name", "adamw")),
            scheduler=str(optimizer_cfg.get("scheduler", "constant")),
            warmup_ratio=float(optimizer_cfg.get("warmup_ratio", 0.0)),
            warmup_steps=int(optimizer_cfg.get("warmup_steps", 0) or 0),
            max_grad_norm=optimizer_cfg.get("max_grad_norm"),
            max_length=int(_coalesce(args.max_length, trainer_cfg.get("max_length"), 2048)),
            max_steps=_optional_int(_coalesce(args.max_steps, trainer_cfg.get("max_steps"))),
            precision=str(_coalesce(args.precision, trainer_cfg.get("precision"), "fp32")),
            dataloader_num_workers=int(_coalesce(
                args.dataloader_num_workers,
                trainer_cfg.get("dataloader_num_workers"),
                0,
            )),
            seed=int(trainer_cfg.get("seed", 42)),
            log_every=int(trainer_cfg.get("log_every", 10)),
            tf32=bool(trainer_cfg.get("tf32", True)),
            gradient_checkpointing=bool(trainer_cfg.get("gradient_checkpointing", False)),
            torch_compile=bool(trainer_cfg.get("torch_compile", False)),
            metric_prefix=str(tracking_cfg.get("metric_prefix", "03_m3_goal_planner")),
            metric_names=metric_names,
            device=args.device,
        ),
        metric_logger=metric_logger,
    )
    history = trainer.train()
    trainer.save()
    if trainer.distributed.is_main_process:
        history_path = Path(output_dir) / "training_history.json"
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "history": history,
            "output_dir": output_dir,
            "profile": profile["name"],
            "dataset_jsonl": dataset_jsonl,
            "records": len(dataset),
        }, indent=2))
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


def _coalesce(*values):
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def _optional_int(value) -> int | None:
    """Convert optional integer-like values."""
    return None if value is None else int(value)


def _torch_dtype(name: str | None):
    """Return torch dtype for a profile string."""
    if name is None:
        return None
    import torch

    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported torch dtype: {name}")


def _apply_peft_if_requested(model, peft_cfg: dict[str, Any] | None):
    """Wrap a causal LM with PEFT LoRA/rsLoRA/DoRA when a profile asks for it."""
    if not peft_cfg or not peft_cfg.get("enabled", False):
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
        target_modules=list(peft_cfg.get("target_modules") or []),
        bias=str(peft_cfg.get("bias", "none")),
        use_rslora=bool(peft_cfg.get("use_rslora", False)),
        use_dora=bool(peft_cfg.get("use_dora", False)),
        init_lora_weights=peft_cfg.get("init_lora_weights", True),
    )
    return get_peft_model(model, config)


def _build_metric_logger(
    args: argparse.Namespace,
    profile: dict[str, Any],
    tracking_cfg: dict[str, Any],
    model_type: str,
    output_dir: str,
    record_count: int,
):
    """Create the configured metric logger, or ``None`` when disabled."""
    report_to = args.metric_report
    if report_to is None:
        report_to = tracking_cfg.get("report_to", "none")
    if str(report_to).lower() in {"", "none", "off", "disabled", "false"}:
        return None
    trainer_cfg = profile.get("trainer") or {}
    optimizer_cfg = profile.get("optimizer") or {}
    peft_cfg = profile.get("peft") or {}
    return MetricLogger(
        str(report_to),
        output_dir=output_dir,
        project=tracking_cfg.get("project"),
        entity=tracking_cfg.get("entity"),
        train="llm",
        llm="goal",
        model_type=model_type,
        num_train_epochs=trainer_cfg.get("epochs"),
        per_device_train_batch_size=trainer_cfg.get("batch_size"),
        num_workers=trainer_cfg.get("dataloader_num_workers"),
        use_peft=bool(peft_cfg.get("enabled", False)),
        lora_r=peft_cfg.get("r"),
        lora_alpha=peft_cfg.get("lora_alpha"),
        sharding="ddp" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "none",
        llm_torch_dtype=(profile.get("model") or {}).get("torch_dtype"),
        device=args.device,
        m3_profile=profile.get("name"),
        m3_record_count=record_count,
        m3_optimizer=optimizer_cfg.get("name"),
        m3_scheduler=optimizer_cfg.get("scheduler"),
    )


def _log_dataset_metrics(metric_logger, metric_names: dict[str, str], record_count: int) -> None:
    """Log rank-zero dataset summary scalars before training starts."""
    metric_logger.log_metric(
        metric_names.get("dataset_records", "m3_dataset/record_count"),
        float(record_count),
        0,
    )


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
