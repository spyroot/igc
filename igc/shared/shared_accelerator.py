import argparse
from accelerate import Accelerator


def build_accelerator(cmd: argparse.Namespace) -> Accelerator:
    """
    Render accelerator object from command line arguments
    :param cmd:
    :return:
    """
    accelerator_args = {}

    if hasattr(cmd, "device_placement"):
        accelerator_args["device_placement"] = cmd.device_placement

    if hasattr(cmd, "split_batches"):
        accelerator_args["split_batches"] = cmd.split_batches

    if hasattr(cmd, "mixed_precision"):
        accelerator_args["mixed_precision"] = cmd.mixed_precision

    if hasattr(cmd, "gradient_accumulation_steps"):
        accelerator_args["gradient_accumulation_steps"] = cmd.gradient_accumulation_steps

    if hasattr(cmd, "cpu"):
        accelerator_args["cpu"] = cmd.cpu

    if hasattr(cmd, "deepspeed_plugin"):
        accelerator_args["deepspeed_plugin"] = cmd.deepspeed_plugin

    if hasattr(cmd, "fsdp_plugin"):
        accelerator_args["fsdp_plugin"] = cmd.fsdp_plugin

    if hasattr(cmd, "megatron_lm_plugin"):
        accelerator_args["megatron_lm_plugin"] = cmd.megatron_lm_plugin

    if hasattr(cmd, "rng_types"):
        accelerator_args["rng_types"] = cmd.rng_types

    if hasattr(cmd, "log_with"):
        accelerator_args["log_with"] = cmd.log_with

    if hasattr(cmd, "project_dir"):
        accelerator_args["project_dir"] = cmd.project_dir

    if hasattr(cmd, "project_config"):
        accelerator_args["project_config"] = cmd.project_config

    if hasattr(cmd, "gradient_accumulation_plugin"):
        accelerator_args["gradient_accumulation_plugin"] = cmd.gradient_accumulation_plugin

    if hasattr(cmd, "dispatch_batches"):
        accelerator_args["dispatch_batches"] = cmd.dispatch_batches

    if hasattr(cmd, "even_batches"):
        accelerator_args["even_batches"] = cmd.even_batches

    if hasattr(cmd, "step_scheduler_with_optimizer"):
        accelerator_args["step_scheduler_with_optimizer"] = cmd.step_scheduler_with_optimizer

    if hasattr(cmd, "kwargs_handlers"):
        accelerator_args["kwargs_handlers"] = cmd.kwargs_handlers

    if hasattr(cmd, "dynamo_backend"):
        accelerator_args["dynamo_backend"] = cmd.dynamo_backend

    accelerator = Accelerator(**accelerator_args)
    return accelerator
