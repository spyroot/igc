"""
All shared specs for trainers.

Author:Mus mbayramo@stanford.edu
"""
import torch


def make_default_optimizer_spec(cmd):
    """
    Default optimizer args

    :param cmd:
    :return:
    """
    default_args = {
        "gradient_checkpointing": getattr(cmd, "gradient_checkpointing", False),
        "gradient_accumulation_steps": getattr(cmd, "gradient_accumulation_steps", 8),
    }
    return default_args


def make_default_data_type_spec(cmd):
    """

    :param cmd:
    :return:
    """
    default_args = {
        "fp16": torch.cuda.is_available() and cmd.fp16 or False,
        "fp16_opt_level": torch.cuda.is_available() and cmd.fp16_opt_level or False,
        "half_precision_backend": torch.cuda.is_available() and cmd.half_precision_backend or False,
        "bf16_full_eval": torch.cuda.is_available() and cmd.bf16_full_eval or False,
        "fp16_full_eval": torch.cuda.is_available() and cmd.fp16_full_eval or False,
        "tf32": torch.cuda.is_available() and cmd.tf32 or False,
    }

    return default_args


def make_default_spec(cmd):
    """
    :param cmd:
    :return:
    """
    default_args = {}
    optimizer_spec = make_default_optimizer_spec(cmd)
    default_args.update(optimizer_spec)

    data_type_spec = make_default_data_type_spec(cmd)
    default_args.update(data_type_spec)
    return default_args

