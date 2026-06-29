"""LoRA via HuggingFace PEFT, so a large decoder fine-tunes on one GPU (bf16 + LoRA).

Replaces the GPT-2-only (and crash-on-construction) ``igc.modules.nn.lora1d``
``LoRAConv1DWrapper`` with PEFT ``LoraConfig`` + ``get_peft_model``, choosing sensible
target modules for GPT-2 (Conv1D: ``c_attn`` / ``c_fc`` / ``c_proj``) or a modern decoder
(``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` / ``gate_proj`` / ``up_proj`` / ``down_proj``).
``peft`` is imported lazily so this module stays importable without it.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

_GPT2_TARGETS = ["c_attn", "c_fc", "c_proj"]
_DECODER_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def default_target_modules(model: Any = None, model_type: Optional[str] = None) -> List[str]:
    """Pick the LoRA target-module names for a backbone.

    :param model: optional HF model; its ``config.model_type`` is used when ``model_type``
        is not given.
    :param model_type: explicit model type/name (e.g. ``"gpt2"``, ``"llama"``).
    :return: the GPT-2 Conv1D targets for a GPT-2 backbone, else the modern-decoder
        projection targets.
    """
    name = (model_type or "").lower()
    if not name and model is not None:
        name = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if "gpt2" in name:
        return list(_GPT2_TARGETS)
    return list(_DECODER_TARGETS)


def apply_lora(
    model: Any,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    model_type: Optional[str] = None,
) -> Any:
    """Wrap a causal-LM with LoRA adapters via PEFT and return the ``PeftModel``.

    Only the injected adapters are trainable; the base weights stay frozen (use bf16 base
    weights for memory). This is the supported path for fine-tuning a large backbone on a
    single GPU.

    :param model: the HF ``AutoModelForCausalLM`` to adapt.
    :param r: LoRA rank.
    :param alpha: LoRA alpha (scaling).
    :param dropout: LoRA dropout.
    :param target_modules: module names to adapt; defaults to :func:`default_target_modules`.
    :param model_type: backbone type hint used for default target selection.
    :return: a ``peft.PeftModel`` wrapping ``model``.
    """
    from peft import LoraConfig, get_peft_model  # lazy: keep this module importable without peft

    if target_modules is None:
        target_modules = default_target_modules(model, model_type)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def trainable_parameter_summary(model: Any) -> Tuple[int, int]:
    """Count trainable vs total parameters (LoRA makes ``trainable << total``).

    :param model: any module exposing ``.parameters()``.
    :return: ``(trainable, total)`` parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Author: Mus mbayramo@stanford.edu
