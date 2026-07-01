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

# The input-embedding module name per backbone. igc extends the tokenizer with the
# @odata/Redfish special tokens and resizes the embedding, so these rows are NEW and
# must be trainable — otherwise LoRA (which freezes the base) leaves them at their
# random init and M1's whole point (learn a latent for the new tokens) is a no-op.
_GPT2_EMB = ["wte"]
_DECODER_EMB = ["embed_tokens"]


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


def default_save_modules(model: Any = None, model_type: Optional[str] = None) -> List[str]:
    """Pick the embedding module PEFT must keep trainable (``modules_to_save``).

    Returns the input-embedding name so the resized (IGC-token) rows train. On a
    modern decoder with tied input/output embeddings (e.g. Qwen2.5,
    ``tie_word_embeddings=True``) saving ``embed_tokens`` alone is correct — the tied
    ``lm_head`` follows it — and avoids the redundant/te-breaking double save.

    :param model: optional HF model; ``config.model_type`` is used when ``model_type``
        is absent.
    :param model_type: explicit model type/name.
    :return: the embedding module name(s) for ``modules_to_save``.
    """
    name = (model_type or "").lower()
    if not name and model is not None:
        name = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if "gpt2" in name:
        return list(_GPT2_EMB)
    return list(_DECODER_EMB)


def apply_lora(
    model: Any,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    model_type: Optional[str] = None,
    modules_to_save: Optional[List[str]] = None,
    train_embeddings: bool = True,
) -> Any:
    """Wrap a causal-LM with LoRA adapters via PEFT and return the ``PeftModel``.

    The injected adapters are trainable and the base weights stay frozen (bf16 base for
    memory). By default the input embedding is ALSO trainable (``modules_to_save``): igc
    extends the tokenizer and resizes the embedding, so the new rows must move or M1's
    representation objective is a no-op. This is the supported path for fine-tuning a
    large backbone on a single GPU.

    :param model: the HF ``AutoModelForCausalLM`` to adapt.
    :param r: LoRA rank.
    :param alpha: LoRA alpha (scaling).
    :param dropout: LoRA dropout.
    :param target_modules: module names to adapt; defaults to :func:`default_target_modules`.
    :param model_type: backbone type hint used for default target/save selection.
    :param modules_to_save: modules kept fully trainable (embeddings); defaults to
        :func:`default_save_modules` when ``train_embeddings`` is set.
    :param train_embeddings: when True (default), keep the resized embedding trainable;
        set False to freeze it (LoRA-adapters-only, the legacy behavior).
    :return: a ``peft.PeftModel`` wrapping ``model``.
    """
    from peft import LoraConfig, get_peft_model  # lazy: keep this module importable without peft

    if target_modules is None:
        target_modules = default_target_modules(model, model_type)
    if modules_to_save is None and train_embeddings:
        modules_to_save = default_save_modules(model, model_type)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
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
