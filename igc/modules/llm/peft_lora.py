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

import math
from typing import Any, List, Optional, Tuple

_GPT2_TARGETS = ["c_attn", "c_fc", "c_proj"]
_DECODER_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# The input-embedding module name per backbone. igc extends the tokenizer with the
# @odata/Redfish special tokens and resizes the embedding, so these rows are NEW and
# must be trainable — otherwise LoRA (which freezes the base) leaves them at their
# random init and Phase 1's new-token objective would otherwise be a no-op.
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
    train_embeddings: bool = False,
    new_token_ids: Optional[List[int]] = None,
    adapter_method: str = "lora",
    init_lora_weights: Any = True,
) -> Any:
    """Wrap a causal-LM with LoRA adapters via PEFT and return the ``PeftModel``.

    The injected adapters are trainable and the base weights (including the embedding)
    stay frozen by default — the stable path. igc extends the tokenizer and resizes the
    embedding, so ideally the NEW rows would train, but on a ``tie_word_embeddings``
    backbone (Qwen) neither PEFT approach is safe today: whole-embedding
    ``modules_to_save`` DIVERGES at the encoder lr, and ``trainable_token_indices``
    breaks the tied ``embed_tokens``/``lm_head`` at runtime. So embedding training is an
    explicit opt-in (``train_embeddings=True``), pending an untie / grad-mask follow-up;
    on an UNtied backbone ``new_token_ids`` trains only the new rows via
    ``trainable_token_indices``.

    :param model: the HF ``AutoModelForCausalLM`` to adapt.
    :param r: LoRA rank.
    :param alpha: LoRA alpha (scaling).
    :param dropout: LoRA dropout.
    :param target_modules: module names to adapt; defaults to :func:`default_target_modules`.
    :param model_type: backbone type hint used for default target/save selection.
    :param modules_to_save: modules kept fully trainable; overrides the embedding defaults.
    :param train_embeddings: opt-in to train the embedding (default False = frozen).
    :param new_token_ids: with ``train_embeddings``, train ONLY these rows (untied models).
    :param adapter_method: ``"lora"`` (default), ``"rslora"`` (rank-stabilized scaling),
        or ``"dora"`` (magnitude/direction decomposition) — the ablation axis of
        ``docs/TRAINING_OPTIMIZATION_PLAN.md``.
    :param init_lora_weights: PEFT adapter init — ``True`` (default), or a string
        initializer such as ``"pissa"`` / ``"eva"`` / ``"loftq"``.
    :return: a ``peft.PeftModel`` wrapping ``model``.
    """
    from peft import LoraConfig, get_peft_model  # lazy: keep this module importable without peft

    if target_modules is None:
        target_modules = default_target_modules(model, model_type)
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_lora_weights,
    )
    method = (adapter_method or "lora").lower()
    if method == "rslora":
        kwargs["use_rslora"] = True
    elif method == "dora":
        kwargs["use_dora"] = True
    elif method != "lora":
        raise ValueError(f"unknown adapter_method {adapter_method!r}; use lora|rslora|dora")
    if train_embeddings and new_token_ids:
        # Preferred: train ONLY the new IGC-token rows; the tied base stays frozen.
        emb = default_save_modules(model, model_type)[0]
        kwargs["trainable_token_indices"] = {emb: list(new_token_ids)}
    elif modules_to_save is not None:
        kwargs["modules_to_save"] = modules_to_save
    elif train_embeddings:
        # Fallback (no new-token ids known): whole-embedding trainable. WARNING: this
        # can diverge on a tied-embedding backbone — pass new_token_ids when possible.
        kwargs["modules_to_save"] = default_save_modules(model, model_type)
    return get_peft_model(model, LoraConfig(**kwargs))


def validate_loaded_adapter_profile(
    model: Any,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
    adapter_method: str,
) -> None:
    """Require a loaded parent adapter to match its training profile.

    Phase 2 and Phase 3 initialize from promoted parent adapters. PEFT reads the
    persisted adapter configuration from the artifact, so the run must compare that
    configuration with the resolved YAML profile before optimization starts.

    :param model: loaded PEFT model exposing ``peft_config``.
    :param r: profile LoRA rank.
    :param alpha: profile LoRA alpha.
    :param dropout: profile LoRA dropout.
    :param target_modules: profile module names; order is not semantic.
    :param adapter_method: profile adapter family (``lora``, ``rslora``, or ``dora``).
    :raises RuntimeError: when the active adapter is ambiguous or any setting differs.
    """
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise RuntimeError("loaded parent does not expose a PEFT adapter configuration")

    active = getattr(model, "active_adapter", None)
    if isinstance(active, str) and active in configs:
        name = active
    elif len(configs) == 1:
        name = next(iter(configs))
    else:
        raise RuntimeError("loaded parent PEFT adapter is ambiguous")
    config = configs[name]

    actual_method = "lora"
    if bool(getattr(config, "use_rslora", False)):
        actual_method = "rslora"
    if bool(getattr(config, "use_dora", False)):
        if actual_method != "lora":
            raise RuntimeError("loaded parent PEFT adapter enables incompatible methods")
        actual_method = "dora"

    if not target_modules:
        raise RuntimeError("resolved parent-adapter profile requires target_modules")
    expected_targets = {str(value) for value in target_modules}
    actual_targets = {str(value) for value in (getattr(config, "target_modules", None) or [])}
    mismatches = []
    if int(getattr(config, "r", -1)) != int(r):
        mismatches.append(f"r={getattr(config, 'r', None)!r} (expected {r!r})")
    if int(getattr(config, "lora_alpha", -1)) != int(alpha):
        mismatches.append(
            f"lora_alpha={getattr(config, 'lora_alpha', None)!r} (expected {alpha!r})"
        )
    actual_dropout = float(getattr(config, "lora_dropout", -1.0))
    if not math.isclose(actual_dropout, float(dropout), rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"lora_dropout={actual_dropout!r} (expected {dropout!r})")
    if actual_targets != expected_targets:
        mismatches.append(
            f"target_modules={sorted(actual_targets)!r} "
            f"(expected {sorted(expected_targets)!r})"
        )
    if actual_method != adapter_method:
        mismatches.append(f"adapter_method={actual_method!r} (expected {adapter_method!r})")
    if mismatches:
        raise RuntimeError(
            "loaded parent adapter does not match the resolved profile: "
            + "; ".join(mismatches)
        )


def trainable_parameter_summary(model: Any) -> Tuple[int, int]:
    """Count trainable vs total parameters (LoRA makes ``trainable << total``).

    :param model: any module exposing ``.parameters()``.
    :return: ``(trainable, total)`` parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Author: Mus mbayramo@stanford.edu
