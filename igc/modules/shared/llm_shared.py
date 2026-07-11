"""
All default llm model creation and loading is done here.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
import pkgutil
from typing import Optional, Union, Dict, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


def _model_id(args: Union[str, argparse.Namespace]) -> str:
    """The HF repo id or local weights path — a bare string, or ``args.model_type``."""
    return args if isinstance(args, str) else args.model_type


def _spec_flag(args: Union[str, argparse.Namespace], name: str, default):
    """Read an optional spec flag; a bare-string ``args`` carries no flags (returns default)."""
    return default if isinstance(args, str) else getattr(args, name, default)


def _resolve_dtype(name: Optional[str]):
    """Map a dtype name (``"bfloat16"``/...) to a ``torch.dtype``; pass ``None``/``"auto"`` through."""
    if name is None or name == "auto":
        return name
    import torch
    return getattr(torch, name)


def _from_pretrained_best_attention(model_id: str, load_kwargs: dict):
    """Load a causal LM preferring the fastest attention kernel the env supports.

    Tries FlashAttention-2 (CUDA only), then SDPA, then falls back to the default
    (eager) path. On CPU or a backbone that doesn't support the faster kernels
    (e.g. GPT-2) this is a no-op that returns the same model as before, so it is
    safe offline while giving the 3B/7B GPU profiles FlashAttention on GB300.

    :param model_id: HF repo id or local model dir.
    :param load_kwargs: kwargs already assembled (trust_remote_code, torch_dtype).
    :return: the loaded ``AutoModelForCausalLM``.
    """
    import torch

    candidates = []
    if torch.cuda.is_available():
        candidates.append("flash_attention_2")
    candidates.append("sdpa")
    for impl in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation=impl, **load_kwargs)
        except (ImportError, ValueError, RuntimeError, OSError):
            continue
        if impl == "flash_attention_2" and not any(
                isinstance(m, torch.nn.Linear) for m in model.modules()):
            # transformers' FA2 dtype probe needs an nn.Linear; Conv1D-only
            # backbones (GPT-2 family) load fine but StopIteration at the first
            # forward — fall through to sdpa for them.
            del model
            continue
        return model
    return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)


def from_pretrained_default(
        args: Union[str, argparse.Namespace],
        only_tokenizer: bool = False,
        only_model: bool = False,
        add_padding: bool = True,
        device_map: Union[str, Dict[str, str]] = "balanced"
) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    """
    Default callback that loads the backbone we fine-tune, for ANY HF model.

    Uses ``AutoModelForCausalLM`` / ``AutoTokenizer`` keyed on ``--model_type`` (a HF repo
    id OR a local weights dir, e.g. ``/home/nvidia/models/DeepSeek-V4-Flash`` — loads with
    no download), honouring ``--trust_remote_code`` (custom architectures like DeepSeek) and
    ``--llm_torch_dtype`` (bf16 for a large model).

    :param only_model:  return only model, no tokenizer.
    :param device_map:  retained for callers; placement of a large model is handled by the
        trainer / accelerate (kept as-is to preserve the existing training flow).
    :param add_padding: add pad tokens to tokenizer, that make sense if we also restore tokenizer.
    :param args: Argument parser namespace or string specifying the model_type.
    :param only_tokenizer: Whether to return only the tokenizer.
    :return: Tuple of model and tokenizer (or None if only_model is True).
    """
    model = None
    tokenizer = None
    # it looks like hugging face doesn't like mps
    if device_map == "mps":
        device_map = "auto"

    model_id = _model_id(args)
    trust_remote_code = bool(_spec_flag(args, "trust_remote_code", False))
    torch_dtype = _resolve_dtype(_spec_flag(args, "llm_torch_dtype", None))

    if not only_tokenizer:
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        model = _from_pretrained_best_attention(model_id, load_kwargs)

    if not only_model:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if add_padding and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if add_padding and model is not None and tokenizer is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token = tokenizer.pad_token

    return model, tokenizer


def safe_resize_token_embeddings(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        force_shrink: bool = False,
) -> PreTrainedModel:
    """Resize the model's token embeddings to the tokenizer, refusing silent shrinks.

    ``resize_token_embeddings`` happily SHRINKS a pretrained vocabulary with no
    warning (e.g. Qwen2.5's 151936 rows down to the cached igc tokenizer's 53147),
    deleting pretrained embedding rows and then training foreign token ids against
    the survivors. Growth (adding the igc JSON special tokens) is the only resize a
    matched backbone/tokenizer pair ever needs, so a shrink means the tokenizer does
    not belong to this backbone and is rejected loudly.

    :param model: the causal-LM backbone.
    :param tokenizer: the tokenizer the embeddings must cover.
    :param force_shrink: explicitly allow a shrinking resize (expert use only).
    :return: the (possibly resized) model, for call-site chaining.
    :raises ValueError: when the resize would shrink the vocabulary and
        ``force_shrink`` is False.
    """
    current = model.get_input_embeddings().num_embeddings
    target = len(tokenizer)
    if target < current and not force_shrink:
        raise ValueError(
            f"Refusing to shrink token embeddings from {current} to {target}: this "
            f"tokenizer does not belong to the backbone (a shrink silently deletes "
            f"pretrained embedding rows). Rebuild the dataset tokenizer for this "
            f"backbone, or pass force_shrink=True to override.")
    if target != current:
        model.resize_token_embeddings(target)
    return model


def igc_base_dir():
    """
    Return the default igc directory based on where igc package
    installed.
    :return:
    """
    resource_module = pkgutil.resolve_name("igc")
    base_dir = os.path.dirname(resource_module.__file__)
    parent_dir = os.path.dirname(base_dir)
    return parent_dir


def igc_default_dataset_dir():
    """
    Return the default dataset directory based on where igc package
    installed.
    :return:
    """
    igc_base = igc_base_dir()
    datasets_dir = os.path.join(igc_base, "datasets")
    return datasets_dir


def igc_default_tokenizer_dir():
    """
    Return the default tokenizer directory based on where igc package
    installed.
    :return:
    """
    igc_ds_base = igc_default_dataset_dir()
    tok_dir = os.path.join(igc_ds_base, "tokenizer")
    return tok_dir


def load_igc_tokenizer(
        tokenizer_dir: str = None
) -> PreTrainedTokenizer:
    """
    Load the tokenizer from the specified directory and return it.

    :param tokenizer_dir: The directory path where the tokenizer is located.
    :return: The loaded tokenizer.
    """

    if tokenizer_dir is None:
        tok_dir = igc_default_tokenizer_dir()
    else:
        tok_dir = tokenizer_dir

    if not os.path.exists(tok_dir):
        raise ValueError(f"Tokenizer directory '{tok_dir}' does not exist.")

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def save_pretrained_default(
        huggingface_dir,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        only_tokenizer=False
):
    """
      Save the default GPT2 model and tokenizer.

    :param huggingface_dir:  we keep it separate
    :param model: fine-tuned model
    :param tokenizer: tokenizer
    :param only_tokenizer:  will save only tokenizer
    :return:
    """

    try:
        if only_tokenizer:
            tokenizer.save_pretrained(huggingface_dir)
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.pad_token = tokenizer.pad_token

            model.save_pretrained(huggingface_dir)
            tokenizer.save_pretrained(huggingface_dir)
        return True
    except Exception as e:
        print(f"Error while saving pretrained default: {str(e)}")

    return False


def load_pretrained_default(
        args: argparse.Namespace,
        path_to_model: str,
        device_map="auto"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
      Load fine-tuned model, the default GPT2 model and tokenizer.

    :param args: argparse namespace that contains various optional parameters for the model.
    :param path_to_model:String specifying the path to the pre-trained model.
    :param device_map: Device map for the model. Can be set to "auto".
    :return: A tuple consisting of the pre-trained GPT-2 model and its tokenizer.
    """

    trust_remote_code = bool(getattr(args, "trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_model,
        cache_dir=args.llm_cache_dir
        if hasattr(args, "llm_cache_dir") else None,
        trust_remote_code=trust_remote_code,
    )

    model_args = {
        "ignore_mismatched_sizes":
            args.llm_ignore_mismatched_sizes if
            hasattr(args, "llm_ignore_mismatched_sizes") else False,
        "output_loading_info":
            args.llm_output_loading_info
            if hasattr(args, "llm_output_loading_info") else False,
        "dtype":
            args.llm_torch_dtype
            if hasattr(args, "llm_torch_dtype") else None,
        "device_map":
            device_map,
    }

    model_args = {k: v for k, v in model_args.items() if v is not None}
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        cache_dir=args.llm_cache_dir if hasattr(args, "llm_cache_dir") else None,
        trust_remote_code=trust_remote_code,
        **model_args,
    )
    return model, tokenizer
