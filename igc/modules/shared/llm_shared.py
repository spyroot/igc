"""
All default llm model creation and loading is done here.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from typing import Optional, Union, Dict, Tuple
from transformers import (GPT2LMHeadModel,
                          GPT2Tokenizer,
                          PreTrainedModel,
                          PreTrainedTokenizer)


def from_pretrained_default(
    args: Union[str, argparse.Namespace],
    only_tokenizer: bool = False,
    only_model: bool = False,
    add_padding: bool = True,
    device_map: Union[str, Dict[str, str]] = "auto"
) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    """
    This is default callback used to load default model that we fine tune.

    :param only_model:  return only model, no tokenizer.
    :param device_map:  where to upload model
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

    if not only_tokenizer:
        if isinstance(args, str):
            model = GPT2LMHeadModel.from_pretrained(
                args, device_map=device_map
            )
        else:
            model = GPT2LMHeadModel.from_pretrained(
                args.model_type, device_map=device_map
            )

    if not only_model:
        if isinstance(args, str):
            tokenizer = GPT2Tokenizer.from_pretrained(args)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)

        if add_padding:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_igc_tokenizer(tokenizer_dir: str = None) -> PreTrainedTokenizer:
    """
    Load the tokenizer from the specified directory and return it.

    :param tokenizer_dir: The directory path where the tokenizer is located.
    :return: The loaded tokenizer.
    """
    if tokenizer_dir is None:
        tok_dir = f"datasets/tokenizer"
    else:
        tok_dir = tokenizer_dir

    if not os.path.exists(tok_dir):
        raise ValueError(f"Tokenizer directory '{tok_dir}' does not exist.")

    tokenizer = GPT2Tokenizer.from_pretrained(tok_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def save_pretrained_default(
    huggingface_dir,
    model: GPT2Tokenizer,
    gpt_tokenizer: GPT2Tokenizer,
    only_tokenizer=False
):
    """
      Save the default GPT2 model and tokenizer.

    :param huggingface_dir:  we keep it separate
    :param model: fine-tuned model
    :param gpt_tokenizer: tokenizer
    :param only_tokenizer:  will save only tokenizer
    :return:
    """

    try:
        if only_tokenizer:
            gpt_tokenizer.save_pretrained(huggingface_dir)
        else:
            model.save_pretrained(huggingface_dir)
            gpt_tokenizer.save_pretrained(huggingface_dir)
        return True
    except:
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

    tokenizer = GPT2Tokenizer.from_pretrained(
        path_to_model,
        cache_dir=args.llm_cache_dir if hasattr(args, "llm_cache_dir") else None,
    )

    model_args = {
        "ignore_mismatched_sizes":
            args.llm_ignore_mismatched_sizes if hasattr(args, "llm_ignore_mismatched_sizes") else False,
        "output_loading_info":
            args.llm_output_loading_info if hasattr(args, "llm_output_loading_info") else False,
        "_fast_init":
            args.llm_fast_init if hasattr(args, "llm_fast_init") else True,
        "torch_dtype":
            args.llm_torch_dtype if hasattr(args, "llm_torch_dtype") else None,
        "device_map":
            device_map,
    }

    model_args = {k: v for k, v in model_args.items() if v is not None}
    model = GPT2LMHeadModel.from_pretrained(
        path_to_model,
        cache_dir=args.llm_cache_dir if hasattr(args, "llm_cache_dir") else None,
        **model_args,
    )
    return model, tokenizer
