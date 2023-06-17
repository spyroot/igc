import argparse
from typing import Optional, Union, Dict, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def from_pretrained_default(
    args: Union[str, argparse.Namespace],
    only_tokenizer: bool = False,
    only_model: bool = False,
    add_padding: bool = True,
    device_map: Union[str, Dict[str, str]] = "auto"
) -> Tuple[Optional[GPT2LMHeadModel], Optional[GPT2Tokenizer]]:
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
) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
      Load fine tuned model, the default GPT2 model and tokenizer.

    :param args:
    :param path_to_model:
    :param device_map:
    :return:
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
