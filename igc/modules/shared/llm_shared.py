from transformers import GPT2LMHeadModel, GPT2Tokenizer


def from_pretrained_default(args, only_tokenizer=False, add_padding=True, device_map="auto"):
    """
    :param device_map:
    :param add_padding:
    :param args: Argument parser namespace or string specifying the model_type.
    :param only_tokenizer: Whether to return only the tokenizer.
    :return: Tuple of model and tokenizer (or just tokenizer if only_tokenizer is True).
    """
    model = None
    if not only_tokenizer:
        if isinstance(args, str):
            model = GPT2LMHeadModel.from_pretrained(args, device_map=device_map)
        else:
            model = GPT2LMHeadModel.from_pretrained(args.model_type, device_map=device_map)

    if isinstance(args, str):
        tokenizer = GPT2Tokenizer.from_pretrained(args)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)

    if add_padding:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
