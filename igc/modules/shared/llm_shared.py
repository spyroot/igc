from transformers import GPT2LMHeadModel, GPT2Tokenizer


def from_pretrained_default(args, only_tokenizer=False):
    """
    :param args: Argument parser namespace or string specifying the model_type.
    :param only_tokenizer: Whether to return only the tokenizer.
    :return: Tuple of model and tokenizer (or just tokenizer if only_tokenizer is True).
    """
    model = None
    if not only_tokenizer:
        if isinstance(args, str):
            model = GPT2LMHeadModel.from_pretrained(args, device_map="auto")
        else:
            model = GPT2LMHeadModel.from_pretrained(args.model_type, device_map="auto")

    if isinstance(args, str):
        tokenizer = GPT2Tokenizer.from_pretrained(args)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)

    return model, tokenizer
