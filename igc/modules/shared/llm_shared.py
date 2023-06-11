from transformers import GPT2LMHeadModel, GPT2Tokenizer


def from_pretrained_default(args, only_tokenizer=False):
    """
    :param args:
    :param only_tokenizer:
    :return:
    """
    model = None
    if not only_tokenizer:
        model = GPT2LMHeadModel.from_pretrained(args.model_type)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    return model, tokenizer
