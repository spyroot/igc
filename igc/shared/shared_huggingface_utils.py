"""
This hugging face shared utility method,
used in different places in the code.

"""
import collections
from typing import Tuple, Any, Type, Dict, List

import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from transformers.models.auto.auto_factory import _BaseAutoModelClass
import deepspeed
import accelerate


class LossMonitorCallback(TrainerCallback):
    def __init__(self, logging_steps=10):
        """

        :param logging_steps:
        """
        self.logging_steps = logging_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        """

        :param args:
        :param state:
        :param control:
        :param logs:
        :param kwargs:
        :return:
        """
        if state.global_step % self.logging_steps == 0:
            print(f"Step: {state.global_step}, Loss: {logs['loss']}")


def hugging_face_info():
    """Prnt version and location of transformers pkg""
    :return:
    """
    print(transformers.__version__)
    print(transformers.__file__)
    print(f"Deepspeed version: {deepspeed.__version__}")
    print(f"Deepspeed location: {deepspeed.__file__}")
    print(f"Accelerate version: {accelerate.__version__}")
    print(f"Accelerate location: {accelerate.__file__}")


def model_hf_name(model_type: str, model_size: str) -> str:
    """
    Return huggingface model name for a given model type and size.

    model_name = model_hf_name('gpt2', 'small')  # returns 'gpt2'
    model_name = model_hf_name('bert', 'large')  # returns 'bert-large-uncased'
    model_name = model_hf_name('neo', 'large')  # returns 'EleutherAI/gpt-neo-2.7B'

    :param model_type: String indicating the model type. Possible values are 'gpt2', 'bert', 'neo', 'roberta'.
    :param model_size: String indicating the model size. Possible values depend on the model type.
    :return: The corresponding HuggingFace model name.
    :raises ValueError: If an invalid model name or size is provided.
    """
    models = {
        'gpt2': {
            'small': 'gpt2',
            'medium': 'gpt2-medium',
            'large': 'gpt2-large',
            'full': 'gpt2-xl'
        },
        'bert': {
            'tiny': 'prajjwal1/bert-tiny',
            'medium': 'prajjwal1/bert-medium',
            'large': 'bert-large-uncased',
            'base': 'bert-base-uncased'
        },
        'neo': {
            'large': 'EleutherAI/gpt-neo-2.7B'
        },
        'roberta': {
            'base': 'roberta-base'
        }
    }

    try:
        model_sizes = models[model_type]
    except KeyError:
        raise ValueError(
            f"Invalid model type: {model_type}. Please choose from {list(models.keys())}.")

    try:
        return model_sizes[model_size]
    except KeyError:
        raise ValueError(
            f"Invalid model size: {model_size}."
            f" For model type {model_type}, please choose from {list(model_sizes.keys())}.")


def get_model_classes() -> Dict[str, Type[transformers.PreTrainedModel]]:
    """Returns a dictionary mapping class name strings to actual HuggingFace model classes.
    :return: A dictionary with class name strings as keys and HuggingFace model classes as values.
    """
    _MODEL_CLASSES = {
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": transformers.AutoModelForSeq2SeqLM,
        "AutoModelForMaskedLM": transformers.AutoModelForMaskedLM,
        "AutoModelForPreTraining": transformers.AutoModelForPreTraining,
        "AutoModelForTokenClassification": transformers.AutoModelForTokenClassification,
        "AutoModelForSequenceClassification": transformers.AutoModelForSequenceClassification,
        "AutoModelForQuestionAnswering": transformers.AutoModelForQuestionAnswering,
    }
    return _MODEL_CLASSES


def model_and_tokenizer(
        model_type: str,
        model_size: str,
        cls: Type[transformers.PreTrainedModel],
        **model_kwargs: Any
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    This function takes a model type, model size, a HuggingFace model class
    and additional model arguments, and returns a HuggingFace model and tokenizer.

    :param model_type: Model type. This will be converted into a HuggingFace model name
                  using the model_hf_name function.
    :param model_size: Model size. This will be converted into a HuggingFace model name
                  using the model_hf_name function.
    :param cls: Class of the model. This should be a subclass of transformers.PreTrainedModel.
    :param model_kwargs: Additional arguments for the model.
    :return: A tuple containing a HuggingFace model and tokenizer.
    """
    try:
        hf_model_name = model_hf_name(model_type, model_size)
    except Exception as e:
        raise ValueError(
            f"Invalid model type or size: {model_type}, {model_size}. Error: {str(e)}")

    # non-existing HuggingFace class
    if not issubclass(cls, transformers.PreTrainedModel) and not issubclass(cls, _BaseAutoModelClass):
        print(
            f"cls: {cls}, type: {type(cls)}, base classes: {cls.__bases__}, "
            f"is PreTrainedModel subclass: {issubclass(cls, transformers.PreTrainedModel)}"
        )
        raise ValueError(f"Invalid class: {cls}. "
                         f"It should be a subclass of transformers.PreTrainedModel.")
    try:
        m = cls.from_pretrained(hf_model_name, **model_kwargs)
    except Exception as e:
        raise ValueError(f"Could not load model from pretrained weights. Error: {str(e)}")

    # Non-GPT2 Model
    if isinstance(m, transformers.GPT2LMHeadModel):
        try:
            m.transformer.gradient_checkpointing_enable()
        except AttributeError:
            print("Warning: The model does not support gradient checkpointing.")

    try:
        tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)
    except Exception as e:
        raise ValueError(f"Could not load tokenizer from pretrained weights. Error: {str(e)}")

    # If your tokenizer is based on the model's name
    if m.name_or_path != tok.name_or_path:
        print(f"Warning: Model type ({m.name_or_path}) "
              f"and tokenizer type ({tok.name_or_path}) do not match.")

    # If your tokenizer is based on the model's pretrained name
    if m.config.model_type not in tok.__class__.__name__.lower():
        print(f"Warning: Model type ({m.config.model_type}) "
              f"and tokenizer type ({tok.__class__.__name__.lower()}) do not match.")

    if tok.pad_token_id is None:
        if cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            # Non-existing special tokens
            try:
                tok.add_special_tokens({'pad_token': '[PAD]'})
                tok.pad_token = '[PAD]'
            except KeyError:
                raise ValueError(
                    "The tokenizer does not have a '[PAD]' "
                    "token in its vocabulary.")
    return m, tok


def generate_spec(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function='gelu_new',
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        output_attentions=False,
        output_hidden_states=False,
        num_return_sequences=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_k=0, top_p=1.0,
        no_repeat_ngram_size=0):
    """Generates a dictionary of model specifications for GPT-2.  Plus a good reminded
    what each parameter does.


    The use_cache parameter controls whether the model should use caching during generation.
    Caching is a technique used to optimize generation by reusing the computed attention probabilities
    from previous steps. When use_cache is set to True, the model caches the attention probabilities,
    allowing for faster generation of subsequent tokens.

    i.e the model avoids recalculating the attention weights for already generated tokens,
    resulting in improved generation speed. This is particularly useful in scenarios where generation
    is performed incrementally, such as generating tokens one by one in a loop.

    However, it's important to note that enabling caching may consume more memory as the cache grows
    with each generated token. Therefore, it's recommended to enable caching when generation is performed
    incrementally and disable it when generating a complete sequence at once
    or when memory usage is a concern.


    BOS/EOS/PAD

    In GPT (Generative Pre-trained Transformer) models,
    the special tokens like BOS (Beginning of Sentence),
    EOS (End of Sentence), and PAD (Padding) tokens play specific roles in the input and output sequences.

    BOS (Beginning of Sentence) Token:
        The BOS token marks the beginning of the input sequence.
        It is used to indicate the start of a generated text or the beginning of an input text for tasks
        like language modeling or text generation.
        The BOS token helps the model understand the context and generate coherent text.

    EOS (End of Sentence) Token:
        The EOS token marks the end of the input or generated sequence.
        It is used to indicate the completion of a generated text or the end of an input text.
         The EOS token helps the model determine where to stop generating text and provides
         a clear boundary for the generated sequence.

    PAD (Padding) Token:
        The PAD token is used for padding sequences to make them of equal length.
        It is typically added to sequences that are shorter than the maximum sequence length to
        ensure consistent input sizes during training or inference.
        The PAD token does not contribute to the actual content of the sequence but is necessary
        for batch processing and maintaining consistent tensor shapes.
        These special tokens are often represented by specific token IDs within the vocabulary of the model.

    The BOS token, EOS token, and PAD token IDs are used to mark the corresponding positions in
    the input and output sequences, allowing the model to understand and process
    the sequences appropriately.

    :param vocab_size: (int): Vocabulary size of the GPT-2 model. Defaults to 50257.
    :param n_positions: n_positions (int, optional, defaults to 1024).
                        The maximum sequence length that this model might ever be used with.
                        Typically, set this to something large just in case (e.g., 512 or 1024 or 2048).
    :param n_embd: (int, optional, defaults to 768) — Dimensionality of the embeddings and hidden states.
    :param n_layer:
    :param n_head:
    :param n_inner:
    :param activation_function:  Activation function, to be selected in
                                   the list ["relu", "silu", "gelu", "tanh", "gelu_new"]. Default glue
    :param resid_pdrop:
    :param embd_pdrop: The dropout ratio for the embeddings. Default 0.1
    :param attn_pdrop: The dropout ratio for the attention.  Default 0.1
    :param layer_norm_epsilon:
    :param initializer_range:
    :param summary_type:
    :param summary_use_proj:
    :param summary_activation:
    :param summary_proj_to_labels:
    :param summary_first_dropout:
    :param scale_attn_weights:
    :param use_cache:  Whether the model should return the last key/values attentions.
    :param scale_attn_by_inverse_layer_idx:
    :param reorder_and_upcast_attn:
    :param bos_token_id: int, beginning of sentence token id. Defaults to 50256.
    :param eos_token_id: int  end of sentence Default 50256
    :param pad_token_id: int  pad Default None
    :param output_attentions:
    :param output_hidden_states:
    :param num_return_sequences:
    :param repetition_penalty:
    :param temperature:
    :param top_k:
    :param top_p:
    :param no_repeat_ngram_size:
    :return:
    """

    """
    Generates a dictionary of model specifications for GPT-2
    """
    return {
        'vocab_size': vocab_size,
        'n_positions': n_positions,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_inner': n_inner,
        'activation_function': activation_function,
        'resid_pdrop': resid_pdrop,
        'embd_pdrop': embd_pdrop,
        'attn_pdrop': attn_pdrop,
        'layer_norm_epsilon': layer_norm_epsilon,
        'initializer_range': initializer_range,
        'summary_type': summary_type,
        'summary_use_proj': summary_use_proj,
        'summary_activation': summary_activation,
        'summary_proj_to_labels': summary_proj_to_labels,
        'summary_first_dropout': summary_first_dropout,
        'scale_attn_weights': scale_attn_weights,
        'use_cache': use_cache,
        'scale_attn_by_inverse_layer_idx': scale_attn_by_inverse_layer_idx,
        'reorder_and_upcast_attn': reorder_and_upcast_attn,
        'bos_token_id': bos_token_id,
        'eos_token_id': eos_token_id,
        'pad_token_id': pad_token_id,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'num_return_sequences': num_return_sequences,
        'repetition_penalty': repetition_penalty,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'no_repeat_ngram_size': no_repeat_ngram_size
    }


def data_collator(features: List[Any]) -> Dict[str, Any]:
    """
    :param features:
    :return:
    """
    if not isinstance(features[0], collections.Mapping):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label_col_name = "label"
    elif "label_ids" in first and first["label_ids"] is not None:
        label_col_name = "label_ids"
    elif "labels" in first and first["labels"] is not None:
        label_col_name = "labels"
    else:
        label_col_name = None

    if label_col_name is not None:
        if isinstance(first[label_col_name], torch.Tensor):
            dtype = torch.int64 if first[label_col_name].dtype.is_integer else torch.float32
        elif isinstance(first[label_col_name], np.ndarray) or isinstance(first[label_col_name], np.generic):
            dtype = torch.int64 if np.issubdtype(first[label_col_name].dtype, np.integer) else torch.float32
        elif isinstance(first[label_col_name], (tuple, list)):
            dtype = torch.int64 if isinstance(first[label_col_name][0], int) else torch.float32
        else:
            dtype = torch.int64 if isinstance(first[label_col_name], int) else torch.float32
        batch["labels"] = torch.tensor([f[label_col_name] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def whole_word_masking_data_collator(
        features, mask_token_id, wwm_probability=0.2, ignore_index=-100):
    """Whole word masking

    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples)

    :param ignore_index:
    :param mask_token_id:
    :param wwm_probability:
    :param features:
    :return:
    """
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [ignore_index] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = mask_token_id
        feature["labels"] = new_labels

    return data_collator(features)
