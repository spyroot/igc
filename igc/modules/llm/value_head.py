"""A value head that maps backbone hidden states to per-token value logits.

Attached on top of a causal-LM backbone for reward-model / RL fine-tuning:
``num_classes=1`` gives the trl-style scalar value head, larger values give a
classification head over the same hidden states. Backbone-agnostic — the input
width comes from the model config (``word_embed_proj_dim`` for OPT-style
models that project embeddings, else ``hidden_size``; GPT-2's ``n_embd`` is
aliased to ``hidden_size`` by transformers), and the forward pass casts inputs
to the head's weight dtype so a bf16 backbone feeds a fp32 head safely.

Author:
Mus mbayramo@stanford.edu
"""

import torch.nn as nn


class ValueHead(nn.Module):
    """Dropout + linear projection from hidden states to ``num_classes`` logits.

    :param config: the backbone's config; provides the hidden width and,
        optionally, ``summary_dropout_prob``.
    :param num_classes: output width (1 for a scalar value head).
    :param kwargs: ``summary_dropout_prob`` fallback when the config lacks it
        (default 0.1; 0 disables dropout entirely).
    """

    def __init__(self, config, num_classes, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # OPT-style models project embeddings to word_embed_proj_dim; every other
        # backbone exposes hidden_size (GPT-2 aliases n_embd to it).
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        """Project hidden states to value logits.

        :param hidden_states: ``(batch, seq, hidden)`` backbone activations.
        :return: ``(batch, seq, num_classes)`` logits in the head's dtype.
        """
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


# Author: Mus mbayramo@stanford.edu
