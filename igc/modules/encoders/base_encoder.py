import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseEncoder:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        :param model: (PreTrainedModel): The pre-trained model.
        :param tokenizer: (PreTrainedTokenizer): The pre-trained tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.encoder_model = model.transformer
        self.model.config.is_decoder = False
        self.model.resize_token_embeddings(len(tokenizer))

        # subtracting 1 to exclude padding index
        input_shape = self.encoder_model.wpe.weight.shape
        self.emb_shape = (input_shape[0] - 1, input_shape[1])

        self.cache = {}

    def encode(self,
               observation: str,
               max_chunk_length: int = 1023,
               max_seq_length=512
               ) -> torch.Tensor:
        """Encode the given observation into embeddings.

        :param max_seq_length:
        :param observation: The input observation.
        :param max_chunk_length: The maximum length of each output chunk.
        :return: torch.Tensor: The encoded embeddings with shape torch.Size([batch_size, seq_len, 768])
        """
        if observation in self.cache:
            return self.cache[observation]

        input_tensor = self.tokenizer.batch_encode_plus(
            [observation],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        with torch.no_grad():
            out = self.encoder_model(input_tensor).last_hidden_state
            attention_mask = out.attention_mask
            last_hidden_state = out.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        self.cache[observation] = mean_embeddings
        return mean_embeddings
