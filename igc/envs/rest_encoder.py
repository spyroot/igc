import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class RestBaseEncoder:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        :param model: (PreTrainedModel): The pre-trained model.
        :param tokenizer: (PreTrainedTokenizer): The pre-trained tokenizer.
        """
        self.model = model

        self.encoder_model = model.transformer
        self.tokenizer = tokenizer
        self.model.config.is_decoder = False
        self.model.resize_token_embeddings(len(tokenizer))

        # subtracting 1 to exclude padding index
        input_shape = self.encoder_model.wpe.weight.shape
        self.emb_shape = (input_shape[0] - 1, input_shape[1])

    def encode(self, observation: str) -> torch.Tensor:
        """Encode the given observation into embeddings.
        :param observation:
        :return: torch.Tensor: The encoded embeddings with shape torch.Size([batch_size, seq_len, 768])

        """
        tokens = self.tokenizer.encode(observation, add_special_tokens=True)
        input_tensor = torch.tensor([tokens])

        with torch.no_grad():
            embeddings = self.encoder_model(input_tensor)

        return embeddings.last_hidden_state
