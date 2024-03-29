import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class RestBaseEncoder:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device=None):
        """
        :param model: (PreTrainedModel): The pre-trained model.
        :param tokenizer: (PreTrainedTokenizer): The pre-trained tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.encoder_model = model.transformer
        self.model.config.is_decoder = False
        self.model.resize_token_embeddings(len(tokenizer))
        self.device = device

        # subtracting 1 to exclude padding index
        input_shape = self.encoder_model.wpe.weight.shape
        self.emb_shape = (input_shape[0] - 1, input_shape[1])
        self.cache = {}

    # def encode(self, observation: str) -> torch.Tensor:
    #     """Encode the given observation into embeddings.
    #     :param observation:
    #     :return: torch.Tensor: The encoded embeddings with shape torch.Size([batch_size, seq_len, 768])
    #
    #     """
    #     tokens = self.tokenizer.encode(observation, add_special_tokens=True)
    #
    #     # truncate or pad the tokens
    #     # to the maximum sequence length TODO
    #     # max_length = self.tokenizer.model_max_length - 2
    #     # tokens = self.tokenizer.truncate_sequences(tokens, max_length)
    #
    #     input_tensor = torch.tensor([tokens])
    #
    #     with torch.no_grad():
    #         embeddings = self.encoder_model(input_tensor)
    #
    #     return embeddings.last_hidden_state
    #
    def encode(self, observation: str, max_chunk_length: int = 1023) -> torch.Tensor:
        """Encode the given observation into embeddings.
        :param observation: The input observation.
        :param max_chunk_length: The maximum length of each output chunk.
        :return: torch.Tensor: The encoded embeddings with shape torch.Size([batch_size, seq_len, 768])
        """
        if observation in self.cache:
            return self.cache[observation]

        start_time = time.time()

        tokens = self.tokenizer.encode(
            observation, truncation=True,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_chunk_length
        )

        embeddings = []
        # Process input in chunks
        for i in range(0, len(tokens), max_chunk_length):
            chunk_tokens = tokens[i:i + max_chunk_length]
            input_tensor = torch.tensor([chunk_tokens])
            with torch.no_grad():
                chunk_embeddings = self.encoder_model(input_tensor).last_hidden_state
            embeddings.append(chunk_embeddings)

        elapsed_time = time.time() - start_time
        print(f"Encoder elapsed time: {elapsed_time} seconds")

        # concat
        embeddings = torch.cat(embeddings, dim=1).squeeze(0)
        self.cache[observation] = embeddings
        return embeddings
