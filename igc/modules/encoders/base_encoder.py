import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


class Conv1DLatent(nn.Module):
    def __init__(self, input_dim, kernel_size=2, padding=1):
        """

        :param input_dim:
        """
        super().__init__()
        #
        self.cn1 = nn.Conv1d(input_dim[1], input_dim[0], kernel_size=kernel_size, padding=padding)
        self.cn2 = nn.Conv1d(input_dim[0], 1, kernel_size=kernel_size, padding=padding)

    def forward(self, last_hidden_state):
        """
        :param last_hidden_state:
        :return: (batch_size, 768)
        """
        # (batch_size, seq_len, 768) -> (batch_size, 768, seq_len)
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        emb = self.cn1(last_hidden_state)
        emb = F.relu(emb)
        emb = self.cn2(emb)
        emb = emb.squeeze(1)
        return emb


class BaseEncoder:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device=None):
        """
        :param model: (PreTrainedModel): The pre-trained model.
        :param tokenizer: (PreTrainedTokenizer): The pre-trained tokenizer.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._encoder_model = model.transformer
        self._encoder_model.eval()
        model.transformer.config.is_decoder = False
        self._model.resize_token_embeddings(len(tokenizer))

        # subtracting 1 to exclude padding index
        input_shape = self._encoder_model.wpe.weight.shape
        self.emb_shape = (input_shape[0] - 1, input_shape[1])
        self._1d_conv = Conv1DLatent(input_dim=input_shape)
        self.cache = {}
        self.device = device
        self._1d_conv.to(self.device)

    def batchify(self, observation, max_chunk_length=1023):
        """
        :param max_chunk_length:
        :param observation:
        :return:
        """
        input_tensors = []
        input_masks = []

        encoded_input = self._tokenizer(
            observation,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors='pt',
            return_attention_mask=True,
            verbose=False
        )

        input_ids = encoded_input.input_ids[0]
        attention_mask = encoded_input.attention_mask[0]

        # Split the input into chunks
        for i in range(0, len(input_ids), max_chunk_length):
            chunk_input_ids = input_ids[i:i + max_chunk_length]
            chunk_attention_mask = attention_mask[i:i + max_chunk_length]

            input_tensors.append(chunk_input_ids.unsqueeze(0))
            input_masks.append(chunk_attention_mask.unsqueeze(0))

        # pad the last chunk if needed
        if len(input_tensors[-1][0]) < max_chunk_length:
            last_chunk_size = len(input_tensors[-1][0])
            padding_length = max_chunk_length - last_chunk_size

            padded_chunk_input_ids = torch.cat([
                input_tensors[-1][0],
                torch.zeros(padding_length, dtype=torch.long)
            ])
            padded_chunk_attention_mask = torch.cat([
                input_masks[-1][0],
                torch.zeros(padding_length, dtype=torch.long)
            ])

            input_tensors[-1] = padded_chunk_input_ids.unsqueeze(0)
            input_masks[-1] = padded_chunk_attention_mask.unsqueeze(0)

        stacked_tensors = (
            torch.cat(input_tensors, dim=0),
            torch.cat(input_masks, dim=0)
        )

        return stacked_tensors

    def encode(self,
               observation: str,
               max_chunk_length: int = 1023,
               ) -> torch.Tensor:
        """Encode the given observation into embeddings.

        :param max_chunk_length:
        :param observation: The input observation.
        :param max_chunk_length: The maximum length of each input chunk to encoder.
        :return: torch.Tensor: The encoded embeddings with shape torch.Size([batch_size, seq_len, 768])
        """
        if observation in self.cache:
            return self.cache[observation]

        input_tensor, input_mask = self.batchify(observation, max_chunk_length=max_chunk_length)

        input_tensor = input_tensor.to(self.device)
        input_mask = input_mask.to(self.device)

        with torch.no_grad():
            hidden_states = [
                self._encoder_model(input_ids=chunk_input_tensor,
                                    attention_mask=chunk_input_mask, use_cache=True).last_hidden_state.unsqueeze(0)
                for chunk_input_tensor, chunk_input_mask, in zip(input_tensor, input_mask)]

            combined_last_hidden_state = torch.cat(hidden_states, dim=0)
            combined_last_hidden_state = combined_last_hidden_state.to(self.device)
            emb = self._1d_conv(combined_last_hidden_state)
            emb = torch.mean(emb, dim=0)

        emb = emb.detach().cpu()
        self.cache[observation] = emb
        return emb
