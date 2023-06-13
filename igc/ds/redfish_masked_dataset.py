from abc import ABC
from typing import Optional, Any, Dict, Union, List

import torch
from transformers import PreTrainedTokenizer

from .redfish_dataset import JSONDataset


class MaskedJSONDataset(JSONDataset, ABC):

    def __init__(self,
                 raw_json_directory_path: str,
                 default_tokenize: Optional[str] = "gpt2-xl",
                 max_len: Optional[int] = 1024,
                 overlap: Optional[int] = 256,
                 dataset_dir: Optional[str] = "datasets",
                 verbose: Optional[bool] = False,
                 recreate_dataset: Optional[bool] = False,
                 tokenizer: Optional[Any] = None,
                 transform=None,
                 target_transform=None,
                 is_force_download=False,
                 do_consistency_check=True):
        """
        :param raw_json_directory_path:
        :param default_tokenize:
        :param max_len:
        :param overlap:
        :param dataset_dir:
        :param verbose:
        :param recreate_dataset:
        :param tokenizer:
        """
        self.token_to_mask = [
            ("odata.id", ["\"},", "\"}"]),
        ]
        super().__init__(
            raw_json_directory_path,
            default_tokenize,
            max_len,
            overlap,
            dataset_dir,
            verbose,
            recreate_dataset,
            tokenizer,
            transform,
            target_transform,
            is_force_download,
            do_consistency_check
        )
        self._cache = [None] * len(self._data["train_data"])

    @staticmethod
    def mask_json_kv_span(
            data: Dict[str, torch.Tensor],
            tokenizer: PreTrainedTokenizer,
            target_key: str,
            end_toks: Union[str, List[str]] = "\"},",
            return_original: bool = False,
    ) -> torch.Tensor:

        """
        It computes Masks for specific key and value in json structure
        It mask both key and value.

        Shapes attention  [1, x]
        Shapes intput_idx [1, x]

        Usage:
            target_key = "@odata.id"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            json_lines = json.dumps(j_data)
            attention_mask = mask_specific_key_and_value(json_lines,
            target_key, tokenizer=tokenizer, debug=True)

        :param return_original: Boolean indicating whether to return the original attention mask
                                if the target key is not found
        :param tokenizer: The tokenizer to use for tokenization
        :param data: The input data containing 'attention_mask', 'input_ids', etc.
        :param target_key: The JSON key to mask the value of
        :param end_toks: The termination sequence to identify the end of the masked value
        :return: The computed new attention mask
        """

        return MaskedJSONDataset.mask_tensor_json_kv_span(
            data['input_ids'], data['attention_mask'], tokenizer, target_key, end_toks, return_original)

        # target_tokens = tokenizer(target_key)['input_ids']
        # end_toks = [tokenizer(end_tok)['input_ids'] for end_tok in end_toks]
        # end_toks_lens = [len(end_toks) for end_toks in end_toks]
        #
        # target_len = len(target_tokens)
        # end_toks_len = len(end_toks)
        #
        # for i in range(input_ids.shape[1] - target_len + 1):
        #     if input_ids[0, i:i + target_len].tolist() == target_tokens:
        #         attention_mask[0, i:i + target_len] = 1
        #         print("FOUND BEING")
        #         # corresponding end tokens after the target key
        #         j = i + target_len
        #         while j < input_ids.shape[1]:
        #             current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
        #             # current_token = input_ids[0, j:j + end_toks_len].tolist()
        #             if any(current_token[-len(et):] == et for et in end_toks):
        #                 attention_mask[0, j:j + max(end_toks_lens)] = 1
        #                 print("BREAKING FOUND")
        #                 break
        #             attention_mask[0, j] = 1
        #             j += 1
        #             i += 1
        #
        # if return_original:
        #     if attention_mask.sum() == 0:
        #         attention_mask[:, :] = 1
        #
        # return attention_mask

    @staticmethod
    def mask_tensor_json_kv_span(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            tokenizer: PreTrainedTokenizer,
            target_key: str,
            end_toks: Union[str, List[str]] = "\"},",
            return_original: bool = False,
    ) -> torch.Tensor:

        """
        It computes Masks for specific key and value in json structure
        It mask both key and value.

        Shapes attention  [1, x]
        Shapes intput_idx [1, x]

        Usage:
            target_key = "@odata.id"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            json_lines = json.dumps(j_data)
            attention_mask = mask_specific_key_and_value(json_lines,
            target_key, tokenizer=tokenizer, debug=True)

        :param attention_mask:
        :param input_ids:
        :param return_original: Boolean indicating whether to return the original attention mask
                                if the target key is not found
        :param tokenizer: The tokenizer to use for tokenization
        :param target_key: The JSON key to mask the value of
        :param end_toks: The termination sequence to identify the end of the masked value
        :return: The computed new attention mask
        """
        print("Original mask shape: ", attention_mask.shape)
        attention_mask = attention_mask.clone()
        attention_mask[:, :] = 0

        print("Original target_key: ", target_key)
        print("Original end_toks: ", end_toks)

        if isinstance(end_toks, str):
            end_toks = [end_toks]

        target_tokens = tokenizer(target_key)['input_ids']
        end_toks = [tokenizer(end_tok)['input_ids'] for end_tok in end_toks]
        end_toks_lens = [len(end_toks) for end_toks in end_toks]

        target_len = len(target_tokens)

        torch.set_printoptions(threshold=10000)  # Set the threshold value as per your requirement

        print(f"Start searching {target_key} tokens {target_tokens}")
        print(f"Start in", input_ids)

        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_tokens:
                attention_mask[0, i:i + target_len] = 1
                # corresponding end tokens after the target key
                j = i + target_len
                while j < input_ids.shape[1]:
                    current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
                    # current_token = input_ids[0, j:j + end_toks_len].tolist()
                    if any(current_token[-len(et):] == et for et in end_toks):
                        attention_mask[0, j:j + max(end_toks_lens)] = 1
                        break
                    attention_mask[0, j] = 1
                    j += 1
                    i += 1

        if return_original:
            if attention_mask.sum() == 0:
                attention_mask[:, :] = 1

        return attention_mask

    def file_path(self, idx) -> str:
        """Retrieve the file path associated with the given index.
        :param idx: The index of the item.
        :return: path to json file.
        """
        return self.__getitem__(idx)['file_path']

    def labels(self, idx):
        """Retrieve the labels associated with the given index.
         :param idx: The index of the item.
         :return: The labels.
        """
        return self.__getitem__(idx)['labels']

    def original_attention_mask(self, idx):
        """Retrieve the original attention mask associated with the given index.

        :param idx:  The index of the item.
        :return: The original attention mask.
        """
        data = self._data["train_data"][idx]
        return data['attention_mask']

    def __len__(self):
        """Return length of dataset"""
        return len(self._data["train_data"])

    def __getitem__(self, idx):
        """Get item from json dataset and mask what we
        need and return it as a dict

        :param idx:
        :return:
        """
        # if self._cache[idx] is not None:
        #     return self._cache[idx]

        data = self._data["train_data"][idx]

        # modified_data = data.copy()
        print("data input_ids", data['input_ids'].shape)
        print("data attention_mask", data['input_ids'].shape)
        print("data mask", data['attention_mask'])

        input_ids = data['input_ids'].unsqueeze(0)
        attention_mask = data['attention_mask'].unsqueeze(0)

        print("data attention_mask", data['input_ids'].shape)
        print("data mask un suizzed", attention_mask)

        for token, end_token in self.token_to_mask:
            new_mask = MaskedJSONDataset.mask_tensor_json_kv_span(
                input_ids, attention_mask, self.tokenizer, token,  end_toks=end_token
            ).squeeze(0)

            print("new mask", new_mask)
            if torch.all(new_mask == 0):
                print("Attention mask contains all zeros")
            else:
                print("Attention mask contains non-zero values")
            data["attention_mask"] = new_mask

        data['file_idx'] = torch.tensor(idx)
        # self._cache[idx] = data
        return data
