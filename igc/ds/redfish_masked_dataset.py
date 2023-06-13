from abc import ABC
from enum import auto, Enum
from typing import Optional, Any, Dict, Union, List

import torch
from transformers import PreTrainedTokenizer
from .redfish_dataset import JSONDataset


class MaskingOption(Enum):
    ALLOWED_VALUE = auto()
    ODATA_ID = auto()
    TARGET = auto()


class MaskedJSONDataset(JSONDataset, ABC):

    def __init__(self,
                 dataset_dir: Optional[str] = "datasets",
                 default_tokenize: Optional[str] = "gpt2",
                 max_len: Optional[int] = 1024,
                 overlap: Optional[int] = 256,
                 verbose: Optional[bool] = False,
                 recreate_dataset: Optional[bool] = False,
                 tokenizer: Optional[Any] = None,
                 transform=None,
                 target_transform=None,
                 is_force_download=False,
                 do_consistency_check=True,
                 raw_json_directory_path: Optional[str] = "~/.json_responses",
                 ):
        """

        :param dataset_dir:
        :param default_tokenize:
        :param max_len:
        :param overlap:
        :param verbose:
        :param recreate_dataset:
        :param tokenizer:
        :param transform:
        :param target_transform:
        :param is_force_download:
        :param do_consistency_check:
        :param raw_json_directory_path:
        """

        self.token_to_mask = [
            ("odata.id", ["\"},", "\"}"]),
        ]

        _object_close = [[92], [92, 13]]
        _object_open = [[90]]
        _array_close = [[60], [60, 11]]
        _array_open = [[58], [60, 11]]

        self._masking_option = {
            MaskingOption.ODATA_ID: (50257, _array_close),
            MaskingOption.ALLOWED_VALUE: (50258, _array_close),
            MaskingOption.TARGET: (16793, _object_close),
        }

        self._current_token_id_mask = [
            (50257, _object_close),
        ]

        super().__init__(
            dataset_dir=dataset_dir,
            default_tokenize=default_tokenize,
            max_len=max_len,
            overlap=overlap,
            verbose=verbose,
            recreate_dataset=recreate_dataset,
            tokenizer=tokenizer,
            transform=transform,
            target_transform=target_transform,
            is_force_download=is_force_download,
            do_consistency_check=do_consistency_check,
            raw_json_directory_path=raw_json_directory_path,
        )
        self._cache = [None] * len(self._data["train_data"])

    def mask_allowed_value(self):
        """
        :return:
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.ALLOWED_VALUE]]

    def mask_odata_id(self):
        """
        :return:
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.ODATA_ID]]

    def mask_targets(self):
        """
        :return:
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.TARGET]]

    @staticmethod
    def mask_json_kv_span(
        data: Dict[str, torch.Tensor],
        tokenizer: PreTrainedTokenizer,
        target_key: Union[str, int],
        end_toks: Union[str, List[Union[str, int]]] = "\"},",
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
            data['input_ids'], data['attention_mask'],
            tokenizer, target_key, end_toks, return_original
        )

    def apply_mask_tensor_json_kv_span(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_key: Union[str, int],
        end_toks: Union[str, List[Union[str, int]]] = "\"},",
        return_original: bool = False,
    ) -> torch.Tensor:
        return MaskedJSONDataset.mask_tensor_json_kv_span(
            input_ids, attention_mask, self.tokenizer,
            target_key, end_toks, return_original=return_original
        )

    @staticmethod
    def mask_tensor_ids_json_kv_span(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Union[int, List[int]],
        end_toks_ids: Union[int, List[int], List[List[int]]] = 92,
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
        :param target_ids: The JSON key to mask the value of
        :param end_toks_ids: The termination sequence to identify the end of the masked value
        :return: The computed new attention mask
        """
        attention_mask = attention_mask.clone()
        attention_mask[:, :] = 0

        if isinstance(end_toks_ids, int):
            end_toks_ids = [[end_toks_ids]]

        if isinstance(target_ids, int):
            target_ids = [target_ids]

        end_toks_lens = [len(end_toks) for end_toks in end_toks_ids]
        target_len = len(target_ids)

        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_ids:
                # print("Found begin")
                attention_mask[0, i:i + target_len] = 1
                # corresponding end tokens after the target key
                j = i + target_len
                while j < input_ids.shape[1]:
                    current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
                    # current_token = input_ids[0, j:j + end_toks_len].tolist()
                    if any(current_token[-len(et):] == et for et in end_toks_ids):
                        attention_mask[0, j:j + max(end_toks_lens)] = 1
                        break
                    attention_mask[0, j] = 1
                    j += 1
                    i += 1

        if return_original:
            if attention_mask.sum() == 0:
                attention_mask[:, :] = 1

        return attention_mask

    @staticmethod
    def mask_tensor_json_kv_span(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        target_key: Union[str, int],
        end_toks: Union[str, List[Union[str, int]]] = "\"},",
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
        attention_mask = attention_mask.clone()
        attention_mask[:, :] = 0

        if isinstance(end_toks, int):
            end_toks = [end_toks]

        target_tokens = tokenizer(target_key, add_special_tokens=True)['input_ids']
        end_toks = [tokenizer(end_tok, add_special_tokens=True)['input_ids'] for end_tok in end_toks]
        end_toks_lens = [len(end_toks) for end_toks in end_toks]
        target_len = len(target_tokens)

        print(f"Start searching {target_key} tokens {target_tokens}")
        print(f"End tokens {end_toks}")
        print("special", tokenizer.additional_special_tokens)
        print("special", tokenizer.additional_special_tokens_ids[3])
        print("special vob size", tokenizer.vocab_size)

        # print(f"End tokens shape {end_toks[0].shape}")
        # print(f"end_toks {input_ids}")

        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_tokens:
                # print("Found begin")
                attention_mask[0, i:i + target_len] = 1
                # corresponding end tokens after the target key
                j = i + target_len
                while j < input_ids.shape[1]:
                    current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
                    # print("Found end")
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

        original_mask = attention_mask = data['attention_mask']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        mask_adjusted = False
        for token, end_token in self._current_token_id_mask:
            mask_adjusted = True
            new_mask = self.mask_tensor_ids_json_kv_span(
                input_ids, attention_mask, target_ids=token, end_toks_ids=end_token
            ).squeeze(0)

            if torch.all(new_mask == 0):
                print("Attention mask contains all zeros")
            else:
                print("Attention mask contains non-zero values")
            data["attention_mask"] = new_mask

        print(f"return data and {mask_adjusted} original shape {original_mask.shape} new mask shape {attention_mask.shape}")

        data['file_idx'] = torch.tensor(idx)
        # self._cache[idx] = data
        return data
