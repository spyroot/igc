import warnings
from abc import ABC
from enum import auto, Enum
from typing import Optional, Any, Dict, Union, List

import torch
from transformers import PreTrainedTokenizer
from .redfish_dataset import JSONDataset


class MaskingOption(Enum):
    """
    """
    ODATA_ID = auto()
    TARGET = auto()
    TARGET_KEY = auto()
    JSON_OBJECT = auto()
    JSON_ARRAY = auto()
    ALLOWED_VALUE = auto()
    KEY_VALUE_PAIR = auto()
    MASK_API_PREFIX = auto()
    MASK_NEW_TOKENS = auto()


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

        self._mask_new_tokens = False
        self._dont_mask = True

        # tokens we use for masking
        special_tokens = self.get_special_tokens()

        redfish_odata = special_tokens["@odata.id"].tolist()
        redfish_odata = [value for sublist in redfish_odata for value in sublist]

        redfish_allow = special_tokens["Redfish.AllowableValues"].tolist()
        redfish_allow = [value for sublist in redfish_allow for value in sublist]

        redfish_target = special_tokens["target"].tolist()
        redfish_target = [value for sublist in redfish_target for value in sublist]

        redfish_target_key = special_tokens["\"target\""]
        if isinstance(redfish_target_key, int):
            redfish_target_key = [redfish_target_key]
        elif isinstance(redfish_target_key, list):
            redfish_target_key = [value for sublist in redfish_target_key for value in sublist]

        array_begin = special_tokens["["].item()
        array_end = special_tokens["]"].item()
        array_end_coma = special_tokens["],"].tolist()
        object_begging = special_tokens["{"].item()
        object_end = special_tokens["}"].item()
        value_terminate = special_tokens["\""].item()
        comma = special_tokens[","].item()

        # self._object_close = [[object_end],
        #                       [object_end, comma],
        #                       [value_terminate, object_end],
        #                       [value_terminate, object_end, comma]]

        self._object_close = [[object_end],
                              [value_terminate, object_end],
                              [value_terminate, object_end, comma]]

        # self._object_close = [[object_end],
        #                       [object_end, comma]]

        self._object_open = [[90]]
        self._array_close = [[60], [60, 11]]
        self._array_open = [[58], [60, 11]]

        api_prefix = "/redfish/v1/"
        api_prefix_ids = self.tokenizer.encode(api_prefix)

        self._masking_option = {
            MaskingOption.ODATA_ID: (redfish_odata, self._object_close),
            MaskingOption.ALLOWED_VALUE: (redfish_allow, self._array_close),
            MaskingOption.TARGET: (redfish_target, self._object_close),
            MaskingOption.TARGET_KEY: (redfish_target_key, self._object_close),
            MaskingOption.JSON_OBJECT: ([object_begging], [[object_end]]),
            MaskingOption.JSON_ARRAY: ([array_begin], [[array_end]]),
            # MaskingOption.KEY_VALUE_PAIR: ([array_begin], [[array_end]]),
            MaskingOption.MASK_API_PREFIX: ([api_prefix_ids], [[self._object_close]])
        }

        # current mask
        self._current_token_id_mask = [
            self._masking_option[MaskingOption.ODATA_ID],
        ]
        self._cache = [None] * len(self._data["train_data"])

    def enable_masking(self):
        self._dont_mask = False

    def disable_masking(self):
        self._dont_mask = True

    def mask_allowed_value(self):
        """
        Mask the allowed value in the JSON dataset.
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.ALLOWED_VALUE]]

    def mask_odata_id(self):
        """
        Mask the odata.id value in the JSON dataset.
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.ODATA_ID]]

    def mask_targets(self):
        """
        Mask the target value in the JSON dataset.
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.TARGET]]

    def mask_targets_key(self):
        """
        Mask the target key in the JSON dataset.
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.TARGET_KEY]]

    def mask_objects(self):
        """
        Mask the object scope in the JSON dataset.
        :return:
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.JSON_OBJECT]]

    def mask_arrays(self):
        """
        Mask the array in the JSON dataset.
        :return:
        """
        self._current_token_id_mask = [self._masking_option[MaskingOption.JSON_ARRAY]]

    def mask_new_tokens(self, is_enabled: bool):
        """
        Mask the array in the JSON dataset.
        :return:
        """
        del self._cache
        self._cache = [None] * len(self._data["train_data"])
        self._mask_new_tokens = is_enabled

    def mask_api_prefix(self):
        self._current_token_id_mask = [self._masking_option[MaskingOption.MASK_API_PREFIX]]

    @staticmethod
    def mask_json_kv_span(
        data: Dict[str, torch.Tensor],
        tokenizer: PreTrainedTokenizer,
        target_key: Union[str],
        end_toks: Union[str, List[Union[str]]] = "\"},",
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
        target_key: Union[str],
        end_toks: Union[str, List[Union[str]]] = "\"},",
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
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask[:, :] = 0

        if isinstance(end_toks_ids, int):
            end_toks_ids = [[end_toks_ids]]

        if isinstance(target_ids, int):
            target_ids = [target_ids]

        end_toks_lens = [len(end_toks) for end_toks in end_toks_ids]
        target_len = len(target_ids)

        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_ids:
                attention_mask[0, i:i + target_len] = 1
                j = i + target_len

                found_end_toks = False
                while j < input_ids.shape[1]:
                    current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
                    if any(current_token[-len(et):] == et for et in end_toks_ids):
                        attention_mask[0, j:j + max(end_toks_lens)] = 1
                        found_end_toks = True
                        break
                    attention_mask[0, j] = 1
                    j += 1
                    i += 1

                if not found_end_toks:
                    attention_mask[0, i + target_len:] = 0

        if return_original:
            if attention_mask.sum() == 0:
                attention_mask[:, :] = 1

        return attention_mask

    def mask_all_new_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a mask that sets 1 for all positions in the
        input IDs that correspond to new token IDs.

        :param input_ids: Tensor of shape (batch_size, sequence_length)
                          containing the input token IDs.
        :param attention_mask: Tensor of shape (batch_size, sequence_length)
                          containing the attention mask.
        :return: Tensor of shape (batch_size, sequence_length) representing the computed attention mask.
        """

        new_tokenizer_size = len(self.tokenizer)
        old_vocab_size = self.tokenizer.vocab_size

        attention_mask = attention_mask.clone()
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask[:, :] = 0

        new_token_ids = torch.arange(old_vocab_size, new_tokenizer_size)
        new_token_mask = torch.isin(input_ids, new_token_ids)
        attention_mask[new_token_mask] = 1
        return attention_mask

    @staticmethod
    def tensor_masking_span(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Union[int, List[int]],
        end_toks_ids: Union[int, List[int], List[List[int]]] = 92,
        return_original: bool = False,
    ) -> torch.Tensor:

        """
        This vectorized version. I need test more.

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
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask[:, :] = 0

        if isinstance(end_toks_ids, int):
            end_toks_ids = [[end_toks_ids]]

        if isinstance(target_ids, int):
            target_ids = [target_ids]

        end_toks_lens = [len(end_toks) for end_toks in end_toks_ids]
        target_len = len(target_ids)

        target_tokens = torch.tensor(target_ids)
        indices = torch.where(torch.all(input_ids == target_tokens, dim=1))[0]
        if indices.numel() > 0:
            start_index = indices[0]
            end_index = start_index + len(target_tokens)
            attention_mask[0, start_index:end_index] = 1
            next_index = end_index + 1

        selected_values = input_ids[0, attention_mask[0] == 1]
        indices = torch.where((input_ids[0] == selected_values[:, None]).all(dim=0))[0]

        end_tokens = torch.tensor(end_toks_ids)
        indices = torch.where((input_ids[0] == end_tokens[:, None]).any(dim=0))[0]

        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_ids:
                attention_mask[0, i:i + target_len] = 1
                j = i + target_len

                found_end_toks = False
                while j < input_ids.shape[1]:
                    current_token = input_ids[0, j:j + max(end_toks_lens)].tolist()
                    if any(current_token[-len(et):] == et for et in end_toks_ids):
                        attention_mask[0, j:j + max(end_toks_lens)] = 1
                        found_end_toks = True
                        break
                    attention_mask[0, j] = 1
                    j += 1
                    i += 1

                if not found_end_toks:
                    attention_mask[0, i + target_len:] = 0

        if return_original:
            if attention_mask.sum() == 0:
                attention_mask[:, :] = 1

        return attention_mask

    @staticmethod
    def mask_tensor_json_kv_span(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        target_key: Union[str],
        end_toks: Union[str, List[Union[str]]] = "\"},",
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
        target_ids = tokenizer(target_key, add_special_tokens=True)['input_ids']
        end_toks_ids = [tokenizer(end_tok, add_special_tokens=True)['input_ids'] for end_tok in end_toks]

        return MaskedJSONDataset.mask_tensor_ids_json_kv_span(
            input_ids,
            attention_mask,
            target_ids,
            end_toks_ids,
            return_original=return_original
        )

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
        if self._cache[idx] is not None:
            return self._cache[idx]

        data = self._data["train_data"][idx]
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        if self._dont_mask:
            data['file_idx'] = torch.tensor(idx)
            self._cache[idx] = data
            return data

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        new_mask = attention_mask
        if self._mask_new_tokens:
            data["attention_mask"] = self.mask_all_new_tokens(input_ids, new_mask)
        else:
            for i, (token, end_token) in enumerate(self._current_token_id_mask, start=1):
                new_mask = self.mask_tensor_ids_json_kv_span(
                    input_ids, new_mask, target_ids=token, end_toks_ids=end_token
                ).squeeze(0)

                # Apply the masked attention mask for each iteration
                data["attention_mask"] = new_mask
                if torch.all(new_mask == 0):
                    warnings.warn("Attention mask contains all zeros")
                    break

        if input_ids.ndim == 2:
            data["attention_mask"] = data["attention_mask"].squeeze(0)
            data["input_ids"] = data["input_ids"].squeeze(0)

        self._cache[idx] = data
        return data

