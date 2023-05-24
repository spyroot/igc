import json
import os
from typing import Optional, Any, List, Tuple
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import logging


class JSONDataset(Dataset):
    def __init__(self,
                 directory_path: str,
                 default_tokenize: Optional[str] = "gpt2-xl",
                 max_len: Optional[int] = 1024,
                 overlap: Optional[int] = 256,
                 dataset_dir: Optional[str] = "datasets",
                 verbose: Optional[bool] = False,
                 recreate_dataset: Optional[bool] = False,
                 tokenizer: Optional[Any] = None,
                 skip_creation: Optional[bool] = False):
        """
        :param directory_path:
        :param default_tokenize:
        :param max_len:
        :param overlap:
        :param dataset_dir:
        :param verbose:
        :param recreate_dataset:
        :param tokenizer:
        :param skip_creation:
        """
        assert isinstance(directory_path, str), 'directory_path should be a string'
        assert isinstance(default_tokenize, str), 'default_tokenize should be a string'
        assert isinstance(max_len, int), 'max_len should be an integer'
        assert isinstance(overlap, int), 'overlap should be an integer'
        assert isinstance(dataset_dir, str), 'dataset_dir should be a string'
        assert isinstance(verbose, bool), 'verbose should be a boolean'

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='dataset.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

        self._data = {}
        self._masked_data = {}
        self.directory_path = directory_path
        self._verbose = verbose
        self._max_len = max_len
        self._overlap = overlap

        if tokenizer is not None:
            self.tokenizer = tokenizer
            tok_name = self.tokenizer.name_or_path
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(default_tokenize)
            tok_name = self.tokenizer.name_or_path

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._default_dir = dataset_dir
        os.makedirs(self._default_dir, exist_ok=True)

        self._dataset_file_name = f"./datasets/processed_dataset_{tok_name}.pt"
        self._dataset_masked_file_name = f"./datasets/processed_masked_dataset_{tok_name}.pt"

        # dictionary to map hash values to action one-hot vectors
        self.num_actions = 0
        self.goals = {}
        self.action_space = {}
        self.action_to_rest = {}

        self._list_masked_keys = ["@odata.id"]

        if skip_creation is False:
            start_time = time.time()  # Start the timer

            if recreate_dataset or not os.path.exists(self._dataset_file_name):
                t1 = time.time()
                self._load_json_files()
                self.logger.debug(f"Time for _load_json_files: {time.time() - t1}")

                t2 = time.time()
                self._load_dicts_from_data()
                self.logger.debug(f"Time for _load_dicts_from_data: {time.time() - t2}")

                t3 = time.time()
                self._construct_action_space()
                self.logger.debug(f"Time for _construct_action_space: {time.time() - t3}")

                t4 = time.time()
                torch.save(self._data, self._dataset_file_name)
                self.logger.debug(f"Time for saving _data: {time.time() - t4}")

                t5 = time.time()
                torch.save(self._masked_data, self._dataset_masked_file_name)
                self.logger.debug(f"Time for saving _masked_data: {time.time() - t5}")

                print(f"Saved dataset to disk. "
                      f"size of dataset: {len(self)} "
                      f"num hash entries: {len(self._data['hash_to_rest_api'])} "
                      f"num hash to action entries: {len(self._data['hash_to_action_idx'])} \\n"
                      f"num action to hash entries: {len(self._data['action_idx_to_hash'])} "
                      f"num action to indices entries: {len(self._data['action_idx_to_hash'])} "
                      f"num masked entries: {len(self._masked_data['train_data'])} ")
            else:
                t1 = time.time()
                self.logger.debug(f"Loading dataset from {self._dataset_file_name}")
                self.logger.debug(f"Loading dataset from disk. {self._dataset_file_name}")
                self._data = torch.load(self._dataset_file_name)
                logging.debug(f"Time for loading _data: {time.time() - t1}")

                t2 = time.time()
                logging.debug(f"Loading masked dataset from disk. {self._dataset_masked_file_name}")
                self._masked_data = torch.load(self._dataset_masked_file_name)
                logging.debug(f"Time for loading _masked_data: {time.time() - t2}")

                t3 = time.time()
                self._load_dicts_from_data()
                logging.debug(f"Time for _load_dicts_from_data: {time.time() - t3}")

                t4 = time.time()
                self._construct_action_space()
                logging.debug(f"Time for _construct_action_space: {time.time() - t4}")

            end_time = time.time()  # End the timer
            self.logger.info(f"Loaded dataset, total time: {end_time - start_time}")

            # set at the end
            self.sample_masked = False

    def chunk_overlap_size(self):
        """Amount of overlap between two chunks
        :return:
        """
        return self._overlap

    def _load_dicts_from_data(self):
        """Load the dictionaries based on the data points in self.data.
        """
        # update num actions
        if "hash_to_rest_api" not in self._data:
            print(f"looks like dataset corrupted is masked.")

        self.num_actions = len(self._data["hash_to_rest_api"])
        for h in self._data["hash_to_rest_api"]:
            hash_idx = self.hash_to_index(h)
            one_hot = self.index_to_one_hot(hash_idx)
            train_data = self._data["train_data"]
            for data_point in train_data:
                if data_point["request_hash"] == h:
                    data_point["labels"] = one_hot
                    # break

    def hash_to_index(self, hash_value: int) -> int:
        """
        :param hash_value:
        :return:
        """
        hash_to_action_idx = self._data["hash_to_action_idx"]
        if hash_value in hash_to_action_idx:
            return hash_to_action_idx[hash_value]

        # assign action index for the unseen hash value
        index = len(hash_to_action_idx)
        self._data["hash_to_action_idx"][hash_value] = index
        self._data["action_idx_to_hash"][index] = hash_value
        return index

    def index_to_hash(self, action_index: int) -> Optional[int]:
        """
        :param action_index:
        :return:
        """
        action_idx = self._data["action_idx_to_hash"]
        if action_index in action_idx:
            return action_idx[action_index]

        return None  # Invalid index

    @staticmethod
    def one_hot_to_index(one_hot_tensor):
        """Take one hot tensor and return index
        :param one_hot_tensor:
        :return:
        """
        index = torch.argmax(one_hot_tensor)
        return index.item() if index is not None else None

    def index_to_one_hot(self, hash_index: int) -> torch.Tensor:
        """Convert an index to its corresponding one-hot tensor.
        :param hash_index:
        :return:
        """
        hash_index = torch.tensor(hash_index)
        one_hot_tensor = F.one_hot(hash_index, self.num_actions)
        return one_hot_tensor

    @staticmethod
    def convert_file_name(file_name: str) -> str:
        """Convert file name back to rest api original request.
        :param file_name:
        :return:
        """
        converted_name = file_name.replace("_", "/")
        return converted_name

    def create_chunks(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create chunks of input_ids and attention_mask, this
        splits the input_ids and attention_mask for large json files.

        :param attention_mask:
        :param input_ids:
        :return:
        """
        idx = 0
        chunks = []
        num_tokens = input_ids.size(1)

        if num_tokens <= self._max_len:
            chunks.append((input_ids, attention_mask))
            return chunks

        while idx < num_tokens:
            if idx + self._max_len < num_tokens:
                chunk_input_ids = input_ids[:, idx:idx + self._max_len]
                chunk_attention_mask = attention_mask[:, idx:idx + self._max_len]
            else:
                chunk_input_ids = input_ids[:, idx:]
                chunk_attention_mask = attention_mask[:, idx:]

            new_chunk_size = self._max_len - chunk_attention_mask.size(1)
            padded_input_ids = torch.nn.functional.pad(
                chunk_input_ids, (0, self._max_len - chunk_input_ids.size(1)), value=self.tokenizer.pad_token_id)

            padded_attention_mask = torch.nn.functional.pad(
                chunk_attention_mask, (0, self._max_len - chunk_attention_mask.size(1)))

            chunks.append((padded_input_ids, padded_attention_mask))
            idx = idx + self._max_len - self._overlap
            num_tokens -= new_chunk_size

        return chunks

    def extract_recursive(self, json_obj, allowable_values, targets):
        """Recursively extracts values from a nested JSON structure.
        """
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if '@odata.id' in key:
                    targets["api_target"] = value
                if "@Redfish.AllowableValues" in key:
                    allowable_values[key] = value
                    # print(f"target: {allowable_values[key]} {action}")
                if "target" in key:
                    targets["api_target"] = value
                    rest = value
                    action = rest.rsplit('/', 1)[-1]
                    self.action_space[rest] = action
                    self.action_to_rest[action] = rest
                self.extract_recursive(value, allowable_values, targets)
        elif isinstance(json_obj, list):
            for item in json_obj:
                self.extract_recursive(item, allowable_values, targets)

    def _extrac_action(self, target):
        """
        :return:
        """
        action = target.rsplit('/', 1)[-1]
        if action not in self.action_space:
            self.action_space[target] = action
        if target not in self.action_to_rest:
            self.action_to_rest[action] = target

    def _construct_action_space(self):
        """
        :return:
        """
        for d in self:
            target = d["targets"]
            if len(target) == 0:
                continue

            t = target['api_target']
            self.goals[t] = []
            allowable_values = d["allowable_values"]
            if len(allowable_values) > 0:
                for key, values in allowable_values.items():
                    parameter, _ = key.split('@')
                    self.goals[t].append({parameter: values})
                    self._extrac_action(t)
            else:
                self._extrac_action(t)

    @staticmethod
    def mask_specific_key(
            json_data,
            target_key: str,
            tokenizer=None,
            debug: Optional[bool] = False):
        """Mask specific keu in json structure, technically will work in other cases
        :param tokenizer:
        :param debug:
        :param json_data:
        :param target_key:
        :return:
        """
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            tokenizer = tokenizer

        if isinstance(json_data, str):
            json_lines = json.dumps(json_data)
        else:
            json_lines = json_data

        encoding = tokenizer(json_lines, return_tensors="pt")
        attention_mask = encoding['attention_mask'].clone()
        attention_mask[:, :] = 0

        target_tokens = tokenizer(target_key)['input_ids']
        input_ids = encoding['input_ids']

        target_len = len(target_tokens)
        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_tokens:
                attention_mask[0, i:i + target_len] = 1
                if debug:
                    print(f"Unmasking tokens "
                          f"at pos {i} to {i + target_len}: "
                          f"{tokenizer.decode(input_ids[0, i:i + target_len])}")

        return attention_mask

    @staticmethod
    def mask_json_key_and_value(encoding, target_key, tokenizer, debug=False):
        """Mask specific key and value in json structure,
         technically will work in other cases.

        this simular to mask_specific_key_and_value but does so for already
        computed mask.

        Usage:
            target_key = "@odata.id"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            json_lines = json.dumps(j_data)
            attention_mask = mask_specific_key_and_value(json_lines,
            target_key, tokenizer=tokenizer, debug=True)

        :param tokenizer:
        :param encoding:
        :param target_key:
        :param debug:
        :return:
        """
        attention_mask = encoding['attention_mask'].clone()
        attention_mask[:, :] = 0

        target_tokens = tokenizer(target_key)['input_ids']
        input_ids = encoding['input_ids']

        period = tokenizer.encode(',', add_special_tokens=False)
        target_len = len(target_tokens)
        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_tokens:
                # umask the key tokens
                attention_mask[0, i:i + target_len] = 1
                #  assume the value starts  immediately after the key.
                j = i + target_len
                unmasked_tokens = []
                while j < input_ids.shape[1] and input_ids[0, j].item() != period[0]:
                    attention_mask[0, j] = 1
                    if debug:
                        unmasked_tokens.append(input_ids[0, j].item())
                    j += 1
                if debug:
                    print(f"Unmasking tokens at positions {i} to {j}: {tokenizer.decode(unmasked_tokens)}")

        return attention_mask

    @staticmethod
    def mask_specific_key_and_value(json_data, target_key, tokenizer=None, debug=False):
        """Mask specific key and value in json structure,
         technically will work in other cases.

        Usage:
            target_key = "@odata.id"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            json_lines = json.dumps(j_data)
            attention_mask = mask_specific_key_and_value(json_lines,
            target_key, tokenizer=tokenizer, debug=True)

        :param tokenizer:
        :param json_data:
        :param target_key:
        :param debug:
        :return:
        """
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            tokenizer = tokenizer

        if isinstance(json_data, str):
            json_lines = json.dumps(json_data)
        else:
            json_lines = json_data

        encoding = tokenizer(json_lines, return_tensors="pt")
        attention_mask = encoding['attention_mask'].clone()
        attention_mask[:, :] = 0

        target_tokens = tokenizer(target_key)['input_ids']
        input_ids = encoding['input_ids']

        period = tokenizer.encode(',', add_special_tokens=False)
        target_len = len(target_tokens)
        for i in range(input_ids.shape[1] - target_len + 1):
            if input_ids[0, i:i + target_len].tolist() == target_tokens:
                # umask the key tokens
                attention_mask[0, i:i + target_len] = 1
                #  assume the value starts  immediately after the key.
                j = i + target_len
                unmasked_tokens = []
                while j < input_ids.shape[1] and input_ids[0, j].item() != period[0]:
                    attention_mask[0, j] = 1
                    if debug:
                        unmasked_tokens.append(input_ids[0, j].item())
                    j += 1
                if debug:
                    print(f"Unmasking tokens at positions {i} to {j}: {tokenizer.decode(unmasked_tokens)}")

        return attention_mask

    def process_and_mask_json_file(
            self,
            json_file_path: str,
            json_file_name: str,
            mask_target_key: str) -> None:
        """This second pass we read the json file and mask what we need.
        :param json_file_path:
        :param json_file_name:
        :param mask_target_key: a key:value that we want to mask
        :return:
        """
        with open(json_file_path, "r") as json_file:
            logging.debug(f"reading {json_file_name}")

            json_lines = json_file.read()
            json_lines += json_lines + "<|endoftext|>"
            tokenized = self.tokenizer(
                json_lines,
                padding='max_length',
                max_length=self._max_len,
                truncation=False,
                return_tensors='pt')

            input_ids = tokenized['input_ids']
            attention_mask = JSONDataset.mask_json_key_and_value(
                tokenized, mask_target_key, tokenizer=self.tokenizer
            )

            chunks = self.create_chunks(input_ids, attention_mask)
            # for each chunk add it as a separate data point
            for i, chunk_tuple in enumerate(chunks):
                padded_chunk, padded_mask = chunk_tuple
                padded_chunk = padded_chunk.squeeze(dim=0)
                padded_mask = padded_mask.squeeze(dim=0)
                self._masked_data["train_data"].append(
                    {
                        "input_ids": padded_chunk,
                        "attention_mask": padded_mask,
                    }
                )

    def _load_json_files(self) -> None:
        """Load json file and construct from raw json presentation
           a dataset.
        """
        self._data["hash_to_rest_api"] = {}
        self._data["hash_to_action_idx"] = {}
        self._data["action_idx_to_hash"] = {}
        self._data["hash_to_rest_api"] = {}
        self._data["train_data"] = []
        self._masked_data["train_data"] = []

        def process_json_file(file_path: str, file_name: str) -> None:
            """
            :param file_path:
            :param file_name:
            :return:
            """
            with open(file_path, "r") as json_file:
                if self._verbose:
                    print(f"reading {file_name}")

                rest_api = self.convert_file_name(file_name)
                json_lines = json_file.read()
                hash_value = hash(rest_api)

                # load JSON as a dictionary
                json_data = json.loads(json_lines)
                allowable_values = {}
                targets = {}
                self.extract_recursive(json_data, allowable_values, targets)

                json_lines += json_lines + "<|endoftext|>"
                tokenizer = self.tokenizer(
                    json_lines,
                    padding='max_length',
                    max_length=self._max_len,
                    truncation=False,
                    return_tensors='pt')

                input_ids = tokenizer['input_ids']
                attention_mask = tokenizer['attention_mask']
                chunks = self.create_chunks(input_ids, attention_mask)

                # for each chunk add it as a separate data point
                for i, chunk_tuple in enumerate(chunks):
                    padded_chunk, padded_mask = chunk_tuple
                    padded_chunk = padded_chunk.squeeze(dim=0)
                    padded_mask = padded_mask.squeeze(dim=0)

                    self._data["hash_to_rest_api"][hash_value] = rest_api
                    self._data["train_data"].append(
                        {
                            "request_hash": hash_value,
                            "request": rest_api,
                            "input_ids": padded_chunk,
                            "attention_mask": padded_mask,
                            "allowable_values": allowable_values,
                            "targets": targets,
                            "file_path": file_path.split(".json_responses")[1],
                        }
                    )

        for root, dirs, files in os.walk(self.directory_path):
            for file_name in files:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    process_json_file(file_path, file_name)
                    for jk in self._list_masked_keys:
                        self.process_and_mask_json_file(file_path, file_name, jk)

    def action(self, one_hot_vec: torch.Tensor) -> str:
        """Return action from one hot vector representation
        :param one_hot_vec:
        :return:
        """
        label_index = self.one_hot_to_index(one_hot_vec)
        hash_value = self.index_to_hash(label_index)
        return self._data["hash_to_rest_api"][hash_value]

    @staticmethod
    def preprocess_sample(sample):
        """

        :param sample:
        :return:
        """
        return sample

    @staticmethod
    def preprocess_json_data(json_data):
        """Preprocess json data
        :param json_data:
        :return:
        """
        preprocessed_data = []
        for item in json_data:
            preprocessed_sample = JSONDataset.preprocess_sample(item)
            preprocessed_data.append(preprocessed_sample)
        return preprocessed_data

    def is_empty(self) -> bool:
        """Return is dataset is empty or not.
        :return:
        """
        if self._data is None or len(self._data) == 0:
            return True

        return False

    def set_masked(self, value):
        """THis will switch to masked data
        :param value:
        :return:
        """
        self._masked_data = value

    def __len__(self):
        """Return length of dataset"""
        return len(self._data["train_data"])

    def __getitem__(self, idx):
        """Get item from dataset
        :param idx:
        :return:
        """
        return self._data["train_data"][idx]


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        """

        :param prompt_dataset:
        :param chosen_dataset:
        :param reject_dataset:
        :param pad_token_id:
        :param train_phase:
        """
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        """

        :return:
        """
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], \
                self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], \
                self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], \
                self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
