import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class JSONDataset(Dataset):
    def __init__(self,
                 directory_path: str,
                 default_tokenize: Optional[str] = "gpt2-xl",
                 max_len: Optional[int] = 1024,
                 overlap: Optional[int] = 256,
                 dataset_dir: Optional[str] = "datasets",
                 verbose: Optional[bool] = False,
                 recreate_dataset: Optional[bool] = False):
        """
        """
        assert isinstance(directory_path, str), 'directory_path should be a string'
        assert isinstance(default_tokenize, str), 'default_tokenize should be a string'
        assert isinstance(max_len, int), 'max_len should be an integer'
        assert isinstance(overlap, int), 'overlap should be an integer'
        assert isinstance(dataset_dir, str), 'dataset_dir should be a string'
        assert isinstance(verbose, bool), 'verbose should be a boolean'

        self._data = {}
        self.directory_path = directory_path
        self._verbose = verbose
        self._max_len = max_len
        self._overlap = overlap
        self.tokenizer = GPT2Tokenizer.from_pretrained(default_tokenize)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._default_dir = dataset_dir
        os.makedirs(self._default_dir, exist_ok=True)

        self._dataset_file_name = f"./datasets/processed_dataset_{default_tokenize}.pt"

        # Dictionary to map hash values to action one-hot vectors
        self.num_actions = 0

        if recreate_dataset or not os.path.exists(self._dataset_file_name):
            self._load_json_files()
            self._load_dicts_from_data()
            torch.save(self._data, self._dataset_file_name)
        else:
            print(f"Loading dataset from {self._dataset_file_name}")
            self._data = torch.load(self._dataset_file_name)
            self._load_dicts_from_data()

    def _load_dicts_from_data(self):
        """Load the dictionaries based on the data points in self.data.
        """
        # update num actions
        self.num_actions = len(self._data["hash_to_rest_api"])

        for h in self._data["hash_to_rest_api"]:
            hash_idx = self.hash_to_index(h)
            one_hot = self.index_to_one_hot(hash_idx)
            train_data = self._data["train_data"]
            for data_point in train_data:
                if data_point["request_hash"] == h:
                    data_point["labels"] = one_hot
                    # break

    def hash_to_index(self, hash_value: str) -> int:
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

    def index_to_hash(self, action_index: int) -> Optional[str]:
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

    def create_chunks(self, input_ids, attention_mask):
        """Create chunks of input_ids and attention_mask, this
        splits the input_ids and attention_mask for large json files.
        :param attention_mask:
        :param input_ids:
        :return:
        """
        num_tokens = input_ids.size(1)

        chunks = []
        idx = 0
        while idx < num_tokens:
            if idx + self._max_len < num_tokens:
                chunk_input_ids = input_ids[:, idx:idx + self._max_len]
                chunk_attention_mask = attention_mask[:, idx:idx + self._max_len]
            else:
                chunk_input_ids = input_ids[:, idx:]
                chunk_attention_mask = attention_mask[:, idx:]

            padded_input_ids = torch.nn.functional.pad(
                chunk_input_ids, (0, self._max_len - chunk_input_ids.size(1)),
                value=self.tokenizer.pad_token_id)

            padded_attention_mask = torch.nn.functional.pad(
                chunk_attention_mask, (0, self._max_len - chunk_attention_mask.size(1)))
            chunks.append((padded_input_ids, padded_attention_mask))
            idx += self._max_len - self._overlap

        return chunks

    @staticmethod
    def extract_recursive(json_obj, allowable_values, targets):
        """Recursively extracts values from a nested JSON structure.
        """
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if "@Redfish.AllowableValues" in key:
                    allowable_values[key] = value
                if "target" in key:
                    targets[key] = value
                JSONDataset.extract_recursive(value, allowable_values, targets)
        elif isinstance(json_obj, list):
            for item in json_obj:
                JSONDataset.extract_recursive(item, allowable_values, targets)

    def _load_json_files(self) -> None:
        """
        """
        self._data["hash_to_rest_api"] = {}
        self._data["hash_to_action_idx"] = {}
        self._data["action_idx_to_hash"] = {}
        self._data["hash_to_rest_api"] = {}
        self._data["train_data"] = []

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
                JSONDataset.extract_recursive(json_data, allowable_values, targets)

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
                    self._data["hash_to_rest_api"][hash_value] = rest_api
                    self._data["train_data"].append(
                        {
                            "rest_api": rest_api,  # this for debug
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

    def action(self, one_hot_vec: torch.Tensor) -> str:
        """return action from one hot vector representation
        :param one_hot_vec:
        :return:
        """
        label_index = self.one_hot_to_index(one_hot_vec)
        hash_value = self.index_to_hash(label_index)
        return self._data["hash_to_rest_api"][hash_value]

    @staticmethod
    def preprocess_sample(sample):
        return sample

    @staticmethod
    def preprocess_json_data(json_data):
        """ preprocess json data
        :param json_data:
        :return:
        """
        preprocessed_data = []
        for item in json_data:
            preprocessed_sample = JSONDataset.preprocess_sample(item)
            preprocessed_data.append(preprocessed_sample)
        return preprocessed_data

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
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
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
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
