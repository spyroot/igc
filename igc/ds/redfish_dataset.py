import json
import os
import shutil
from pathlib import Path
from random import random
from typing import Optional, Any, List, Tuple, Union, Iterator
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from transformers import GPT2Tokenizer
import logging

from .ds_downloadable_ds import DownloadableDataset
from .ds_rest_trajectories import RestTrajectory
from .ds_utils import (
    create_tar_gz,
    unpack_tar_gz
)
from igc.interfaces.rest_mapping_interface import RestMappingInterface
from igc.interfaces.rest_one_hot_interface import RestActionEncoderInterface
import zlib
import random


class DatasetConsistencyError(Exception):
    """Base class for other exceptions"""
    pass


class JSONDataset(DownloadableDataset, RestMappingInterface, RestActionEncoderInterface):

    def __init__(self,
                 directory_path: str,
                 default_tokenize: Optional[str] = "gpt2-xl",
                 max_len: Optional[int] = 1024,
                 overlap: Optional[int] = 256,
                 dataset_dir: Optional[str] = "datasets",
                 verbose: Optional[bool] = False,
                 recreate_dataset: Optional[bool] = False,
                 tokenizer: Optional[Any] = None,
                 transform=None,
                 target_transform=None):
        """
        :param directory_path:
        :param default_tokenize:
        :param max_len:
        :param overlap:
        :param dataset_dir:
        :param verbose:
        :param recreate_dataset:
        :param tokenizer:
        """
        assert isinstance(directory_path, str), 'directory_path should be a string'
        assert isinstance(default_tokenize, str), 'default_tokenize should be a string'
        assert isinstance(max_len, int), 'max_len should be an integer'
        assert isinstance(overlap, int), 'overlap should be an integer'
        assert isinstance(dataset_dir, str), 'dataset_dir should be a string'
        assert isinstance(verbose, bool), 'verbose should be a boolean'

        self._recreate_dataset = recreate_dataset

        # torch dataset mirror
        # dataset mirror
        self._mirrors = [
            {"train_dataset": 'http://192.168.254.78/ds/igc.tar.gz'},
            {"json_data": 'http://192.168.254.78/ds/json_data.tar.gz'},
        ]

        self._resources = [
            ("igc.tar.gz", "12af9db8f37d80b695bf84117e53cb05", "train_dataset"),
            ("json_data.tar.gz", "f5f3f54b39a2e2b8f63ec099fed8e677", "json_data"),
        ]

        # this could types train val if we want store separately.
        self._dataset_file_type = ["train_dataset", "json_data"]

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='dataset.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

        self.transform = transform
        self.target_transform = target_transform

        self._data = {}
        self._masked_data = {}
        _unprocessed = os.path.abspath(directory_path)
        _unprocessed = Path(_unprocessed).resolve()
        self._unprocessed = str(_unprocessed)
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

        dataset_path = os.path.abspath(dataset_dir)
        dataset_path = Path(dataset_path).resolve()

        self._default_dir = str(dataset_path)
        self._default_raw_dir = str(dataset_path / 'raw')
        self._default_original_dir = str(dataset_path / 'orig')
        self._json_directory_path = self._default_original_dir

        os.makedirs(self._default_dir, exist_ok=True)
        os.makedirs(self._default_raw_dir, exist_ok=True)

        self._dataset_file_name = str(
            Path(self._default_raw_dir) / f"processed_dataset_{tok_name}.pt")
        self._dataset_masked_file_name = str(
            Path(self._default_raw_dir) / f"processed_masked_dataset_{tok_name}.pt")
        self._rest_api_to_method_file_name = str(
            Path(self._default_raw_dir) / f"rest_api_to_method_{tok_name}.pt")
        self._rest_api_to_respond_file_name = str(
            Path(self._default_raw_dir) / f"rest_api_to_respond_{tok_name}.pt")
        self._dataset_tarball_name = str(
            Path(self._default_dir) / 'igc.tar.gz')
        self._json_data_tarball_name = str(
            Path(self._default_dir) / 'json_data.tar.gz')
        self._tarball_hash = ""

        print(f"Dataset dir {self._default_dir}")
        print(f"Dataset dir {self._default_raw_dir}")

        # all dataset files
        self._dataset_file_names = [
            self._dataset_file_name,
            self._dataset_masked_file_name,
            self._rest_api_to_method_file_name,
            self._rest_api_to_respond_file_name,
        ]

        # dictionary to map hash values to action one-hot vectors
        self.num_actions = 0
        self.goals = {}
        self.action_space = {}
        self.action_to_rest = {}

        self._list_masked_keys = ["@odata.id"]
        self._rest_trajectories = None

        # call super method to download dataset
        super().__init__(dataset_root_dir=self._default_dir)
        # create all tarballs if we have raw files.
        self._create_tarball()
        # load or build dataset
        self._load_dataset()
        # check consistency
        self._check_consistency()

    @staticmethod
    def required_keys():
        """
        :return:
        """

        required_keys = ["hash_to_rest_api",
                         "hash_to_action_idx",
                         "action_idx_to_hash",
                         "train_data"]
        return required_keys

    def _check_dataset_files(self) -> bool:
        """Check if all dataset files are present.
        :return: True if all files are present, False otherwise.
        """
        dataset_files = [
            self._dataset_file_name,
            self._dataset_masked_file_name,
            self._rest_api_to_respond_file_name,
            self._rest_api_to_method_file_name,
        ]

        result = all(os.path.exists(file) for file in dataset_files)
        if result:
            print("Found all required dataset file.")

        return all(os.path.exists(file) for file in dataset_files)

    def _create_tarball(self):
        """Create tarball if needed.
        :return:
        """
        # copy all json if not present already
        if not os.path.exists(self._json_directory_path):
            print(f"Copy all discovered data to {self._json_directory_path}")
            shutil.copytree(self._unprocessed, f"{self._json_directory_path}/")

        # create tarball if not present.
        if not os.path.exists(self._json_data_tarball_name):
            print(f"Creating tarball {self._json_data_tarball_name}")
            full_path_tarball, self._tarball_hash = create_tar_gz(
                self._json_directory_path, self._json_data_tarball_name)

        if self._check_dataset_files():
            if not os.path.exists(self._dataset_tarball_name):
                os.makedirs(self._default_dir, exist_ok=True)
                print(f"Creating tarball {self._dataset_tarball_name} file.")
                full_path_tarball, self._tarball_hash = create_tar_gz(
                    self._default_raw_dir, self._dataset_tarball_name)

    def _build_dataset(self):
        """Build a dataset from all json responds
        :return:
        """

        # self._default_dir, self._default_original_dir
        self._rest_trajectories = RestTrajectory(self._unprocessed, self._default_original_dir)
        self._rest_trajectories.load()

        # all api mapping
        self._rest_api_to_respond, self._rest_api_to_method = self._rest_trajectories.merged_view()
        self._respond_to_api = {value: key for key, value in self._rest_api_to_respond.items()}

        # build a dataset
        if not self._check_dataset_files():
            print(f"Re-building dataset {self._dataset_file_name}")

            self._load_json_files()
            self._load_dicts_from_data()
            self._construct_action_space()

            print(f"Saving data to: {self._dataset_file_name}")
            torch.save(self._data, self._dataset_file_name)

            print(f"Saving masked data to: {self._dataset_masked_file_name}")
            torch.save(self._masked_data, self._dataset_masked_file_name)

            print(f"Saving api respond data to: {self._rest_api_to_respond_file_name}")
            torch.save(self._rest_api_to_respond, self._rest_api_to_respond_file_name)

            print(f"Saving api respond method data to: {self._rest_api_to_respond_file_name}")
            torch.save(self._rest_api_to_method, self._rest_api_to_method_file_name)

            print(f"Saved dataset to disk. "
                  f"size of dataset: {len(self)} "
                  f"num hash entries: {len(self._data['hash_to_rest_api'])} \n"
                  f"num hash to action entries: {len(self._data['hash_to_action_idx'])} \n"
                  f"num action to hash entries: {len(self._data['action_idx_to_hash'])} "
                  f"num action to indices entries: {len(self._data['action_idx_to_hash'])} "
                  f"num masked entries: {len(self._masked_data['train_data'])} ")

            self._check_consistency()

            # create tarball
            self._create_tarball()

    def _load_dataset(self):
        """
        :return:
        """
        start_time = time.time()
        print("Loading dataset.")

        if self._recreate_dataset:
            print("Forcing rebuild a dataset.")
            self._build_dataset()

        # if some file not present and tar present unpack
        if not self._check_dataset_files():
            # if tar file present unpack other create new dataset.
            if os.path.exists(self._dataset_tarball_name):
                print(f"Found tarball unpack {self._dataset_tarball_name} files to {self._default_raw_dir}")
                unpack_tar_gz(self._dataset_tarball_name, self._default_raw_dir)
                if not self._check_dataset_files():
                    raise FileNotFoundError("Dataset files not found after unpacking the tarball.")
            else:
                self._build_dataset()

        # if tarball of all api responds.
        if os.path.exists(self._json_data_tarball_name):
            print(f"Found tarball unpack {self._json_data_tarball_name } files to {self._default_original_dir}")
            unpack_tar_gz(self._json_data_tarball_name, self._default_original_dir)

        self.logger.debug(f"Loading dataset from {self._dataset_file_name}")
        self.logger.debug(f"Loading dataset from disk. {self._dataset_file_name}")

        self._data = torch.load(self._dataset_file_name)
        for k in JSONDataset.required_keys():
            if k not in self._data:
                raise DatasetConsistencyError(f"Loaded dataset has no mandatory key {k}")

        self._masked_data = torch.load(self._dataset_masked_file_name)
        if 'train_data' not in self._masked_data:
            raise DatasetConsistencyError(f"Loaded dataset has no mandatory key train_data")

        self._load_dicts_from_data()
        self._construct_action_space()

        self._rest_api_to_respond = torch.load(self._rest_api_to_respond_file_name)
        self._rest_api_to_method = torch.load(self._rest_api_to_method_file_name)
        self._respond_to_api = {value: key for key, value in self._rest_api_to_respond.items()}

        self.logger.debug(f"Loaded dataset length: {len(self._data)}")
        self.logger.info(f"Loaded dataset, total time: {time.time() - start_time}")

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
            DatasetConsistencyError("looks like dataset corrupted hash_to_rest_api not present")

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
        """Method take a hash of rest api and return index of hash.
        Index used to map hash to index and index to one hot vector.
        :param hash_value: a hash of rest api
        :return: index of hash
        """
        hash_to_action_idx = self._data["hash_to_action_idx"]
        if hash_value in hash_to_action_idx:
            return hash_to_action_idx[hash_value]

        index = len(hash_to_action_idx)
        self._data["hash_to_action_idx"][hash_value] = index
        self._data["action_idx_to_hash"][index] = hash_value
        return index

    def index_to_hash(self, action_index: int) -> Optional[int]:
        """Method take index of hash and return corresponding hash.
        :param action_index:  index of hash
        :return:
        """
        action_idx = self._data["action_idx_to_hash"]
        if action_index in action_idx:
            return action_idx[action_index]

        return None

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
                    # action = rest.rsplit('/', 1)[-1]
                    try:
                        action = rest.rsplit('/', 1)[-1]
                    except AttributeError:
                        print("Error: Failed to rsplit. Value:", rest)
                        raise

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
        """Actions is rest api and corresponding arguments
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

    def get_rest_api_mappings(self) -> Iterator[Tuple[str, str]]:
        """
        Abstract method to provide the dict of all mapping of REST APIs.
        :return: An iterator of tuples, each containing the REST API and its corresponding response.
        """
        for rest_api, resp in self._rest_api_to_respond.items():
            yield rest_api, resp

    def get_rest_api_methods(self) -> Iterator[Tuple[str, str]]:
        """
        Abstract method to provide the dict of all mapping of REST APIs. Methods
        :return: An iterator of tuples, each containing the REST API and its corresponding method.
        """
        for rest_api, method in self._rest_api_to_method.items():
            yield rest_api, method

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
                    self.logger.debug(f"reading {file_name}")

                # extract the extra file name it key so, we get rest api
                json_lines = json_file.read()

                if file_path not in self._respond_to_api:
                    raise DatasetConsistencyError(
                        f"Inconsistency we have file {file_path} but no corresponding "
                        f"rest api size of resp_api {len(self._respond_to_api)}.")

                _rest_api = self._respond_to_api[file_path]
                hash_value = zlib.adler32(_rest_api.encode())

                # load JSON as a dictionary
                json_data = json.loads(json_lines)
                allowable_values = {}
                targets = {}

                # skip schema there is no point to extract target from that or api
                if "$schema" not in json_data:
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

                    self._data["hash_to_rest_api"][hash_value] = _rest_api
                    self._data["train_data"].append(
                        {
                            "request_hash": hash_value,
                            "request": _rest_api,
                            "input_ids": padded_chunk,
                            "attention_mask": padded_mask,
                            "allowable_values": allowable_values,
                            "targets": targets,
                            "file_path": file_path.split(self._default_original_dir)[1],
                        }
                    )

        for root, dirs, files in os.walk(self._json_directory_path):
            for file_name in files:
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    process_json_file(file_path, file_name)
                    for jk in self._list_masked_keys:
                        self.process_and_mask_json_file(file_path, file_name, jk)

    def action(self, one_hot_vec: torch.Tensor) -> str:
        """Return action from one hot vector representation
        :param one_hot_vec: is action in rest api
        :return: returns api /redfish/v1/Fabrics
        """
        label_index = self.one_hot_to_index(one_hot_vec)
        hash_value = self.index_to_hash(label_index)
        return self._data["hash_to_rest_api"][hash_value]

    def get_accepted_method(self, action: str) -> str:
        """Returns the accepted method for the given action.
        :param action: The action for which to retrieve the accepted method.
        :return: The accepted method for the given action.
        """
        return self._rest_api_to_method.get(action, "Unknown")

    def action_to_one_hot(self, rest_api: str) -> Union[np.ndarray, torch.Tensor]:
        """Must take a string and return one hot vector either as tensor or ndarray
        :param rest_api: The REST API to lookup.
        :return: The response associated with the REST API.
        """
        hash_value = zlib.adler32(rest_api.encode())
        action_index = self.hash_to_index(hash_value)
        one_hot_tensor = self.index_to_one_hot(action_index)
        return one_hot_tensor

    def one_hot_vector_to_action(self, one_hot_vector: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Takes a one-hot vector and returns the corresponding REST API.

        :param one_hot_vector: The one-hot vector representing the REST API.
                               It can be a tensor or a numpy array.
        :return: The REST API corresponding to the one-hot vector.
        """
        return self.action(one_hot_vector)

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

    def sample_random_masked(self):
        """Get item from dataset
        :param idx:
        :return:
        """
        index = random.choice(range(len(self._masked_data['train_data'])))
        return self._masked_data['train_data'][index]

    def sample_masked(self, idx):
        """Get item from dataset
        :param idx:
        :return:
        """
        return self._masked_data['train_data'][idx]

    def sample_masked_iter(self):
        """
        :return:
        """
        for s in self._masked_data['train_data']:
            yield s

    def lookup_rest_api_to_respond(self, rest_api: str):
        """Lookup the response for a given REST API.
        :param rest_api: The REST API to lookup.
        :return: The response associated with the REST API
        if found, or a default value if not found.
        """
        return self._rest_api_to_respond.get(rest_api, None)

    def lookup_rest_api_to_method(self, rest_api: str):
        """Lookup the method for a given REST API.
        :param rest_api: The REST API to lookup.
        :return: The method associated with the REST API if
        found, or raise a KeyError if not found.
        """
        return self._rest_api_to_method.get(rest_api, None)

    def sample_rest_api(self) -> Tuple[str, List[str], torch.Tensor]:
        """Randomly sample a REST API endpoint from the dataset.
        It good for testing etc.
        :return: A tuple containing the sampled REST API endpoint,
                supported method, and one-hot vector representation.
        """
        rest_api = random.choice(list(self._rest_api_to_respond.keys()))
        supported_method = self._rest_api_to_method[rest_api]
        return rest_api, supported_method, self.action_to_one_hot(rest_api)

    def resources_tarball(self):
        """We need return list of resources.
        :return:
        """
        return self._resources

    def is_tarball(self) -> bool:
        """Implement this method indicate yes tarball.
        :return:
        """
        return True

    def mirrors_tarballs(self):
        """Our dataset in tarball
        :return:
        """
        return self._mirrors

    def is_overwrite(self):
        """Implement this method indicate yes overwrite.
        :return:
        """
        return True

    def root_dir(self) -> str:
        """Downloaded dataset return the root directory of the dataset.
        :return:
        """
        return self._default_dir

    def dataset_types(self):
        """Downloaded dataset requires each file has dataset type.
        i.e. dataset type implied small , medium , large, or any other keys.
        :return:
        """
        return self._dataset_file_type

    def _get_unique_values(self):
        """
        :return:
        """
        unique_requests = set()
        unique_hashes = set()
        unique_labels = set()

        for data_point in self._data["train_data"]:
            unique_requests.add(data_point["request"])
            unique_hashes.add(data_point["request_hash"])
            unique_labels.add(tuple(data_point["labels"].tolist()))

        return unique_requests, unique_hashes, unique_labels

    def _check_consistency(self):
        """Now because we have bunch data structure data and mapping
         want Check the consistency between data structures.
        """

        _required_keys = JSONDataset.required_keys()
        for key in _required_keys:
            if key not in self._data:
                raise ValueError(f"Consistency check failed. Missing mandatory key: {key}")
            print(f"Length of {key}: {len(self._data[key])}")

        hash_to_rest_api = self._data["hash_to_rest_api"]
        hash_to_action_idx = self._data["hash_to_action_idx"]
        action_idx_to_hash = self._data["action_idx_to_hash"]

        unique_requests, unique_hashes, unique_labels = self._get_unique_values()
        print(f"Number of unique REST API requests: {len(unique_requests)}")
        print(f"Number of unique hashes: {len(unique_hashes)}")
        print(f"Number of unique labels: {len(unique_labels)}")
        print(f"Number of unique rest_api: {len(self._rest_api_to_respond)}")
        print(f"Number of unique rest_api: {len(self._rest_api_to_method)}")
        print(f"Number of unique hash to rest: {len(hash_to_rest_api)}")
        print(f"Number of unique hash to action: {len(hash_to_action_idx)}")
        print(f"Number of unique action to hash: {len(action_idx_to_hash)}")

        # a) check consistency between _rest_api_to_respond and _rest_api_to_method
        for rest_api in self._rest_api_to_respond:
            if rest_api not in self._rest_api_to_method:
                print(f"Error consistence check 1: Missing method for REST API: {rest_api}")
                raise ValueError(f"Consistency check failed. Missing key: {rest_api}")

        # b) Check consistency between _rest_api_to_method and _data["hash_to_rest_api"]
        for rest_api in self._rest_api_to_respond:
            rest_api_hash = zlib.adler32(rest_api.encode())
            if rest_api_hash not in self._data["hash_to_rest_api"]:
                print(f"Error consistence check 2: Inconsistent "
                      f"REST API mapping for method: api {rest_api} no in dataset hash_to_rest_api")
                raise ValueError(f"Error consistence check 2: Inconsistent "
                                 f"REST API mapping for method: api {rest_api} "
                                 f"no in dataset hash_to_rest_api")

        # c) Check consistency between _data["hash_to_action_idx"] and _data["action_idx_to_hash"]
        for index in self._data["action_idx_to_hash"]:
            if index not in self._data["hash_to_action_idx"].values():
                print(f"Error consistence check 3: Inconsistent action "
                      f"index mapping for hash value: {self._data['action_idx_to_hash'][index]}")

        # d) Iterate over all REST APIs and perform checks
        #   that we can recover hash -> one hot and reverse
        for rest_api in self._rest_api_to_respond:
            rest_api_req_hash = zlib.adler32(rest_api.encode())
            # check if rest_api exists in hash_to_rest_api
            if rest_api_req_hash not in self._data["hash_to_rest_api"]:
                print(f"Error: Missing hash value for REST API: {rest_api}")

            # do check if hash value exists in hash_to_action_idx
            if rest_api_req_hash not in self._data["hash_to_action_idx"]:
                print(f"Error: Missing action index for hash value: {rest_api_req_hash}")

            # get the hash index and get it one hot
            action_index = self._data["hash_to_action_idx"][rest_api_req_hash]
            one_hot_tensor = self.index_to_one_hot(action_index)

            # take one hot recover rest
            recovered_rest_api = self.one_hot_vector_to_action(one_hot_tensor)
            if recovered_rest_api != rest_api:
                print(f"Error: Inconsistent recovery for REST API: {rest_api}")
                raise ValueError(f"Inconsistent recovery for REST API: {rest_api}")

        # reverse check
        for entry in self._data["train_data"]:
            one_hot_vector = entry["labels"]
            hash_index = self.one_hot_to_index(one_hot_vector)
            hash_value = self.index_to_hash(hash_index)

            # check if the hash value exists in hash_to_action_idx
            if hash_value not in self._data["hash_to_action_idx"]:
                print(f"Error: Missing action index for hash value: {hash_value}")

            # recover the REST API URL from the one-hot vector
            recovered_rest_api = self.one_hot_vector_to_action(one_hot_vector)
            if recovered_rest_api != self._data["hash_to_rest_api"][hash_value]:
                print(f"Error: Inconsistent recovery for hash value: {hash_value}")

        print("Consistency check completed.")
