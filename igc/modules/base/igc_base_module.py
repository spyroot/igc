"""
This class is base module for all trainers
inherit from this class.  It has basic functionality
for all trainers.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from collections import namedtuple
from pathlib import Path
from typing import Optional, Any

import loguru
import torch
from torch.utils.data import random_split, Subset

from igc.ds.redfish_dataset import JSONDataset
from igc.modules.base.metric_logger import MetricLogger
from igc.shared.shared_torch_utils import get_device

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])
from loguru import logger


class IgcBaseModule:
    """
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 ds: JSONDataset,
                 metric_logger: MetricLogger,
                 llm_model,
                 llm_tokenizer):
        """
        Note module name is important for saving
        and logging make sure it has no collision.

        :param module_name: name of the module
        :param spec: store all specs.
        :param ds: dataset used to trainer IGC
        :param metric_logger: a metric logger to store metrics
        :param llm_model: pre-trained language model
        :param llm_tokenizer: pre-trained tokenizer
        """

        # validate arguments
        if spec.num_train_epochs <= 0:
            raise ValueError("Invalid value for num_train_epochs. "
                             "It should be greater than 0.")

        if spec.per_device_train_batch_size <= 0:
            raise ValueError("Invalid value for per_device_train_batch_size. "
                             "It should be greater than 0.")

        if spec.overfit and spec.per_device_train_batch_size > len(ds):
            raise ValueError("Invalid combination of overfit and per_device_train_batch_size. "
                             "per_device_train_batch_size should be smaller than "
                             "the dataset size when overfit is True.")

        if llm_model is None:
            raise ValueError("llm_model cannot be None.")

        if llm_tokenizer is None:
            raise ValueError("llm_tokenizer cannot be None.")

        if ds is None:
            raise ValueError("ds (dataset) cannot be None.")

        self.model = llm_model
        self.tokenizer = llm_tokenizer
        self.module_name = module_name

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size

        self.dataset = ds
        self.device = get_device()
        self.metric_logger = metric_logger

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self._trainer_args = spec
        self._batch_log = 10
        self._default_lr = 1e-5
        self._overfit = spec.overfit

        self.optimizer = None

        # model saving
        self.save_strategy = spec.save_strategy
        checkpoint_path_dir = Path(spec.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        logger.info(f"Model {self.module_name} saving dir {checkpoint_path_dir}")

        if not checkpoint_path_dir.is_dir():
            raise ValueError(f"Indicate path to checkpoint dir {checkpoint_path_dir}.")

        self.checkpoint_dir = str(checkpoint_path_dir)
        self.rank = int(os.environ.get('LOCAL_RANK', -1))

        # logging
        log_level = spec.llm_log_level.upper()
        log_file = os.path.join(spec.log_dir, f"{module_name}.log")

        # configure loguru logger
        loguru.logger.remove()
        loguru.logger.add(log_file, level=log_level)

        self.metric_logger.set_logger(loguru.logger)
        self.metric_logger.set_log_level(log_level)

    def split_dataset(self, ratio: float = 0.8):
        """
        Split datasets,  train and eval

        :param ratio: ratio of split
        :return:
        """
        if ratio <= 0 or ratio >= 1:
            raise ValueError(
                "Invalid ratio. The ratio value should be between 0 and 1 (exclusive).")

        train_size = int(len(self.dataset) * ratio)
        eval_size = len(self.dataset) - train_size

        if train_size <= 0 or eval_size <= 0:
            raise ValueError(
                "Invalid dataset sizes. Adjust the ratio value to ensure non-zero splits.")

        return random_split(
            self.dataset, [train_size, eval_size])

    def split_slice_dataset(
            self,
            train_ratio: float = 0.8,
            sample_ratio: float = 0.01) -> list[Subset[Any]]:
        """
        Split a subset of the dataset and specify the amount of sample used.

        :param train_ratio: The ratio of the split for the subset.
        :param sample_ratio: The ratio of the sample to be used.
        :return: A tuple containing the train dataset and eval dataset of the subset.
        """
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError(
                "Invalid train_ratio. The train_ratio "
                "value should be between 0 and 1 (exclusive).")

        if sample_ratio <= 0 or sample_ratio >= 1:
            raise ValueError(
                "Invalid sample_ratio. The sample_ratio value "
                "should be between 0 and 1 (exclusive).")

        sample_size = int(len(self.dataset) * sample_ratio)

        # randomly select a subset of the dataset
        indices = torch.randperm(len(self.dataset))
        self.dataset = torch.utils.data.Subset(self.dataset, indices[:sample_size])

        train_size = int(len(self.dataset) * train_ratio)
        eval_size = len(self.dataset) - train_size

        if train_size <= 0 or eval_size <= 0:
            raise ValueError(
                "Invalid dataset sizes. Adjust the ratio "
                "value to ensure non-zero splits.")

        return random_split(
            self.dataset, [train_size, eval_size])

    def save_strategy(self):
        return

    def save_model(self, checkpoint_dir):
        """Save model, this call at the end for last save.
        :param checkpoint_dir:
        :return:
        """
        if self.rank > 0:
            return

        checkpoint_file = f"{checkpoint_dir}/last.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_file)
        print(f"Rank: {self.rank} checkpoint saved to {checkpoint_file}")

    def save_checkpoint(self, checkpoint_dir, epoch):
        """
        :param checkpoint_dir:
        :param epoch:
        :return:
        """
        if self.rank > 0:
            return

        epoch_mod = epoch % 3
        checkpoint_file = f"{checkpoint_dir}/{self.module_name}_epoch_{epoch_mod}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_file)
        print(f"Rank: {self.rank} {self.module_name} checkpoint saved to {checkpoint_file}.")

    def load_checkpoint(self, checkpoint_dir) -> int:
        """

        Loading module for resuming training.

        :param checkpoint_dir: Directory location of the checkpoints.
        :return: Last saved epoch from the checkpoint.
        """
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if checkpoint_files:
            checkpoint_file = checkpoint_files[0]
            checkpoint = torch.load(checkpoint_file, map_location={'cuda:1': 'cuda:0'})

            # Check if all required keys are present in the checkpoint
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Checkpoint file {self.module_name} {checkpoint_file} "
                               f"is missing the following keys: {missing_keys}")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Rank: {self.rank} module {self.module_name} "
                  f"loading checkpoint loaded from "
                  f"{checkpoint_file}, epoch: {epoch}")
            return epoch
        else:
            print(f"No checkpoint files found in dir {checkpoint_dir}")
            return 0

    @staticmethod
    def load_model_inference(
            model: torch.nn.Module,
            args: argparse.Namespace,
            device: torch.device = "cpu",
            requires_grad: bool = False,
    ) -> Optional[int]:
        """
        Load model from checkpoint for inference.

        :param model: The model to load the checkpoint into.
        :type model: torch.nn.Module
        :param args: The command-line arguments.
        :type args: argparse.Namespace
        :param device: The device to load the model onto, defaults to "cpu".
        :param requires_grad: by default load and set to inference.
        :type device: str, optional
        :return: The epoch of the loaded checkpoint, or None if no checkpoint is found.
        :rtype: Optional[int]
        """

        checkpoint_path_dir = Path(args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        if not checkpoint_path_dir.is_dir():
            raise ValueError("Indicate path to checkpoint dir.")

        checkpoint_dir = str(checkpoint_path_dir)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if checkpoint_files:
            checkpoint_file = checkpoint_files[0]
            print(f"Found candidate file {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Loading checkpoint loaded from {checkpoint_file}, epoch: {epoch}")

            model.eval()
            for param in model.parameters():
                param.requires_grad = requires_grad
            return epoch

        return 0

    @staticmethod
    def dataset_checker(dataset, _logger):
        """
        Dataset checker

        :param dataset: The dataset to check.
        :param _logger: The logger object to use for logging.
        """
        required_keys = ["label", "rest_api"]

        for data_point in dataset:
            for key in required_keys:
                if key not in data_point:
                    _logger.error(f"Key '{key}' not found in the dataset.")

            rest_call = dataset.action(data_point["label"])
            _logger.info(f"rest recovered: {rest_call}")
            _logger.info(f"rest original: {data_point['rest_api']}")
            _logger.info(f"rest original: {data_point['label']}")

