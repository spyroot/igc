"""
This class is base module for all trainers
inherit from this class.  It has basic functionality
for all trainers.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from collections import namedtuple
from typing import Optional, Any

import loguru
import torch
from torch.utils.data import random_split, Subset
from transformers import PreTrainedModel, PreTrainedTokenizer

from igc.ds.redfish_dataset import JSONDataset
from igc.shared.shared_torch_utils import get_device

from .igc_metric_logger import MetricLogger
from .igc_specs import make_default_spec

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class IgcBaseModule:
    """
    This Base igc module, it encapsulates shared logic for all trainers.
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = False):
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
        if not isinstance(module_name, str):
            raise TypeError(f"module_name should be a string, received {type(module_name).__name__}.")

        if not isinstance(spec, argparse.Namespace):
            raise TypeError(f"spec should be an instance of argparse.Namespace, received {type(spec).__name__}.")

        if not isinstance(llm_model, PreTrainedModel):
            raise TypeError(f"llm_model should be an instance of PreTrainedModel, received {type(llm_model).__name__}.")

        if not isinstance(llm_tokenizer, PreTrainedTokenizer):
            raise TypeError(
                f"llm_tokenizer should be an instance of PreTrainedTokenizer, received {type(llm_tokenizer).__name__}.")

        if ds is not None and not isinstance(ds, JSONDataset):
            raise TypeError(f"ds should be an instance of JSONDataset or None, received {type(ds).__name__}.")

        if metric_logger is not None and not isinstance(metric_logger, MetricLogger):
            raise TypeError(
                f"metric_logger should be an instance of MetricLogger or None, received {type(metric_logger).__name__}.")

        if not isinstance(is_inference, bool):
            raise TypeError(f"is_inference should be a boolean, received {type(is_inference).__name__}.")

        self._is_inference = is_inference

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

        if not is_inference and ds is None:
            raise ValueError("ds (dataset) cannot be None.")

        self._log_file = None
        self.logger = loguru.logger
        self._is_trained = False

        self.model = llm_model
        self.tokenizer = llm_tokenizer
        self.update_tokenizer_settings(self.tokenizer)
        self.module_name = module_name

        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size
        self.on_epoch_eval = spec.eval_mode == "on_epoch"

        if not is_inference:
            self.dataset = ds

        self.device = get_device()
        self.metric_logger = metric_logger

        self._trainer_args = spec
        self._batch_log = 10
        self._default_lr = 1e-5
        self._overfit = spec.overfit

        self.optimizer = None

        # model saving
        self.save_strategy = spec.save_strategy
        self.checkpoint_dir = self._prepare_checkpoint_dir()
        self.rank = int(os.environ.get('LOCAL_RANK', -1))

        # update specs and add all defaults
        self._trainer_specs = make_default_spec(self._trainer_args)

        # configure logger
        self._configure_logger(module_name)
        self.logger.info(f"Model {self.module_name} saving dir {self.checkpoint_dir}")
        self._debug_info()

    def set_tokenizer(self, tokenizers):
        """Update tokenizer
        :param tokenizers:
        :return:
        """
        self.tokenizer = tokenizers
        self.update_tokenizer_settings(self.tokenizer)

    def update_tokenizer_settings(self, llm_tokenizer):
        """Update tokenizer settings
        :return:
        """
        self.tokenizer.pad_token = llm_tokenizer.eos_token
        self.tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        self.model.config.pad_token_id = llm_tokenizer.pad_token_id
        self.pad_token = llm_tokenizer.pad_token
        self.pad_token_id = llm_tokenizer.pad_token_id

    def _debug_info(self):
        """
        :return:
        """
        # Debug logging for initialized parameters
        self.logger.debug(f"IgcBaseModule.__init__ - module_name: {self.module_name}")
        self.logger.debug(f"IgcBaseModule.__init__ - is_inference: {self._is_inference}")

        self.logger.debug("Internal variables:")
        self.logger.debug(f"  - device: {self.device}")
        self.logger.debug(f"  - pad_token: {self.pad_token}")
        self.logger.debug(f"  - pad_token_id: {self.pad_token_id}")
        self.logger.debug(f"  - num_epochs: {self.num_epochs}")
        self.logger.debug(f"  - batch_size: {self.batch_size}")
        self.logger.debug(f"  - checkpoint_dir: {self.checkpoint_dir}")
        self.logger.debug(f"  - rank: {self.rank}")

    def _prepare_checkpoint_dir(self):
        """
        Prepares the checkpoint directory.

        :return:
        """
        checkpoint_path_dir = Path(self._trainer_args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        if not checkpoint_path_dir.is_dir():
            raise ValueError(f"Indicate path to checkpoint dir {checkpoint_path_dir}.")

        return str(checkpoint_path_dir)

    def _configure_logger(self, module_name: str):
        """
        Configures the logger for the module.

        :param module_name: The name of the module.
        """
        logs_dir = self._trainer_args.log_dir or "logs"
        os.makedirs(logs_dir, exist_ok=True)

        self._log_file = os.path.join(logs_dir, f"{module_name}.log")
        self._log_level = self._trainer_args.log_level.upper()
        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()

        if self._trainer_args.log_to_file:
            log_file = os.path.join(logs_dir, f"{module_name}.log")
            self.logger.add(log_file, level=self._log_level)
        else:
            self.logger.add(sys.stdout, level=self._log_level)

        if self.metric_logger is not None:
            self.metric_logger.set_logger(self.logger)
            self.metric_logger.set_log_level(self._log_level)

    def get_model(self):
        """Return module model.
        :return:
        """
        return self.model

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

    def _model_file(self, checkpoint_dir):
        return f"{checkpoint_dir}/{self.module_name}_last.pt"

    @staticmethod
    def model_file(model_dir: str, name: str):
        return f"{model_dir}/{name}_last.pt"

    @staticmethod
    def can_resume(model_dir: str, name: str):
        """
        Return if model can resume training.

        :param model_dir:
        :param name:
        :return:
        """
        model_file = IgcBaseModule.model_file(model_dir, name)
        if not os.path.exists(model_file):
            warnings.warn(f"Checkpoint file {model_file} not found.")
            return False

    def save_model(self, checkpoint_dir):
        """

        Save model, after we're done training,
        this call at the end for last save.

        All modules save to separate spot, during dataset creation
        Dataset need pull this.

        :param checkpoint_dir:
        :return:
        """
        if self.rank > 0:
            return

        checkpoint_file = self._model_file(checkpoint_dir)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'is_trained': True,
        }, checkpoint_file)

        print(f"Rank: {self.rank} "
              f"module name {self.module_name} "
              f"checkpoint saved to {checkpoint_file}")

    def load_model(self, checkpoint_dir, map_location=None) -> bool:
        """
        Load the last saved model.

        :param map_location:
        :param checkpoint_dir: The directory containing the model.
        """
        if map_location is None:
            map_location = {'cuda:1': 'cuda:0'}

        model_file = self._model_file(checkpoint_dir)
        if not os.path.exists(model_file):
            self.logger.warning(f"Checkpoint file {model_file} not found.")
            return False

        checkpoint = torch.load(model_file, map_location=map_location)
        required_keys = ['model_state_dict', 'is_trained']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            self.logger.warning(f"Checkpoint file {model_file} "
                                f"is missing the following keys: {missing_keys}")
            return False

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint['is_trained']

    def save_checkpoint(self, checkpoint_dir, epoch: int, num_check_points_to_keep: Optional[int] = 3):
        """
        Save model checkpoint.

        :param checkpoint_dir: a directory for checkpoint
        :param epoch: a checkpoint we are saving.
        :param num_check_points_to_keep:   number of checkpoints to keep.
        :return:
        """
        if self.rank > 0:
            return

        epoch_mod = epoch % num_check_points_to_keep
        checkpoint_file = f"{checkpoint_dir}/{self.module_name}_epoch_{epoch_mod}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'is_trained': True,
        }, checkpoint_file)
        self.logger.info(f"Rank: {self.rank} {self.module_name} checkpoint saved to {checkpoint_file}.")

    def load_checkpoint(self, checkpoint_dir: str, resuming="True") -> int:
        """
        Load model checkpoint for resuming training.

        :param resuming:
        :param checkpoint_dir: Directory location of the checkpoints.
        :return: Last saved epoch from the checkpoint.
        """
        # during re-resume we don't load model, we load from checkpoint
        model_file = self._model_file(checkpoint_dir)
        base_model_name = os.path.basename(self._model_file(checkpoint_dir))

        if not resuming:
            if not os.path.exists(model_file):
                self.logger.info(f"Model file {model_file} not found.")

        self.logger.info(f"Searching for latest checkpoint.")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if
                            f.endswith('.pt') and f != base_model_name]
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if checkpoint_files:
            checkpoint_file = checkpoint_files[0]
            self.logger.info(f"Found latest checkpoint, loading {checkpoint_file}.")
            checkpoint = torch.load(checkpoint_file, map_location={'cuda:1': 'cuda:0'})

            required_keys = ['model_state_dict', 'epoch']
            if resuming:
                required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']

            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise KeyError(f"Checkpoint file {self.module_name} {checkpoint_file} "
                               f"is missing the following keys: {missing_keys}")

            optional_keys = ['is_trained']
            missing_keys = [key for key in optional_keys if key not in checkpoint]
            if missing_keys:
                warnings.warn("Optional key is missing from the checkpoint file. ")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            if resuming:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint['epoch']
            self.logger.info(
                f"Rank: {self.rank} module {self.module_name} "
                f"loading checkpoint loaded from "
                f"{checkpoint_file}, epoch: {epoch}")
            return epoch
        else:
            self.logger.info(f"No checkpoint files found in dir {checkpoint_dir}")
            return 0

    @staticmethod
    def load(
            module_name: str,
            model: torch.nn.Module,
            specs: argparse.Namespace,
            device: torch.device = "cpu",
            is_inference: bool = True,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,

    ) -> tuple[Optional[int], bool]:
        """
        Load model from checkpoint for inference.

        :param scheduler:
        :param optimizer:
        :param module_name:  module name.
        :param model: The model to load the checkpoint into.
        :type model: torch.nn.Module
        :param specs: The command-line arguments.
        :type specs: argparse.Namespace
        :param device: The device to load the model onto, defaults to "cpu".
        :param is_inference: by default load and set to inference.
        :type device: str, optional
        :return: The epoch of the loaded checkpoint, or None if no checkpoint is found.
        :rtype: Optional[int] bool
        """

        checkpoint_path_dir = Path(specs.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        if not checkpoint_path_dir.is_dir():
            raise ValueError("Indicate path to checkpoint dir.")

        checkpoint_dir = str(checkpoint_path_dir)

        model_file = IgcBaseModule.model_file(module_name, checkpoint_dir)
        if os.path.exists(model_file):
            checkpoint_file = model_file
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
            checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            if checkpoint_files:
                checkpoint_file = checkpoint_files[0]
            else:
                print(f"No checkpoint files found in dir {checkpoint_dir}")
                return 0, False

        print(f"Found model file {checkpoint_file} loading to device {device}")

        checkpoint = torch.load(checkpoint_file, map_location=device)
        required_keys = ['model_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            print(f"Checkpoint file {checkpoint_file} "
                  f"is missing the following keys: {missing_keys}")
            return 0, False

        optional_keys = ['is_trained']
        missing_keys = [key for key in optional_keys if key not in checkpoint]
        if missing_keys:
            warnings.warn("Optional key is missing from the checkpoint file. ")
            is_trained = True
        else:
            is_trained = checkpoint['is_trained']

        if optimizer is not None:
            required_keys.append("optimizer_state_dict")
        if scheduler is not None:
            required_keys.append("scheduler_state_dict")

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

        print(f"Loading checkpoint loaded from {checkpoint_file}, epoch: {epoch}")

        if is_inference:
            model.eval()
            for param in model.parameters():
                if is_inference:
                    param.requires_grad = False
        else:
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return epoch, is_trained

    def is_trained(self) -> bool:
        """
        Check if the model has been trained,
        this flag post model train procedure.

        :return: True if the model has been trained, False otherwise.
        """
        return self._is_trained

    @staticmethod
    def dataset_checker(dataset, global_logger):
        """
        Dataset checker,  checks if the dataset is valid and
        has all data that we need.

        :param dataset: The dataset to check.
        :param global_logger: The logger object to use for logging.
        """
        required_keys = ["label", "rest_api"]

        for data_point in dataset:
            for key in required_keys:
                if key not in data_point:
                    global_logger.error(f"Key '{key}' not found in the dataset.")

            rest_call = dataset.action(data_point["label"])
            global_logger.info(f"rest recovered: {rest_call}")
            global_logger.info(f"rest original: {data_point['rest_api']}")
            global_logger.info(f"rest original: {data_point['label']}")
