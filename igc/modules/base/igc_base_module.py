"""
This class is base module for all trainers inherit from this class.
It has basic functionality for all trainers.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import json
import os
import shutil
import tempfile
import urllib
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Optional, Any, Union, Dict, List, Tuple, Callable, Set
from urllib.error import URLError

import accelerate
import pkg_resources
import torch
from torch.utils.data import random_split, Subset, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from .igc_state import IgcBaseState
from ..shared.llm_shared import from_pretrained_default
from ...ds import ds_utils as igc_util
from .igc_specs import make_default_spec
from .igc_metric_logger import MetricLogger
from ...ds.redfish_dataset import JSONDataset
from .igc_tokenize_state import GenericTokenizeState
from ...modules.base.igc_abstract_logger import AbstractLogger
from ...shared.modules_typing import SaveStrategy
from ...shared.shared_torch_utils import get_device

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class DownloadModuleError(Exception):
    """Base class for module download """
    pass


class IgcModule(IgcBaseState):
    """
    This Base igc module, it encapsulates shared logic for all trainers.
    """
    logger = AbstractLogger.create_logger()

    def __init__(
        self,
        module_name: str,
        spec: argparse.Namespace,
        llm_model, llm_tokenizer,
        ds: Optional[Union[JSONDataset, Dataset]] = None,
        metric_logger: Optional[MetricLogger] = None,
        is_inference: Optional[bool] = False,
        device=None
    ):
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
        super().__init__(module_name, spec, device)

        if not isinstance(module_name, str):
            raise TypeError(f"module_name should be a string, received "
                            f"{type(module_name).__name__}.")

        if not isinstance(spec, argparse.Namespace):
            raise TypeError(f"spec should be an instance of argparse.Namespace, "
                            f"received {type(spec).__name__}.")

        if not isinstance(llm_model, PreTrainedModel):
            raise TypeError(f"llm_model should be an instance of PreTrainedModel, "
                            f"received {type(llm_model).__name__}.")

        if not isinstance(llm_tokenizer, PreTrainedTokenizer):
            raise TypeError(
                f"llm_tokenizer should be an instance of PreTrainedTokenizer, "
                f"received {type(llm_tokenizer).__name__}.")

        if ds is not None and not isinstance(ds, (JSONDataset, Dataset)):
            raise TypeError(f"ds should be an instance of JSONDataset or None, "
                            f"received {type(ds).__name__}.")

        if metric_logger is not None and not isinstance(metric_logger, MetricLogger):
            raise TypeError(
                f"metric_logger should be an instance of MetricLogger or None, "
                f"received {type(metric_logger).__name__}.")

        if not isinstance(is_inference, bool):
            raise TypeError(f"is_inference should be a boolean, "
                            f"received {type(is_inference).__name__}.")

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

        self._is_trained = False

        # model param
        self.model = llm_model
        self.model.resize_token_embeddings(len(llm_tokenizer))
        self.tokenizer = llm_tokenizer
        self.module_name = module_name

        self._tokenize_state = None
        self.update_tokenizer_settings(self.tokenizer)
        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size
        self.on_epoch_eval = spec.eval_mode == "on_epoch"

        if not is_inference:
            self.dataset = ds

        self.metric_logger = metric_logger

        self._trainer_args = spec
        self._batch_log = getattr(spec, "batch_log", 10)
        self._default_lr = getattr(spec, "default_lr", 1e-5)
        self._overfit = spec.overfit

        self.optimizer = None
        self._num_workers = getattr(spec, "num_workers", 1)

        # model saving
        self._save_strategy = getattr(spec, "save_strategy", SaveStrategy.EPOCH)
        self._checkpoint_dir = self._prepare_checkpoint_dir()
        self._module_checkpoint_dir = f"{self._checkpoint_dir}/{module_name}"
        os.makedirs(self._module_checkpoint_dir, exist_ok=True)

        # update specs and add all defaults
        self._trainer_specs = make_default_spec(self._trainer_args)

        # configure logger
        self._configure_metric_logger(module_name)
        self.logger.info(f"Model {self.module_name} saving dir {self._module_checkpoint_dir}")
        self._debug_info()

    @property
    def trainer_specs(self):
        """
        Return trainer specs
        :return:
        """
        return self._trainer_specs

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Update tokenizer
        :param tokenizer:
        :return:
        """
        if not isinstance(tokenizer, PreTrainedTokenizer):
            raise ValueError("Invalid Hugging Face tokenizer provided.")

        self.tokenizer = tokenizer
        self.update_tokenizer_settings(self.tokenizer)

    def update_tokenizer_settings(self, llm_tokenizer: PreTrainedTokenizer):
        """
        Update tokenize state

        :param llm_tokenizer:  pre-trained tokenizer
        :return: nothing
        """
        self._tokenize_state = GenericTokenizeState(
            pad_token=llm_tokenizer.pad_token,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token=llm_tokenizer.eos_token,
            eos_token_id=llm_tokenizer.eos_token_id,
            model_pad_token_id=self.model.config.pad_token_id,
            model_eos_token_id=self.model.config.eos_token_id
        )

        self.model.config.eos_token = llm_tokenizer.eos_token
        self.model.config.pad_token = llm_tokenizer.pad_token
        self.model.config.pad_token_id = llm_tokenizer.pad_token_id
        self.model.config.eos_token_id = llm_tokenizer.eos_token_id

    def _debug_info(self):
        """
        :return:
        """
        # Debug logging for initialized parameters
        self.logger.debug(f"IgcBaseModule.__init__ - module_name: {self.module_name}")
        self.logger.debug(f"IgcBaseModule.__init__ - is_inference: {self._is_inference}")

        self.logger.debug("Internal variables:")
        self.logger.debug(f"  - device: {self.device}")
        self.logger.debug(f"  - pad_token: {self._tokenize_state}")
        self.logger.debug(f"  - num_epochs: {self.num_epochs}")
        self.logger.debug(f"  - batch_size: {self.batch_size}")
        self.logger.debug(f"  - checkpoint_dir: {self._module_checkpoint_dir}")
        self.logger.debug(f"  - rank: {self.rank}")

    def _prepare_checkpoint_dir(self):
        """
        Prepares the checkpoint directory.
        :return:
        """
        output_dir = getattr(self._trainer_args, "output_dir", None)
        if output_dir:
            checkpoint_path_dir = Path(output_dir).resolve()
            self._trainer_args.output_dir = str(checkpoint_path_dir)
        else:
            warnings.warn("output_dir is not provided. Using a temporary directory as a fallback.")
            checkpoint_path_dir = Path(tempfile.mkdtemp())
            self._trainer_args.output_dir = str(checkpoint_path_dir)
            os.makedirs(self._trainer_args.output_dir, exist_ok=True)

        checkpoint_path_dir = Path(self._trainer_args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        if not checkpoint_path_dir.is_dir():
            raise ValueError(f"Indicate path to checkpoint dir {checkpoint_path_dir}.")

        return str(checkpoint_path_dir)

    def _configure_metric_logger(self, module_name: str):
        """
        Configures the logger for the module.

        :param module_name: The name of the module.
        """
        if self.metric_logger is not None:
            self.metric_logger.set_logger(self.logger)
            self.metric_logger.set_log_level(self._log_level)

    def get_model(self) -> PreTrainedModel:
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

        return random_split(self.dataset, [train_size, eval_size])

    def split_slice_dataset(
        self,
        train_ratio: float = 0.8,
        sample_ratio: float = 0.01
    ) -> list[Subset[Any]]:
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

        if sample_ratio <= 0. or sample_ratio >= 1.:
            raise ValueError(
                "Invalid sample_ratio. The sample_ratio value "
                "should be between 0 and 1 (exclusive).")

        sample_size = int(len(self.dataset) * sample_ratio)

        indices = torch.randperm(len(self.dataset))
        data = torch.utils.data.Subset(self.dataset, indices[:sample_size])

        train_size = int(len(self.dataset) * train_ratio)
        eval_size = len(self.dataset) - train_size

        if train_size <= 0 or eval_size <= 0:
            raise ValueError(
                "Invalid dataset sizes. Adjust the ratio "
                "value to ensure non-zero splits.")

        return random_split(
            data, [train_size, eval_size])

    @property
    def save_strategy(self):
        """
        :return:
        """
        return self._save_strategy

    def _model_file(self, checkpoint_dir: str = None) -> str:
        """
        :param checkpoint_dir:
        :return:
        """
        if checkpoint_dir is None:
            checkpoint_dir = self._checkpoint_dir

        if len(checkpoint_dir) == 0:
            raise ValueError("Invalid checkpoint dir.Please specify a valid name.")

        if not os.path.isdir(checkpoint_dir):
            raise ValueError(f"Invalid checkpoint dir: {checkpoint_dir}. "
                             "Please specify a valid directory.")

        return os.path.join(checkpoint_dir, f"{self.module_name}_last.pt")

    @staticmethod
    def model_file(model_dir: str, name: str) -> str:
        """
        Save model file as last model

        :param model_dir: a directory to save model
        :param name: a name of module
        :return: a path to model file
        """
        if len(model_dir) == 0:
            raise ValueError("Invalid model_dir. Please specify "
                             "a valid model_dir.")

        if len(name) == 0:
            raise ValueError("Invalid module name. "
                             "Please specify a valid module name.")

        if not os.path.isdir(model_dir):
            raise ValueError(f"Invalid model_dir: {model_dir}. "
                             f"Please specify a valid directory.")

        return os.path.join(model_dir, f"{name}_last.pt")

    @staticmethod
    def can_resume(model_dir: str, name: str):
        """
        Return if model can resume training.

        :param model_dir:
        :param name:
        :return:
        """
        model_file = IgcModule.model_file(model_dir, name)
        if not os.path.exists(model_file):
            warnings.warn(f"Checkpoint file {model_file} not found.")
            return False

    @staticmethod
    def last_checkpoint(module_checkpoint_dir: str):
        """Return last checkpoint file.
        :return:
        """
        checkpoint_file = None

        checkpoint_files = [f for f in os.listdir(module_checkpoint_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(module_checkpoint_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        if checkpoint_files:
            checkpoint_file = checkpoint_files[0]

        return checkpoint_file

    @staticmethod
    def checkpoint_dir(specs: Union[str, argparse.Namespace], module_name: str):
        """Return a path to checkpoint dir given module name and main experiment dir.
        :param specs:
        :param module_name:
        :return:
        """
        if isinstance(specs, argparse.Namespace):
            _checkpoint_path_dir = Path(specs.output_dir)
        elif isinstance(specs, str):
            _checkpoint_path_dir = Path(specs)
        else:
            raise TypeError(
                "Invalid type for 'specs'. Expected argparse.Namespace or str.")

        experiment_dir = _checkpoint_path_dir.resolve()
        if not experiment_dir.is_dir():
            raise ValueError("Path to checkpoint dir invalid.")

        experiment_dir = str(experiment_dir)
        module_checkpoint_dir = f"{experiment_dir}/{module_name}"
        return module_checkpoint_dir, experiment_dir

    @staticmethod
    def copy_checkpoint(
        specs: Union[str, argparse.Namespace],
        module_name: str,
        pre_trained: PreTrainedModel,
        checkpoint_file: str = None,
        device: str = None
    ) -> Tuple[Union[None, PreTrainedModel], int, str]:
        """
        Copy last checkpoint to last saved model to module dir and return
        a new pre-trained model with loaded state.

        if checkpoint_file provided it will use that as source and create
        final model.

        :param specs: specs that contains output_dir attribute
        :param module_name: llm module name
        :param pre_trained: pre-trained model where we load last state
        :param checkpoint_file:  if checkpoint path provided it will use that.
        :param device: pass device to load model
        :return: return pre-trained model, epoch and checkpoint file
        """

        if device is None:
            device = torch.device("cpu")

        if checkpoint_file is None:
            module_dir, experiment_dir = IgcModule.checkpoint_dir(specs, module_name)
            checkpoint_file = IgcModule.last_checkpoint(module_dir)
            last_model_path = IgcModule.model_file(experiment_dir, module_name)

            if checkpoint_file is None:
                print(f"No checkpoint files found in dir {module_dir}")
                return None, 0, last_model_path
        else:
            experiment_dir = os.path.dirname(checkpoint_file)
            last_model_path = IgcModule.model_file(experiment_dir, module_name)

        model = torch.load(checkpoint_file, map_location=device)
        required_keys = ['model_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in model]
        if missing_keys:
            warnings.warn(f"Checkpoint file {checkpoint_file} is "
                          f"missing the following keys: {missing_keys}")
            return None, 0, last_model_path

        if 'optimizer_state_dict' in model:
            model.pop('optimizer_state_dict', None)
        if 'scheduler_state_dict' in model:
            model.pop('scheduler_state_dict', None)

        epoch = model['epoch'] if 'epoch' in model else 0
        try:
            pre_trained.load_state_dict(model['model_state_dict'])
        except RuntimeError as e:
            warnings.warn(f"Error in loading the model state_dict: {e}")
            return None, 0, last_model_path

        for param in pre_trained.parameters():
            param.requires_grad = False

        pre_trained.eval()
        torch.save(
            model, last_model_path
        )

        return model, epoch, last_model_path

    def save_model(self, checkpoint_dir):
        """

        Save model, after we're done training,
        this call at the end for last save.

        All modules save to separate spot, during dataset creation
        Dataset need pull this.

        :param checkpoint_dir:
        :return:
        """
        if self.is_rank_zero():
            # weights
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
            self.logger.warning(
                f"Checkpoint file {model_file} "
                f"is missing the following keys: {missing_keys}")
            return False

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = checkpoint['is_trained']
        return True

    def save_checkpoint(
        self,
        checkpoint_dir,
        epoch: int,
        num_check_points_to_keep: Optional[int] = 3,
        model: Optional[PreTrainedModel] = None,
        optimizer=None,
        scheduler=None,
    ) -> str:
        """
        Save model checkpoint.

        :param scheduler:
        :param optimizer:
        :param model:
        :param checkpoint_dir: a directory for checkpoint
        :param epoch: a checkpoint we are saving.
        :param num_check_points_to_keep:   number of checkpoints to keep.
        :return:  return path where checkpoint saved
        """

        if self.rank > 0:
            return ""

        epoch_mod = epoch % num_check_points_to_keep
        checkpoint_file = f"{checkpoint_dir}/{self.module_name}_epoch_{epoch_mod}.pt"

        _model = model if model is not None else self.model
        _shed = scheduler if scheduler is not None else self.scheduler
        _optimizer = optimizer if optimizer is not None else self.optimizer

        checkpoint = {
            'model_state_dict': _model.state_dict(),
            'epoch': epoch,
            'is_trained': True,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = _optimizer.state_dict()

        if _shed is not None:
            if isinstance(_shed, list):
                checkpoint['scheduler_state_dicts'] = [
                    scheduler.state_dict() for scheduler in _shed]
            else:
                checkpoint['scheduler_state_dict'] = _shed.state_dict()

        torch.save(checkpoint, checkpoint_file)
        self.logger.info(
            f"Rank: {self.rank} {self.module_name} checkpoint saved to {checkpoint_file}.")

        return checkpoint_file

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        resuming: Optional[bool] = True,
        map_location=None
    ) -> int:
        """
        Load model checkpoint for resuming training.

        :param map_location:
        :param resuming:
        :param checkpoint_dir: Directory location of the checkpoints.
        :return: Last saved epoch from the checkpoint.
        """

        map_to = {'cuda:1': 'cuda:0'} if map_location is None else map_location
        get_device()

        # during re-resume we don't load model, we load from checkpoint
        if os.path.isfile(checkpoint_dir):
            checkpoint_file = checkpoint_dir
        else:
            #
            model_file = self._model_file(checkpoint_dir)
            if not resuming:
                if not os.path.exists(model_file):
                    self.logger.info(f"Model file {model_file} not found.")

            self.logger.info(f"Searching for latest checkpoint.")
            checkpoint_file = self.last_checkpoint(checkpoint_dir)

        if checkpoint_file:
            self.logger.info(f"Found latest checkpoint, loading {checkpoint_file}.")
            checkpoint = torch.load(checkpoint_file)

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
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                elif 'scheduler_state_dicts' in checkpoint:
                    for idx, scheduler_state_dict in enumerate(checkpoint['scheduler_state_dicts']):
                        self.scheduler[idx].load_state_dict(scheduler_state_dict)

            epoch = checkpoint['epoch']
            self.logger.info(
                f"Rank: {self.rank} module {self.module_name} "
                f"loading checkpoint loaded from "
                f"{checkpoint_file}, epoch: {epoch}")
            return epoch

        self.logger.info(f"No checkpoint files found in dir {checkpoint_dir}")
        return 0

    @staticmethod
    def load(
        module_name: str,
        model: torch.nn.Module,
        specs: Union[str, argparse.Namespace],
        is_inference: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        map_location=None,
    ) -> tuple[Optional[int], bool]:
        """
        Load model from checkpoint for inference.

        :param module_name: igc module name.
        :param model: The model to load the checkpoint into.
        :param specs: The command-line arguments that must contain the "output_dir" arg or path
        :param is_inference: by default load and set to inference.
        :return: The epoch of the loaded checkpoint, or None if no checkpoint is found.
        :param scheduler:  if resume it should lr_scheduler that we are using.
        :param optimizer: if resume it should optimize that we are using.
        :param map_location: where are mapping the model to.

        """
        if map_location is None:
            map_to = {'cuda:1': 'cuda:0'}
        else:
            map_to = map_location

        module_dir, experiment_dir = IgcModule.checkpoint_dir(specs, module_name)
        last_checkpoint_file = IgcModule.last_checkpoint(module_dir)
        last_module_file = IgcModule.model_file(experiment_dir, module_name)

        last_file = None
        if os.path.exists(last_module_file):
            last_file = last_module_file
        else:
            if last_checkpoint_file is not None and os.path.exists(last_checkpoint_file):
                last_file = last_checkpoint_file

        if last_file is None:
            print(f"No checkpoint or module file found")
            return 0, False

        print(f"Found model file {last_file} loading mapping to {map_to}")
        checkpoint = torch.load(last_file, map_location=map_to)
        required_keys = ['model_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            print(f"Checkpoint file {last_file} "
                  f" is missing the following keys: {missing_keys}")
            return 0, False

        is_trained = checkpoint['is_trained'] if 'is_trained' in checkpoint else False
        if not is_inference:
            if optimizer is not None:
                required_keys.append("optimizer_state_dict")
            if scheduler is not None:
                required_keys.append("scheduler_state_dict")

        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

        print(f"Loading checkpoint loaded from {last_file}, epoch: {epoch}")

        if is_inference:
            # in inference mod we don't need to load optimizer and scheduler
            # so if we loaded from checkpoint we drop that
            if 'optimizer_state_dict' in checkpoint:
                checkpoint.pop('optimizer_state_dict')
            if 'scheduler_state_dict' in checkpoint:
                checkpoint.pop('scheduler_state_dict')
            model.eval()
            for param in model.parameters():
                if is_inference:
                    param.requires_grad = False
        else:
            # if we want train
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

    def log_memory_usage(self):
        """
        Log memory usage.
        :return:
        """
        if torch.cuda.is_available():
            mem_get_info = torch.cuda.memory_stats()
            self.logger.info("Memory allocated:", mem_get_info["allocated_bytes.all.current"] / 1024 ** 3, "GB")
            # additional CUDA statistics if available
            if hasattr(torch.cuda, 'utilization'):
                self.logger.info(f"CUDA utilization:", torch.cuda.utilization())
            if hasattr(torch.cuda, 'memory_summary'):
                torch.cuda.memory_summary()

    @staticmethod
    def read_model_specs(
        default_model_file: str = "../datasets/models.json"
    ):
        """Read model specs file.

        :return:  The model specs data and root dir where the modules need to be saved
        """

        dataset_path = pkg_resources.resource_filename(
            "igc", default_model_file)

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"The model specs file '{dataset_path}' does not exist.")

        try:
            with open(dataset_path, "r") as file:
                models_data = json.load(file)

        except Exception as e:
            raise ValueError(
                f"Failed to parse the model specs file '{dataset_path}': {str(e)}")

        required_keys = ["mirrors"]
        for key in required_keys:
            if key not in models_data or not isinstance(models_data[key], dict):
                raise ValueError(
                    f"Invalid model specs file. '{key}' key is missing or not a dictionary.")

        for mirror_url, mirror_entries in models_data["mirrors"].items():
            if not isinstance(mirror_entries, list):
                raise ValueError(
                    f"Invalid model specs file. Mirror entries for '{mirror_url}' is not a list.")

            mandatory_keys = ["spec", "files", "local_file"]

            for entry in mirror_entries:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Invalid model specs file. "
                        f"Mirror entry for '{mirror_url}' is not a dictionary.")

        return models_data, os.path.abspath(os.path.dirname(dataset_path))

    @staticmethod
    def _download_module(
        url: str,
        path: str,
        filename: Optional[str] = None,
        checksum: Optional[str] = None,
        overwrite: Optional[bool] = False,
        retry: int = 5,
        is_strict=False
    ) -> tuple[bool, str]:
        """
        Download igc module file from url and store in default location.
        Each module is under datasets/modules.

        Method will create all directory.

        :param overwrite: if we need overwrite, no checksum check.
        :param is_strict: if we couldn't find any raise exception otherwise it just warnings.
        :param path: where want to save a file.
        :param url: link to a file.
        :param filename:  Name to save the file under. If None, use the basename of the URL.
        :param checksum:  Checksum of the download. If None, or empty string will not do check.
        :param retry: num retry
        :return:
        """
        logger = AbstractLogger.create_logger()
        logger.info("Downloading module...")

        if not isinstance(url, str):
            raise DownloadModuleError(f"The 'url' argument must "
                                      f"be a string, not {type(url).__name__}.")

        if not isinstance(path, str):
            raise DownloadModuleError(f"The 'path' argument must "
                                      f"be a string, not {type(path).__name__}.")

        if filename is not None and not isinstance(filename, str):
            raise DownloadModuleError(f"The 'filename' argument must "
                                      f"be a string or None, not {type(filename).__name__}.")

        if checksum is not None and not isinstance(checksum, str):
            raise DownloadModuleError(f"The 'checksum' argument must "
                                      f"be a string or None, not {type(checksum).__name__}.")

        if not isinstance(overwrite, bool):
            raise DownloadModuleError(f"The 'overwrite' argument must "
                                      f"be a boolean, not {type(overwrite).__name__}.")

        if not isinstance(retry, int):
            raise DownloadModuleError(f"The 'retry' argument must "
                                      f"be an integer, not {type(retry).__name__}.")

        if not isinstance(is_strict, bool):
            raise DownloadModuleError(f"The 'is_strict' argument must "
                                      f"be a boolean, not {type(is_strict).__name__}.")

        root_dir = Path(path).expanduser()
        if Path(root_dir).is_dir():
            logger.debug("Creating directory structure.".format(str(root_dir)))
            os.makedirs(root_dir, exist_ok=True)

        if not filename:
            filename = os.path.basename(url)

        full_path = root_dir / filename
        full_path = full_path.resolve()

        # check if file is already present locally
        if not overwrite:
            # we check checksum if needed.
            if checksum is not None and len(checksum) > 0 and full_path.exists():
                # check integrity
                if not igc_util.check_integrity(str(full_path), checksum):
                    warnings.warn(f"Checksum mismatched for a file: {str(full_path)}")
                    return False, ""
                else:
                    return True, str(full_path)
            else:
                if full_path.exists():
                    hash_checksum = igc_util.md5_checksum(str(full_path))
                    warnings.warn("File already exists. hash {}".format(hash_checksum))
                    return full_path.exists(), str(full_path)
                else:
                    logger.debug("File not not found {}".format(str(full_path)))

        logger.debug("Making http head request {}".format(url))
        final_url = igc_util.do_http_head(url, max_redirect=retry)
        try:
            logger.info(
                f"Fetching {url} location {full_path}."
            )
            igc_util.fetch_content(final_url, str(full_path))

        except (urllib.error.URLError, OSError) as e:
            warnings.warn("Failed to fetch".format(final_url))
            if is_strict:
                raise e

        # check integrity of downloaded file
        if checksum is not None and full_path.exists():
            if not igc_util.check_integrity(str(full_path), checksum):
                warnings.warn(f"Checksum {checksum} mismatch.")
                return False, ""

        logger.info(f"Dataset exists {full_path.exists()} and path {str(full_path)}")
        return full_path.exists(), str(full_path)

    @staticmethod
    def download_module(
        mirror_url: str,
        module_dir: str,
        module_filename: str,
        checksum: str = None,
        is_overwrite: Optional[bool] = False,
    ):
        """
        Download a module file from the specified mirror URL.

        :param module_dir: a directory where we want to save a file.
        :param mirror_url: The URL of the mirror from which to download the file.
        :param module_filename: The name of the file to be downloaded.
        :param checksum: Optional. The checksum value for the file. If provided,
                         the downloaded file's checksum will be
                         verified against this value.

        :param is_overwrite:  overwrite existing file.
        :return: True if the file is downloaded successfully, False otherwise.
        """

        logger = IgcModule.logger
        _dataset_file = []

        if not isinstance(mirror_url, str):
            raise DownloadModuleError(
                f"mirror_url should be a string, received {type(mirror_url)}")
        if not isinstance(module_filename, str):
            raise DownloadModuleError(
                f"_filename should be a string, received {type(module_filename)}")
        if checksum is not None and not isinstance(checksum, str):
            raise DownloadModuleError(
                f"checksum should be a string or None, received {type(checksum)}")

        try:
            logger.debug(f"Downloading from mirror: {mirror_url} file: {module_filename}")
            if checksum is not None:
                logger.debug(f"Using checksum: {checksum}")
            else:
                logger.debug(f"No md5 checksum provided.")

            if checksum is not None:
                if igc_util.check_integrity(f"{module_dir}/{module_filename}", md5=checksum):
                    return True

            _is_downloaded_done, file_path = IgcModule._download_module(
                url=mirror_url,
                path=module_dir,
                filename=module_filename,
                checksum=checksum,
                overwrite=is_overwrite
            )
            _dataset_file.append(file_path)
            if _is_downloaded_done:
                logger.debug("All file in the system: {}".format(file_path))
                return True

        except URLError as e:
            logger.debug(
                "Failed to download {} {}. "
                "Moving to the next mirror.".format(mirror_url, module_filename))
            logger.error(e)

        return False

    @staticmethod
    def download_modules(mirrors: Union[str, Dict]):
        """
        Download multiple module files from the specified mirror URLs.
        Either single string or dictionary of mirror URLs can be provided.
        :return:
        """
        # If mirrors is a string,  single mirror URL
        if isinstance(mirrors, str):
            mirrors = {mirrors: None}
        elif isinstance(mirrors, dict):
            pass
        else:
            raise DownloadModuleError(
                f"mirrors should be a string "
                f"or a dictionary, received {type(mirrors).__name__}.")

        logger = IgcModule.logger
        logger.info("Downloading module files...")

        all_files_downloaded = True

        for mirror_url, files in mirrors.items():
            if isinstance(files, str):
                # If files is a string, assume it's a single file without a specified local path
                files = {files: None}
            elif not isinstance(files, dict) and files is not None:
                raise DownloadModuleError(
                    f"The files for mirror URL {mirror_url} should be a string or a dictionary, "
                    f"received {type(files).__name__}.")

            logger.debug(f"Downloading from mirror: {mirror_url}")

            for remote_file, local_file in files.items():
                if isinstance(local_file, str) and local_file:
                    local_path = local_file
                elif local_file is None:
                    local_path = None
                else:
                    raise DownloadModuleError(
                        f"The local file path for {remote_file} should be a string or None, "
                        f"received {type(local_file).__name__}.")

                logger.debug(f"Downloading file: {remote_file}")
                download_result = IgcModule.download_module(
                    mirror_url, remote_file, is_overwrite=True
                )

                if download_result:
                    logger.info(f"Downloaded file: {remote_file}")
                    if local_path is not None:
                        shutil.move(download_result[1], local_path)
                        logger.info(f"Moved file to: {local_path}")
                else:
                    logger.error(f"Failed to download file: {remote_file}")
                    all_files_downloaded = False

        return all_files_downloaded

    @staticmethod
    def download() -> Set[str]:
        """
        Download the modules specified in the 'models.json'
        file and save them to the appropriate locations for each model.

        :return: A set of downloaded module files.
        """

        module_files = set()
        models_data, package_root_dir = IgcModule.read_model_specs()
        logger = IgcModule.logger
        logger.info(f"Downloading module to {package_root_dir}")

        for mirror, mirror_entries in models_data["mirrors"].items():
            module_remote_files = [mirror_entry[k] for mirror_entry in mirror_entries
                                   for k in mirror_entry if k == "files"][0]
            module_local_files = [mirror_entry[k] for mirror_entry in mirror_entries
                                  for k in mirror_entry if k == "local_file"][0]

            module_local_dirs = [os.path.join(package_root_dir, os.path.dirname(file))
                                 for file in module_local_files]
            module_local_files = [os.path.join(package_root_dir, file)
                                  for file in module_local_files]

            _download_result = []
            for url, module_dir, module_file in zip(
                module_remote_files, module_local_dirs, module_local_files):
                logger.debug(f"Downloading module from {url}")
                logger.debug(f"Downloading module to {module_file}")
                os.makedirs(module_dir, exist_ok=True)
                if os.path.exists(module_file):
                    logger.debug(f"File {module_file} is already present. Skipping download.")
                    _download_result.append(True)
                    module_files.add(module_file)
                else:
                    logger.debug(f"Downloading module from {url}")
                    logger.debug(f"Downloading module to {module_file}")
                    os.makedirs(module_dir, exist_ok=True)
                    download_result = IgcModule.download_module(
                        url, module_dir, module_file, is_overwrite=True
                    )
                    _download_result.append(download_result)
                    module_files.add(module_file)

            if all(_download_result):
                break

        return module_files

    @staticmethod
    def checkpoint_to_module(
        spec: Union[str, argparse.Namespace],
        module_name: str,
        pre_trained_tokenizer: Optional[PreTrainedTokenizer] = None,
        pre_trained_callback: Optional[Callable] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[Union[None, PreTrainedModel], int, str]:
        """
        Download all modules, if modules contains only checkpoints from last save.
        Copy each checkpoint for a module and convert to final model file.

        Note this is mainly required if you train on one host and another host
        you want to consume fined tuned llm for downstream task.

        :param spec: The specification of the model to download (e.g., model name or configuration file path).
        :param module_name: The name of the module to download.
        :param pre_trained_callback: Optional. A callback function to customize the loading of the pre-trained model.
                    If not provided, a default callback will be used.
        :param device:  a default device where we use the model.
        :param pre_trained_tokenizer: a callback function to create a tokenizer.
        :return: A tuple containing the pre-trained model, the epoch of the loaded checkpoint,
                and the file path of the final model.
        """

        if not isinstance(spec, (str, argparse.Namespace)):
            raise ValueError("The 'spec' argument must be a string or argparse.Namespace.")

        if not isinstance(module_name, str):
            raise ValueError("The 'module_name' argument must be a string.")

        if pre_trained_callback is not None and not callable(pre_trained_callback):
            raise ValueError("The 'pre_trained_callback' argument must be a callable or None.")

        if device is None:
            device = torch.device("cpu")

        if pre_trained_callback is None:
            pre_trained_callback = from_pretrained_default

        pre_trained, _ = pre_trained_callback(
            spec, only_model=True, only_tokenizer=False
        )

        if pre_trained_tokenizer is not None:
            pre_trained.resize_token_embeddings(len(pre_trained_tokenizer))
            pre_trained_tokenizer.pad_token = pre_trained_tokenizer.eos_token
            pre_trained_tokenizer.pad_token_id = pre_trained_tokenizer.eos_token_id
            pre_trained.config.pad_token_id = pre_trained_tokenizer.pad_token_id
            pre_trained.config.eos_token_id = pre_trained_tokenizer.eos_token_id
            pre_trained.config.pad_token = pre_trained_tokenizer.pad_token
            pre_trained.config.eos_token = pre_trained_tokenizer.eos_token

        print(pre_trained)

        files = IgcModule.download()
        if not files:
            raise DownloadModuleError(
                f"Failed to download module files for {module_name}.")

        largest_epoch = 0
        last_checkpoint_file = None
        for checkpoint_file in files:
            if module_name in checkpoint_file and os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file, map_location=device)
                epoch = checkpoint['epoch']
                if epoch > largest_epoch:
                    largest_epoch = epoch
                    last_checkpoint_file = checkpoint_file
                del checkpoint

        if last_checkpoint_file is None:
            raise ValueError("No last checkpoint file found.")

        return IgcModule.copy_checkpoint(
            spec, module_name,
            pre_trained=pre_trained,
            checkpoint_file=last_checkpoint_file
        )
