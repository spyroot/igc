"""
This class is base module for all trainers
inherit from this class.  It has basic functionality
for all trainers.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import loguru
import torch
import torch.distributed as dist
from accelerate import Accelerator

from igc.shared.shared_torch_utils import get_device
from ...shared.shared_accelerator import build_accelerator


class IgcBaseState:
    """
    This base state shared in trainer , modules.
    Rank, Devices, Logging
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 device=None):
        """
        This only shared states,  devices, accelerators, logging, etc.

        :param module_name:
        :param spec:
        :param device:
        """
        if not isinstance(spec, argparse.Namespace):
            raise TypeError(f"spec should be an instance of argparse.Namespace, "
                            f"received {type(spec).__name__}.")

        self._log_file = None
        self._is_trained = False
        self._trainer_args = spec

        self.logger = loguru.logger
        self._configure_logger(module_name)
        self.rank = int(os.environ.get('LOCAL_RANK', -1))

        self.is_accelerator = False
        if device is not None:
            self._device = torch.device(device)

        self._accelerator = None

        if hasattr(spec, 'use_accelerator') and spec.use_accelerator:
            self._accelerator = build_accelerator(spec)
            self.is_accelerator = True
            # let accelerator choose device.
            self._device = self._accelerator.device
            self.logger.info(f"Rank {self.rank}, Running accelerator , accelerate selected device {self.device}")
        elif hasattr(spec, "device"):
            self._device = spec.device
        else:
            self.logger.info(f"Rank {self.rank}, Running on selected device {self.device}")
            # if we are not using accelerator, we need to set device
            self._device = get_device(self.rank) if device is None else device
            raise

        self.scheduler = None



    @property
    def accelerator(self) -> Accelerator:
        """Return accelerator.
        :return:
        """
        return self._accelerator

    @property
    def device(self):
        """Return ether cuda or cpu device or accelerator device.
        :return:
        """
        if self._accelerator:
            return self.accelerator.device
        else:
            return self._device

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
        logs_dir = getattr(self._trainer_args, "log_dir", None)
        if not logs_dir:
            warnings.warn("log_dir is not provided. "
                          "Using a temporary directory as a fallback.")
            logs_dir = tempfile.mkdtemp()
        else:
            os.makedirs(logs_dir, exist_ok=True)

        self._log_file = os.path.join(logs_dir, f"{module_name}.log")
        self._log_level = getattr(
            self._trainer_args, "log_level", "ERROR"
        )
        if self._log_level:
            self._log_level = self._log_level.upper()

        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()

        log_to_file = getattr(self._trainer_args, "log_to_file", None)
        if not log_to_file:
            self.logger.add(sys.stdout, level=self._log_level)
        else:
            if self._trainer_args.log_to_file:
                log_file = os.path.join(logs_dir, f"{module_name}.log")
                self.logger.add(log_file, level=self._log_level)
            else:
                self.logger.add(sys.stdout, level=self._log_level)

    def is_distributed(self):
        """
        :return:
        """
        return self.rank != -1

    def is_rank_zero(self):
        """
        :return:
        """
        return self.rank == -1 or self.rank == 0

    def is_last_worker(self):
        """

        :return:
        """
        if self.rank == -1:
            return True

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        is_last_worker = rank == (world_size - 1)
        return is_last_worker

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

    def show_accelerator_info(self):
        """
        Show information about the accelerator used.
        """
        if self.is_accelerator:
            self.logger.debug("Accelerator Configuration:")
            self.logger.debug(f"is main process: {self.accelerator.is_main_process}")
            self.logger.debug(f"Distributed Training: {self.accelerator.distributed_type}")
            self.logger.debug(f"Device Placement: {self.accelerator.device_placement}")
            self.logger.debug(f"Split Batches: {self.accelerator.split_batches}")
            self.logger.debug(f"Mixed Precision: {self.accelerator.mixed_precision}")
            self.logger.debug(f"Project Directory: {self.accelerator.project_dir}")
            self.logger.debug(f"Dispatch Batches: {self.accelerator.dispatch_batches}")
            self.logger.debug(f"Even Batches: {self.accelerator.even_batches}")
            self.logger.debug(f"RNG Types: {self.accelerator.rng_types}")
            self.logger.debug(f"Log With: {self.accelerator.log_with}")
            self.logger.debug(f"Step Scheduler with Optimizer: {self.accelerator.step_scheduler_with_optimizer}")
