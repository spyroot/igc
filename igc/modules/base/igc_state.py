"""
This class is base module for all trainers
inherit from this class.  It has basic functionality
for all trainers.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
import sys
from pathlib import Path

import loguru
import torch

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

        self.is_accelerator = False
        if "use_accelerator" in spec and spec.use_accelerator:
            self.accelerator = build_accelerator(spec)
            self.is_accelerator = True
            # let accelerator choose device.
            self.device = self.accelerator.device
        elif hasattr(spec, "device"):
            self.device = spec.device
        else:
            # if we are not using accelerator, we need to set device
            self.device = get_device() if device is None else device

        self._log_file = None
        self._is_trained = False
        self._trainer_args = spec

        self.logger = loguru.logger
        self._configure_logger(module_name)

        self.rank = int(os.environ.get('LOCAL_RANK', -1))

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
