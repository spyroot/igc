import argparse
import os

import torch
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_llm_module import IgcLanguageModule
from .igc_rl_module import IgcRlModule
from ..ds.redfish_dataset import JSONDataset
from .shared.llm_shared import from_pretrained_default


class IgcMain:
    """
    IGC main class

    """

    def __init__(self, specs: argparse.Namespace, from_pretrained=from_pretrained_default):
        """
        :param specs:
        """
        self._from_pretrained_fn = from_pretrained
        self._metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        self._directory_path = os.path.expanduser(specs.raw_data_dir)
        self._specs = specs

    def train(self):
        """
        Main igc trainer.

        :return:
        """
        if self._specs.train:
            _, tokenizer = self._from_pretrained_fn(self._specs, only_tokenizer=True)
            dataset = JSONDataset(
                self._directory_path, verbose=True, tokenizer=tokenizer)

            if (self._specs.train == "llm" or self._specs.train == "all") and self._specs.llm is not None:
                print("Starting RL training")
                llm_module = IgcLanguageModule(self._specs, self._metric_logger, dataset)
                llm_module.train()

            if (self._specs.train == "agent" or self._specs.train == "all") and self._specs.rl is not None:
                print("Starting RL training")
                rl_module = IgcRlModule(self._specs, self._metric_logger, dataset)
                rl_module.train()

    def load(self, specs: argparse.Namespace, module_name: str,  device: torch.device = "cpu"):
        """Load a module.

        :param device:
        :param specs:
        :param module_name:
        :return:
        """
        _metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        _, tokenizer = self._from_pretrained_fn(self._specs, only_tokenizer=True)

        dataset = JSONDataset(
            self._directory_path,
            verbose=True,
            tokenizer=tokenizer,
            do_consistency_check=specs.do_consistency_check)

        llm_module = IgcLanguageModule(self._specs, self._metric_logger, dataset)
        modules = llm_module.load(specs, device=device, module_name=module_name)
        return modules

    def run(self):
        self.train()
