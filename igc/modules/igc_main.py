"""
This class is main class that encapsulate trainer logic.
The reason it's done this , I'm going some trainer logic to async io
since we can parallelize some of the training logic.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os

import torch
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_llm_module import IgcLanguageModule
from .igc_rl_module import IgcRlModule
from ..ds.redfish_dataset import JSONDataset
from .shared.llm_shared import (
    from_pretrained_default,
    load_pretrained_default,
    save_pretrained_default
)
from ..ds.redfish_masked_dataset import MaskedJSONDataset


class IgcMain:
    """
    IGC main class

    """
    def __init__(
        self, specs: argparse.Namespace,
        from_pretrained=from_pretrained_default,
        from_pretrained_load_fn=load_pretrained_default,
        from_pretrained_save_fn=save_pretrained_default,
    ):
        """
        from_pretrained_default creates initial model.
        if model saved in hugging face format we ue load_pretrained_default and save_pretrained_default
        to save and load model.

        :param specs: An `argparse.Namespace` object containing the specifications and arguments.
        :param from_pretrained: A function for loading the pretrained model and tokenizer.
        :param from_pretrained_load_fn: A function for loading a pretrained model from a directory.
        :param from_pretrained_save_fn: A function for saving the pretrained model and tokenizer.
        """
        """
        :param specs:
        """
        self._from_pretrained_fn = from_pretrained
        self._from_pretrained_load_fn = from_pretrained_load_fn
        self._from_pretrained_save_fn = from_pretrained_save_fn

        self._metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        self._directory_path = os.path.expanduser(specs.raw_data_dir)
        self._specs = specs

    def train(self):
        """
        Main igc trainer.

        * Fine tune llm and save the model experiments/state_encoder
        * Training auto encoder and save the model experiments/state_auto_encoder
        Optional:
                * Use fine-tuned model and train goal extractor.
                * Use fined-tuned model and trains sub-goal and parameters extractor.

        * Use tune tuned model and load state_auto_encoder
        * Create vectorized rest api vectorized env and use auto encoder to compress state.
        * Train RL agent.

        :return:
        """
        if self._specs.train:
            dataset = MaskedJSONDataset(
                "datasets",
                do_consistency_check=False
            )
            if (self._specs.train == "llm" or self._specs.train == "all") and self._specs.llm is not None:
                llm_module = IgcLanguageModule(self._specs, self._metric_logger, dataset)
                llm_module.train()

            if (self._specs.train == "agent" or self._specs.train == "all") and self._specs.rl is not None:
                rl_module = IgcRlModule(self._specs, self._metric_logger, dataset)
                rl_module.train()

    def load(self, specs: argparse.Namespace, module_name: str,  device: torch.device = "cpu"):
        """

        Load a module.

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
            do_consistency_check=specs.do_consistency_check)

        llm_module = IgcLanguageModule(
            self._specs, self._metric_logger, dataset
        )
        modules = llm_module.load(
            specs, device=device, module_name=module_name
        )
        return modules

    def run(self):
        self.train()
