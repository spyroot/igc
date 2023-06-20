"""
This class is main class that encapsulate trainer logic.
The reason it's done this , I'm going some trainer logic to async io
since we can parallelize some of the training logic.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from typing import Optional, Union, List, Dict

import torch

from .base.igc_base_module import IgcModule
from .base.igc_llm_base_module import LlmModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_llm_module import IgcLanguageModule
from .igc_rl_module import IgcRlModule
from .llm_train_state_encoder import LlmEmbeddingsTrainer
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
        self,
        specs: argparse.Namespace,
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
        self._dataset = None
        """
        :param specs:
        """
        self._dataset = None
        self._metric_logger = None
        self._from_pretrained_fn = from_pretrained
        self._from_pretrained_load_fn = from_pretrained_load_fn
        self._from_pretrained_save_fn = from_pretrained_save_fn

        self._metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        self._directory_path = os.path.abspath(os.path.expanduser(specs.json_data_dir))
        self._dataset_dir = os.path.abspath(specs.dataset_dir)
        self._specs = specs

    @property
    def metric_logger(self):
        """
        :return:
        """
        if self._metric_logger is None:
            self._metric_logger = MetricLogger(
                self._specs.metric_report, **vars(self._specs))
        return self._metric_logger

    @property
    def dataset(self):
        """

        :return:
        """
        if self._dataset is None:
            self._dataset = MaskedJSONDataset(
                self._dataset_dir,
                do_consistency_check=self._specs.do_consistency_check
            )
        return self._dataset

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

            if (self._specs.train == "llm" or self._specs.train == "all") and self._specs.llm is not None:
                llm_module = IgcLanguageModule(self._specs, self.metric_logger, self.dataset)
                llm_module.train()

            if (self._specs.train == "agent" or self._specs.train == "all") and self._specs.rl is not None:
                rl_module = IgcRlModule(self._specs, self.metric_logger, self.dataset)
                rl_module.train()

    def load(
        self,
        specs: argparse.Namespace,
        module_names: Optional[Union[str, List[str]]] = None,
        device: torch.device = "cpu"
    ) -> Dict[str, Union[LlmModule, IgcModule]]:
        """

        Load a module.

        :param module_names: 
        :param device:
        :param specs:
        :param module_names:
        :return:
        """

        modules = IgcLanguageModule.load(specs, device=device, module_names=module_names)
        # IgcRlModule.load(specs, device=device, module_names=module_names)
        return modules

    def run(self):
        """

        :return:
        """
        self._dataset = MaskedJSONDataset(
            self._dataset_dir,
            do_consistency_check=self._specs.do_consistency_check
        )

        # copy last checkpoint as last model with opt etc. so we can use it.
        if self._specs.copy_llm:
            model, _ = from_pretrained_default(self._specs, only_model=True)
            model.to(torch.device("cpu"))
            model.resize_token_embeddings(len(self._dataset.tokenizer))
            model = model.to_bettertransformer()
            model, epoch, model_path = IgcModule.copy_checkpoint(self._specs, "state_encoder", model)
            print("Saved model to checkpoint file: ", model_path)
        elif self._specs.test_llm:
            modules = self.load(self._specs, "state_encoder")
            llm_trainer = LlmEmbeddingsTrainer(
                "llm_trainer", self._specs,
                llm_model=modules["state_encoder"].model,
                llm_tokenizer=self.dataset.tokenizer,
                dataset=self.dataset,
                metric_logger=self.metric_logger,
                is_inference=True
            )
            llm_trainer.test_inference()

        else:
            self.train()
