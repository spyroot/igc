import sys
import argparse
from typing import Optional, Dict, Union, List

import loguru
import torch
from igc.ds.redfish_dataset import JSONDataset
from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.llm_train_goal_extract import GoalExtractorTrainer
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.modules.base.igc_llm_base_module import LlmBaseModule
from igc.modules.base.igc_metric_logger import MetricLogger


class IgcLanguageModule:
    """
    """
    modules = ["goal_extractor", "parameter_extractor", "state_encoder", "state_autoencoder"]

    def __init__(self,
                 spec: argparse.Namespace,
                 metric_logger: MetricLogger,
                 ds: JSONDataset,
                 from_pretrained=from_pretrained_default):
        """
        :param spec:
        """

        if spec is None:
            raise ValueError("Specs cannot be None")

        self.modules = ["goal_extractor", "parameter_extractor", "state_encoder", "state_autoencoder"]

        self._from_pretrained_fn = from_pretrained
        self.metric_logger = metric_logger
        self.spec = spec
        self.ds = ds
        self._configure_logger()

    def _configure_logger(self):
        """
        Configures the logger for the module.
        """
        module_name = self.__class__.__name__
        self._log_level = self.spec.log_level.upper()
        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()
        self.logger.add(sys.stdout, level=self._log_level)

    def train(self):
        """Main call to train all language models.
        :return:
        """

        model, tokenizer = self._from_pretrained_fn(self.spec)
        self.logger.info("Starting training.")

        # we train State Encoder the goal here take rest api response
        # and re-present as state.
        if self.spec.llm == "latent" or self.spec.llm == "all":
            self.logger.info("Starting training state encoder.")
            llm_embeddings = LlmEmbeddingsTrainer(
                "state_encoder",
                self.spec, model, tokenizer,
                ds=self.ds, metric_logger=self.metric_logger)
            llm_embeddings.train()
            model = llm_embeddings.model
        # we train goal extractor the goal here extract
        # goal from high level sentence
        if self.spec.llm == "goal" or self.spec.llm == "all":
            self.logger.info("Starting training goal extractor.")
            # note we first fine tune LLM then we tune all other models.
            goal_extractor = GoalExtractorTrainer(
                "goal_extractor",
                self.spec,
                model, tokenizer,
                ds=self.ds,
                metric_logger=self.metric_logger)
            goal_extractor.train_goal_representation()
        # we train goal and parameter extractor, the goal here to extract
        # high level goal and parameters for that goal.
        if self.spec.llm == "parameter" or self.spec.llm == "all":
            self.logger.info("Starting training goal parameter extractor.")
            parameter_extractor = GoalExtractorTrainer(
                "parameter_extractor",
                self.spec,
                self.ds,
                self.metric_logger,
                model,
                tokenizer)
            parameter_extractor.train_goal_and_parameter_extractor()
        # we train auto encoder the aim here to reduce state re-presentation
        if self.spec.llm == "encoder" or self.spec.llm == "all":
            self.logger.info("Starting training state auto encoder.")
            autoencoder = AutoencoderTrainer(
                "state_autoencoder",
                self.spec, self.ds,
                self.metric_logger,
                model,
                tokenizer)
            autoencoder.train()

        # self.llm_autoencoder.train_autoencoder()
        # self.goal_extractor.train_goal_and_parameter_extractor()
        # self.goal_extractor.train_goal_representation()

    @staticmethod
    def make_base_model(spec: argparse.Namespace):
        return from_pretrained_default(spec)

    @staticmethod
    def make_model(
            spec: argparse.Namespace,
            module_name: str,
            base_model: None,
            base_tokenizer) -> Optional[LlmBaseModule]:

        """
        Create an igc module based on the module name.

        :param spec: specs for each module
        :param module_name: igc module name
        :param base_model: base llm model
        :param base_tokenizer:  base llm tokenizer
        :return:
        """
        return LlmBaseModule(module_name, spec, base_model, base_tokenizer, is_inference=True)

    @staticmethod
    def load_tokenizer(spec: argparse.Namespace):
        """
        Return tokenizer used for the llm models.

        :param spec:
        :return:
        """
        _, tokenizer = from_pretrained_default(spec.model_type, only_tokenizer=True)
        return tokenizer

    @staticmethod
    def load(
            spec: argparse.Namespace,
            device: torch.device = "cpu",
            module_name: Union[str, List[str]] = None,
    ) -> Dict[str, LlmBaseModule]:
        """

        Load the all llm models embedding model for inference.
        i.e. agent will use this as encoder

        :param spec: specs for modules
        :param module_name: The name or list of names of the specific modules to load, or None to load all modules.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """

        base_model, base_tokenizer = IgcLanguageModule.make_base_model(spec)

        modules = {}
        if module_name is not None:
            if module_name not in IgcLanguageModule.modules:
                raise ValueError(f"Invalid module_name: {module_name}. "
                                 f"Must be one of {IgcLanguageModule.modules}")

            module = IgcLanguageModule.make_model(spec, module_name, base_model, base_tokenizer)
            module.load(module_name, module.model, spec,
                        device=device, is_inference=True, optimizer=None, scheduler=None)
            modules[module_name] = module
            modules[module_name].set_tokenizer(base_tokenizer)

        else:
            for model_name in IgcLanguageModule.modules:
                module = IgcLanguageModule.make_model(spec, model_name, base_model, base_tokenizer)
                module.load(model_name, module.model, spec,
                            device=device, is_inference=True, optimizer=None, scheduler=None)
                modules[model_name] = module
                modules[model_name].set_tokenizer(base_tokenizer)

        return modules
