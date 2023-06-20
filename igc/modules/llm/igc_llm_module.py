"""
This class re-present IGC llm submodule.

Essentially it composite object of many module that might form LLM.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import sys
import warnings
from typing import Optional, Dict, Union, List, Tuple

import loguru
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...ds.redfish_dataset import JSONDataset
from ...ds.redfish_masked_dataset import MaskedJSONDataset
from ...modules.base.igc_llm_base_module import LlmModule
from ...modules.base.igc_metric_logger import MetricLogger
from ...modules.igc_train_auto_state_encoder import AutoencoderTrainer
from ...modules.llm_train_goal_extract import GoalExtractorTrainer
from ...modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from ...modules.shared.llm_shared import from_pretrained_default, load_igc_tokenizer
from ...shared.modules_typing import ModelType, IgcModuleType


class IgcLanguageModule:
    """
    """
    modules = [
        "goal_extractor",
        "parameter_extractor",
        "state_encoder",
        "state_autoencoder"
    ]

    def __init__(self,
                 spec: argparse.Namespace,
                 metric_logger: MetricLogger,
                 ds: Union[JSONDataset, MaskedJSONDataset],
                 from_pretrained=from_pretrained_default):
        """

        :param spec: all model specs.
        :param metric_logger: a metric logger objet use to report metric.
        :param ds: A dataset used to train llm model.
        :param from_pretrained:
        """
        if spec is None:
            raise ValueError("Specs cannot be None")

        self.modules = ["goal_extractor",
                        "parameter_extractor",
                        "state_encoder",
                        "state_autoencoder"]

        self._from_pretrained_fn = from_pretrained
        self._metric_logger = metric_logger
        self._spec = spec
        self._dataset = ds
        self._configure_logger()

    def _configure_logger(self):
        """
        Configures the logger for the module.
        """
        module_name = self.__class__.__name__
        self._log_level = self._spec.log_level.upper()
        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()
        self.logger.add(sys.stdout, level=self._log_level)

    def load_finetuned_llm(
        self, use_pretrained_only: Optional[bool] = False
    ):

        """Load either pre-trained model only or fine-tuned llm.
        If we want experiment with auto encoder with some pre-trained model
        without doing any fine-tuning.

        :param use_pretrained_only:  if we only want to use stock pre-training model. i.e not fined.
        :return:
        """
        _is_llm_pre_trained = False
        llm_state = ModelType.UNTRAINED
        llm_tokenizer = self._dataset.tokenizer

        if use_pretrained_only:
            _model, _ = self._from_pretrained_fn(
                self._spec, only_tokenizer=False, only_model=True
            )
            _model.resize_token_embeddings(len(self._dataset.tokenizer))
            llm_state = ModelType.PRETRAINED
            return _model, llm_tokenizer, llm_state

        self.logger.info("Starting training.")

        llm_model = None
        llm_tokenizer = None

        # we train State Encoder the goal here take rest api response, and re-present as state.
        if self._spec.llm == "latent" or self._spec.llm == "all":
            self.logger.info("Starting training state encoder.")
            pretrained_model, t = self._from_pretrained_fn(
                self._spec,
                only_tokenizer=False,
                device_map=self._spec.device
            )

            llm_embeddings = LlmEmbeddingsTrainer(
                module_name="state_encoder",
                spec=self._spec,
                llm_model=pretrained_model,
                llm_tokenizer=self._dataset.tokenizer,
                dataset=self._dataset,
                metric_logger=self._metric_logger,
                is_inference=False,
                device=self._spec.device
            )

            if hasattr(pretrained_model, 'resize_token_embeddings'):
                pretrained_model.resize_token_embeddings(len(self._dataset.tokenizer))

            llm_embeddings.train()
            llm_model = llm_embeddings.get_model()
            llm_tokenizer = llm_embeddings.get_tokenizer()

        return llm_model, llm_tokenizer, llm_state

    def load_finetuned_state_encoder(self, path: str = None):
        """
        Load fine tuned state encoder.
        
        :return:
        """

        _model = None
        self.logger.info("Loading state encoder fined tuned state.")

        modules = IgcLanguageModule.load(
            self._spec if path is None else path,
            device=self._spec.device,
            module_names=["state_encoder"],
        )
        module = modules["state_encoder"]
        _model = module.model

        if _model is None:
            warnings.warn("Please train state encoder first.")
            return

        return _model

    def train(self, use_pretrained_only: bool = False):
        """Main call to train all language models.
        :return:
        """
        _model = None
        self.logger.info("Starting training.")
        _model, tokenizer, model_state = self.load_finetuned_llm(
            use_pretrained_only=use_pretrained_only
        )

        if model_state == ModelType.FINETUNED:
            _model = self.load_finetuned_state_encoder()

        # we train goal extractor
        if self._spec.llm == "goal" or self._spec.llm == "all":
            self.logger.info("Starting training goal extractor.")
            goal_extractor = GoalExtractorTrainer(
                "goal_extractor",
                self._spec,
                _model,
                tokenizer,
                ds=self._dataset,
                metric_logger=self._metric_logger
            )
            if hasattr(_model, 'resize_token_embeddings'):
                _model.resize_token_embeddings(len(tokenizer))
            goal_extractor.train_goal_representation()

        # we train goal and parameter extractor, the goal here to extract
        # high level goal and parameters for that goal.
        if self._spec.llm == "parameter" or self._spec.llm == "all":
            self.logger.info("Starting training goal parameter extractor.")
            parameter_extractor = GoalExtractorTrainer(
                "parameter_extractor",
                self._spec,
                _model,
                tokenizer,
                ds=self._dataset,
                metric_logger=self._metric_logger,
                is_inference=False
            )
            if hasattr(_model, 'resize_token_embeddings'):
                _model.resize_token_embeddings(len(tokenizer))
            parameter_extractor.train_goal_and_parameter_extractor()

        # we train auto encoder the aim here to reduce state re-presentation
        if self._spec.llm == "encoder" or self._spec.llm == "all":
            self.logger.info("Starting training state auto encoder.")
            autoencoder = AutoencoderTrainer(
                "state_autoencoder",
                spec=self._spec,
                llm_model=_model,
                llm_tokenizer=tokenizer,
                ds=self._dataset,
                metric_logger=self._metric_logger,
                is_inference=False,
                device=self._spec.device)

            if hasattr(_model, 'resize_token_embeddings'):
                _model.resize_token_embeddings(len(tokenizer))
            autoencoder.train()

        # self.llm_autoencoder.train_autoencoder()
        # self.goal_extractor.train_goal_and_parameter_extractor()
        # self.goal_extractor.train_goal_representation()

    def _register_tokens(self):
        pass

    @staticmethod
    def make_base_model(
        spec: Union[str, argparse.Namespace]
    ) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """
        Create base llm model with igc tokenizer.

        :param spec: path or spec
        :return: pretrained model and tokenizer
        """
        model, _ = from_pretrained_default(
            spec, only_model=True,
            only_tokenizer=False,
            device_map=spec.device
        )
        tokenizer = load_igc_tokenizer()
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer

    @staticmethod
    def make_model(
        spec: argparse.Namespace,
        module_name: str,
        base_model: None,
        base_tokenizer,
        device=None,
    ) -> Optional[LlmModule]:
        """
        Create an igc module based on the module name.

        :param device:
        :param spec: specs for each module
        :param module_name: igc module name
        :param base_model: base llm model
        :param base_tokenizer:  base llm tokenizer
        :return:
        """
        return LlmModule(
            module_name=module_name,
            spec=spec,
            llm_model=base_model,
            llm_tokenizer=base_tokenizer,
            is_inference=True,
            device=device,
        )

    @staticmethod
    def load_tokenizer(spec: argparse.Namespace):
        """
        Return tokenizer used for the llm models.

        :param spec:
        :return:
        """
        _, tokenizer = from_pretrained_default(
            spec.model_type,
            only_tokenizer=True
        )
        return tokenizer

    @staticmethod
    def load(
        spec: argparse.Namespace,
        device: torch.device = "cpu",
        module_names: Union[str, List[str], IgcModuleType] = None,
    ) -> Dict[str, LlmModule]:
        """

        Load all igc llm modules.

        :param spec: specs for modules
        :param module_names: The name or list of names of the specific modules to load,
                             or None to load all modules.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """

        base_model, base_tokenizer = IgcLanguageModule.make_base_model(spec)
        if module_names is None:
            raise ValueError(
                f"Invalid module_name: {module_names}. "
                f"Must be one of {IgcLanguageModule.modules}")

        if isinstance(module_names, str):
            module_names = [module_names]

        invalid_modules = set(module_names) - set(IgcLanguageModule.modules)
        if invalid_modules:
            raise ValueError(
                f"Invalid module_name(s): {invalid_modules}. "
                f"Must be one of {IgcLanguageModule.modules}")

        modules = {}
        for module_name in module_names:
            print(f"Loading {module_name} module.")
            module = IgcLanguageModule.make_model(
                spec, module_name, base_model, base_tokenizer
            )

            module.load(
                module_name,
                module.model,
                specs=spec,
                is_inference=True,
                optimizer=None,
                scheduler=None,
                map_location=device
            )

            modules[module_name] = module
            modules[module_name].set_tokenizer(base_tokenizer)

        return modules
