import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from transformers import (GPT2LMHeadModel,
                          GPT2Tokenizer,
                          PreTrainedModel,
                          PreTrainedTokenizer)

from .igc_auto_state_encoder import AutoencoderTrainer
from .base.igc_metric_logger import MetricLogger
from .llm_goal_extract_trainer import GoalExtractorTrainer
from .llm_representation_trainer import LlmEmbeddingsTrainer
from ..ds.redfish_dataset import JSONDataset


def from_pretrained_default(args):
    """
    :param args:
    :return:
    """
    model = GPT2LMHeadModel.from_pretrained(args.model_type)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    return model, tokenizer


class IgcLanguageModule:
    """
    """
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

        self._from_pretrained_fn = from_pretrained
        self.metric_logger = metric_logger
        self.spec = spec

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename='igc_llm_module.log',
            level=logging.DEBUG, format='%(asctime)s %(message)s')

        self.ds = ds

    def train(self):
        """Main call to train all language models.
        :return:
        """

        model, tokenizer = self._from_pretrained_fn(self.spec.model_type)

        # we train State Encoder the goal here take rest api response
        # and re-present as state.
        if self.spec == "latent" or self.spec == "all":
            llm_embeddings = LlmEmbeddingsTrainer(
                "state_encoder",
                self.spec, self.ds, self.metric_logger, model, tokenizer)
            llm_embeddings.train()
            model = llm_embeddings.model
        # we train goal extractor the goal here extract
        # goal from high level sentence
        if self.spec == "goal" or self.spec == "all":
            # note we first fine tune LLM then we tune all other models.
            goal_extractor = GoalExtractorTrainer(
                "goal_extractor",
                self.spec,
                self.ds,
                self.metric_logger,
                model,
                tokenizer)
            goal_extractor.train_goal_representation()
        # we train goal and parameter extractor, the goal here to extract
        # high level goal and parameters for that goal.
        if self.spec == "parameter" or self.spec == "all":
            parameter_extractor = GoalExtractorTrainer(
                "parameter_extractor",
                self.spec,
                self.ds,
                self.metric_logger,
                model,
                tokenizer)
            parameter_extractor.train_goal_and_parameter_extractor()
        # we train auto encoder the aim here to reduce state re-presentation
        if self.spec == "encoder" or self.spec == "all":
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
    def load_llm_embeddings_model(
            args: argparse.Namespace,
            only_tokenizer: Optional[bool] = False,
            device: torch.device = "cpu"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[int]]:
        """
        Load the LLM embedding model for inference.
        i.e. agent will use this as encoder

        :param only_tokenizer:  if we only need get tokenizer.
        :param args: The command-line arguments.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
        if only_tokenizer:
            return tokenizer

        checkpoint_path_dir = Path(args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        checkpoint_path_dir = checkpoint_path_dir.absolute()

        logging.info(F"absolute {str(args.output_dir)}")
        if not checkpoint_path_dir.is_dir():
            raise ValueError("Invalid checkpoint directory.")

        checkpoint_files = [f for f in os.listdir(checkpoint_path_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_path_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if not checkpoint_files:
            raise ValueError(
                f"No checkpoint files found "
                f"in the checkpoint directory {checkpoint_files}.")

        model = GPT2LMHeadModel.from_pretrained(args.model_type)
        last_epoch = LlmEmbeddingsTrainer.load_model_inference(
            model, args, device=device)

        return model, tokenizer, last_epoch
