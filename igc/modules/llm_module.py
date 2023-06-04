import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import torch

from .metric_logger import MetricLogger
from .llm_goal_extract_trainer import GoalExtractor
from .llm_representation_trainer import LlmEmbeddingsTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer

from ..ds.redfish_dataset import JSONDataset
from ..shared.shared_torch_builder import TorchBuilder
import inspect


class IgcLllModule:
    def __init__(self, args: argparse.Namespace):
        """
        :param args:
        """
        self.metric_logger = MetricLogger(args.metric_report, **vars(args))
        self.model = GPT2LMHeadModel.from_pretrained(args.model_type)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
        directory_path = os.path.expanduser(args.raw_data_dir)

        self.dataset = JSONDataset(
            directory_path, verbose=True, tokenizer=self.tokenizer)

        self.goal_extractor = GoalExtractor(
            args, self.dataset, self.metric_logger, self.model, self.tokenizer)
        self.llm_embeddings = LlmEmbeddingsTrainer(
            args, self.dataset, self.metric_logger, self.model, self.tokenizer)

        # goal_extractor.train_goal_representation()
        # self.goal_extractor.train_goal_and_parameter_extractor()

    def train(self):
        """
        :return:
        """
        self.llm_embeddings.train_observation()

        # self.goal_extractor.train_goal_and_parameter_extractor()
        # self.goal_extractor.train_goal_representation()

    @staticmethod
    def load_llm_embeddings_model(
        args: argparse.Namespace,
        device: torch.device = "cpu"
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Optional[int]]:
        """
        Load the LLM embedding model for inference.
        i.e. agent will use this as encoder

        :param args: The command-line arguments.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """
        print(Path(args.output_dir).absolute())

        checkpoint_path_dir = Path(args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        checkpoint_path_dir = checkpoint_path_dir.absolute()

        print(F"absolute {str(args.output_dir)}")
        if not checkpoint_path_dir.is_dir():
            raise ValueError("Invalid checkpoint directory.")

        checkpoint_files = [f for f in os.listdir(checkpoint_path_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_path_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        print(F"FILS {checkpoint_files}")

        if not checkpoint_files:
            raise ValueError(
                f"No checkpoint files found "
                f"in the checkpoint directory {checkpoint_files}.")

        model = GPT2LMHeadModel.from_pretrained(args.model_type)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
        last_epoch = LlmEmbeddingsTrainer.load_model_inference(
            model, args, device=device)

        return model, tokenizer, last_epoch
