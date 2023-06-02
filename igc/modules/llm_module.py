import argparse
import os

from .metric_logger import MetricLogger
from .llm_goal_extract_trainer import GoalExtractor
from .llm_representation_trainer import LlmEmbeddingsTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

        self.goal_extractor = GoalExtractor(args, self.dataset, self.metric_logger, self.model, self.tokenizer)
        self.llm_embeddings = LlmEmbeddingsTrainer(args, self.dataset, self.metric_logger, self.model, self.tokenizer)

        # goal_extractor.train_goal_representation()
        # self.goal_extractor.train_goal_and_parameter_extractor()

    def train(self):
        """
        :return:
        """
        self.llm_embeddings.train_observation()

        # self.goal_extractor.train_goal_and_parameter_extractor()
        # self.goal_extractor.train_goal_representation()
