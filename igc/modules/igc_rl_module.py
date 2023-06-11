import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (GPT2LMHeadModel)

from .base.igc_metric_logger import MetricLogger
from .igc_agent_trainer import IgcAgentTrainer
from .igc_auto_state_encoder import AutoencoderTrainer
from .igc_llm_module import IgcLanguageModule
from .llm_representation_trainer import LlmEmbeddingsTrainer
from ..ds.redfish_dataset import JSONDataset
from ..envs.rest_gym_batch_env import VectorizedRestApiEnv


class IgcRlModule:
    """
    """
    def __init__(self, spec: argparse.Namespace, metric_logger: MetricLogger, ds: JSONDataset):
        """
        :param spec:
        """
        model, tokenizer, last_epoch = IgcLanguageModule.load_llm_embeddings_model(spec, only_tokenizer=False)
        # create env
        self.env = VectorizedRestApiEnv(
            args=spec,
            model=model,
            tokenizer=tokenizer,
            discovered_rest_api=ds,
            max_episode=spec.max_episode_len,
            num_envs=spec.batch_size
        )

        directory_path = os.path.expanduser(spec.raw_data_dir)
        self.cmd = spec

        self.autoencoder = AutoencoderTrainer(
            "state_autoencoder", spec, ds, metric_logger, model, tokenizer)

        self.rl_gent = IgcAgentTrainer(
            "rl_agent", spec, ds, metric_logger, model, tokenizer
        )

    def train(self):
        """Main call to train all language models.
        :return:
        """
        self.rl_gent.train()

    @staticmethod
    def load_rl_model(
            args: argparse.Namespace,
            device: torch.device = "cpu"
    ):
        """
        Load RL Agent.
        i.e. agent will use this as encoder

        :param args: The command-line arguments.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """
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
        last_epoch = IgcLanguageModule.load_llm_embeddings_model(
            model, args, device=device)

        return model, last_epoch
