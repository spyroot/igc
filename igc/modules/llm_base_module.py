"""
This class is used to train a goal extractor from input query.

Given input text provided by the user, or external system.
The goal is to extract a goal for the agent and parameters
that agent need used.

For example given input text: "Update raid with raid0"
The goal here update raid configuration and the
parameter is raid0.

In downstream task the goal encoded as one hot vector.
This what used to train RL agent.

Parameters just passed to agent. i.e. we don't train on parameters.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from collections import namedtuple
from pathlib import Path

import torch
from torch.utils.data import random_split

from igc.ds.redfish_dataset import JSONDataset
from igc.modules.metric_logger import MetricLogger
from igc.shared.shared_torch_utils import get_device

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class LlmBaseModule:
    """
    """

    def __init__(self,
                 args: argparse.Namespace,
                 ds: JSONDataset,
                 metric_logger: MetricLogger,
                 llm_model,
                 llm_tokenizer):
        """
        :param args:
        :param ds:
        :param metric_logger:
        :param llm_model:
        :param llm_tokenizer:
        """
        # Define the GPT model and tokenizer
        self.model = llm_model
        self.tokenizer = llm_tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.num_epochs = args.num_train_epochs
        self.batch_size = args.per_device_train_batch_size

        self.dataset = ds
        self.device = get_device()
        self.metric_logger = metric_logger

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.batch_log = 10
        self._overfit = args.overfit
        self._default_lr = 1e-5
        self.trainer_args = args

        self.optimizer = None

        # model saving
        self.save_strategy = args.save_strategy
        checkpoint_path_dir = Path(args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()

        if not checkpoint_path_dir.is_dir():
            raise ValueError("Indicate path to checkpoint dir.")

        self.checkpoint_dir = str(checkpoint_path_dir)
        self.rank = int(os.environ.get('LOCAL_RANK', -1))

    def split_dataset(self, ratio: float = 0.8):
        """split dataset
        :param ratio:
        :return:
        """
        if ratio <= 0 or ratio >= 1:
            raise ValueError(
                "Invalid ratio. The ratio value should be between 0 and 1 (exclusive).")

        train_size = int(len(self.dataset) * ratio)
        eval_size = len(self.dataset) - train_size

        if train_size <= 0 or eval_size <= 0:
            raise ValueError(
                "Invalid dataset sizes. Adjust the ratio value to ensure non-zero splits.")

        return random_split(
            self.dataset, [train_size, eval_size])

    def split_slice_dataset(self, train_ratio: float = 0.8, sample_ratio: float = 0.01):
        """split dataset
        :param train_ratio:
        :param sample_ratio:
        :return:
        """
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError(
                "Invalid train_ratio. The train_ratio value should be between 0 and 1 (exclusive).")

        if sample_ratio <= 0 or sample_ratio >= 1:
            raise ValueError(
                "Invalid sample_ratio. The sample_ratio value should be between 0 and 1 (exclusive).")

        sample_size = int(len(self.dataset) * sample_ratio)

        # randomly select a subset of the dataset
        indices = torch.randperm(len(self.dataset))
        self.dataset = torch.utils.data.Subset(self.dataset, indices[:sample_size])

        train_size = int(len(self.dataset) * train_ratio)
        eval_size = len(self.dataset) - train_size

        if train_size <= 0 or eval_size <= 0:
            raise ValueError(
                "Invalid dataset sizes. Adjust the ratio value to ensure non-zero splits.")

        return random_split(
            self.dataset, [train_size, eval_size])

    def save_strategy(self):
        return

    def save_checkpoint(self, checkpoint_dir, epoch):
        """
        :param checkpoint_dir:
        :param epoch:
        :return:
        """
        if self.rank > 0:
            return

        epoch_mod = epoch % 3
        checkpoint_file = f"{checkpoint_dir}/epoch{epoch_mod}.pt"
        # if self.accelerator is not None:
        #     self.accelerator.save_state(output_dir=checkpoint_dir)
        #     print(f"Checkpoint saved to {checkpoint_file}")
        # else:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_file)
        print(f"Rank: {self.rank} checkpoint saved to {checkpoint_file}")

    def load_checkpoint(self, checkpoint_dir):
        """
        :param checkpoint_dir:
        :return:
        """
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if checkpoint_files:
            # if self.accelerator is not None:
            #     self.accelerator.load_state(input_dir=checkpoint_dir)
            #     epoch = self.accelerator.state.epoch
            #     print(f"Checkpoint loaded from {checkpoint_file}, epoch: {epoch}")
            # else:
            checkpoint_file = checkpoint_files[0]
            checkpoint = torch.load(checkpoint_file, map_location={'cuda:1': 'cuda:0'})
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"Rank: {self.rank} loading checkpoint loaded from {checkpoint_file}, epoch: {epoch}")
            return epoch
        else:
            print(f"No checkpoint files found in dir {checkpoint_dir}")
            return 0
