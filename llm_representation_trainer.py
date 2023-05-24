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
import itertools
import os
import random
from typing import List, Optional, Tuple
import re

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

import readline
from ds.redfish_dataset import JSONDataset, MaskedSampler
from shared_torch_utils import get_device
from rest_action import RestActionSpace, ActionWithoutParam

from collections import namedtuple
from torch.utils.data import DataLoader, RandomSampler

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class LlmEmbeddingsTrainer:
    """
    """

    def __init__(self, model_name='gpt2'):
        """
        :param model_name:
        """
        # Define the GPT model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.num_epochs = 1000
        self.batch_size = 4

        directory_path = os.path.expanduser("~/.json_responses")
        self.dataset = JSONDataset(
            directory_path, verbose=True, tokenizer=self.tokenizer)

        self.device = get_device()
        self.writer = SummaryWriter(log_dir="logs")
        self.batch_log = 10
        self.num_workers = 1
        self.shuffle = False

        self.pad_token = self.dataset.tokenizer.pad_token
        self.pad_token_id = self.dataset.tokenizer.pad_token_id
        self._default_lr = 1e-5

    def val_observation_space_batch(self, input_seqs: List[List[str]], goals: List[str]):
        """Validate LLM model for goal extraction.
        :param input_seqs:
        :param goals:
        :return:
        """
        self.model.eval()
        prefix = len("RedfishGoal: ")
        max_length = max(len(goal) for goal in goals)
        max_seq_length = max(len(seq) for seq in input_seqs)
        max_total_len = max_seq_length + max_length + prefix + 10

        correct_predictions = 0.0
        for i, seq in enumerate(input_seqs):
            eval_encoded_inputs = self.tokenizer(
                seq, padding=True, truncation=True, return_tensors='pt')

            eval_input_ids = eval_encoded_inputs['input_ids'].to(self.device)
            eval_attr_mask = eval_encoded_inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=eval_input_ids,
                    attention_mask=eval_attr_mask,
                    max_length=128)

            generated_prompts = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            target_goal = self.extract_goal(generated_prompts)
            if target_goal is not None and target_goal.lower() == goals[i].lower():
                correct_predictions += 1.0

        accuracy = (correct_predictions / len(goals)) * 100.0
        return accuracy

    def val_goal_representation_epoch(self, overfit: Optional[bool] = True):
        """Validate LLM model for goal extraction at the end epoch.
        :param overfit:
        :return:
        """
        if overfit:
            batch_generator = iter([next(self.generate_prompts())])
        else:
            batch_generator = self.generate_prompts()
        epoch_accuracy = 0.0
        num_batches = 0

        for i, (batch, input_seqs, goals) in enumerate(batch_generator):
            # evaluate entire batch
            num_batches += 1
            epoch_accuracy += self.val_goal_representation(input_seqs, goals)

        return epoch_accuracy / num_batches

    @staticmethod
    def custom_collate_fn(samples):
        """

        :param samples:
        :return:
        """
        # included_keys = ['input_ids', 'attention_mask', 'request_hash']
        # batched_samples = []
        # for i, s in enumerate(samples):
        #     # print(f"{i} input", s["input_ids"])
        #     # print(f"{i}attention_mask", s["attention_mask"])
        #     # print(f"{i}request_hash", s["request_hash"])
        #     batched_samples.append([s["input_ids"], s["attention_mask"], ["request_hash"]])
        included_keys = ['input_ids', 'attention_mask']
        batch = {key: torch.stack([s[key] for s in samples]) for key in included_keys}
        for i, s in enumerate(samples):
            print("Collatting")
        return batch
        # return batched_samples

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    @staticmethod
    def get_batch(src: Tensor, idx: int, chunk_size=35) -> Tuple[Tensor, Tensor]:
        """
        :param src: [full_seq_len, batch_size]
        :param idx
        :param chunk_size:
        :return: tuple (data, target),  shape [seq_len, batch_size], [seq_len * batch_size]
        """
        seq_len = min(chunk_size, len(src) - 1 - idx)
        data = src[idx:idx + seq_len]
        target = src[idx + 1:idx + 1 + seq_len].reshape(-1)
        return data, target

    def train_observation(self, overfit: Optional[bool] = True):
        """Train LLM model to map high level goal to redfish actions.

        For example
                "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"
        :param overfit:
        :return:
        """
        optimizer = AdamW(self.model.parameters(), lr=self._default_lr)
        self.model.to(self.device)
        self.model.train()

        # sampler = RandomSampler(self.dataset)
        # sampler = MaskedSampler(self.dataset)
        dataloader1 = DataLoader(self.dataset,
                                 batch_size=self.batch_size,
                                 # sampler=sampler,
                                 num_workers=self.num_workers,
                                 shuffle=self.shuffle,
                                 collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        if overfit:
            dataloader_overfit = [next(iter(dataloader1))]

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            if overfit:
                dataloader = iter(dataloader_overfit)

            for i, batch in enumerate(dataloader):
                labels = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.pad_token_id)
                labels = labels.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                if epoch % 10 == 0:
                    for j in range(batch_inputs['input_ids'].size(0)):
                        # def mask_json_key_and_value(encoding, target_key, tokenizer, debug=False):
                        batch_inputs["attention_mask"] = self.dataset.mask_json_key_and_value(
                            batch_inputs, "@odata.id", self.tokenizer)

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                # print(f"Epoch {epoch + 1}/{self.num_epochs} - Batch {i + 1}/{len(dataloader)}")
                # print("before", batch_inputs['input_ids'].shape)
                # print("before", batch_inputs['attention_mask'].shape)
                # print("before", batch_inputs['input_ids'].shape)
                # # shifted
                #
                # print(batch_inputs['input_ids'].shape)
                # print(batch_inputs['attention_mask'].shape)
                # print(labels.shape)

                outputs = self.model(**batch_inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # accumulate loss
                total_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                average_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

        print("Embedding extractor training complete.")


def main():
    """
    :return:
    """
    llm_embeddings_trainer = LlmEmbeddingsTrainer()
    llm_embeddings_trainer.train_observation(overfit=True)
    # llm_embeddings_trainer.writer.close()


if __name__ == '__main__':
    """
    """
    main()
