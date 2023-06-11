"""

we pre-train the model using the shift generation method:

In this step, we train the model to predict the next token in
the sequence by shifting the input sequence and using a masked language modeling objective.
This helps the model learn the language patterns and dependencies.

Fine-tune the pretrained model using the random span method:
After pre-training, you can further fine-tune the model
using the random span method. In this case, you replace random spans of
text with a single mask token and train the
model to predict the original text. This helps the model learn to fill in missing
information and generate coherent text.

By combining these two methods, you can benefit from both the language modeling capabilities learned through shift
generation and the ability to generate missing text using the random span method. This two-step process
allows the model to capture a broader range of language understanding and generation capabilities.
"""
import argparse
import os
from enum import Enum

import numpy as np
import torch
from torch import tensor
from torch.utils.data import random_split

from igc.ds.redfish_dataset import JSONDataset

torch.cuda.empty_cache()


class MetricType(Enum):
    """Metric type
    """
    ROUGE = 'rouge'
    PERPLEXITY = 'perplexity'
    EXACT_MATCH = 'exact match'
    CLASSIFICATION_ACCURACY = 'classification accuracy'
    F1_SCORE = 'f1'
    BLEU = 'bleu'
    MAP = 'map'
    SPEARMAN_CORRELATION = 'spearman correlation'
    WER = 'wer'
    MSE = 'mse'

    def __str__(self):
        return self.value


class LLmBaseTrainer:
    """
    Base trainer for language model used in IGC
    """

    def __init__(self, cmd: argparse.Namespace):
        """
        :param cmd:
        """
        super().__init__()

        self.cmd = cmd
        self.collate_fn = self.collate_input_shift_fn
        self.metrics_fn = self.compute_metrics
        # dataset
        self.directory_path = os.path.expanduser("~/.json_responses")
        self.dataset = JSONDataset(
            self.directory_path, verbose=False)
        self.pad_token = self.dataset.tokenizer.pad_token
        self.pad_token_id = self.dataset.tokenizer.pad_token_id

        self.train_dataset = self.dataset
        self.eval_dataset = self.dataset

        # split dataset
        self.split_dataset()

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

        self.train_dataset, self.eval_dataset = random_split(
            self.dataset, [train_size, eval_size])

    @staticmethod
    def dataset_checker(self):
        """Dataset checker
        :return:
        """
        for data_point in self.dataset:
            rest_call = self.dataset.action(data_point["label"])
            print("rest recovered:", rest_call)
            print("rest original:", data_point["rest_api"])
            print("rest original:", data_point["label"])



    @staticmethod
    def print_batch_shapes(input_ids, attention_mask, labels):
        """
        :param input_ids:
        :param attention_mask:
        :param labels:
        :return:
        """
        print(f"shapes "
              f"input:{input_ids.shape} "
              f"mask:{attention_mask.shape} "
              f"label:{labels.shape}")

    def collate_input_shift_fn(self, batch):
        """
        :param batch:
        :return:
        """
        input_ids = torch.cat(
            [item['input_ids'].squeeze(1) for item in batch]
        )

        attention_mask = torch.cat(
            [item['attention_mask'].squeeze(1) for item in batch]
        )

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # shifting
        labels = input_ids[:, 1:].clone()
        labels[:, -1] = -100  # ignore index
        mask = torch.tensor(input_ids == self.pad_token_id)
        labels = labels.masked_fill(mask, -100)

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


    @staticmethod
    def normalize_prediction(prediction, string_pair="Q"):
        """
        :param string_pair:
        :param prediction:
        :return:
        """
        if prediction.endswith(string_pair):
            prediction = prediction[:-1]
        elif 'Q:' in prediction:
            prediction = prediction[:prediction.index('Q:')]
        return prediction.strip('. ').lower()




def test_compute_metrics():
    # Mock evaluation predictions
    logits = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 1, 0])
    eval_prediction = (logits, labels)
    metrics = LLmBaseTrainer.compute_metrics(eval_prediction)
    print(f"Metrics: {metrics}")


def test_exact_match():
    """
    :return:
    """
    predictions = ["Hello", "World", "Python"]
    targets = ["Hello", "world", "Python"]

    exact_match = LLmBaseTrainer.compute_exact_match(predictions, targets)
    print(f"Exact Match: {exact_match}")


def test_collate():
    # Sample batch of data points
    batch = [
        {
            'input_ids': tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1]])
        },
        {
            'input_ids': tensor([[6, 7, 8, 9, 10]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1]])
        },
    ]

    base_trainer = LLmBaseTrainer(None)
    print("Input batch:")
    print(batch)

    collated_data = base_trainer.collate_random_span_fn(batch)
    print("Collated data:")
    print(collated_data["input_ids"])
    print(collated_data["attention_mask"])
    print(collated_data["labels"])


def test_json_dataset():
    """
    :return:
    """
    directory_path = os.path.expanduser("~/.json_responses")
    dataset = JSONDataset(directory_path, verbose=False, recreate_dataset=True)
    for d in dataset:
        if len(d["targets"]) > 0:
            print(d["targets"])
            print(d["allowable_values"])


if __name__ == '__main__':
    test_json_dataset()
    # test_collate()
    # test_exact_match()
    # test_compute_metrics()
