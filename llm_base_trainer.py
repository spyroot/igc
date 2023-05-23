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
import json
import os
from collections import defaultdict
from typing import List

import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from rouge_score import rouge_scorer
from torch import tensor
from torch.utils.data import random_split
from enum import Enum
from ds.redfish_dataset import JSONDataset


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
        self.pad_token_id = self.dataset.tokenizer.pad_token
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

    def collate_random_span_fn(self, input_batch):
        """Collates and masks a random span of text in each input.

        When collating and masking a random span of text in each input,
        the labels should be the original input_ids with the
        masked spans replaced with the pad_token_id.


        Usage:

        batch = [
            {
            'input_ids': tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1]])
            },
            {
            'input_ids': tensor([[6, 7, 8, 9, 10]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1]])
            },
            ...
        ]

        collated_data = collate_random_span_fn(batch)

        :param input_batch: List of data points to be collated.
        :return: Collated data with masked spans.
        """
        input_ids = torch.cat(
            [item['input_ids'].squeeze(1) for item in input_batch]
        )

        attention_mask = torch.cat(
            [item['attention_mask'].squeeze(1) for item in input_batch]
        )

        # we need a copy of input_ids as labels
        labels = input_ids.clone()
        input_ids_clone = input_ids.clone()

        # Mask a random span of text in each input
        for i in range(len(input_batch)):
            input_length = input_ids[i].size(0)
            # randomly choose start position for masking
            mask_start = torch.randint(1, input_length - 1, (1,)).item()
            # randomly choose end position for masking
            mask_end = mask_start + torch.randint(1, input_length - mask_start, (1,)).item()
            # replace the selected span with pad_token_id
            input_ids[i, mask_start:mask_end] = self.pad_token_id
            # set the labels to the original span
            labels[i, mask_start:mask_end] = input_ids[i, mask_start:mask_end]
            # labels[i, :mask_start] = self.pad_token_id
            # labels[i, mask_end:] = self.pad_token_id

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

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
    def compute_rouge_metric(
            predictions:
            List[str], targets: List[str],
            default_rouge: str = 'rouge1') -> float:
        """
        
        :param predictions: 
        :param targets: 
        :param default_rouge: 
        :return: 
        """
        scorer = rouge_scorer.RougeScorer([default_rouge], use_stemmer=True)
        scores = [scorer.score(p, t)[default_rouge].fmeasure
                  for p, t in zip(predictions, targets)]
        return sum(scores) / len(scores)

    @staticmethod
    def compute_f1_score(predictions, targets) -> float:
        pass

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

    @staticmethod
    def compute_exact_match(
            predictions: List[str], targets: List[str]) -> float:
        """

        :param predictions:
        :param targets:
        :return:
        """
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip()
                 for p, t in zip(predictions, targets)]) / len(predictions)

    @staticmethod
    def performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
        """
        :param predictions:
        :param targets:
        :param metric:
        :return:
        """
        if metric == 'rouge':
            return LLmBaseTrainer.compute_rouge_metric(predictions, targets)
        elif metric == 'f1':
            return LLmBaseTrainer.compute_f1_score(predictions, targets)
        elif metric == 'exact match':
            if isinstance(targets[0], str):
                return sum([p.strip() == t.strip() for p, t in zip(predictions, targets)]) / len(predictions)
            else:
                def _normalize(prediction):
                    if prediction.endswith('Q'):
                        prediction = prediction[:-1]
                    elif 'Q:' in prediction:
                        prediction = prediction[:prediction.index('Q:')]
                    return prediction.strip('. ').lower()

                normalized = [_normalize(p) for p in predictions]

                def contains(key, candidates):
                    for c in candidates:
                        if key in c:
                            return True
                    return False

                return sum([contains(n, t)
                            for n, t in zip(normalized, targets)]) / len(normalized)
        else:
            raise NotImplementedError()

    @staticmethod
    def metric_for_dataset():
        """
        :return:
        """
        return {
            MetricType.ROUGE,
            MetricType.PERPLEXITY,
            MetricType.EXACT_MATCH,
            MetricType.CLASSIFICATION_ACCURACY,
            MetricType.F1_SCORE,
            MetricType.BLEU,
            MetricType.MAP,
            MetricType.SPEARMAN_CORRELATION,
            MetricType.WER,
            MetricType.MSE
        }

    @staticmethod
    def compute_metrics(eval_prediction):
        """Task names: "cola", "sst", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"

        CoLA (Corpus of Linguistic Acceptability): This task involves determining whether
                a given sentence is grammatically acceptable or not.

        SST (Stanford Sentiment Treebank): The task is to predict the sentiment
                (positive or negative) of a given sentence.

        MRPC (Microsoft Research Paraphrase Corpus): This task involves determining
         whether two sentences are paraphrases of each other or not.

        QQP (Quora Question Pairs): The goal is to determine whether two questions
        are semantically equivalent or not.

        STS-B (Semantic Textual Similarity Benchmark): This task involves predicting
        the degree of semantic similarity between two sentences.

        MNLI (Multi-Genre Natural Language Inference): The task is to determine the
        logical relationship (entailment, contradiction, or neutral) between two sentences.

        QNLI (Question-answering Natural Language Inference): Similar to MNLI,
        but focused on question-answering scenarios.

        RTE (Recognizing Textual Entailment): This task involves determining whether a
        given hypothesis can be inferred from a given premise.

        WNLI (Winograd Schema Challenge): The task is to resolve pronoun references in
        sentences based on world knowledge.

        :param eval_prediction:
        :return:
        """
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def plot(self, ks, prompts, default_log_scale=4):
        """
        :param default_log_scale:
        :param ks:
        :param prompts:
        :return:
        """
        data = defaultdict(lambda: defaultdict(list))
        symbols = ['solid', 'dashed', 'dotted', 'dashdot']
        x_vals = set()
        for prompt in prompts:
            for k in ks:
                method = '_'.join([self.cmd.model, self.dataset, str(k), prompt])
                id_ = '_'.join([self.cmd.model, self.dataset, prompt])
                with open(f'{method}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)
                    data[id_]['linestyle'] = symbols[0]

        [
            plt.plot(v['x'], v['y'],
                     label=k,
                     linestyle=v['linestyle'])
            for k, v in data.items()
        ]

        plt.xscale('symlog') if max(x_vals) > default_log_scale else None

        ax = plt.gca()
        ax.xaxis.set_major_formatter(
            mticker.ScalarFormatter()
        )

        for v in data.values():
            ax.xaxis.set_ticks(v['x'])

        plt.legend()
        plt.title(self.dataset)
        plt.ylabel(str(MetricType[self.dataset]))
        plt.xlabel('Number of support examples')
        plt.show()


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
