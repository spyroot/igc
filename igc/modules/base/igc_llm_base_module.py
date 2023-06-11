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
from typing import List, Optional
from collections import namedtuple

import evaluate
import numpy as np
import torch
from rouge_score import rouge_scorer

from igc.ds.redfish_dataset import JSONDataset
from .igc_base_module import IgcBaseModule
from .igc_llm_metrics_type import MetricType
from .igc_metric_logger import MetricLogger

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class LlmBaseModule(IgcBaseModule):
    """
    """
    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = "False"):
        """
        Base LLM module

        :param spec:  specs for the particular module
        :param ds: dataset
        :param metric_logger: metric logger used for training
        :param llm_model:
        :param llm_tokenizer:
        """
        super().__init__(module_name,
                         spec,
                         llm_model,
                         llm_tokenizer,
                         ds=ds,
                         metric_logger=metric_logger,
                         is_inference=is_inference)

        self.logger.info("Starting llm module")
        self._log_level = spec.llm_log_level.upper()
        self.metric_logger.set_log_level(self._log_level)

    @staticmethod
    def compute_rouge_metric(
            predictions:
            List[str], targets: List[str],
            default_rouge: str = 'rouge1') -> float:
        """
        Compute_rouge_metric
        
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
    def compute_exact_match(
            predictions: List[str], targets: List[str]) -> float:
        """
        Compute exact match score.

        :param predictions:
        :param targets:
        :return:
        """
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip()
                 for p, t in zip(predictions, targets)]) / len(predictions)

    @staticmethod
    def compute_f1_score(predictions, targets) -> float:
        pass

    @staticmethod
    def performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
        """
        :param predictions:
        :param targets:
        :param metric:
        :return:
        """
        if metric == 'rouge':
            return LlmBaseModule.compute_rouge_metric(predictions, targets)
        elif metric == 'f1':
            return LlmBaseModule.compute_f1_score(predictions, targets)
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
        """
        It uses HuggingFace evaluate.

        Task names: "cola", "sst", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"

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
        _logits, _labels = eval_prediction
        predictions = np.argmax(_logits, axis=-1)
        return metric.compute(predictions=predictions, references=_labels)

    @staticmethod
    def collate_random_span_fn(input_batch, pad_token_id):
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

        :param pad_token_id:
        :param input_batch: List of data points to be collated.
        :return: Collated data with masked spans.
        """
        input_ids = torch.cat(
            [item['input_ids'].squeeze(1) for item in input_batch]
        )

        attention_mask = torch.cat(
            [item['attention_mask'].squeeze(1) for item in input_batch]
        )

        labels = input_ids.clone()

        # Mask a random span of text in each input
        for i in range(len(input_batch)):
            input_length = input_ids[i].size(0)
            # randomly choose start position for masking
            mask_start = torch.randint(1, input_length - 1, (1,)).item()
            # randomly choose end position for masking
            mask_end = mask_start + torch.randint(1, input_length - mask_start, (1,)).item()
            # replace the selected span with pad_token_id
            input_ids[i, mask_start:mask_end] = pad_token_id
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
    def collate_input_shift_fn(batch, pad_token_id, ignore_index=-100):
        """
        :param batch:
        :param pad_token_id:
        :param ignore_index:
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
        labels[:, -1] = ignore_index  # ignore index
        mask = torch.tensor(input_ids == pad_token_id)
        labels = labels.masked_fill(mask, ignore_index)

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
