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
import warnings
from typing import List, Optional, Union, Callable
from collections import namedtuple

import evaluate
import numpy as np
import torch
from rouge_score import rouge_scorer

from igc.ds.redfish_dataset import JSONDataset
from .igc_base_module import IgcModule
from .igc_llm_metrics_type import MetricType
from .igc_metric_logger import MetricLogger
from sklearn.metrics import f1_score

from .prompt_types import PromptType
from ..shared.llm_shared import (
    load_pretrained_default, save_pretrained_default
)

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class LlmModule(IgcModule):
    """
    This base LLM Module share between different part IGC.
    """

    def __init__(
            self,
            module_name: str,
            spec: argparse.Namespace,
            llm_model,
            llm_tokenizer,
            ds: Optional[JSONDataset] = None,
            metric_logger: Optional[MetricLogger] = None,
            is_inference: Optional[bool] = False,
            device=None):
        """
        Base LLM module, shared by all LLM submodules.

        :param module_name: name of the module
        :param spec: store all specs.
        :param llm_model: pre-trained language model
        :param llm_tokenizer: pre-trained tokenizer
        :param ds: dataset used to train IGC
        :param metric_logger: a metric logger to store metrics
        :param is_inference: flag indicating if the module is for inference

        """
        super().__init__(
            module_name,
            spec,
            llm_model,
            llm_tokenizer,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=is_inference,
            device=device
        )

        self._log_level = spec.llm_log_level.upper()
        if self.metric_logger is not None:
            self.metric_logger.set_log_level(self._log_level)

        if hasattr(llm_model, 'resize_token_embeddings'):
            llm_model.resize_token_embeddings(len(llm_tokenizer))
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            warnings.warn("Model does not have the 'resize_token_embeddings' method.")

        self.logger.info("Starting llm module")

    def finetuned_dir(self):
        """
        Get the directory path for saving/loading the fine-tuned model.
        This huggingface format.

        :return: The directory path.
        """
        return f"{self._module_checkpoint_dir}/fine_tuned"

    def save_finetuned(self):
        """
        Save the fine-tuned model and tokenizer. This huggingface format.
        :return: None
        """
        if not self.is_rank_zero():
            return

        fine_tuned = self.finetuned_dir()
        if not os.path.exists(fine_tuned):
            os.makedirs(fine_tuned)
        save_pretrained_default(fine_tuned, self.model, self.tokenizer)

    def load_finetuned(self, device_map="auto"):
        """
        Load the fine-tuned model and tokenizer. This huggingface format.
        :param device_map: The device map for placing the model on specific devices. Defaults to "auto".
        :return: Tuple of the loaded model and tokenizer.
        """
        model, tokenizer = load_pretrained_default(
            self._trainer_specs, self.finetuned_dir(), device_map=device_map
        )
        return model, tokenizer

    @staticmethod
    def compute_rouge_metric(
            predictions: List[str],
            targets: List[str],
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
            predictions: List[Union[str, int]],
            targets: List[Union[str, int]]
    ) -> float:
        """
        Compute exact match score.

        :param predictions:
        :param targets:
        :return:
        """
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip() for p, t
                 in zip(predictions, targets)]) / len(predictions)
        else:
            return sum(
                [p == t for p, t
                 in zip(predictions, targets)]) / len(predictions)

    @staticmethod
    def compute_f1_score(
            predictions: List[str],
            targets: List[str]
    ) -> float:
        """
        Compute the F1 score.

        :param predictions: List of predicted values.
        :param targets: List of target values.
        :return: F1 score.
        """
        f1 = f1_score(targets, predictions)
        return f1

    @staticmethod
    def _normalize(prediction: str, substring: str) -> str:
        """
        Normalize the prediction by removing a specified substring.

        :param prediction: Prediction string.
        :param substring: Substring to remove from the prediction.
        :return: Normalized prediction string.
        """
        if prediction.startswith(substring):
            prediction = prediction[len(substring):]
        elif prediction.endswith(substring):
            prediction = prediction[:-len(substring)]
        elif substring.lower() in prediction.lower():
            prediction = prediction.replace(substring.lower(), "").replace(substring.upper(), "").strip()

        return prediction.strip()

    @staticmethod
    def _contains(key: str, candidates: Union[str, List[str]]) -> bool:
        """
        Check if the key string is contained in any of the candidate strings.

        :param key: Key string to search for.
        :param candidates: List of candidate strings.
        :return: True if key is contained in any candidate, False otherwise.
        """
        if isinstance(candidates, str):
            return key.lower() in candidates.lower()
        else:
            for c in candidates:
                if key.lower() in c.lower():
                    return True
            return False

    @staticmethod
    def sentiment_accuracy_metric(predictions: List[str], targets: List[str]) -> float:
        return LlmModule.accuracy_metric(predictions, targets)

    @staticmethod
    def intent_accuracy_metric(predictions: List[str], targets: List[str]) -> float:
        return LlmModule.accuracy_metric(predictions, targets)

    @staticmethod
    def accuracy_metric(predictions: List[str], targets: List[str]) -> float:
        """
        Compute the accuracy metric between predictions and targets.

        :param predictions: List of predicted values.
        :param targets: List of target values.
        :return: Accuracy metric score.
        """
        total_count = len(predictions)
        correct_count = sum(1 for prediction, target in zip(predictions, targets) if prediction == target)
        accuracy = correct_count / total_count
        return accuracy

    @staticmethod
    def performance_metric(
            predictions: List[str],
            targets: List[str],
            metric: Union[str, MetricType],
            prompt_type: Optional[PromptType] = None,
            callback: Optional[Callable] = None,
            prefix_to_remove=None) -> float:
        """
        Compute the performance metric between predictions and targets.

        :param prefix_to_remove:
        :param predictions: List of predicted values from the language model.
        :param targets: List of target values.
        :param metric: Metric type as a string or an enumeration value.
        :param prompt_type: Type of the prompt.
        :param callback: Callback function to invoke for custom prompt types.
        :return: Performance metric score.
        """
        if isinstance(metric, str):
            metric = MetricType(metric)

        if prompt_type == PromptType.SENTIMENT:
            if metric == MetricType.F1_SCORE:
                return LlmModule.compute_f1_score(predictions, targets)
            elif metric == MetricType.ROUGE:
                return LlmModule.compute_rouge_metric(predictions, targets)
            else:
                return LlmModule.sentiment_accuracy_metric(predictions, targets)

        if prompt_type == PromptType.INTENT:
            if metric == MetricType.F1_SCORE:
                return LlmModule.compute_f1_score(predictions, targets)
            elif metric == MetricType.ROUGE:
                return LlmModule.compute_rouge_metric(predictions, targets)
            else:
                return LlmModule.intent_accuracy_metric(predictions, targets)

        if metric == MetricType.ROUGE:
            return LlmModule.compute_rouge_metric(predictions, targets)
        elif metric == MetricType.F1_SCORE:
            return LlmModule.compute_f1_score(predictions, targets)
        elif metric == MetricType.EXACT_MATCH:
            exact = LlmModule.compute_exact_match(predictions, targets)
            # in all case if we have exact match we return it
            if exact == 1.0:
                return exact
            # if we have exact match skip this step
            if prompt_type == PromptType.EXACT_MATCH:
                return exact
            if prompt_type == PromptType.SUMMARY:
                return LlmModule.compute_exact_match(predictions, targets)
            elif prompt_type == PromptType.QUESTION:
                default_prefix = "Q:"
                if prefix_to_remove is None:
                    prefix_to_remove = default_prefix
                normalized = [LlmModule._normalize(p, prefix_to_remove) for p in predictions]
                return sum([LlmModule._contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
            elif prompt_type == PromptType.TLDR:
                default_prefix = "TLDR"
                if prefix_to_remove is None:
                    prefix_to_remove = default_prefix
                normalized = [LlmModule._normalize(p, prefix_to_remove) for p in predictions]
                s = sum([LlmModule._contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
                print("S", s)
                return sum([LlmModule._contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
            elif prompt_type == PromptType.CUSTOM:
                if callback is not None:
                    return callback(predictions, targets)
                else:
                    raise ValueError("Callback function must be provided for CUSTOM prompt type.")
        else:
            raise NotImplementedError()

    @staticmethod
    def metric_for_dataset():
        """Returns metric types
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
        labels[:, -1] = ignore_index
        mask = torch.tensor(input_ids == pad_token_id)
        labels = labels.masked_fill(mask, ignore_index)

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
