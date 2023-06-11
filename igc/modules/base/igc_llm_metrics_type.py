"""
Typing for all llm metrics

Author:Mus mbayramo@stanford.edu
"""
from enum import Enum


class MetricType(Enum):
    """Metric type
    """
    F1_SCORE = 'f1'
    ROUGE = 'rouge'
    BLEU = 'bleu'
    MAP = 'map'
    WER = 'wer'
    MSE = 'mse'
    PERPLEXITY = 'perplexity'
    EXACT_MATCH = 'exact match'
    SPEARMAN_CORRELATION = 'spearman correlation'
    CLASSIFICATION_ACCURACY = 'classification accuracy'

    def __str__(self):
        return self.value
