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

from igc.ds.redfish_dataset import JSONDataset
from .igc_base_module import IgcBaseModule
from .igc_metric_logger import MetricLogger


class RlBaseModule(IgcBaseModule):
    """
    """
    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 ds: JSONDataset,
                 metric_logger:
                 MetricLogger, llm_model,
                 llm_tokenizer):
        """
        :param spec: 
        :param ds:
        :param metric_logger:
        :param llm_model:
        :param llm_tokenizer:
        """
        super().__init__(module_name, spec, ds, metric_logger, llm_model, llm_tokenizer)