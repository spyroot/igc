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
from typing import Optional

from igc.ds.redfish_dataset import JSONDataset
from .igc_base_module import IgcModule
from .igc_metric_logger import MetricLogger


class RlBaseModule(IgcModule):
    """
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = "False",
                 device=None):
        """
        Base RL module

        :param module_name: 
        :param spec: 
        :param ds: 
        :param metric_logger: 
        :param llm_model: 
        :param llm_tokenizer: 
        """
        super().__init__(
            module_name,
            spec,
            llm_model,
            llm_tokenizer,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=is_inference,
            device=device)

        self._log_level = spec.rl_log_level.upper()
        self.logger.info("Starting RL module")
        self.metric_logger.set_log_level(self._log_level)
