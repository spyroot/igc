"""
Shared class for logging metrics to console and file.
Author:Mus mbayramo@stanford.edu
"""
from typing import Optional

import loguru
from igc.modules.metric_factory import create_logger


class MetricLogger:
    def __init__(self, report_to: str, **kwargs: Optional[str]):
        """
        :param report_to:
        :param kwargs:
        """
        self.logger = create_logger(report_to, **kwargs)
        self._file_terminal_logger = loguru.logger
        self._file_terminal_log_level = loguru.logger.level

    def log_metric(self, key: str, value: float, step: int):
        """
        :param key:
        :param value:
        :param step:
        :return:
        """
        if self.logger is not None:
            self.logger.log_scalar(key, value, step)

    def set_logger(self, logger):
        """
        Set the logger for logging to console or file.

        :param logger: The logger object to use.
        """
        self._file_terminal_logger = logger

    def set_log_level(self, log_level):
        """
        Set the log level for the MetricLogger.

        :param log_level: The log level to use.
        """
        self._file_terminal_log_level = log_level

