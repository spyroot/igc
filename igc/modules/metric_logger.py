from typing import Optional
from .metric_factory import create_logger


class MetricLogger:
    def __init__(self, report_to: str, **kwargs: Optional[str]):
        """

        :param report_to:
        :param kwargs:
        """
        self.logger = create_logger(report_to, **kwargs)

    def log_metric(self, key: str, value: float, step: int):
        """
        :param key:
        :param value:
        :param step:
        :return:
        """
        if self.logger is not None:
            self.logger.log_scalar(key, value, step)
