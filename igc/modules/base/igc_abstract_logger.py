"""
This class abstract logging facility used
all almost every class.

Author:Mus mbayramo@stanford.edu
"""
import os
import sys
from logging import Handler
from pathlib import Path
from typing import Optional, Any
from typing import Union, TextIO, Callable

import loguru


class AbstractLogger:
    """
    This class abstract logging facility used in IGC
    """
    _is_log_to_file = False

    def __init__(self, module_name: str = None):
        """
        Initializes the logger for the module.

        :param module_name: The name of the module (optional).
        """
        if module_name is None:
            module_name = self.__class__.__name__

        self._log_file = None
        self._log_level = "INFO"

        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()
        self.logger.add(sys.stdout, level=self._log_level)

    def _configure_logger(self, module_name: str = None):
        """
        Configures the logger for the module.

        :param module_name: The name of the module (optional).
        """
        if module_name is None:
            module_name = self.__class__.__name__

        logs_dir = "experiments/logs"
        os.makedirs(logs_dir, exist_ok=True)

        self._log_file = os.path.join(logs_dir, f"{module_name}.log")
        self._log_level = "INFO"

        self.logger = loguru.logger.bind(module_name=module_name)
        self.logger.remove()
        self.logger.add(sys.stdout, level=self._log_level)

    @staticmethod
    def create_logger(
        module_name: str = None,
        level: str = "INFO"
    ) -> loguru.logger:
        """
        Creates a logger instance for the specified module.

        :param module_name: The name of the module (optional).
        :param level: The log level (default: "INFO").
        :return: The logger instance.
        """
        logger = loguru.logger
        logger.remove()
        logger.add(sys.stdout, level=level)

        if module_name:
            logs_dir = "experiments/logs"
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"{module_name}.log")
            logger.add(log_file, level=level)

        return logger

    @classmethod
    def generate_handler_config(
        cls,
        sink: Optional[
            Union[str, Path, TextIO, Callable, Handler]
        ] = None,
        level: Union[str, int] = "DEBUG",
        log_format: Union[str, Any] = "{time} - {level} - {message}",
        log_filter: Optional[Union[str, Any]] = None,
        colorize: Optional[bool] = True,
        serialize: bool = False,
        backtrace: bool = True,
        diagnose: bool = False,
        enqueue: bool = True,
        catch: bool = False
    ):
        """
        :param sink: The sink for log messages.
        :param level: The log level.
        :param log_format: The log message format.
        :param log_filter: The log message filter.
        :param colorize: Whether to colorize log messages.
        :param serialize: Whether to serialize log messages.
        :param backtrace: Whether to include backtrace in log messages.
        :param diagnose: Whether to include diagnostic information in log messages.
        :param enqueue: Whether to enqueue log messages.
        :param catch: Whether to catch exceptions in log messages.
        :return: The handler configuration.
        """
        logs_dir = "experiments/logs"
        os.makedirs(logs_dir, exist_ok=True)

        if sink is None:
            sink = logs_dir

        return {
            "sink": sink,
            "level": level,
            "format": log_format,
            "filter": log_filter,
            "colorize": colorize,
            "serialize": serialize,
            "backtrace": backtrace,
            "diagnose": diagnose,
            "enqueue": enqueue,
            "catch": catch
        }

    @classmethod
    def is_log_to_file(cls) -> bool:
        """
        Check if logging to a file is enabled.

        :return: True if logging to a file is enabled, False otherwise.
        """
        return cls._is_log_to_file

    @classmethod
    def enable_log_to_file(cls, level: str = "INFO"):
        """
        Enable logging to separate log files per level.

        :param level: The log level (default: "INFO").
        """
        logs_dir = "experiments/logs"
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"{level.lower()}.log")
        loguru.logger.add(log_file, level=level)
        cls._is_log_to_file = True

    @staticmethod
    def register_handler(handler_config):
        loguru.logger.add(**handler_config)
