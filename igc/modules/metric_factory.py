from abc import abstractmethod, ABC
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
from typing import Optional


class BaseLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """

        :param tag:
        :param value:
        :param step:
        :return:
        """
        pass


class TensorBoardLogger(BaseLogger):
    def __init__(self, output_dir: str):
        """
        :param output_dir: a dir where store tensorboard log for a particular experiment
        """
        from torch.utils.tensorboard import SummaryWriter
        p = Path(output_dir).resolve()
        if not p.is_dir():
            raise ValueError(f'Please pass a directory for '
                             f'tensorboard logger. Unable resolve dir {str(p)}')

        self.writer = SummaryWriter(log_dir=f"{str(p)}/log")

    def log_scalar(self, tag: str, value: float, step: int):
        """
        :param tag:
        :param value:
        :param step:
        :return:
        """
        self.writer.add_scalar(tag, value, step)


class CometMLLogger(BaseLogger):
    def __init__(self, api_key: str, project_name: str, workspace: str):
        import comet_ml
        self.experiment = comet_ml.Experiment(api_key=api_key, project_name=project_name, workspace=workspace)

    def log_scalar(self, tag: str, value: float, step: int):
        self.experiment.log_metric(tag, value, step=step)


class MlFlowLogger(BaseLogger):
    def __init__(self, tracking_uri: str, run_name: str):
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_name=run_name)
        self.mlflow = mlflow

    def log_scalar(self, tag: str, value: float, step: int):
        self.mlflow.log_metric(tag, value, step=step)


class NeptuneLogger(BaseLogger):
    def __init__(self, api_token: str, project_qualified_name: str):
        import neptune.new as neptune
        self.run = neptune.init(project=project_qualified_name, api_token=api_token)

    def log_scalar(self, tag: str, value: float, step: int):
        self.run[tag].log(value, step=step)


class WandbLogger(BaseLogger):
    def __init__(self, project: str, entity: str):
        """

        :param project:
        :param entity:
        """
        import wandb
        self.run = wandb.init(project=project, entity=entity)

    def log_scalar(self, tag: str, value: float, step: int):
        """

        :param tag:
        :param value:
        :param step:
        :return:
        """
        self.run.log({tag: value}, step=step)


class ClearMLLogger(BaseLogger):
    def __init__(self, project_name: str, task_name: str):
        """

        :param project_name:
        :param task_name:
        """
        from clearml import Task
        self.task = Task.init(project_name=project_name, task_name=task_name)

    def log_scalar(self, tag: str, value: float, step: int):
        """

        :param tag:
        :param value:
        :param step:
        :return:
        """
        self.task.get_logger().report_scalar(title=tag, series='series', iteration=step, value=value)


def create_logger(report_to: str, **kwargs: Optional[str]):
    try:
        common_args = {key: kwargs[key] for key in ["api_key", "project_name", "workspace",
                                                    "tracking_uri", "run_name",
                                                    "project_qualified_name",
                                                    "project", "entity",
                                                    "task_name"]
                       if key in kwargs}

        if report_to == 'comet_ml':
            return CometMLLogger(**common_args)
        elif report_to == 'mlflow':
            return MlFlowLogger(**common_args)
        elif report_to == 'neptune':
            return NeptuneLogger(**common_args)
        elif report_to == 'tensorboard':
            return TensorBoardLogger(output_dir=kwargs["output_dir"])
        elif report_to == 'wandb':
            return WandbLogger(**common_args)
        elif report_to == 'clearml':
            return ClearMLLogger(**common_args)
        else:
            raise ValueError(f'Invalid logger type {report_to}')
    except ImportError:
        print(f"Cannot create {report_to} logger, please make sure the {report_to} module is installed.")
        return None
