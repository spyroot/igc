"""
This factory class, which is used to create where
we serialize all metrics.

Author:Mus mbayramo@stanford.edu
"""
import os
import tempfile
import warnings
from abc import abstractmethod, ABC
from pathlib import Path
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
    def __init__(self, project: str = None, entity: str = None,
                 name: str = None, group: str = None, job_type: str = None,
                 tags: list = None, config: dict = None):
        """Start a W&B run; project/entity default to ``$WANDB_PROJECT`` / ``$WANDB_ENTITY``.

        Defaulting from the environment (which `.internal/wandb.env` or the launcher sets)
        is what lets the logger build when the parsed spec carries no project/entity — the
        prior required-arg signature fell through to a None logger, so nothing was tracked.
        ``name``/``group``/``job_type``/``tags``/``config`` (built by
        :func:`_wandb_run_meta` from the spec) make the training STAGE (e.g. m1 state
        encoder), model, batch, and epochs legible in the W&B UI instead of a random name.

        :param project: W&B project (default: ``$WANDB_PROJECT``).
        :param entity: W&B entity/team (default: ``$WANDB_ENTITY``).
        :param name: human-readable run name (e.g. ``m1-state-encoder-qwen2.5-0.5b-e5-bs8``).
        :param group: run group (the stage), so all runs of a stage cluster together.
        :param job_type: ``train`` / ``eval``.
        :param tags: filterable tags (stage, model, bs, epochs, lora, sharding).
        :param config: the run's hyperparameters, shown in the W&B config panel.
        """
        import os

        import wandb
        self.run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT"),
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=config,
        )

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


def _wandb_run_meta(kw: dict) -> dict:
    """Build W&B run metadata (name/group/job_type/tags/config) from the spec kwargs.

    Maps the ``--train``/``--llm``/``--rl`` selection to a readable curriculum label so
    a W&B run reads as e.g. ``m1-state-encoder`` grouped, tagged, and configured — not a
    random name. The old goal/parameter selections are labelled as legacy.

    :param kw: the flattened spec (``vars(specs)``).
    :return: a dict of ``name``, ``group``, ``job_type``, ``tags``, ``config``.
    """
    train, llm = kw.get("train"), kw.get("llm")
    stage_map = {
        ("llm", "latent"): "m1-state-encoder",
        ("llm", "all"): "m1m2-encoder",
        ("llm", "goal"): "goal-extractor-legacy",
        ("llm", "parameter"): "param-extractor-legacy",
    }
    if train in ("agent",) or kw.get("rl"):
        stage = "m6-rl-agent"
    else:
        stage = stage_map.get((train, llm), f"{train or 'run'}-{llm}" if llm else (train or "run"))

    model = str(kw.get("model_type") or "").rstrip("/").split("/")[-1].lower() or "model"
    epochs, bs = kw.get("num_train_epochs"), kw.get("per_device_train_batch_size")

    tags = [stage, model]
    if epochs is not None:
        tags.append(f"ep{epochs}")
    if bs is not None:
        tags.append(f"bs{bs}")
    if kw.get("use_peft"):
        tags.append("lora")
    if kw.get("sharding") and kw.get("sharding") != "none":
        tags.append(str(kw.get("sharding")))

    name_bits = [stage, model]
    if epochs is not None:
        name_bits.append(f"e{epochs}")
    if bs is not None:
        name_bits.append(f"bs{bs}")

    config_keys = [
        "model_type", "train", "llm", "rl", "num_train_epochs",
        "per_device_train_batch_size", "num_workers", "use_peft",
        "lora_r", "lora_alpha", "sharding", "llm_torch_dtype", "device",
        "lora_dropout", "lora_method", "use_rslora", "use_dora",
        "m3_profile", "m3_record_count", "m3_optimizer", "m3_scheduler",
        "m3_learning_rate", "m3_weight_decay", "m3_warmup_ratio",
        "m3_gradient_accumulation_steps", "m3_max_length", "m3_max_steps",
        "m3_precision", "m3_gradient_checkpointing", "m3_torch_compile",
        "m3_dataloader_num_workers", "m3_total_parameters",
        "m3_trainable_parameters", "m3_trainable_parameter_ratio",
        "hf_home", "hf_hub_cache", "hf_token_available",
        "nccl_mnnvl_enable", "nccl_cumem_enable", "nccl_nvls_enable",
    ]
    config = {k: kw[k] for k in config_keys if k in kw}

    return {
        "name": "-".join(name_bits),
        "group": stage,
        "job_type": "train",
        "tags": tags,
        "config": config,
    }


class NullLogger(BaseLogger):
    """No-op metric logger for non-main ranks.

    Under multi-GPU (accelerate/torchrun) every rank constructs the
    MetricLogger, so a network backend like W&B would open one run PER RANK
    (four runs for a 4-GPU job). Non-main ranks get this no-op instead, so a
    job produces exactly one clean W&B run driven by rank 0.
    """

    def log_scalar(self, tag: str, value: float, step: int):
        pass


def _is_main_process() -> bool:
    """True on the main distributed rank (or a single-process run).

    Reads ``RANK`` then ``LOCAL_RANK`` (set by accelerate/torchrun); absent =>
    single process => main.

    :return: whether this process should own the metric backend.
    """
    return int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0) == 0


def create_logger(report_to: str, **kwargs: Optional[str]):
    """
    Create logger

    :param report_to:
    :param kwargs:
    :return:
    """
    # only the main rank owns the metric backend (one W&B run per job, not one per GPU).
    if not _is_main_process():
        return NullLogger()
    try:
        common_args = {key: kwargs[key] for key in [
            "api_key", "project_name", "workspace",
            "tracking_uri", "run_name",
            "project_qualified_name",
            "project", "entity",
            "task_name"
        ] if key in kwargs}

        if report_to == 'comet_ml':
            return CometMLLogger(**common_args)
        elif report_to == 'mlflow':
            return MlFlowLogger(**common_args)
        elif report_to == 'neptune':
            return NeptuneLogger(**common_args)
        elif report_to == 'tensorboard':
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                warnings.warn("output_dir is not provided. Using a temporary directory as a fallback.")
                output_dir = tempfile.mkdtemp()
            else:
                os.makedirs(output_dir, exist_ok=True)
            return TensorBoardLogger(output_dir=output_dir)
        elif report_to == 'wandb':
            wb_args = {k: common_args[k] for k in ("project", "entity") if k in common_args}
            return WandbLogger(**wb_args, **_wandb_run_meta(kwargs))
        elif report_to == 'clearml':
            return ClearMLLogger(**common_args)
        else:
            raise ValueError(f'Invalid logger type {report_to}')
    except ImportError:
        print(f"Cannot create {report_to} logger, please make sure the {report_to} module is installed.")
        return None
