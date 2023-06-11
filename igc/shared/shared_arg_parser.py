import argparse
from typing import Optional

import deepspeed
import loguru
from accelerate import Accelerator
from .shared_torch_builder import TorchBuilder
from .shared_torch_utils import get_device


def set_logger(spec):
    """
    Set up the logger based on the log level specified in the spec.

    :param spec: The argparse.Namespace object containing the program arguments.
    """
    log_level = spec.llm_log_level.upper()

    # Configure loguru logger
    loguru.logger.remove()  # Remove any existing handlers
    loguru.logger.add("logfile.log", level=log_level)  # Add a handler with the specified log level


def add_optimizer_group(parser):
    """
    Optimizer parameters.

    :param parser:
    :return:
    """
    optimizer_group = parser.add_argument_group('Optimizer')
    optimizer_group.add_argument(
        "--llm_optimizer",
        type=str,
        default="AdamW2",
        choices=TorchBuilder.get_supported_optimizers(),
        help="LLM Optimizer to use."
    )

    optimizer_group.add_argument(
        "--llm_learning_rate",
        type=float, default=1e-5,
        # type=float, default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.")

    optimizer_group.add_argument(
        "--llm_weight_decay",
        type=float, default=0.0,
        help="Weight decay to use.")

    optimizer_group.add_argument(
        "--gradient_checkpointing",
        type=bool, default=True,
        help="Gradient checkpointing to save memory.")

    optimizer_group.add_argument(
        "--gradient_accumulation_steps",
        type=int, default=8,
        help="Gradient checkpointing to save memory.")

    return parser


def add_model_type_group(parser):
    """
    LLM Model parameters.

    :param parser:
    :return:
    """
    model_type_group = parser.add_argument_group('Model Type')
    model_type_group.add_argument(
        "--model_type",
        type=str, default="gpt2",
        choices=['gpt2-xl', 'gpt2-large', 'gpt2-medium', 'gpt2'],
        help="Model type."
    )
    return parser


def add_scheduler_group(parser):
    """
    Scheduler parameters.

    :param parser:
    :return:
    """
    scheduler_group = parser.add_argument_group('Scheduler')
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="StepLR",
        choices=TorchBuilder.get_supported_schedulers(),
        help="Scheduler to use."
    )
    scheduler_group.add_argument(
        "--num_warmup_steps",
        type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    return parser


def add_rl_trainer_group(parser):
    """
    This all RL agent trainer parameters.

    :param parser:
    :return:
    """

    trainer_group = parser.add_argument_group('RL Trainer')

    trainer_group.add_argument(
        "--rl_num_train_epochs",
        type=int, default=1000,
        help="Total number of training epochs to perform.")

    trainer_group.add_argument(
        "--rl_max_train_steps",
        type=int, default=None,
        help="Total number of training steps to perform."
             " If provided, overrides num_train_epochs.")

    trainer_group.add_argument(
        "--max_trajectory_length",
        type=int, default=None,
        help="Maximum length of a trajectory.")

    trainer_group.add_argument(
        "--rl_buffer_size",
        type=float, default=1e6,
        help="Experience buffer size.")

    trainer_group.add_argument(
        "--rl_batch_size",
        type=float, default=8,
        help="Batch size we use for rl agent. "
             "Note number environments will be the same since it vectorized.")

    trainer_group.add_argument(
        "--rl_steps_per_episode",
        type=int, default=16,
        help="Batch size we use for rl agent. "
             "Note number environments will be the same since it vectorized.")

    trainer_group.add_argument(
        "--rl_gamma_value",
        type=float, default=0.98,
        help="Gamma value for the rl agent.")

    trainer_group.add_argument(
        "--rl_num_optimization_steps",
        type=int, default=40,
        help="Number of optimization steps.")

    return parser


def add_trainer_group(parser):
    """
    This all trainer parameters.

    :param parser:
    :return:
    """

    trainer_group = parser.add_argument_group('Trainer')
    trainer_group.add_argument(
        "--per_device_train_batch_size",
        type=int, default=4,
        help="The batch size per GPU/TPU core/CPU for training.")

    trainer_group.add_argument(
        "--per_device_eval_batch_size",
        type=int, default=8,
        help="Batch size (per device) "
             "for the evaluation dataloader.")

    trainer_group.add_argument(
        "--auto_batch",
        type=bool, default=False,
        help="Whether to find a batch size that will fit into memory "
             "automatically through exponential decay, avoiding "
             "CUDA Out-of-Memory errors")

    trainer_group.add_argument(
        "--num_train_epochs",
        type=int, default=1000,
        help="Total number of training epochs to perform.")

    trainer_group.add_argument(
        "--max_train_steps",
        type=int, default=None,
        help="Total number of training steps to perform."
             " If provided, overrides num_train_epochs.")

    trainer_group.add_argument(
        "--output_dir",
        type=str, default=None,
        help="Where to store the model.")

    trainer_group.add_argument(
        "--seed",
        type=int, default=42,
        help="A seed for reproducible training.")

    parser.add_argument(
        "--data_seed",
        type=int, default=42,
        help="Random seed to be used with data samplers."
    )

    # indicate that we train
    parser.add_argument(
        "--train",
        choices=["agent", "llm", "all", "none"],
        type=str, default="none",
        help="Training mode, we either train all or agent or llm."
    )

    # setting what llm model we train
    parser.add_argument(
        "--llm",
        choices=["all", "latent", "goal", "parameter", "encoder", "none"],
        type=str, default="none",
        help="if we training llm we can train all or particular sub-model."
             "(A model we use for state encoder, goal encoder, "
             "goal and parameter encoder.)"
    )

    parser.add_argument(
        "--rl",
        choices=["all", "none"],
        type=str, default="none",
        help="Training rl agent"
    )

    # do we want to run evaluation for each model during a training
    parser.add_argument(
        "--eval",
        type=bool, default=True,
        help="Run evaluation for each model we train, based"
             " the validation set for a particular model and strategy."
    )

    # do we use gradient norm or not.
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm (for gradient clipping)."
    )

    # bound to a total number of steps.
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set to a positive number, the total number of training steps to perform."
    )

    # this mainly for debug model to overfit on a batch.
    parser.add_argument(
        "--overfit",
        type=bool, default=False,
        help="By default do just overfit pass. This is mainly for debug."
    )

    return parser


def add_logging_group(parser):
    """
    Logging argument parameters.

    :param parser:
    :return:
    """
    group = parser.add_argument_group('Logging')

    group.add_argument("--log_dir",
                       type=str, default="logs",
                       help="Log.")

    group.add_argument("--log_level",
                       type=str, default='warning',
                       choices=['info', 'warning', 'error', 'critical'],
                       help="llm log level..")

    group.add_argument("--llm_log_level",
                       type=str, default='warning',
                       choices=['info', 'warning', 'error', 'critical'],
                       help="llm log level..")

    group.add_argument("--rl_log_level",
                       type=str, default='warning',
                       choices=['info', 'warning', 'error', 'critical'],
                       help="rl agent log level.")

    parser.add_argument('--log-to-file',
                        action='store_true',
                        help='By detail logger output to console, this will switch logs to a file')

    group.add_argument("--dataset_builder",
                       type=str, default='warning',
                       choices=['info', 'warning', 'error', 'critical'],
                       help="dataset log level.")

    group.add_argument("--log_strategy",
                       type=str, default="epoch",
                       choices=['no', 'epoch', 'steps'],
                       help="logging strategy to adopt during training.")

    group.add_argument("--log_on_each_node",
                       type=bool, default=False,
                       help=" In multinode distributed training, whether to log using "
                            "`log_level` once per node, or only on the main")

    group.add_argument("--log_steps",
                       type=int, default=10,
                       help=" Number of update steps between two logs.")

    return parser


def add_data_types_group(parser):
    """
    Tensor data types parameters.

    :param parser:
    :return:
    """
    data_types_group = parser.add_argument_group('Data Types')

    data_types_group.add_argument(
        "--bf16",
        action='store_true',
        help="Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. "
             "Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). "
             "This is an experimental API and it may change.")

    data_types_group.add_argument(
        "--fp16", action='store_true',
        default=True,
        help="Whether to use fp16 16-bit (mixed) precision "
             "training instead of 32-bit training.")

    data_types_group.add_argument(
        "--fp16_opt_level", type=str, default='O1',
        choices=['O0', 'O1', 'O2', 'O3'],
        help="For fp16 training, Apex AMP optimization level selected in "
             "['O0', 'O1', 'O2', and 'O3'].")

    data_types_group.add_argument(
        "--half_precision_backend",
        type=str, default="auto",
        choices=["auto", "cuda_amp", "apex", "cpu_amp"],
        help="The backend to use for mixed precision training. "
             "Must be one of 'auto', 'cuda_amp', 'apex', 'cpu_amp'.")

    data_types_group.add_argument(
        "--bf16_full_eval",
        action='store_true',
        help="Whether to use full bfloat16  instead of 32-bit. "
             "This will be faster and save memory but can harm metric values."
             "This is an experimental API and it may change.")

    data_types_group.add_argument(
        "--fp16_full_eval",
        action='store_true',
        help="Whether to use full float16 evaluation instead of 32-bit. "
             "This will be faster and save memory but can harm metric values.")

    data_types_group.add_argument(
        "--tf32",
        action='store_true',
        help="Whether to enable the TF32 mode, available in Ampere and newer GPU architectures. "
             "This is an experimental API and it may change.")

    return parser


def add_checkpoint_group(parser):
    """
    Saving checkpoint strategies, and other checkpoint related parameters.

    :param parser:
    :return:
    """
    group = parser.add_argument_group('Checkpoint')
    group.add_argument("--save_strategy",
                       type=str, default="epoch",
                       choices=['no', 'epoch', 'steps'],
                       help="Save is done at the end of each epoch, or given step")

    group.add_argument("--save_steps",
                       type=int, default=500,
                       help="Number of updates steps before two checkpoint saves.")

    group.add_argument("--save_total_limit",
                       type=int, default=None,
                       help="limit the total amount of checkpoints.")

    group.add_argument("--save_on_each_node",
                       type=bool, default=False, required=False,
                       help="When doing multi-node distributed training, "
                            "whether to save models and checkpoints on each node, or only on."
                            "the main one.")
    return parser


def add_distribute_backends(parser):
    """
    Distribute backend settings, by default we use accelerate.

    :param parser:
    :return:
    """
    group = parser.add_argument_group('Checkpoint')
    group.add_argument("--ddp_backend",
                       type=str, default="nccl",
                       choices=['nccl', 'mpi', 'ccl', 'gloo'],
                       help="backend to use for distributed training.")

    group.add_argument("--sharded_ddp",
                       type=str, default="simple",
                       choices=["simple", "zero_dp_2", "zero_dp_3", "offload"],
                       help="Use Sharded DDP training from FairScale "
                            "(in distributed training only). "
                            "This is an experimental feature.")

    group.add_argument(
        "--deepspeed",
        type=str, help="Use Deepspeed. The value can be either"
                       " the location of the DeepSpeed JSON config file "
                       "(e.g., `ds_config.json`) or an already loaded JSON file as a `dict`.")

    return parser


def add_dataset_dataloader(parser):
    """
    Dataloader settings.

    :param parser:
    :return:
    """
    group = parser.add_argument_group('Dataset and Dataloader')
    group.add_argument(
        "--dataloader_pin_memory",
        type=bool, default=False,
        help="Whether pin dataset to memory in torch data loaders or not.")

    group.add_argument(
        "--num_workers",
        type=int, default=1,
        help="Number of workers to use for data loading.")

    group.add_argument(
        "--raw_data_dir",
        type=str, default="~/.json_responses",
        help="A directory where all discovered rest API json files. "
             "This mainly need a node that build dataset")

    group.add_argument(
        "--do_consistency_check",
        type=bool, default=True,
        help="Whether we perform dataset consistency check.")

    return parser


def add_reporting_group(parser):
    """
    :param parser:
    :return:
    """
    group = parser.add_argument_group('Reporting and Metric')
    group.add_argument(
        "--metric_report", type=str, default="tensorboard",
        choices=['comet_ml', 'mlflow',
                 'neptune', 'tensorboard',
                 'wandb', 'clearml'],
        help="Where we want report metrics.")
    return parser


def shared_arg_parser(
        is_deepspeed_arg_parser: Optional[bool] = False,
        is_accelerate_arg_parser: Optional[bool] = False,
        is_fairscale_arg_parser: Optional[bool] = False):
    """

    :param is_deepspeed_arg_parser:
    :param is_accelerate_arg_parser:
    :param is_fairscale_arg_parser:
    :return:
    """
    parser = argparse.ArgumentParser(
        description="IGC Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = add_trainer_group(parser)
    parser = add_optimizer_group(parser)
    parser = add_scheduler_group(parser)
    parser = add_logging_group(parser)
    parser = add_checkpoint_group(parser)
    parser = add_data_types_group(parser)
    parser = add_distribute_backends(parser)
    parser = add_dataset_dataloader(parser)
    parser = add_reporting_group(parser)
    parser = add_model_type_group(parser)
    parser = add_rl_trainer_group(parser)

    parser.add_argument("--local-rank",
                        type=int, default=-1,
                        help="local_rank for distributed training on GPUs")

    if is_accelerate_arg_parser:
        try:
            accelerator = Accelerator()
            parser = accelerator.inject_arguments(parser)
        except ImportError:
            pass

    if is_fairscale_arg_parser:
        optimizer_group = parser.add_argument_group('Optimizer')
        optimizer_group.add_argument("--optimizer",
                                     type=str, default="adamw2",
                                     choices=['adamw', 'adamw2', 'sgd', 'adagrad'],
                                     help="Optimizer to use. adamw2 is default hugging face version of adam.")

        optimizer_group.add_argument(
            "--scale-loss", action='store_true',
            help="Enable gradient scaling using fairscale.")

    if is_deepspeed_arg_parser:
        parser = deepspeed.add_config_arguments(parser)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=['cuda', 'cpu', 'mps', 'auto'],
        help="Device to use (overrides the default device)."
             " Options: 'cuda', 'cpu', or 'auto'. 'auto' selects 'cuda' if available."
    )

    args = parser.parse_args()
    args.device = get_device() if args.device == "auto" else args.device

    # Set up the logger based on the log level in the arguments
    set_logger(args)
    return args
