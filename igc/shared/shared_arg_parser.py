import argparse
from typing import Optional

import deepspeed
import loguru
from accelerate import Accelerator
from .shared_torch_builder import TorchBuilder
from .shared_torch_utils import get_device
from ..ds.redfish_masked_dataset import MaskingOption, MaskingType


def set_logger(spec):
    """
    Set up the logger based on the log level specified in the spec.

    :param spec: The argparse.Namespace object containing the program arguments.
    """
    log_level = spec.llm_log_level.upper()
    loguru.logger.remove()
    loguru.logger.add("logfile.log", level=log_level)


def add_accelerator_parser_group(parser) -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments

    :return: ArgumentParser object
    """
    accelerate_group = parser.add_argument_group(description="Accelerator Argument Parser")
    accelerate_group.add_argument(
        "--use_accelerator", action='store_true',
        default=False, help="Enable accelerator. By default, IGC will use it.")
    accelerate_group.add_argument(
        "--device_placement", type=bool, default=True, help="Device placement flag")
    accelerate_group.add_argument(
        "--split_batches", type=bool, default=False, help="Split batches flag")
    accelerate_group.add_argument(
        "--mixed_precision", type=str, default=None, help="Mixed precision setting")
    accelerate_group.add_argument(
        "--cpu", type=bool, default=False, help="CPU flag")
    accelerate_group.add_argument(
        "--deepspeed_plugin", type=str, default=None, help="DeepSpeed plugin")
    accelerate_group.add_argument(
        "--fsdp_plugin", type=str, default=None, help="FullyShardedDataParallel plugin")
    accelerate_group.add_argument(
        "--megatron_lm_plugin", type=str, default=None, help="MegatronLM plugin")
    accelerate_group.add_argument(
        "--rng_types", nargs="+", default=None, help="RNG types")
    accelerate_group.add_argument(
        "--log_with", nargs="+", default=None, help="Log with")
    accelerate_group.add_argument(
        "--project_dir", type=str, default=None, help="Project directory")
    accelerate_group.add_argument(
        "--project_config", type=str, default=None, help="Project configuration")
    accelerate_group.add_argument(
        "--gradient_accumulation_plugin", type=str, default=None,
        help="Gradient accumulation plugin")
    accelerate_group.add_argument(
        "--dispatch_batches", type=bool, default=None, help="Dispatch batches flag")
    accelerate_group.add_argument(
        "--even_batches", type=bool, default=True, help="Even batches flag")
    accelerate_group.add_argument(
        "--step_scheduler_with_optimizer", type=bool, default=True,
        help="Step scheduler with optimizer flag")
    accelerate_group.add_argument(
        "--kwargs_handlers", nargs="+", default=None, help="Kwargs handlers")
    accelerate_group.add_argument(
        "--dynamo_backend", type=str, default=None, help="Dynamo backend")
    return parser


def add_optimizer_group(parser):
    """
    Optimizer parameters.

    :param parser:
    :return:
    """
    group_name = "optimizer"
    optimizer_group = parser.add_argument_group('Optimizer')
    optimizer_group.add_argument(
        "--llm_optimizer",
        type=str,
        default="AdamW",
        choices=TorchBuilder.get_supported_optimizers(),
        help="LLM Optimizer to use."

    )

    optimizer_group.add_argument(
        "--llm_learning_rate",
        type=float, default=1e-5,
        # type=float, default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    optimizer_group.add_argument(
        "--llm_weight_decay",
        type=float, default=0.01,
        help="Weight decay to use."
    )

    optimizer_group.add_argument(
        "--gradient_checkpointing",
        type=bool, default=True,
        help="Gradient checkpointing to save memory."
    )

    optimizer_group.add_argument(
        "--gradient_accumulation_steps",
        type=int, default=8,
        help="Gradient checkpointing to save memory.")

    return parser


def add_redfish_group(parser):
    """
    This for a Mock server.  If we want to execute live,
    we need to add the following parameters:

    :param parser: existing parser that we use to add
    :return:
    """
    redfish_group = parser.add_argument_group('Mock Server Optional')
    redfish_group.add_argument(
        "--live", action='store_true',
        default=False, help="Enable live mode. All request will be send to redfish host.")
    redfish_group.add_argument(
        "--live_test", action='store_true', default=False,
        help="will execute couple of test before we start RL trainer.")
    redfish_group.add_argument(
        "--redfish-ip", type=str, default="https://10.252.252.209",
        help="IP address of the Redfish server")
    redfish_group.add_argument(
        "--redfish-username", type=str, default="root",
        help="Username for authentication")
    redfish_group.add_argument(
        "--redfish-password", type=str, default="",
        help="Password for authentication")
    redfish_group.add_argument(
        "--redfish-port", type=int,
        help="Port number for the Redfish server.")
    redfish_group.add_argument(
        "--insecure", action="store_true",
        help="Disable SSL certificate verification.")
    redfish_group.add_argument(
        "--is-http", action="store_true",
        help="Use HTTP transport instead of HTTPS.")
    redfish_group.add_argument(
        "--x-auth", type=str,
        help="Enables X-Auth-Token for authentication otherwise use Basic Auth.")

    return parser


def add_model_type_group(parser):
    """
    LLM Model parameters, a name of pre-trained model

    :param parser: existing parser
    :return:
    """
    model_type_group = parser.add_argument_group('Model Type')
    model_type_group.add_argument(
        "--model_type",
        type=str, default="gpt2",
        choices=['gpt2-xl', 'gpt2-large', 'gpt2-medium', 'gpt2'],
        help="Model type, note for anything beyond we need huge memory."
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
        "--llm_scheduler",
        type=str,
        default="OneCycleLR",
        choices=TorchBuilder.get_supported_schedulers(),
        help="Scheduler to use."
    )

    scheduler_group.add_argument(
        "--num_warmup_steps",
        type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )

    scheduler_group.add_argument(
        "--base_lr",
        type=int, default=0.01,
        help="Initial learning rate which is the lower boundary in the cycle for each parameter group.."
    )

    scheduler_group.add_argument(
        "--max_lr",
        type=int, default=0.1,
        help="Upper learning rate boundaries in the cycle for each parameter group. ."
    )

    return parser


def add_auto_encoder_group(parser):
    """
    This all RL agent trainer parameters.

    :param parser:
    :return:
    """

    trainer_group = parser.add_argument_group('Autoencoder Trainer')

    trainer_group.add_argument(
        "--auto_encoder_lr",
        type=float,
        default=0.001,
        help="Auto encoder learning rate.")

    trainer_group.add_argument(
        "--auto_encoder_train_steps",
        type=int, default=None,
        help="Total number of training steps to perform."
             " If provided, overrides num_train_epochs.")

    trainer_group.add_argument(
        "--auto_encoder_optimizer",
        type=str,
        default="Adam",
        choices=TorchBuilder.get_supported_optimizers(),
        help="Autoencoder optimizer to use."
    )

    trainer_group.add_argument(
        "--auto_encoder_weight_decay",
        type=float, default=0.0,
        help="Weight decay to use.")

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
        type=int, default=32,
        help="Maximum length of a trajectory.")

    trainer_group.add_argument(
        "--rl_buffer_size",
        type=float, default=1e6,
        help="Experience buffer size.")

    trainer_group.add_argument(
        "--rl_num_episodes",
        type=int, default=32,
        help="Number of episode to collect.")

    trainer_group.add_argument(
        "--rl_batch_size",
        type=float, default=8,
        help="Batch size we use for rl agent. "
             "Note number environments will be the same since it vectorized.")

    trainer_group.add_argument(
        "--rl_lr",
        type=float, default=1e-3,
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
        "--copy_llm", action='store_true', default=False,
        help="The batch size per GPU/TPU core/CPU for training.")

    trainer_group.add_argument(
        "--test_llm",
        action='store_true', default=False,
        help="Run a test inference check on llm fine tuned model.")

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
        "--llm_mask_freq",
        type=int, default=5,
        help="How frequent we change mask for llm model."
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

    # do we use gradient norm or not?
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
        help="If set to a positive number, the total "
             "number of training steps to perform."
    )

    # this mainly for debug model to overfit on a batch.
    parser.add_argument(
        "--overfit",
        type=bool, default=False,
        help="By default do just overfit pass. This is mainly for debug."
    )

    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["on_epoch", "freq"],
        default="on_epoch",
        help="By default uses freq otherwise at the end of each epoch."
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
                       help=" In distribute setting use log on each node."
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
    group = parser.add_argument_group('Checkpoint and Saving')
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


def masking_option_type(option_str):
    try:
        return MaskingOption[option_str.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid masking option: {option_str}")


def masking_type_type(type_str):
    try:
        return MaskingType[type_str.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid masking type: {type_str}")


def add_dataset_dataloader(parser):
    """
    Dataset and dataloader settings.

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
        "--json_data_dir",
        type=str, default="~/.json_responses",
        help="A directory where all discovered rest API json files. "
             "This mainly need a node that build dataset")

    group.add_argument(
        "--dataset_dir",
        type=str, default="datasets",
        help="A location where we unpack or build a dataset.")

    group.add_argument(
        "--do_consistency_check",
        type=bool, default=False,
        help="Whether we perform dataset consistency check, "
             "post dataset build or during a load procedure.")

    group.add_argument(
        "--masking_option",
        type=masking_option_type, choices=list(MaskingOption), default=MaskingOption.ODATA_ID,
        help="The masking option to apply to the dataset."
    )

    group.add_argument(
        "--masking_type",
        type=masking_type_type, choices=list(MaskingType), default=MaskingType.NO_MASK,
        help="The type of masking to apply to the dataset."
    )

    group.add_argument(
        "--do_random_masking",
        action="store_true",
        help="Whether to choose random masking option from on the dataset. Default is False."
    )

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
        is_fairscale_arg_parser: Optional[bool] = False
):
    """

    :param is_deepspeed_arg_parser:
    :param is_accelerate_arg_parser:
    :param is_fairscale_arg_parser:
    :return:
    """
    parser = argparse.ArgumentParser(
        description="IGC Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True)

    global_trainer_group = add_trainer_group(parser)
    global_optimizer_group = add_optimizer_group(parser)
    global_scheduler = add_scheduler_group(parser)
    global_logging = add_logging_group(parser)
    checkpointing = add_checkpoint_group(parser)
    data_type = add_data_types_group(parser)
    distributed = add_distribute_backends(parser)
    data_loader = add_dataset_dataloader(parser)
    reporting_parser = add_reporting_group(parser)
    model_types_group = add_model_type_group(parser)
    rl_trainer_group = add_rl_trainer_group(parser)
    autoencoder_group = add_auto_encoder_group(parser)
    accelerator = add_accelerator_parser_group(parser)
    mock_and_redfish_group = add_redfish_group(parser)
    parser = mock_and_redfish_group

    sections = [
        ("main", global_trainer_group),
        ("optimizer", global_optimizer_group),
        ("scheduler", global_scheduler),
        ("logging", global_logging),
        ("checkpoint", checkpointing),
        ("data_types", data_type),
        ("distribute_backends", distributed),
        ("dataset_dataloader", data_loader),
        ("reporting", reporting_parser),
        ("model_type", model_types_group),
        ("rl_trainer", rl_trainer_group),
        ("auto_encoder", autoencoder_group),
        ("accelerator", accelerator),
        ("redfish", mock_and_redfish_group)
    ]

    parser.add_argument(
        "--local-rank",
        type=int, default=-1,
        help="local_rank for distributed training on GPUs")

    if is_fairscale_arg_parser:
        optimizer_group = parser.add_argument_group('Optimizer')
        optimizer_group.add_argument(
            "--optimizer",
            type=str, default="adamw2",
            choices=['adamw', 'adamw2', 'sgd', 'adagrad'],
            help="Optimizer to use. adamw2 is default hugging face version of adam.")

        optimizer_group.add_argument(
            "--scale-loss", action='store_true',
            help="Enable gradient scaling using fairscale.")

    if is_deepspeed_arg_parser:
        parser = deepspeed.add_config_arguments(parser)

    available_gpus = TorchBuilder.available_gpus_string()
    device_choices = ['cuda', 'cpu', 'mps', 'auto'] + available_gpus

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=device_choices,
        help="Device to use (overrides the default device)."
             " Options: 'cuda', 'cpu', or 'auto'. 'auto' selects 'cuda' if available."
    )

    args = parser.parse_args()
    args.device = get_device() if args.device == "auto" else args.device

    if is_accelerate_arg_parser:
        try:
            if args.use_accelerator:
                accelerator = Accelerator()
                parser = accelerator.inject_arguments(parser)
        except ImportError:
            pass

    set_logger(args)

    return args, sections
