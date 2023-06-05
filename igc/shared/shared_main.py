import os
import re
import torch
from datetime import datetime
from transformers import deepspeed
from .shared_arg_parser import shared_arg_parser
from .shared_torch_utils import get_device


def shared_main(
    is_deepspeed_dd_init=False,
    is_cuda_empty_cache=True,
    device_list_visibility: str = None
):
    """Share main function for all trainers.
    :return:
    """
    args = shared_arg_parser()
    if args.local_rank == -1:
        args.device = get_device()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        if is_deepspeed_dd_init:
            deepspeed.init_distributed()

    if is_cuda_empty_cache:
        torch.cuda.empty_cache()

    if device_list_visibility:
        # Validate device_list_visibility format
        if isinstance(device_list_visibility, str):
            if not re.match(r'^\d(,\d)*$', device_list_visibility):
                raise ValueError("Invalid format for device_list_visibility. "
                                 "It should be a comma-separated string of integers.")
        else:
            raise TypeError("device_list_visibility should be a string.")

        os.environ["CUDA_VISIBLE_DEVICES"] = device_list_visibility

    # create output directory if not provided
    if args.output_dir is None:
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{args.model_type}_" \
                   f"{args.per_device_train_batch_size}_" \
                   f"{args.llm_optimizer}_" \
                   f"{args.scheduler}_lr_" \
                   f"{args.llm_learning_rate}"

        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.output_dir = os.path.join(package_dir, "experiments")
        args.output_dir = os.path.join(args.output_dir, run_name)

    args.log_dir = os.path.join(args.output_dir, "../../logs")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args


