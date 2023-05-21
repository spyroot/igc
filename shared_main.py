import argparse
import os

import torch
from transformers import deepspeed
from shared_arg_parser import shared_arg_parser
import re


def shared_main(
    is_deepspeed_dd_init=False,
    is_cuda_empty_cache=True,
    device_list_visibility: str = None
):
    """

    :return:
    """
    args = shared_arg_parser()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
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



    return args, device
