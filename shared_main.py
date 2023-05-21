import argparse

import torch
from transformers import deepspeed
from arg_parser_shared import parse_args


def shared_main(
    is_deepspeed_dd_init=False,
    is_cuda_empty_cache=True,
):
    """

    :return:
    """
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        if is_deepspeed_dd_init:
            deepspeed.init_distributed()

    if is_cuda_empty_cache:
        torch.cuda.empty_cache()

    return args, device
