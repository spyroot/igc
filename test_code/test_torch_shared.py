# this will download all models
import os
import socket
import argparse

from igc.shared.shared_torch_utils import (
    get_device,
    is_bf16_supported,
    is_amp_supported,
    torch_runtime_details,
    cuda_memory,
    check_torch_distributed_ops,
    torch_distributed_operations_test)


def main(args):
    # Run the distributed operations test
    torch_distributed_operations_test(args.local_rank, args.world_size)


def get_network_interfaces():
    """
    Get a list of network interface names.

    :return: List of interface names.
    """
    interfaces = []
    for interface in socket.if_nameindex():
        interfaces.append(interface[1])
    return interfaces


if __name__ == '__main__':
    dev = get_device()
    torch_runtime_details()
    print(is_bf16_supported())
    print(is_amp_supported())
    cuda_memory(is_verbose=False)
    check_torch_distributed_ops()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training.")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of processes.")
    args = parser.parse_args()

    # Set the environment variables for distributed training
    os.environ["RANK"] = str(args.local_rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)

    # Example usage
    interface_names = get_network_interfaces()
    print(interface_names)
