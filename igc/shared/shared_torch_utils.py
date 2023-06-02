from distutils.version import LooseVersion
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.backends import cudnn
from torch.cuda import nccl
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tabulate import tabulate
from pynvml import *
# for nccl to figure interface we need
import socket


def get_device(rank: Optional[int] = None) -> torch.device:
    """Get a torch.device. If rank is specified,
    CUDA devices are distributed in a round-robin fashion.

    :param rank: The rank of the current process, for CUDA device distribution.
                  If None, no distribution is done.
    :return: A torch.device.
    """
    # Get the total number of CUDA devices
    num_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_cuda_devices > 0:
        if rank is not None:
            # If a rank is specified, distribute CUDA devices in a round-robin fashion
            device_index = rank % num_cuda_devices
        else:
            # If no rank is specified, just use the first CUDA device
            device_index = 0

        cuda_device = torch.device(f"cuda:{device_index}")
        print(f"Using CUDA device: {cuda_device}")
        return cuda_device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print(f"Using MPS device: {dev}")
        return torch.device("mps")
    else:
        print("CUDA and MPS not available, falling back to CPU.")
        return torch.device("cpu")


def shift_and_mask(batch, pad_token_id, ignore=-100):
    """
    :return:
    """
    labels = batch["input_ids"][:, 1:].clone().detach()
    mask = (batch["input_ids"] == pad_token_id).clone().detach()
    labels = labels.masked_fill(mask[:, 1:], ignore)
    batch['input_ids'] = batch['input_ids'][:, :-1]
    batch['attention_mask'] = batch['attention_mask'][:, :-1]
    return batch['input_ids'][:, :-1], batch['attention_mask'][:, :-1], labels


def mask_random_span(self, input_ids, attention_mask):
    """

    :param self:
    :param input_ids:
    :param attention_mask:
    :return:
    """
    batch_size = input_ids.shape[0]
    labels = input_ids.clone()
    input_ids_clone = input_ids.clone()

    for i in range(batch_size):
        input_length = input_ids[i].size(0)
        # randomly choose start  and pos for masking
        mask_start = torch.randint(1, input_length - 1, (1,)).item()
        mask_end = mask_start + torch.randint(1, input_length - mask_start, (1,)).item()
        #     # replace the selected span with pad_token_id
        input_ids[i, mask_start:mask_end] = self.pad_token_id
        #     # set the labels to the original span
        labels[i, mask_start:mask_end] = input_ids[i, mask_start:mask_end]
        # labels[i, :mask_start] = self.pad_token_id
        # labels[i, mask_end:] = self.pad_token_id

    input_ids = input_ids.squeeze(1)
    attention_mask = attention_mask.squeeze(1)
    return input_ids, attention_mask, labels


def masked_select(inputs, token_logits, mask_token_id, tokn=10):
    """Pick the [MASK] candidates with the highest logits
    :param tokn:
    :param inputs:
    :param token_logits:
    :param mask_token_id:
    :return:
    """
    # [MASK] and extract its logits
    mask_token_index = np.argwhere(inputs == mask_token_id)[0, 1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    np.argsort(-mask_token_logits)[:tokn].tolist()
    return np.argsort(-mask_token_logits)[:tokn].tolist()


def get_network_interfaces():
    """Get a list of network interface names.
    :return: List of interface names.
    """
    interfaces = []
    for interface in socket.if_nameindex():
        interfaces.append(interface[1])
    return interfaces


def print_gpu_utilization():
    """
    :return:
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_dict_as_table(data):
    """Print dict as table, this mainly for mem utilization.
    :param data:
    :return:
    """
    table = []
    for key, value in data.items():
        table.append([key, value])
    headers = ["Key", "Value"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def cuda_memory(is_verbose=False):
    """
    :return:
    """
    if is_verbose:
        torch.cuda.memory_summary()
        print_dict_as_table(torch.cuda.memory_stats())

    torch.cuda.mem_get_info()


def torch_runtime_details():
    """
    Print details about the PyTorch runtime environment, including
    CUDA version, CuDNN version, and NCCL version (if available).
    if AMP supported BGF16 supported or not, NCCL available or not.

    """
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    cudnn_version = cudnn.version() if hasattr(torch.backends, 'cudnn') else None
    print("CuDNN version:", cudnn_version if cudnn_version is not None else "Not available")

    if not hasattr(torch._C, '_nccl_all_reduce'):
        print('PyTorch is not compiled with NCCL support')

    print("NCCL available:", torch.distributed.is_nccl_available())
    print("MPI available:", torch.distributed.is_mpi_available())
    print("GLO available:", torch.distributed.is_gloo_available())
    print("UCC available:", torch.distributed.is_ucc_available())

    if torch.cuda.is_available():
        print("BF16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("BF16 not supported:")

    if torch.cuda.is_available():
        mem_get_info = torch.cuda.memory_stats()
        print("Memory allocated:", mem_get_info["allocated_bytes.all.current"] / 1024 ** 3, "GB")
        # additional CUDA statistics if available
        if hasattr(torch.cuda, 'utilization'):
            print("CUDA utilization:", torch.cuda.utilization())
        if hasattr(torch.cuda, 'memory_summary'):
            print("CUDA memory summary:")
            torch.cuda.memory_summary()

    if torch.cuda.is_available() and hasattr(torch.cuda, "amp") and torch.cuda.amp:
        print("AMP is supported")
    else:
        print("AMP is not supported")

    print("distribute supported:", torch.distributed.is_available())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        fp16_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device=device)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device=device, requires_grad=True)
        y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float16, device=device, requires_grad=True)
        z = torch.sqrt(x ** 2 + y ** 2)
        w = torch.sin(z)
        loss = w.sum()
        loss.backward()
        loss = loss.detach()
        fp16_tensor_np = fp16_tensor.cpu().numpy()

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()

        # Check if all tensors have dtype float16
        all_fp16 = all(
            np.issubdtype(arr.dtype, np.float16)
            for arr in [fp16_tensor_np, x_np, y_np, z_np, w_np]
        )
        if all_fp16:
            print("All tensors are in the supported float16 format.")
        else:
            print("Not all tensors have dtype float16.")

        # Float64 tensors
        try:
            x64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device, requires_grad=True)
            y64 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, device=device, requires_grad=True)
            z64 = torch.sqrt(x64 ** 2 + y64 ** 2)
            w64 = torch.sin(z64)
            loss64 = w64.sum()
            loss64.backward()
            print("\nFloat64 tensors:")
            print("loss:", loss64)

            # Convert to NumPy arrays
            x64_np = x64.cpu().numpy()
            y64_np = y64.cpu().numpy()
            z64_np = z64.cpu().numpy()
            w64_np = w64.cpu().numpy()

            # Check if all tensors have dtype float64
            all_fp64 = all(
                np.issubdtype(arr.dtype, np.float64)
                for arr in [x64_np, y64_np, z64_np, w64_np]
            )

            # Print the corresponding message
            if all_fp64:
                print("\nAll float64 tensors have dtype float64.")
            else:
                print("\nNot all float64 tensors have dtype float64.")

        except TypeError as e:
            print("\nUnsupported float64 tensors:", e)


def is_amp_supported():
    return (
            torch.version.cuda
            and torch.cuda.is_available()
            and LooseVersion(torch.version.cuda) >= "11.0"
    )


def is_bf16_supported():
    return (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and torch.distributed.is_nccl_available()
            and nccl.version() >= (2, 10)
    )


def print_summary(result):
    """
    :param result:
    :return:
    """
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def chat_with_gpt2(model_name="gpt2-xl", user_input="Hello!", device="cpu"):
    """
    Function to interact with a GPT-2 model.

    :param device:
    :param model_name: (str) Name of the pretrained model.
    :param user_input: (str) User input string.
    :return: (str) Model's response.
    """
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token to be the eos_token
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    print("You are now chatting with GPT-2, type 'exit' or 'quit' to end the chat.")
    while True:
        # Get user input
        user_input = input("User: ")

        # Break the loop if user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            break

        # Encode user input and end-of-string (EOS) token, then add return_tensors parameter
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.long).to(device)
        attention_mask[:, :len(input_ids[0])] = 1

        # Generate a response with a length of 512 tokens
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"GPT-2: {response}")


def generate_observation(model, dataset):
    """
    :param model:
    :param dataset:
    :return:
    """
    input_ids = dataset[0]['respond']
    # # Need to unsqueeze to add a batch dimension
    input_ids = input_ids.unsqueeze(0)
    output = model(input_ids)
    embeddings = output.last_hidden_state

    print(embeddings.shape)
    print(embeddings)

    data_collator = lambda data: {
        'input_ids': torch.stack([item['input_ids'] for item in data]),
        'attention_mask': torch.stack([item['attention_mask'] for item in data]),
        'labels': torch.stack([item['labels'] for item in data])
    }

    sample_batch = [dataset[i] for i in range(3)]  # Get a sample batch of size 3 from the dataset
    data_sample = data_collator(sample_batch)
    print(data_sample["input_ids"].shape)
    print(data_sample["attention_mask"].shape)
    print(data_sample["labels"].shape)


def check_torch_distributed_ops():
    """Print all dist ops supported or not
    :return:
    """
    table = []
    operations = [
        "is_available",
        "get_rank",
        "get_world_size",
        "is_initialized",
        "init_process_group",
        "destroy_process_group",
        "backend",
        "new_group",
        "all_reduce",
        "reduce",
        "broadcast",
        "all_gather",
        "gather",
        "scatter",
        "reduce_scatter",
        "barrier"
    ]

    for op in operations:
        is_available = hasattr(torch.distributed, op)
        table.append([op, "Available" if is_available else "Not available"])

    headers = ["Operation", "Availability"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def debug_nccl(debug_level="WARN"):
    """Set the debug level for NCCL.
    :param debug_level: Debug level for NCCL. Default is "WARN".
    """
    os.environ["NCCL_DEBUG"] = debug_level


def set_ifname(if_name: str = "enp7s0"):
    """Set the interface name for NCCL socket.
    :param if_name: Interface name for NCCL socket. Default is "ens1f0".
    """
    os.environ["NCCL_SOCKET_IFNAME"] = if_name


def torch_distributed_operations_test(rank, world_size):
    """I need properly test it.
    Test torch.distributed operations (all_reduce, all_gather) in a distributed environment.
    :param rank: Rank of the current process.
    :param world_size: Total number of processes.
    """
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    print(f"Rank: {rank}, World Size: {world_size}")

    # Test all_reduce operation
    input_tensor = torch.tensor(rank).cuda()
    output_tensor = torch.tensor([0])
    input_tensor = input_tensor.to_dense()

    dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
    print(f"Rank: {rank}, All Reduce Result: {output_tensor.item()}")

    # # Test gather operation
    # input_tensor = torch.tensor(rank).cuda()
    # output_tensors = [torch.tensor([0]).cuda() for _ in range(world_size)]
    # input_tensor = input_tensor.to_dense()
    #
    # dist.gather(input_tensor, gather_list=output_tensors, dst=0)
    # if rank == 0:
    #     gather_result = [tensor.item() for tensor in output_tensors]
    #     print(f"Rank: {rank}, Gather Result: {gather_result}")

    # Cleanup
    dist.destroy_process_group()


def batchify(data: Tensor, batch_size: int) -> Tensor:
    """Divides the data into separate sequences, removing extra elements
    :param data:  shape [N]
    :param batch_size: int, batch size
    :return:
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(batch_size)
