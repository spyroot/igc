from distutils.version import LooseVersion
from typing import Optional

import numpy as np
import torch
from pynvml import *
from torch import dist
from torch.backends import cudnn
from torch.cuda import nccl
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_device(rank: Optional[int] = None) -> torch.device:
    """
    Get a torch.device. If rank is specified,
    CUDA devices are distributed in a round-robin fashion.

    :param rank: The rank of the current process, for CUDA device distribution. If None, no distribution is done.
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


def print_gpu_utilization():
    """
    :return:
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


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

    print("NCCL available:", nccl.is_available(torch.tensor([1, 2, 3])))
    print("NCCL version:", nccl.version() if nccl.is_available(torch.tensor([1, 2, 3])) else "Not available")
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

    if torch.cuda.is_available() and hasattr(torch.cuda, "amp") and torch.cuda.amp.is_available():
        print("AMP is supported")
    else:
        print("AMP is not supported")

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
#
# >>> dir(torch.cuda)
# ['Any', 'BFloat16Storage', 'BFloat16Tensor', 'BoolStorage', 'BoolTensor', 'ByteStorage', 'ByteTensor', 'CUDAGraph', 'CUDAPluggableAllocator', 'CharStorage',
#  'CharTensor', 'ComplexDoubleStorage', 'ComplexFloatStorage', 'CudaError',
#  'DeferredCudaCallError', 'Device', 'DoubleStorage', 'DoubleTensor', 'Event', 'ExternalStream',
#  'FloatStorage', 'FloatTensor', 'HalfStorage', 'HalfTensor', 'IntStorage', 'IntTensor', 'List', 'LongStorage', 'LongTensor', 'Optional', 'OutOfMemoryError',
#  'ShortStorage', 'ShortTensor', 'Stream', 'StreamContext', 'Tuple', 'Union', '_CudaBase', '_CudaDeviceProperties',
#  '_DeviceGuard', '_LazySeedTracker', '__all__', '__annotations__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
#  '__name__', '__package__', '__path__', '__spec__', '_check_capability', '_check_cubins',
#  '_cudart', '_device', '_device_count_nvml', '_device_t', '_dummy_type', '_exchange_device', '_get_device_index', '_initialization_lock',
#  '_initialized', '_is_compiled', '_is_in_bad_fork', '_lazy_call', '_lazy_init', '_lazy_new',
#  '_lazy_seed_tracker', '_memory_viz', '_nvml_based_avail', '_parse_visible_devices',
#  '_queued_calls', '_raw_device_count_nvml', '_raw_device_uuid_nvml', '_sleep', '_tls',
#  '_transform_uuid_to_ordinals', '_utils', '_warn_typed_storage_removal', 'amp', 'caching_allocator_alloc',
#  'caching_allocator_delete', 'can_device_access_peer', 'cast', 'change_current_allocator', 'check_error',
#  'classproperty', 'contextlib', 'cudaStatus', 'cudart', 'current_blas_handle', 'current_device', 'current_stream',
#  'default_generators', 'default_stream', 'device', 'device_count', 'device_of', 'empty_cache', 'get_allocator_backend',
#  'get_arch_list', 'get_device_capability', 'get_device_name', 'get_device_properties', 'get_gencode_flags', 'get_rng_state',
#  'get_rng_state_all', 'get_sync_debug_mode', 'graph', 'graph_pool_handle', 'graphs', 'has_half', 'has_magma'
#     , 'init', 'initial_seed', 'ipc_collect', 'is_available', 'is_bf16_supported', 'is_current_stream_capturing',
#  'is_initialized', 'jiterator', 'list_gpu_processes', 'lru_cache', 'make_graphed_callables',
#  'manual_seed', 'manual_seed_all', 'max_memory_allocated', 'max_memory_cached', 'max_memory_reserved',
#  'mem_get_info', 'memory', 'memory_allocated', 'memory_cached', 'memory_reserved', 'memory_snapshot',
#  'memory_stats', 'memory_stats_as_nested_dict', 'memory_summary', 'memory_usage', 'nccl', 'nvtx',
#  'os', 'profiler', 'random', 'reset_accumulated_memory_stats', 'reset_max_memory_allocated',
#  'reset_max_memory_cached', 'reset_peak_memory_stats', 'seed', 'seed_all', 'set_device',
#  'set_per_process_memory_fraction', 'set_rng_state', 'set_rng_state_all', 'set_stream', 'set_sync_debug_mode',
#  'sparse', 'stream', 'streams', 'synchronize', 'threading', 'torch', 'traceback', 'utilization', 'warnings']

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
            and dist.is_nccl_available()
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
