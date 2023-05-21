# this will download all models
from shared_torch_utils import get_device, is_bf16_supported, is_amp_supported, torch_runtime_details

if __name__ == '__main__':
    dev = get_device()

    torch_runtime_details()

    print(is_bf16_supported())
    print(is_amp_supported())
