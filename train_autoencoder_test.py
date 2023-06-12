import os

import torch
from accelerate import Accelerator
from igc.ds.redfish_dataset import JSONDataset
from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_accelerator import build_accelerator
from igc.shared.shared_main import shared_main
from igc.shared.shared_torch_builder import TorchBuilder


def main(cmd):
    """
    :return:
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    accelerator = build_accelerator(cmd)
    rank = int(os.environ.get('LOCAL_RANK', -1))
    gpus = TorchBuilder.get_available_gpus()
    print(gpus)

    model, tokenizers = from_pretrained_default(cmd)
    dataset = JSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizers,
        do_consistency_check=True)

    igc_autoencoder = AutoencoderTrainer(
        "autoencoder", cmd, model, tokenizers, ds=dataset, metric_logger=None,
        is_inference=False)

    igc_autoencoder.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
