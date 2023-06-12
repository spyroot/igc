import os

import torch
from accelerate import Accelerator

from igc.ds.redfish_dataset import JSONDataset
from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    accelerator = Accelerator(device_placement=True, split_batches=True, mixed_precision="fp16")
    rank = int(os.environ.get('LOCAL_RANK', -1))
    if rank == 0:
        device = torch.device("cuda:0")
    if rank == 1:
        device = torch.device("cuda:1")

    cmd.device = device

    model, tokenizers = from_pretrained_default(cmd)
    dataset = JSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizers,
        do_consistency_check=True)

    igc_autoencoder = AutoencoderTrainer(
        "autoencoder", cmd, model, tokenizers, ds=dataset, metric_logger=None,
        is_inference=False, device=device)
    igc_autoencoder.device = accelerator.device
    igc_autoencoder.accelerator = accelerator
    igc_autoencoder.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
