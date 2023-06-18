"""

This separate trainer so we can test different blocks

Author: Mus mbayramo@stanford.edu

"""
import logging

import torch
from transformers import GPT2LMHeadModel
from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.modules.igc_rl_module import IgcRlModule
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        do_consistency_check=False
    )

    cmd.device = "cuda:1"
    cpu_device = torch.device("cpu")

    metric_logger = MetricLogger(cmd.metric_report, **vars(cmd))
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map=cpu_device)
    model = model.to(cpu_device)
    model.resize_token_embeddings(len(dataset.tokenizer))

    latent_module = LlmEmbeddingsTrainer(
        "test_mod", cmd, model, dataset.tokenizer, dataset=dataset,
        metric_logger=metric_logger, is_inference=False, device=cpu_device)
    latent_module.load_checkpoint("experiments/gpt2_4_AdamW2_StepLR_lr_1e-05/state_encoder",
                                  resuming=False, map_location=cmd.device)

    # module_name: str,
    # spec: argparse.Namespace,
    # metric_logger: MetricLogger,
    # ds: JSONDataset,
    # llm_model = None,
    #

    logging.basicConfig(level=logging.INFO)
    rl_module = IgcRlModule(
        module_name="rl_module",
        spec=cmd,
        metric_logger=metric_logger,
        ds=dataset,
        llm_model=latent_module.model,
        device=cmd.device
    )

    rl_module.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
