"""

This separate trainer so we can test different blocks

Author: Mus mbayramo@stanford.edu

"""
from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        do_consistency_check=False
    )

    metric_logger = MetricLogger(cmd.metric_report, **vars(cmd))
    model, _ = from_pretrained_default("gpt2", only_model=True)
    model.resize_token_embeddings(len(dataset.tokenizer))

    latent_model = LlmEmbeddingsTrainer(
        "test_mod", cmd, model, dataset.tokenizer, dataset=dataset,
        metric_logger=metric_logger, is_inference=False)
    latent_model.train()


if __name__ == '__main__':
    args, _ = shared_main()
    main(args)
