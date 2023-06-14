from transformers import GPT2LMHeadModel

from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.base.igc_metric_logger import MetricLogger
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

    metric_logger = MetricLogger(cmd.metric_report, **vars(cmd))
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(dataset.tokenizer))

    latent_model = LlmEmbeddingsTrainer(
        "test_mod", cmd, model, dataset.tokenizer, ds=dataset,
        metric_logger=metric_logger, is_inference=False)
    latent_model.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
