from transformers import GPT2LMHeadModel

from igc.ds.redfish_dataset import JSONDataset
from igc.shared.shared_main import shared_main
from igc.shared.shared_torch_builder import TorchBuilder
from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.modules.base.igc_metric_logger import MetricLogger


def main(cmd):
    """
    :return:
    """
    gpus = TorchBuilder.get_available_gpus()
    model, tokenizers = from_pretrained_default(cmd)
    _metric_logger = MetricLogger(cmd.metric_report, **vars(cmd))

    dataset = JSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizers,
        do_consistency_check=False)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(dataset.tokenizer))

    igc_autoencoder = AutoencoderTrainer(
        "autoencoder", cmd, model, tokenizers, ds=dataset, metric_logger=_metric_logger,
        is_inference=False)

    igc_autoencoder.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
