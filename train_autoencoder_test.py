from igc.ds.redfish_dataset import JSONDataset
from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_main import shared_main
from igc.shared.shared_torch_builder import TorchBuilder


def main(cmd):
    """
    :return:
    """
    gpus = TorchBuilder.get_available_gpus()
    model, tokenizers = from_pretrained_default(cmd)
    dataset = JSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizers,
        do_consistency_check=True)

    igc_autoencoder = AutoencoderTrainer(
        "autoencoder", cmd, model, tokenizers, ds=dataset, metric_logger=None,
        is_inference=False)

    # print(gpus)
    # igc_autoencoder.show_accelerator_info()
    igc_autoencoder.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
