import logging
from igc.ds.redfish_dataset import JSONDataset
from igc.ds.redfish_masked_dataset import MaskedJSONDataset


def custom_path():
    """
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    dataset = JSONDataset(
        "test_datasets",
        do_consistency_check=False,
        skip_download=True
    )


def download_custom_path():
    """
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    dataset = MaskedJSONDataset(
        "test_datasets",
        do_consistency_check=False
    )
    if dataset.is_downloaded():
        print("Dataset is already downloaded")

    if dataset.is_tokenizer_loader():
        print("Tokenizer loaded")

    print(len(dataset.filtered_api_keys))
    sample = dataset.sample_batch(1)
    print(sample)


def main():
    """
    :return:
    """
    download_custom_path()


if __name__ == '__main__':
    main()
