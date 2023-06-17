import logging
from igc.ds.redfish_masked_dataset import MaskedJSONDataset


def custom_path():
    """
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    dataset = MaskedJSONDataset(
        "test_datasets",
        do_consistency_check=False
    )


def main():
    """

    :return:
    """
    custom_path()


if __name__ == '__main__':
    main()
