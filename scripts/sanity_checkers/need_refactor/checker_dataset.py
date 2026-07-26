"""Parked legacy dataset checker.

This helper is retained for refactor reference only. It is not a unit test, not
a current acceptance gate, and must not run on the operator laptop.

Author: Mus mbayramo@stanford.edu
"""
import logging

from igc.ds.redfish_dataset import JSONDataset
from igc.ds.redfish_masked_dataset import MaskedJSONDataset


def main():
    """

    :return:
    """
    toks = JSONDataset.build_special_tok_table()
    print(toks)
    load_tokenizer = JSONDataset.load_tokenizer()
    print(load_tokenizer)
    dataset_dir = JSONDataset.dataset_default_root()
    print(dataset_dir)

    # logging check
    logging.basicConfig(level=logging.INFO)
    dataset = JSONDataset(
        "datasets",
        do_consistency_check=False
    )

    # logging check
    logging.basicConfig(level=logging.INFO)
    dataset = MaskedJSONDataset(
        "datasets",
        do_consistency_check=False
    )


if __name__ == '__main__':
    main()
