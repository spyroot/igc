"""
This is a demo script to show how to download a dataset and
module from the IGC

Note the modules download all you to fine tune one machine
and use it in downstream tasks.

Author: Mus mbayramo@stanford.edu
"""
import logging
from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.base.igc_base_module import IgcModule
from igc.shared.modules_typing import IgcModuleType
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    logging.basicConfig(level=logging.INFO)
    dataset = MaskedJSONDataset(
        "datasets",
        do_consistency_check=False
    )

    fine_tuned, epochs, path = IgcModule.checkpoint_to_module(
        cmd, IgcModuleType.STATE_ENCODER.value,
        pre_trained_tokenizer=dataset.tokenizer
    )


if __name__ == '__main__':
    args, groups = shared_main()
    main(args)
