import os
from igc.modules.llm_module import IgcLllModule
from igc.modules.trainer import IgcAgentTrainer
from igc.shared.shared_main import shared_main


def main(cmd):
    """

    :return:
    """
    # igc = IgcLllModule(cmd)
    # igc.train()

    args = shared_main()
    igc_agent = IgcAgentTrainer(args)


if __name__ == '__main__':
    args = shared_main()
    main(args)
