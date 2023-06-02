import os

from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main


def main(cmd):
    """

    :return:
    """
    igc = IgcLllModule(cmd)
    igc.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
