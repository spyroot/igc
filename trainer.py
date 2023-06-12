import os

from igc.modules.igc_main import IgcMain
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    igc = IgcMain(cmd)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    igc.run()


if __name__ == '__main__':
    args = shared_main()
    main(args)

