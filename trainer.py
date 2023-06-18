from igc.modules.igc_main import IgcMain
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """

    igc = IgcMain(cmd)
    igc.run()


if __name__ == '__main__':
    args = shared_main()
    main(args)

