from igc.modules.llm_module import IgcLanguageModule
from igc.modules.trainer import IgcAgentTrainer
from igc.shared.shared_main import shared_main


def main(cmd):
    """
    :return:
    """
    if cmd.train and cmd.llm is not None:
        igc = IgcLanguageModule(cmd)
        igc.train()

    igc_rl = IgcAgentTrainer(cmd)
    igc_rl.train()


if __name__ == '__main__':
    args = shared_main()
    main(args)
