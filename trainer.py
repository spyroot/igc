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
    igc_agent = IgcAgentTrainer(cmd)
    ds = igc_agent.dataset

    # for entry in ds.rest_api_iterator():
    #     rest_api, path = entry
    #     print(f" path {path} ")

    # for rest_api, methods in ds.get_rest_api_methods():
    #     print(f" {rest_api} path {methods} ")

    # for resp_files in ds.respond_to_api_iterator():
    #     resp_file, api = resp_files
    #     print(f" resp files {resp_file} ")


if __name__ == '__main__':
    args = shared_main()
    main(args)
