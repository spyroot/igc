import os
from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_gym_env import RestApiEnv
from igc.envs.rest_mock_server import MockServer
from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main
import time



def main(cmd):
    """
    :return:
    """
    mock_test_all_rest_api(cmd)


if __name__ == '__main__':
    args = shared_main()
    main(args)
