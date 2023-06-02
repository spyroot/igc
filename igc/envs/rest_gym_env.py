import argparse
from typing import Optional, Tuple

import argparse
from typing import Optional, Tuple

import gym
import numpy as np
import requests
from gym.core import ObsType, ActType
from gym.vector.utils import spaces
from transformers import PreTrainedModel, PreTrainedTokenizer

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_encoder import RestBaseEncoder
from igc.envs.rest_mock_server import MockServer


class RestApiEnv(gym.Env):
    """
    """
    def __init__(self,
                 args: argparse.Namespace,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 default_rest_base: Optional[str] = "http://localhost",
                 render_mode: Optional[str] = None,
                 directory_path=None):
        """
        Initialize the RestApiEnv environment.

        :param args: The command-line arguments.
        :param model: The pre-trained model for encoding responses.
        :param tokenizer: The pre-trained tokenizer for encoding responses.
        :param default_rest_base: The default base URL for REST API requests.
        :param render_mode: The rendering mode, defaults to None.
        :param directory_path: The directory path for the JSON dataset, defaults to None.
        """
        super().__init__()

        if default_rest_base is None:
            raise ValueError("Invalid default_rest_base. Please provide a valid URL.")

        # mock rest API where agent will send request.
        self.mock_rest = MockServer(args)
        # encoder that take rest api respond and transform to embeddings.
        self.encoder = RestBaseEncoder(model=model, tokenizer=tokenizer)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.encoder.emb_shape,
            dtype=np.float32
        )

        self.reward_range = (-1, 1)
        self.base_url = default_rest_base

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        :param action:
        :return:
        """
        action_index = np.argmax(action)
        response = self.mock_server.request(f"{self.base_url}/endpoint/{action_index}", "GET")

        if response.status_code == 200 or response.status_code == 202:
            observation = response.json()
            encoded_observation = self.encoder.encode(observation)
            reward = 1
            done = False
        else:
            # Use zero vector for invalid response
            encoded_observation = np.zeros(self.observation_space.shape)
            reward = -1
            done = True

        info = {}
        return encoded_observation, reward, done, info

    def reset(self):
        """

        :return:
        """
        return self._get_observation()

    def _get_observation(self):
        # Generate and return an example observation
        return (
            np.random.randint(32),
            np.random.randint(11),
            np.random.randint(2)
        )

    def goal(self) -> float:
        response = requests.post(f"{self.base_url}/goal_endpoint", json={"param": "value"})

        if response.status_code == 200:
            return 100.0  # Large reward for successfully reaching the goal
        else:
            return -1.0  # Negative reward for not reaching the goal

    # def goal(self) -> float:
    #     response = self.mock_server.request(f"{self.base_url}/goal_endpoint", "POST", {"param": "value"})
    #
    #     if response.status_code == 200:
    #         return 100.0  # Large reward for successfully reaching the goal
    #     else:
