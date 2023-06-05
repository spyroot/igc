import argparse
from typing import Optional, Tuple, List

import gym
import numpy as np
import requests
import torch
from gym.core import ObsType, ActType
from gym.vector.utils import spaces
from transformers import PreTrainedModel, PreTrainedTokenizer

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_encoder import RestBaseEncoder
from igc.envs.rest_mock_server import MockServer


class RestApiEnv(gym.Env):
    """
    """
    METHOD_MAPPING = [
        "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"
    ]

    def __init__(self,
                 args: argparse.Namespace,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 discovered_rest_api: JSONDataset,
                 default_rest_base: Optional[str] = "http://localhost",
                 max_episode: int = 200,
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

        self.last_observation = None
        self.max_steps = max_episode
        self.step_count = 0

        if default_rest_base is None:
            raise ValueError("Invalid default_rest_base. Please provide a valid URL.")

        # mock rest API where agent will send request.
        self._discovered_rest_api = discovered_rest_api
        self._rest_api_methods = self._discovered_rest_api.get_rest_api_methods()

        self._mock_rest = MockServer(args, discovered_rest_api)

        # encoder that take rest api respond and transform to embeddings.
        self.encoder = RestBaseEncoder(model=model, tokenizer=tokenizer)

        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(list(self._rest_api_methods)),), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.encoder.emb_shape,
            dtype=np.float32
        )

        self.reward_range = (-1, 1)
        self.base_url = default_rest_base

    def max_reward(self) -> float:
        """Max reward value.
        :return: The maximum reward value.
        """
        return 1.0

    def sample_action(self):
        """ Sampling from action space.
        :returns: Random action from action space
        """
        return self.action_space.sample()

    def sample_observation(self):
        """
        Sampling from action space.
        :return: Tuple containing random observation and method
        """
        rest_api, supported_method, supported_method = self._discovered_rest_api.sample_rest_api()
        response = self._mock_rest.request(rest_api, supported_method)
        if 200 <= response.status_code <= 300:
            observation = response.json()
            encoded_observation = self.encoder.encode(observation)
        else:
            # todo this
            encoded_observation = self.encoder.initialize()
        return encoded_observation

    def obs_shape(self):
        """Return shape of observation
        :return:
        """
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_shape = self.observation_space.shape
        else:
            raise ValueError("Unsupported observation space")
        return obs_shape

    def action_shape(self):
        """Return shape of action.
        :return:
        """
        if isinstance(self.action_space, gym.spaces.Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (1,)
        else:
            raise ValueError("Unsupported action space")
        return action_shape

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        :param action:
        :return:
        """
        self.step_count += 1
        info = {}

        done = False
        terminated = False

        if self.step_count >= self.max_steps:
            done = True
            terminated = True
            reward = -1.0

        if not done:

            # Remove the additional dimension from the one-hot vector
            rest_api_one_hot = action[:-len(RestApiEnv.METHOD_MAPPING)]
            method_one_hot = action[-len(RestApiEnv.METHOD_MAPPING):]
            action_index = torch.argmax(rest_api_one_hot)
            method_index = torch.argmax(method_one_hot)

            print(f"Action index {action_index}")
            print(f"method_index {method_index} ")

            method = RestApiEnv.METHOD_MAPPING[method_index.item()]
            print(f"method {method} ")

            rest_api_one_hot, method_index = action[:-1], int(action[-1].item())
            # split the action into the one-hot vector and method
            # rest_api_one_hot, method_index = action[:-1], action[-1]
            action_index = np.argmax(rest_api_one_hot)
            method = RestApiEnv.METHOD_MAPPING[method_index]
            if method not in RestApiEnv.METHOD_MAPPING:
                reward = -0.1
                observation = self._mock_rest.generate_error_response()
                encoded_observation = self.encoder.encode(observation)
            else:
                rest_api = self._discovered_rest_api.one_hot_vector_to_action(rest_api_one_hot)
                response = self._mock_rest.request(rest_api, method)
                if 200 <= response.status_code <= 300:
                    observation = response.json()
                    encoded_observation = self.encoder.encode(observation)
                    reward = 0.1
                    done = False
                    # Update the current observation with the successful observation
                    self.last_observation = encoded_observation
                elif response.status_code == 500:
                    reward = -0.5
                    done = True
                    terminated = False
                    observation = self._mock_rest.generate_error_response()
                    encoded_observation = self.encoder.encode(observation)
                else:
                    observation = self._mock_rest.generate_error_response()
                    encoded_observation = self.encoder.encode(observation)
                    reward = -0.2

        # # Check if the current observation is the same as the previous one
        # if np.array_equal(encoded_observation, self.current_observation):
        #     # Return the previous successful observation and assign a small negative reward
        #     encoded_observation = self.current_observation
        #     reward = -0.1

        reward = np.clip(reward, -1.0, 1.0)
        return encoded_observation, reward, done, terminated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        if seed is not None:
            super().reset(seed=seed, options=options)

        self.step_count = 0

        observation, info = super().reset(options=options)
        return observation, info

    def goal(self) -> float:
        """
        :return:
        """
        response = requests.post(f"{self.base_url}/goal_endpoint", json={"param": "value"})
        if response.status_code == 200:
            return 100.0  # Large reward for successfully reaching the goal
        else:
            return -1.0  # Negative reward for not reaching the goal

    @staticmethod
    def concat_rest_api_method(rest_api_vector: torch.Tensor, method_vector: torch.Tensor) -> torch.Tensor:
        """
        Concatenate the one-hot encoded vectors representing the REST API and method.
        :param rest_api_vector: The one-hot encoded vector representing the REST API.
        :param method_vector: The one-hot encoded vector representing the method.
        :return: The concatenated vector.
        """

        if rest_api_vector.dim() != method_vector.dim():
            raise ValueError("Input tensors must have the same number of dimensions.")

        concatenated_vector = torch.cat((rest_api_vector, method_vector), dim=rest_api_vector.dim() - 1)
        return concatenated_vector

    @staticmethod
    def encode_rest_api_method(method: str) -> torch.Tensor:
        """
        Encode the REST API method as a one-hot encoded tensor.

        :param method: The REST API method as a string.
        :return: A one-hot encoded tensor representing the method.
        """
        if method not in RestApiEnv.METHOD_MAPPING:
            raise ValueError("Invalid REST API method.")

        method_index = RestApiEnv.METHOD_MAPPING.index(method)
        one_hot_tensor = torch.zeros(len(RestApiEnv.METHOD_MAPPING))
        one_hot_tensor[method_index] = 1
        return one_hot_tensor

    @staticmethod
    def one_hot_to_method_string(method_vector: torch.Tensor) -> str:
        """
        Convert a one-hot encoded tensor representing a
        REST API method to its string representation.

        :param method_vector: The one-hot encoded tensor representing the method.
        :return: The string representation of the REST API method.
        """
        if method_vector.dim() != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        method_index = torch.argmax(method_vector).item()
        return RestApiEnv.METHOD_MAPPING[method_index]

    @staticmethod
    def batch_one_hot_to_method_string(method_vector: torch.Tensor) -> List[str]:
        """
        Convert a batch of one-hot encoded tensors representing
        REST API methods (GET,POST) to their string representations.

        :param method_vector: The batched one-hot encoded tensor representing the methods.
                              Shape: (batch_size, method_dim)
        :return: The list of string representations of the REST API methods for each element in the batch.
        """
        if method_vector.dim() != 2:
            raise ValueError("Input tensor must have 2 dimensions (batch_size, method_dim).")
        method_indices = torch.argmax(method_vector, dim=1)
        method_strings = [RestApiEnv.METHOD_MAPPING[index.item()] for index in method_indices]
        return method_strings


    @staticmethod
    def extract_single_action_method(input_tensor: torch.Tensor):
        """
        Extract the rest_api_one_hot and method_one_hot from the input_tensor without batch dimension.

        :param input_tensor: The input tensor with concatenated rest_api_one_hot and method_one_hot.
        :return: The extracted rest_api_one_hot and method_one_hot tensors.
        """
        method_sz = len(RestApiEnv.METHOD_MAPPING)
        rest_api_one_hot = input_tensor[:-method_sz]
        method_one_hot = input_tensor[-method_sz:]
        return rest_api_one_hot, method_one_hot

    @staticmethod
    def extract_batched_action_method(input_tensor: torch.Tensor):
        """
        Extract the rest_api_one_hot and method_one_hot from the input_tensor.

        :param input_tensor: The input tensor with concatenated rest_api_one_hot and method_one_hot.
                             Shape: (batch_size, rest_api_dim + method_dim)
        :return: The extracted rest_api_one_hot and method_one_hot tensors.
                 Shapes: (batch_size, rest_api_dim), (batch_size, method_dim)
        """
        method_sz = len(RestApiEnv.METHOD_MAPPING)
        rest_api_one_hot = input_tensor[:, :-method_sz]
        method_one_hot = input_tensor[:, -method_sz:]
        return rest_api_one_hot, method_one_hot

    @staticmethod
    def extract_action_method(input_tensor: torch.Tensor):
        """
        Extract the rest_api_one_hot and method_one_hot from the input_tensor.

        :param input_tensor: The input tensor with concatenated rest_api_one_hot and method_one_hot.
        :return: The extracted rest_api_one_hot and method_one_hot tensors.
        """
        if input_tensor.dim() == 2:
            return RestApiEnv.extract_batched_action_method(input_tensor)
        elif input_tensor.dim() == 1:
            return RestApiEnv.extract_single_action_method(input_tensor)
        else:
            raise ValueError("Invalid input tensor dimensions. Expected 1 or 2.")

