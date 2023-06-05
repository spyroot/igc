import argparse
from enum import Enum
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


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"


class GoalTypeState(Enum):
    # State represent a state that agent need reach
    State = 0
    # Action is goal that agent need execute with
    # particular method and/or list of parameters
    # i.e. HTTP post with particular payload
    Action = 1
    # fixed state client provide state
    FixedState = 2


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
                 directory_path=None,
                 goal=None):
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
        self.goal_action = goal
        self.goal_type = None
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
            low=0, high=1,
            shape=(len(list(self._rest_api_methods)) + len(RestApiEnv.METHOD_MAPPING),), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.encoder.emb_shape,
            dtype=np.float32
        )

        self.reward_range = (-1, 1)
        self.base_url = default_rest_base

    def mock_server(self) -> MockServer:
        return self._mock_rest

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

    def is_goal_reached(self, action: ActType, response=None) -> bool:
        """
        Check if the given action is the goal action.

        :param response:
        :param action: The action to check.
        :return: True if the action is the goal action, False otherwise.
        """
        if self.goal_type == GoalTypeState.State.value:
            if response is not None:
                observation = self.encoder.encode(response.json())
                if torch.allclose(observation, self.goal_action) and 200 <= response.status_code <= 300:
                    return True
        # simple goal execute particular http api with particular method.
        elif self.goal_type == GoalTypeState.Action.value:
            if torch.allclose(action, self.goal_action):
                return True
        # fixed state goal provided as a tensor
        elif self.goal_type == GoalTypeState.FixedState.value:
            if response is not None:
                observation = self.encoder.encode(response.json())
                if torch.allclose(observation, self.goal_action) and 200 <= response.status_code <= 300:
                    return True

        return False

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
            self.extract_action_method(action)
            rest_api_one_hot, method_one_hot = RestApiEnv.extract_action_method(action)
            method = RestApiEnv.one_hot_to_method_string(method_one_hot)
            # Remove the additional dimension from the one-hot vector
            if method not in RestApiEnv.METHOD_MAPPING:
                print("Method not in mapping")
                reward = -0.1
                observation = self._mock_rest.generate_error_response()
                self.last_observation = self.encoder.encode(observation)
            else:
                rest_api = self._discovered_rest_api.one_hot_vector_to_action(rest_api_one_hot)
                print(f"rest api {rest_api} method {method}")
                response = self._mock_rest.request(rest_api, method)
                # agent execute goal action
                if self.is_goal_reached(action, response):
                    reward = 1.0
                    done = True
                    self.last_observation = self.encoder.encode(response.json())
                elif 200 <= response.status_code <= 300:
                    print(f"status code {response.status_code}")
                    self.last_observation = self.encoder.encode(response.json())
                    reward = 0.1
                    done = False
                    # Update the current observation with the successful observation
                elif response.status_code == 500:
                    print(f"status code {response.status_code}")
                    reward = -0.5
                    done = True
                    terminated = False
                    observation = self._mock_rest.generate_error_response()
                    self.last_observation = self.encoder.encode(observation)
                else:
                    print(f"status code {response.status_code}")
                    observation = self._mock_rest.generate_error_response()
                    self.last_observation = self.encoder.encode(observation)
                    reward = -0.2

        # # Check if the current observation is the same as the previous one
        # if np.array_equal(encoded_observation, self.current_observation):
        #     # Return the previous successful observation and assign a small negative reward
        #     encoded_observation = self.current_observation
        #     reward = -0.1

        reward = np.clip(reward, -1.0, 1.0)
        return self.last_observation, reward, done, terminated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        goal: Optional[ObsType] = None,
        goal_type: GoalTypeState = None
    ) -> Tuple[ObsType, dict]:
        """Set initial observation to entry point to rest API.
        :param goal_type:
        :param goal:
        :param seed:
        :param options:
        :return:
        """
        if seed is not None:
            super().reset(seed=seed, options=options)

        self.step_count = 0
        self.goal_action = goal
        self.goal_type = goal_type

        # add option to execute list of action and get final goal
        if goal_type == GoalTypeState.State:
            if goal is not None and isinstance(goal, dict):
                # execute rest api and observe goal
                response = self.mock_server().request(
                    goal["rest_api"], goal["method"], goal["parameters"])
                if 200 <= response.status_code <= 300:
                    self.goal_action = self.encoder.encode(response.json())
                else:
                    # Handle the case where the goal REST API returns an error code
                    raise ValueError("Goal REST API returned an error code.")
        # simple goal execute particular http api with particular method.
        elif goal_type == GoalTypeState.Action:
            self.goal_action = goal
        # fixed state goal provided as a tensor
        elif goal_type == GoalTypeState.FixedState:
            if goal.shape != self.observation_space.shape:
                raise ValueError("Fixed state goal dimensions do not match observation space dimensions.")
            self.goal_action = goal

        # initial observation
        rest_api, one_hot_vector = self._discovered_rest_api.entry_rest_api()
        response = self.mock_server().request(rest_api, HttpMethod.GET.value)
        self.last_observation = self.encoder.encode(response.json())
        print(f"Last self.last_observation.shape  {self.last_observation.shape}")
        print(f"Last self.goal_action shape {self.goal_action.shape}")
        return self.last_observation, {"goal": self.goal_action}

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
