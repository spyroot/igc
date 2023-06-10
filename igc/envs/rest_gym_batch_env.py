"""
This batch version,  the idea is to pass a batch of action
and get batch of observation. That should provide speed up
for encoder since we need pass each time observation to encoder.
in order obtain the embedding.

Mus mbayramo@stanfor.edu

"""
import argparse
from typing import Optional, Tuple, List, Union

import gym
import numpy as np
import torch

from gym.core import ObsType, ActType
from gym.vector import VectorEnv
from gym.vector.utils import spaces
from transformers import PreTrainedModel, PreTrainedTokenizer

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_encoder import RestBaseEncoder
from igc.envs.rest_mock_server import MockServer
from .rest_gym_base import (
    RestApiBaseEnv, GoalTypeState, HttpMethod
)


class VectorizedRestApiEnv(VectorEnv, RestApiBaseEnv):
    """
    """
    def __init__(self,
                 args: argparse.Namespace,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 discovered_rest_api: JSONDataset,
                 default_rest_base: Optional[str] = "http://localhost",
                 max_episode: int = 200,
                 render_mode: Optional[str] = None,
                 directory_path=None,
                 goal=None,
                 num_envs=2):
        """
        Initialize the RestApiEnv environment.

        :param args: The command-line arguments.
        :param model: The pre-trained model for encoding responses.
        :param tokenizer: The pre-trained tokenizer for encoding responses.
        :param discovered_rest_api: The discovered REST API dataset.
        :param default_rest_base: The default base URL for REST API requests.
        :param max_episode: The maximum number of episodes.
        :param render_mode: The rendering mode, defaults to None.
        :param directory_path: The directory path for the JSON dataset, defaults to None.
        :param goal: The goal, defaults to None.

        """
        self.goal = goal
        self.goal_type = None
        self._num_envs = num_envs
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
            low=0, high=1,
            shape=(len(list(self._rest_api_methods)) + len(RestApiBaseEnv.METHOD_MAPPING),), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.encoder.emb_shape,
            dtype=np.float32
        )

        self.reward_range = (-1, 1)
        self.base_url = default_rest_base

        self.responses = []
        self.dones = [False] * num_envs
        self.rewards = [0.0] * num_envs
        self.terminateds = [False] * num_envs

        # unit testing.
        self._simulate_goal_reward = False
        self._simulate_goal_reward_idx = None

        VectorEnv.__init__(self, num_envs=num_envs,
                           observation_space=self.observation_space,
                           action_space=self.action_space)

    def _reset_state(self):
        """Reset the state of the environment.
        :return:
        """
        self.responses = []
        self.dones = [False] * self._num_envs
        self.rewards = [0.0] * self._num_envs
        self.terminateds = [False] * self._num_envs

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

    def reset(
            self,
            *,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
            goal: Optional[ObsType] = None,
            goal_type: GoalTypeState = None
    ):
        """
        Reset the environment.

        :return: The initial observation.
        """
        self.goal = goal
        self.goal_type = goal_type
        self.step_count = 0

        self._reset_state()
        observations = []

        for _ in range(self.num_envs):
            rest_api, _ = self._discovered_rest_api.entry_rest_api()
            response = self.mock_server().request(rest_api, HttpMethod.GET.value)
            observation = self.encoder.encode(response.json())
            observations.append(observation)

        observations_tensor = torch.stack(observations, dim=0)
        self.last_observation = observations_tensor
        # observations = super().reset(seed=seed, options=options)
        return self.last_observation, {}

    def step_async(self, actions: List[ActType]) -> None:
        """Asynchronously execute a batch of actions.

        :param actions: The list of actions.
        """
        del self.responses
        self.responses = []

        self.step_count += 1
        rest_api_one_hot_batch, method_one_hot_batch = RestApiBaseEnv.extract_action_method(actions)
        methods = RestApiBaseEnv.batch_one_hot_to_method_string(method_one_hot_batch)

        for i, (rest_api_one_hot, method) in enumerate(zip(rest_api_one_hot_batch, methods)):
            if self.terminateds[i] or self.dones[i]:
                print("#### Skipping terminated environment")
                # Skip processing for already terminated environment
                continue

            rest_api = self._discovered_rest_api.one_hot_vector_to_action(rest_api_one_hot)
            response = self._mock_rest.request(rest_api, method)
            if 200 <= response.status_code <= 300:
                self.rewards[i] = 0.1
            elif 300 <= response.status_code <= 499:
                self.rewards[i] = -0.2
            else:
                self.rewards[i] = -0.5
                self.dones[i] = True
                self.terminateds[i] = False

            # add response to list
            self.responses.append(response)

            # Mark the environment as terminated if the termination condition is met
            if self.step_count >= self.max_steps:
                self.dones[i] = True
                self.terminateds[i] = True
                self.rewards[i] = -1.0

    def step_wait(
            self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """
        Wait for the asynchronous actions to complete and return the results.

        :return: Tuple containing the observations,
                 rewards, done flags, terminated flags, and info dictionaries.
        """
        observations = []
        terminateds = []
        dones = []

        info = [{}] * self.num_envs

        _responses = 0
        for response in self.responses:
            if 200 <= response.status_code <= 300:
                observations.append(self.encoder.encode(response.json()))
                dones.append(False)
                terminateds.append(False)
                _responses += 1
            else:
                observations.append(self.encoder.encode(
                    self._mock_rest.generate_error_response())
                )
                dones.append(True)
                terminateds.append(False)
                _responses += 1

        # print(f"Total responses {_responses} observations len {len(observations)}")
        done = torch.tensor(self.dones, dtype=torch.bool)
        terminated = torch.tensor(self.terminateds, dtype=torch.bool)
        _observations = torch.stack(observations, dim=0)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        # pass list of observations to check if any of batch reached goal
        # if self.check_goal(observations):
        #     goal_reward = torch.tensor([1.0] * self.num_envs, dtype=torch.float32)
        #     rewards = torch.cat((rewards, goal_reward), dim=0)
        #     done.fill_(True)
        #     terminated.fill_(False)

        goal_reached = self.check_goal(_observations)
        print("Goal reached: ", goal_reached)
        rewards[goal_reached] = 1.0
        done[goal_reached] = True
        terminated[goal_reached] = False

        if self.step_count >= self.max_steps:
            done.fill_(True)
            terminated.fill_(True)

        self.last_observation = _observations
        return _observations, rewards, done, terminated, info

    def simulate_goal_reached(self, batch_id: int):
        """Simulate that particular trajectory in batch reached a goal state.

        :param batch_id: a batch id that reached goal state
        :return:
        """
        self._simulate_goal_reward = True
        self._simulate_goal_reward_idx = batch_id

    @staticmethod
    def is_goal_reached_exact(current_state, goal_state):
        """
        Check if the current state matches the goal state.
        :param current_state: The current state.
        :param goal_state: The goal state.
        :return: True if the goal is reached, False otherwise.
        """
        return torch.eq(current_state, goal_state).all()

    @staticmethod
    def is_goal_reached(current_state, goal_state, tolerance=1e-6):
        """
        Check if the current state matches,
        the goal state within a certain tolerance.

        :param current_state: The current state.
        :param goal_state: The goal state.
        :param tolerance: Tolerance for comparison.
        :return: True if the goal is reached, False otherwise.
        """
        diff = torch.norm(current_state - goal_state)
        return torch.all(diff <= tolerance)

    def check_goal(self, observations: torch.Tensor):
        """ For now, it false.
        :param observations:
        :return:
        """
        # case when we want to simulate some goal reached
        goal_reached = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._simulate_goal_reward:
            goal_reached = torch.zeros(self.num_envs, dtype=torch.bool)
            goal_reached[self._simulate_goal_reward_idx] = True

        if self.goal is not None:
            print(type(self.goal))
            print(type(observations))

            goal_reached = torch.allclose(
                observations, self.goal, rtol=1e-3, atol=1e-3, equal_nan=True)

            # goal_reached_check2 = self.is_goal_reached(observations, self.goal)
            # goal_reached3 = torch.all(observations == self.goal)
            # print(f"goal compare methods m1: {goal_reached}, m2: {goal_reached_check2}, m3: {goal_reached3}")

        return goal_reached

    def sample_different_goals(self):
        """This method samples different goals, each batch dim has different goal.
        :return:
        """
        # sample goal
        rest_apis, supported_methods, one_hot_vectors = self._discovered_rest_api.sample_batch_of_rest_api(self._num_envs)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", self._num_envs)
        goal_action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot
        )

        emb = None
        respond = None

        observations = []
        for api_req in rest_apis:
            if respond is None:
                respond = self.mock_server().request(api_req, "GET")
                emb = self.encoder.encode(respond.json())
            observations.append(emb)

        goal_observation = torch.stack(observations, dim=0)
        return goal_observation, goal_action_vector, rest_apis, supported_methods

    def _do_sanity_check(self, rest_apis, goal_action_vector, goal_state):

        # we do reverse sanity, that emb goal we can compare
        # and recover the goal state.
        reached_goal_states = []
        rest_api_one_hot_batch, method_one_hot_batch = RestApiBaseEnv.extract_action_method(goal_action_vector)
        methods = RestApiBaseEnv.batch_one_hot_to_method_string(method_one_hot_batch)
        for i, (rest_api_one_hot, method) in enumerate(zip(rest_api_one_hot_batch, methods)):
            # check one compare rest api
            rest_api = self._discovered_rest_api.one_hot_vector_to_action(rest_api_one_hot)
            assert rest_apis[i] == rest_api, "Sampled rest API does not match the decoded rest API"

            # check compare state with goal_observation
            response = self._mock_rest.request(rest_api, method)
            state = self.encoder.encode(response.json())
            assert self.is_goal_reached(goal_state[i], state), "Goal not reached for env: {}".format(i)
            reached_goal_states.append(state)

        # stack and pass so we compare with self.goal.
        reached_goal_states = torch.stack(reached_goal_states, dim=0)
        check_goal = self.check_goal(reached_goal_states)
        print("CHECK GOAL RETURN", check_goal)

    def sample_same_goal(self, do_sanity_check=False):
        """Method sample the same goal for all the environments, batch size
        :return:
        """

        # sample goal
        rest_apis, supported_methods, one_hot_vectors = self._discovered_rest_api.sample_batch(self._num_envs)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", self._num_envs)
        goal_action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot
        )

        emb = None
        respond = None

        observations = []
        for api_req in rest_apis:
            if respond is None:
                respond = self.mock_server().request(api_req, "GET")
                emb = self.encoder.encode(respond.json())
            observations.append(emb)

        goal_observation = torch.stack(observations, dim=0)
        self.goal = goal_observation

        if do_sanity_check:
            self._do_sanity_check(rest_apis, goal_action_vector, goal_observation)

        return goal_observation, goal_action_vector, rest_apis, supported_methods

    def add_goal_state(
            self, goal_observation: torch.Tensor,
            goal_type: GoalTypeState = GoalTypeState.State
    ):
        """
        :param goal_observation:
        :param goal_type:
        :return:
        """
        if goal_observation.size(0) != self.num_envs:
            raise ValueError("Invalid batch dimension for goal state.")

        self.goal = goal_observation
        self.goal_type = goal_type







