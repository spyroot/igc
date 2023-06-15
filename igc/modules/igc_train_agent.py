"""

This class is used to train a RL agent.
It consists two trainer, auto encoder used to
reduce state encoder dimensionality , and RL trainer.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from typing import Optional

import loguru
import numpy as np
import torch
from torch import optim, nn
from transformers import PreTrainedTokenizer

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.envs.rest_gym_env import RestApiEnv, GoalTypeState
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.modules.base.igc_rl_base_module import RlBaseModule
from igc.modules.igc_experience_buffer import Buffer
from igc.modules.igc_q_network import Igc_QNetwork


def env_reward_function(state, goal):
    """
    Reward function for the goal_all_4_trajectory_reward_state_goal_single_trajectory method.
    Rewards +1 when at least one state matches the goal state, and -1 otherwise.
    """
    # print(f"state {state.shape} {goal.shape}")
    is_goal_reached = torch.any(torch.all(state == goal, dim=1), dim=0)
    goal_reached = torch.where(is_goal_reached, torch.tensor(1.0), torch.tensor(0.0))
    goal_reached = goal_reached.unsqueeze(dim=0).expand(state.size(0))
    # print("Goal reached shape", goal_reached.shape)
    return goal_reached


class IgcAgentTrainer(RlBaseModule):
    """
    A class representing the IGC Agent Trainer.

    :param module_name: The name of the module.
    :type module_name: str
    :param spec: The specifications for the trainer.
    :type spec: argparse.Namespace
    :param llm_model: The LLM model for training.
    :type llm_model: torch.nn.Module
    :param llm_tokenizer: The LLM tokenizer for training.
    :type llm_tokenizer: PreTrainedTokenizer
    :param env: The vectorized REST API environment.
    :type env: VectorizedRestApiEnv
    :param ds: The JSONDataset for training, if available.
    :type ds: Optional[JSONDataset]
    :param metric_logger: The metric logger for tracking training metrics, if available.
    :type metric_logger: Optional[MetricLogger]
    :param is_inference: Flag indicating whether the trainer is for inference.
    :type is_inference: Optional[bool]
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 env: VectorizedRestApiEnv,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = "False",
                 device=None):
        """
        :param module_name:
        :param spec:
        :param llm_model:
        :param llm_tokenizer:
        :param ds:
        :param metric_logger:
        :param is_inference:
        """
        super().__init__(
            module_name,
            spec,
            llm_model,
            llm_tokenizer,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=is_inference,
            device=device
        )

        # Validate inputs
        self.epsilon_decay_factor = 0.99

        if not isinstance(module_name, str):
            raise TypeError("module_name must be a string.")

        if not isinstance(spec, argparse.Namespace):
            raise TypeError("spec must be an instance of argparse.Namespace.")

        if not isinstance(llm_model, torch.nn.Module):
            raise TypeError("llm_model must be an instance of torch.nn.Module.")

        if not isinstance(llm_tokenizer, PreTrainedTokenizer):
            raise TypeError("llm_tokenizer must be an instance of PreTrainedTokenizer.")

        if ds is not None and not isinstance(ds, JSONDataset):
            raise TypeError("ds must be an instance of JSONDataset.")

        if metric_logger is not None and not isinstance(metric_logger, MetricLogger):
            raise TypeError("metric_logger must be an instance of MetricLogger.")

        if not isinstance(is_inference, bool):
            raise TypeError("is_inference must be a boolean value.")

        self._args = spec
        self.env = env

        self.batch_size = spec.rl_batch_size
        self.max_episode_len = spec.max_trajectory_length

        self.buffer_size = spec.rl_buffer_size

        self.num_episodes = spec.rl_num_episodes
        self.num_epochs = spec.rl_num_train_epochs
        self.gamma = spec.rl_gamma_value
        self.num_optimization_steps = spec.rl_num_optimization_steps
        self.steps_per_episode = spec.rl_steps_per_episode

        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(package_dir)

        self.current_goal_action = None
        self.action_dim = self.env.action_space.shape[-1]
        self.observation_space = 1025

        self.device = device
        self.agent_model = Igc_QNetwork(
            self.observation_space * 2,
            self.action_dim,
            hidden_dim=self.env.observation_space.shape[-1])

        self.target_model = Igc_QNetwork(
            self.observation_space * 2,
            self.action_dim,
            hidden_dim=self.env.observation_space.shape[-1]
        )

        self.optimizer = optim.Adam(self.agent_model.parameters(), lr=spec.rl_lr)
        self.agent_model.to(self.device)
        self.target_model.to(self.device)

        self.current_goal = None
        self.replay_buffer = Buffer(self.buffer_size, self.batch_size)

        loguru.logger.level("INFO")
        self.logger.info(f"Creating igc rl trainer with buffer_size={self.buffer_size}, lr={spec.rl_lr}, "
                         f"batch_size={self.batch_size}, max_episode_len={self.max_episode_len}, "
                         f"num_episodes={self.num_episodes}, num_epochs={self.num_epochs}, "
                         f"gamma={self.gamma}, num_optimization_steps={self.num_optimization_steps}, "
                         f"steps_per_episode={self.steps_per_episode}")
        print(f"buffer size {spec.rl_buffer_size}")

        self.env_reward_function = env_reward_function
        self.num_relabeled = 4
        self.goal_reached = []

    @staticmethod
    def update_target(model: nn.Module, target_model: nn.Module):
        """
        Update the target model by copying the weights from the source model.

        :param model: The source model.
        :type model: nn.Module
        :param target_model: The target model.
        :type target_model: nn.Module
        """
        target_model.load_state_dict(model.state_dict())

    def _create_action(self):
        """
        Create rest api action, that consists of one hot vector for rest api
        and one hot vector for http method.
        :return:
        """
        rest_api, http_supported_method, one_hot_action = self.dataset.sample_rest_api()
        action = RestApiEnv.concat_rest_api_method(
            one_hot_action, RestApiEnv.encode_rest_api_method("GET")
        )
        return action, rest_api, http_supported_method

    def _create_goal(self, http_method: Optional[str] = "GET") -> dict:
        """
        Sample a goal from the dataset.

        :return:
        """
        goal_state, action_vector, rest_apis, supported_methods = self.env.sample_same_goal()
        goal = {
            "state": goal_state,
            "rest_apis": rest_apis,
            "action_vector": action_vector,
            "http_method": http_method,
            "parameters": None,
        }
        return goal


    def train_goal(self, epsilon=0.0):
        """
        :return:
        """
        _state, info = self.env.reset(
            goal=self.current_goal["state"],
            goal_type=GoalTypeState.State
        )

        self.env.add_goal_state(self.current_goal["state"])

        if not torch.is_same_size(_state, self.current_goal["state"]):
            raise ValueError("State and goal have different dimensions.")

        if not isinstance(_state, torch.Tensor) or not isinstance(
            self.current_goal["state"], torch.Tensor):
            raise TypeError("State and goal must be tensors.")

        episode_experience = []
        rewards_per_trajectory = []

        i = 0
        terminated = [False] * self.env.num_envs
        truncated = [False] * self.env.num_envs

        while (not any(terminated) or not any(truncated)) and i < self.max_episode_len:

            goal_state = self.current_goal["state"]
            state_flat = _state.view(_state.size(0), -1)
            goal_state_flat = goal_state.view(goal_state.size(0), -1)
            input_state = torch.cat([state_flat, goal_state_flat], dim=1)

            if np.random.uniform(0, 1) < epsilon:
                # random action
                rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(self.env.num_envs)
                http_methods_one_hot = self.env.encode_batched_rest_api_method("GET", self.env.num_envs)
                concatenated_vector = self.env.concat_batch_rest_api_method(
                    one_hot_vectors, http_methods_one_hot)
            else:
                out = self.agent_model.forward(input_state.to(self.device))
                # greedy action
                rest_tensor_slice, method_tensor_slice = self.env.extract_action_method(out)
                rest_api_indices = torch.argmax(rest_tensor_slice, dim=1)
                rest_api_method_indices = torch.argmax(method_tensor_slice, dim=1)

                num_rest_api_class = rest_tensor_slice.shape[1]
                num_rest_api_methods = method_tensor_slice.shape[1]

                rest_api_one_hot = torch.nn.functional.one_hot(
                    rest_api_indices,
                    num_classes=num_rest_api_class
                ).float().float().detach().cpu()

                rest_api_method_one_hot = torch.nn.functional.one_hot(
                    rest_api_method_indices,
                    num_classes=num_rest_api_methods
                ).float().float().detach().cpu()

                at_most_one_1_rest_api = torch.sum(rest_api_one_hot == 1) <= 1
                at_most_one_1_rest_api_method = torch.sum(rest_api_method_one_hot == 1) <= 1
                concatenated_vector = torch.cat([rest_api_one_hot, rest_api_method_one_hot], dim=1)

            next_state, rewards, done, truncated, info = self.env.step(concatenated_vector)
            next_state_flat = next_state.view(next_state.size(0), -1).detach().cpu()
            episode_experience.append(
                (state_flat, concatenated_vector, rewards, next_state_flat, goal_state_flat)
            )

            rewards_per_trajectory.append(rewards)
            _state = next_state
            i += 1

        goal_reached_flags = [goals['goal_reached'].item() for goals in info]
        goal_reached_count = sum(goal_reached_flags)

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        return episode_experience, torch.sum(rewards_sum_per_trajectory, dim=0), goal_reached_count

    def update_replay_buffer(self, episode_experience):
        """

        :param episode_experience:
        :return:
        """
        num_experiences_added = 0
        for timestep in range(len(episode_experience)):
            # copy experience from episode_experience to replay_buffer
            _state, action_one_hot, rewards, _next_state, goal = episode_experience[timestep]
            combined_current_state = torch.cat([_state, goal], dim=1).clone()
            combined_next_state = torch.cat([_next_state, goal], dim=1).clone()
            self.replay_buffer.add(combined_current_state, action_one_hot.clone(), rewards.clone(), combined_next_state)

            for _ in range(self.num_relabeled):
                final_state, _, _, _, _ = episode_experience[-1]
                relabeled_goal = final_state.clone()

                # print("relabeled_goal shape:", relabeled_goal.shape)
                # print("final_state shape:", final_state.shape)

                relabeled_reward = self.env_reward_function(_state.clone(), relabeled_goal.clone())
                relabeled_current_state = torch.cat([_state, relabeled_goal], dim=1).clone()
                relabeled_next_state = torch.cat([_next_state, relabeled_goal], dim=1).clone()

                # print("relabeled_goal shape:", relabeled_goal.shape)
                # print("relabeled_current_state shape:", relabeled_current_state.shape)
                # print("action_one_hot shape:", action_one_hot.shape)
                # print("relabeled_reward shape:", relabeled_reward.shape)
                # print("relabeled_next_state shape:", relabeled_next_state.shape)

                self.replay_buffer.add(
                    relabeled_current_state,
                    action_one_hot.clone(),
                    relabeled_reward.clone(),
                    relabeled_next_state
                )

            num_experiences_added += 1

    def train(self):
        """

        :return:
        """

        epsilon = 1.0
        self.current_goal = self._create_goal()
        # start by making Q-target and Q-policy the same
        self.update_target(self.agent_model, self.target_model)

        # Run for a fixed number of epochs
        for epoch_idx in range(self.num_epochs):

            # total reward for the epoch
            total_reward = 0.0
            total_goal_reached = 0

            goal_reached_counts = []
            mean_losses = []
            losses = []

            for _ in range(self.num_episodes):
                episode_experience, rewards_sum_per_trajectory, goal_reached_count = self.train_goal(epsilon)
                total_goal_reached += goal_reached_count
                self.update_replay_buffer(episode_experience)
                total_reward += rewards_sum_per_trajectory.item()

            for _ in range(self.num_optimization_steps):
                state, action_one_hot, reward, next_state = self.replay_buffer.sample_batch()
                self.optimizer.zero_grad()
                next_state = next_state.to(self.device)
                target_q_vals = self.target_model(next_state).detach()

                reward = reward.to(self.device)
                target_q_vals = target_q_vals.to(self.device)

                q_loss_target = torch.clip(
                    reward + self.gamma * torch.max(
                        target_q_vals, dim=-1).values, -1.0 / (1 - self.gamma), 0)

                state = state.to(self.device)
                action_one_hot = action_one_hot.to(self.device)

                model_predict = self.agent_model(state)
                q_val = torch.sum(model_predict * action_one_hot, dim=1)
                criterion = nn.MSELoss()

                loss = criterion(q_val, q_loss_target)
                losses.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()

            goal_reached_counts.append(total_goal_reached)
            mean_losses.append(np.mean(losses))

            # log metrics
            self.metric_logger.log_metric("epoch_mean_loss", np.mean(losses), epoch_idx)
            self.metric_logger.log_metric("epoch_cumulative_reward", total_reward, epoch_idx)
            self.metric_logger.log_metric("epoch_goal_reached_count", total_goal_reached, epoch_idx)

            self.update_target(self.agent_model, self.target_model)
            print(
                f"Epoch: {epoch_idx} "
                f"Goals reached {total_goal_reached} "
                f"Cumulative reward: {total_reward} "
                f"Mean loss: {np.mean(losses)}")

            # Decay epsilon
            epsilon *= self.epsilon_decay_factor
