"""
Train the legacy Redfish DQN agent with HER relabeling.

This module owns rollout collection, replay-buffer updates, DQN targets, and
target-network synchronization for ``IgcAgentTrainer``. Autoencoder setup is
orchestrated outside this file by the RL module wiring.

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
from igc.modules.rl.q_targets import q_learning_target, relabel_future


def env_reward_function(state, goal):
    """
    Return a binary reward for states that match the goal tensor.

    :param state: Batched environment state tensor.
    :param goal: Goal state tensor to compare against each row in ``state``.
    :return: Reward vector with ``1.0`` when any state row matches the goal,
        otherwise ``0.0``.
    """
    # per-row match with the same tolerance VectorizedRestApiEnv.check_goal uses,
    # so HER-recomputed success and env success agree; a cross-batch any() here
    # previously leaked one env's success to every row of the relabeled batch.
    matched = torch.all(torch.isclose(state, goal, rtol=1e-3, atol=1e-3), dim=1)
    return matched.to(torch.float32)


class IgcAgentTrainer(RlBaseModule):
    """
    Collect goal-conditioned REST rollouts and train the legacy DQN policy.

    The trainer owns replay insertion, HER relabeling, target-network sync, and
    epoch-level metric logging for the current one-hot action path.
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 env: VectorizedRestApiEnv,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = False,
                 device=None):
        """
        Initialize the DQN trainer, target network, and replay buffer.

        :param module_name: Stable module name used for logging and checkpoint paths.
        :param spec: Training configuration namespace parsed by the shared CLI.
        :param llm_model: LLM module used by the base trainer contract.
        :param llm_tokenizer: Tokenizer paired with ``llm_model``.
        :param env: Vectorized REST environment used for rollout collection.
        :param ds: Optional Redfish JSON dataset used for sampled REST actions.
        :param metric_logger: Optional metric sink for training metrics.
        :param is_inference: Whether the trainer is being built for inference.
        :param device: Torch device for networks and sampled batches.
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
        # observation width comes from the env's encoder, not a legacy constant;
        # the base module already resolved self.device (read-only property here).
        self.observation_space = self.env.observation_space.shape[-1]
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
        self._rng = np.random.default_rng()

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

        :return: Tuple of encoded action tensor, REST API path, and supported methods.
        """
        rest_api, http_supported_method, one_hot_action = self.dataset.sample_rest_api()
        action = RestApiEnv.concat_rest_api_method(
            one_hot_action, RestApiEnv.encode_rest_api_method("GET")
        )
        return action, rest_api, http_supported_method

    def _create_goal(self, http_method: Optional[str] = "GET") -> dict:
        """
        Sample a goal from the vectorized REST environment.

        :param http_method: HTTP method associated with the sampled goal.
        :return: Goal dictionary with state, REST APIs, action vector, method,
            and parameters fields consumed by the rollout loop.
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
        Roll out one goal-conditioned episode with epsilon-greedy actions.

        :param epsilon: Probability of sampling a random action instead of the policy action.
        :return: Episode transitions, cumulative reward, and reached-goal count.
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
        # standard contract: position 3 = terminated (goal/dead-end), position 4 =
        # truncated (time-limit). The rollout stops on either; replay bootstrapping
        # distinguishes them downstream (q_targets masks on terminal-only dones).
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.env.num_envs, dtype=torch.bool)

        while not bool((terminated | truncated).any()) and i < self.max_episode_len:

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
                # out = self.agent_model.forward(input_state.to(self.device))
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

                concatenated_vector = torch.cat([rest_api_one_hot, rest_api_method_one_hot], dim=1)

            next_state, rewards, terminated, truncated, info = self.env.step(concatenated_vector)
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
        Insert episode transitions and HER relabeled transitions into replay.

        :param episode_experience: Sequence of rollout tuples from ``train_goal``.
        """
        num_experiences_added = 0
        for timestep in range(len(episode_experience)):
            # copy experience from episode_experience to replay_buffer
            _state, action_one_hot, rewards, _next_state, goal = episode_experience[timestep]
            combined_current_state = torch.cat([_state, goal], dim=1).clone()
            combined_next_state = torch.cat([_next_state, goal], dim=1).clone()
            # terminal iff the env goal was reached on this transition; truncation
            # (running out of steps) is not terminal and must keep bootstrapping.
            done = (rewards >= 1.0).to(rewards.dtype)
            self.replay_buffer.add(
                combined_current_state, action_one_hot.clone(),
                rewards.clone(), combined_next_state, done.clone())

            # HER "future" relabeling: substitute goals achieved at a sampled future
            # next-state and recompute the reward against this transition's own
            # achieved next-state (episode_experience tuple index 3 == next_state).
            for relabeled_goal, relabeled_reward, relabeled_done in relabel_future(
                    episode_experience, timestep, _next_state,
                    self.num_relabeled, self.env_reward_function, self._rng):
                relabeled_current_state = torch.cat([_state, relabeled_goal], dim=1).clone()
                relabeled_next_state = torch.cat([_next_state, relabeled_goal], dim=1).clone()
                self.replay_buffer.add(
                    relabeled_current_state,
                    action_one_hot.clone(),
                    relabeled_reward.clone(),
                    relabeled_next_state,
                    relabeled_done.clone(),
                )

            num_experiences_added += 1

    def train(self):
        """
        Train the DQN policy with rollout collection, HER relabeling, and target updates.
        """
        # epsilon = 0
        epsilon = 0.98
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
                # a fresh goal per episode: HER relabels alone do not vary the
                # conditioning goal, so a single fixed goal starves generalization.
                self.current_goal = self._create_goal()
                episode_experience, rewards_sum_per_trajectory, goal_reached_count = self.train_goal(epsilon)
                total_goal_reached += goal_reached_count
                self.update_replay_buffer(episode_experience)
                total_reward += rewards_sum_per_trajectory.item()

            for _ in range(self.num_optimization_steps):
                state, action_one_hot, reward, next_state, done = self.replay_buffer.sample_batch()
                self.optimizer.zero_grad()
                next_state = next_state.to(self.device)
                target_q_vals = self.target_model(next_state).detach()

                reward = reward.to(self.device)
                done = done.to(self.device)
                target_q_vals = target_q_vals.to(self.device)

                # reward + gamma * (1 - done) * max_a' Q_target(s', a'): the done mask
                # stops bootstrapping at terminals and preserves the +1 success reward
                # (the prior clip(..., ceiling=0) erased every {0, +1} success).
                q_loss_target = q_learning_target(reward, done, target_q_vals, self.gamma)

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

            # persist the agent: without this a full run discarded every weight.
            if self._module_checkpoint_dir is not None:
                self.save_checkpoint(
                    self._module_checkpoint_dir,
                    epoch=epoch_idx,
                    model=self.agent_model,
                    optimizer=self.optimizer,
                )

        if self._module_checkpoint_dir is not None:
            self.save_checkpoint(
                self._module_checkpoint_dir,
                epoch=self.num_epochs,
                model=self.agent_model,
                optimizer=self.optimizer,
            )
