import argparse
import os

import numpy as np
import torch
from torch import optim, nn

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.envs.rest_gym_env import RestApiEnv, GoalTypeState
from igc.modules import experience_buffer, q_network
from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main


class IgcAgentTrainer:
    """
    """

    def __init__(self, args: argparse.Namespace):
        """

        :param args:
        """
        self.args = args
        self.env = None

        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(package_dir)
        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        self.dataset = JSONDataset(
            raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        self.max_episode_len = 10
        self.batch_size = 4

        self.env = VectorizedRestApiEnv(
            args=args,
            model=model,
            tokenizer=tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=self.max_episode_len,
            num_envs=self.batch_size
        )
        # self.env = VectorizedRestApiEnv(
        #     args=args, model=model,
        #     tokenizer=tokenizer,
        #     discovered_rest_api=self.dataset,
        #     max_episode=max_episode_len,
        #     num_envs=2
        # )
        #

        self.current_goal_action = None
        self.steps_per_episode = 10

        self.action_dim = self.env.action_space.shape[-1]

        self.agent_model = q_network.QNetwork(
            1571328,
            self.action_dim, hidden_dim=self.env.observation_space.shape[-1])
        self.target_model = q_network.QNetwork(
            1571328,
            self.action_dim, hidden_dim=self.env.observation_space.shape[-1]
        )

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # set a goal
        self.current_goal = None

    @staticmethod
    def update_target(model: nn.Module, target_model: nn.Module):
        """
        :param model:
        :param target_model:
        :return:
        """
        target_model.load_state_dict(model.state_dict())

    def create_action(self):
        """
        :return:
        """
        rest_api, supported_method, one_hot_action = self.dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

    def create_goal(self, http_method="GET"):
        """Sample a goal from the dataset,
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

    def train_goal(self):
        """
        :return:
        """

        print("train goal")

        print("Training goal:", self.current_goal["state"].shape)

        _state, info = self.env.reset(
            goal=self.current_goal["state"],
            goal_type=GoalTypeState.State
        )

        # self.current_goal["goal_state"] = info["goal"]

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

        while (not any(terminated) or not any (truncated)) and i < self.max_episode_len:
            goal_state = self.current_goal["state"]
            state_flat = _state.view(_state.size(0), -1)
            goal_state_flat = goal_state.view(goal_state.size(0), -1)
            input_state = torch.cat([state_flat, goal_state_flat], dim=1)

            out = self.agent_model.forward(input_state)
            rest_tensor_slice, method_tensor_slice = self.env.extract_action_method(out)
            rest_api_indices = torch.argmax(rest_tensor_slice, dim=1)
            rest_api_method_indices = torch.argmax(method_tensor_slice, dim=1)

            num_rest_api_class = rest_tensor_slice.shape[1]
            num_rest_api_methods = method_tensor_slice.shape[1]

            rest_api_one_hot = torch.nn.functional.one_hot(
                rest_api_indices,
                num_classes=num_rest_api_class
            ).float()

            rest_api_method_one_hot = torch.nn.functional.one_hot(
                rest_api_method_indices,
                num_classes=num_rest_api_methods
            ).float()

            at_most_one_1_rest_api = torch.sum(rest_api_one_hot == 1) <= 1
            at_most_one_1_rest_api_method = torch.sum(rest_api_method_one_hot == 1) <= 1

            concatenated_vector = torch.cat([rest_api_one_hot, rest_api_method_one_hot], dim=1)
            next_state, rewards, done, truncated, info = self.env.step(concatenated_vector)
            rewards_per_trajectory.append(rewards)

            next_state_flat = next_state.view(next_state.size(0), -1)
            episode_experience.append(
                (state_flat, concatenated_vector, rewards, next_state_flat, goal_state_flat)
            )
            _state = next_state
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        print("train goal done")
        print(f"episode reward {rewards_sum_per_trajectory}")

        return episode_experience

    def update_replay_buffer(self,
                             replay_buffer,
                             episode_experience,
                             env_reward_function=None,
                             num_relabeled=4, ):
        """Adds experience to the replay buffer. Training is done with
        episodes from the replay buffer. When HER is used, relabeled
        experiences are also added to the replay buffer.

        Args:

            replay_buffer (ReplayBuffer): replay buffer to store experience

            episode_experience (list): list containing the transitions
                (state, action, reward, next_state, goal_state)

            HER (HERType): type of hindsight experience replay to use

            env_reward_function ((ndarray, ndarray) -> float):
                reward function for relabelling transitions

            num_relabeled (int): number of relabeled transition per transition
        """

        for timestep in range(len(episode_experience)):
            # copy experience from episode_experience to replay_buffer
            _state, action_one_hot, rewards, _next_state, goal = episode_experience[timestep]
            combined_current_state = torch.cat([_state, goal], dim=1)
            combined_next_state = torch.cat([_next_state, goal], dim=1)
            replay_buffer.add(combined_current_state, action_one_hot, rewards, combined_next_state)

            # # get final goal
            # final_state, _, _, _, _ = episode_experience[-1]
            # relabeled_goal = final_state
            #
            # # compute new reward
            # relabeled_reward = env_reward_function(next_state, relabeled_goal)
            #
            # # add to buffer
            # replay_buffer.add(np.append(state.copy(), relabeled_goal.copy()),
            #                   action,
            #                   relabeled_reward,
            #                   np.append(next_state.copy(), relabeled_goal.copy()))

    def train(
            self,
            num_epochs,
            env_reward_function=None,
            num_relabeled=4,
            buffer_size=1e6,
            num_episodes=16,
            steps_per_episode=50,
            gamma=0.98,
            opt_steps=40,
            log_interval=5,
    ):

        self.current_goal = self.create_goal()
        replay_buffer = experience_buffer.Buffer(buffer_size, self.batch_size)

        # start by making Q-target and Q-policy the same
        self.update_target(self.agent_model, self.target_model)

        # Run for a fixed number of epochs
        for epoch_idx in range(num_epochs):
            # total reward for the epoch
            total_reward = 0.0
            # record success rate for each episode of the epoch
            successes = []
            # loss at the end of each epoch
            losses = []

            for _ in range(num_episodes):
                episode_experience = self.train_goal()
                # successes.append(succeeded)
                # add to the replay buffer; use specified HER policy
                self.update_replay_buffer(
                    replay_buffer,
                    episode_experience,
                    env_reward_function=env_reward_function,
                    num_relabeled=num_relabeled
                )

            # optimize the Q-policy network
            for _ in range(opt_steps):
                # sample from the replay buffer, each (batch_size, )
                state, action_one_hot, reward, next_state = replay_buffer.sample_batch()
                self.optimizer.zero_grad()

                # forward pass through target network
                target_q_vals = self.target_model(next_state).detach()
                print("Target q vals shape", target_q_vals.shape)

                # calculate target reward
                q_loss_target = torch.clip(
                    reward + gamma * torch.max(target_q_vals, dim=-1).values, -1.0 / (1 - gamma), 0)

                model_predict = self.agent_model(state)
                q_val = torch.sum(model_predict * action_one_hot, dim=1)

                criterion = nn.MSELoss()
                loss = criterion(q_val, q_loss_target)
                losses.append(loss.detach().numpy())

                loss.backward()
                self.optimizer.step()

            # update target model by copying Q-policy to Q-target
            self.update_target(self.agent_model, self.target_model)

            if epoch_idx % log_interval == 0:
                print(
                    f"Epoch: {epoch_idx} Cumulative reward: "
                    f"{total_reward} Success rate: {np.mean(successes)} Mean loss: {np.mean(losses)}"
                    # pylint: disable=line-too-long
                )
                # writer.add_scalar(
                #     "eval_metrics/total_reward", total_reward, epoch_idx)
                # writer.add_scalar(
                #     "eval_metrics/success_rate", np.mean(successes), epoch_idx)
                # writer.add_scalar(
                #     "train_metrics/td_loss", np.mean(losses), epoch_idx)
