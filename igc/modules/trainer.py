import argparse
import os

import numpy as np
import torch
from torch import optim, nn

from igc.ds.redfish_dataset import JSONDataset
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

        max_episode_len = 10

        # # sample some random goal and choose method
        # # this our goal
        # rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        # http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        # goal_action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
        #

        self.env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=max_episode_len)

        self.current_goal_action = None
        self.steps_per_episode = 10

        self.action_dim = self.env.action_space.shape[0]
        self.model = q_network.QNetwork(self.env.observation_space.shape[1], self.action_dim)
        self.target_model = q_network.QNetwork(self.env.observation_space.shape[1], self.action_dim)
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

    def run_episode(self, q_net):
        """
        :param q_net:
        :return:
        """
        # list for recording what happened in the episode
        episode_experience = []
        succeeded = False
        episodic_return = 0.0

        # reset the environment to get the initial state
        state, goal_state = self.env.reset()

        for _ in range(self.steps_per_episode):

            # append goal state to input, and prepare for feeding to the q-network
            input_state = np.concatenate([state, goal_state])
            input_state_tensor = torch.tensor(input_state, dtype=torch.float32)

            # forward pass to find action and do it greedy
            action = q_net(input_state_tensor)
            action = torch.argmax(action)
            action = action.detach().numpy()
            action = int(action)

            # take action, use env.step
            next_state, reward, done, truncated, info = self.env.step(action)
            # add transition to episode_experience as a tuple of
            # (state, action, reward, next_state, goal)
            episode_experience.append((state.copy(), action, reward.copy(), next_state.copy(), goal_state.copy()))
            episodic_return += reward

            # update state
            state = next_state

            # update succeeded bool from the info returned by env.step
            succeeded = succeeded or info.get('successful_this_state', False)

            # break the episode if done=True
            if done:
                break

        return episode_experience, episodic_return, succeeded

    def create_goal(self, method="GET"):
        """Sample a goal from the dataset,
        :return:
        """
        rest_api, supported_method, one_hot_action = self.dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        self.current_goal_action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

        goal = {
            "rest_api": rest_api,
            "supported_method": supported_method,
            "one_hot_action": self.current_goal_action,
            "action": self.current_goal_action,
            "method": "GET",
            "parameters": None,
        }

        return goal

    def train_goal(self):
        """
        :return:
        """
        _state, info = self.env.reset(goal=self.current_goal, goal_type=GoalTypeState.State)
        self.current_goal["goal_state"] = info["goal"]
        if not torch.is_same_size(_state, self.current_goal["goal_state"]):
            raise ValueError("State and goal have different dimensions.")

        if not isinstance(_state, torch.Tensor) or not isinstance(self.current_goal["goal_state"], torch.Tensor):
            raise TypeError("State and goal must be tensors.")

        episodic_return = 0
        episode_experience = []

        for _ in range(self.steps_per_episode):

            input_state = torch.cat([_state, self.current_goal["goal_state"]])
            input_state = input_state.unsqueeze(0)

            # print("Input state", input_state.shape)
            # print("Input state type:", input_state.dtype)

            out = self.model.forward(input_state)

            # print("Output shape out:", out.shape)
            # print("Output dtype out:", out.dtype)

            # output (batch size, obs shape, num_actions) torch.Size([1, 2046, 2472])
            action_one_hot = torch.argmax(out, dim=1)
            action_one_hot = torch.squeeze(action_one_hot, dim=0)

            print("action_one_hot shape out", action_one_hot.shape)
            print("action_one_hot shape out", action_one_hot.dtype)

            next_state, reward, done, truncated, info = self.env.step(action_one_hot)
            episodic_return += reward

            episode_experience.append(
                (_state.copy(), action_one_hot, reward, next_state, self.current_goal["goal_state"]))

            # Print types of all elements in episode_experience
            for exp in episode_experience:
                state_type = type(exp[0])
                action_type = type(exp[1])
                reward_type = type(exp[2])
                next_state_type = type(exp[3])
                goal_state_type = type(exp[4])
                print("Experience types:", state_type, action_type, reward_type, next_state_type, goal_state_type)

            _state = next_state
            # break the episode if done=True
            if done:
                break

        print(f"episode reward {episodic_return}")
        return episodic_return, episode_experience

    def update_replay_buffer(self,
                             replay_buffer,
                             episode_experience,
                             env_reward_function=None,
                             num_relabeled=4,):
        """Adds past experience to the replay buffer. Training is done with
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
            state, action, reward, next_state, goal = episode_experience[timestep]
            # use replay_buffer.add
            replay_buffer.add(np.append(state, goal), action, reward, np.append(next_state, goal))

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
            batch_size=128,
            log_interval=5,
    ):

        self.current_goal = self.create_goal()

        # create replay buffer
        replay_buffer = experience_buffer.Buffer(buffer_size, batch_size)

        # start by making Q-target and Q-policy the same
        self.update_target(self.model, self.target_model)

        # Run for a fixed number of epochs
        for epoch_idx in range(num_epochs):
            # total reward for the epoch
            total_reward = 0.0
            # record success rate for each episode of the epoch
            successes = []
            # loss at the end of each epoch
            losses = []

            for _ in range(num_episodes):
                # collect data in the environment
                episode_experience, ep_reward = self.train_goal()
                # track eval metrics in the environment
                total_reward += ep_reward
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
                # sample from the replay buffer
                state, action, reward, next_state = replay_buffer.sample()
                state = torch.from_numpy(state.astype(np.float32))
                action = torch.from_numpy(action)
                reward = torch.from_numpy(reward.astype(np.float32))
                next_state = torch.from_numpy(next_state.astype(np.float32))

                self.optimizer.zero_grad()
                # forward pass through target network
                target_q_vals = self.target_model(next_state).detach()

                # calculate target reward
                q_loss_target = torch.clip(
                    reward + gamma * torch.max(target_q_vals, axis=-1).values, -1.0 / (1 - gamma), 0)

                # calculate predictions and loss
                model_predict = self.model(state)
                model_action_taken = torch.reshape(action, [-1])
                action_one_hot = nn.functional.one_hot(model_action_taken, self.action_dim)
                q_val = torch.sum(model_predict * action_one_hot, axis=1)
                criterion = nn.MSELoss()
                loss = criterion(q_val, q_loss_target)
                losses.append(loss.detach().numpy())

                loss.backward()
                self.optimizer.step()

            # update target model by copying Q-policy to Q-target
            self.update_target(self.model, self.target_model)

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
