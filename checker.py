import json
import os
import torch

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_gym_base import RestApiBaseEnv
from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.envs.rest_gym_env import RestApiEnv
from igc.envs.rest_mock_server import MockServer, MockResponse
from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main
import time


def enable_secure_boot_callback(json_data, handler_state) -> MockResponse:
    """Register single callback for enabling secure boot.

    :param handler_state:
    :param json_data: The JSON data received in the request.
    :return: A mock response indicating the status of enabling secure boot.
    """
    try:
        new_data = json.loads(json_data)
    except json.JSONDecodeError:
        return MockResponse({"message": "Invalid JSON data"}, 400)

    current_data = handler_state.get("json_data")
    if current_data:
        if isinstance(current_data, str):
            print("Loading from string")
            current_data = json.loads(current_data)
        print("updating current_data")
        current_data.update(new_data)

    # if "SecureBootCurrentBoot" in current_data:
    #     return MockResponse({"message": "Invalid SecureBootCurrentBoot value"}, 400)
    #
    # print("CAlled2")
    #
    # if "SecureBootEnable" in current_data:
    #     return MockResponse({"message": "Invalid SecureBootEnable value"}, 400)
    #
    # print("CAlled3")
    # if "SecureBootMode" in current_data:
    #     return MockResponse({"message": "Invalid SecureBootMode value"}, 400)

    print("current data data type", type(current_data))
    resp = MockResponse(
        {"message": "Secure boot is enabled"},
        200, error=False, new_state=current_data)

    return resp


def mock_test_all_rest_api(cmd):
    """
    Iterate over all REST API in the dataset, send mock requests to the server,
    and assert that all GET requests return a status code of 200.

    :return:
    """
    tokenizer = IgcLllModule.load_llm_embeddings_model(cmd, only_tokenizer=True)
    dataset = JSONDataset(
        raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
        dataset_dir=f"datasets",
        verbose=True, tokenizer=tokenizer)
    mock_rest = MockServer(args, dataset)

    start_time = time.time()
    for rest_api, resp, one_hot in dataset.sample_all_rest_api():
        response = mock_rest.request(rest_api, "GET")
        assert response.status_code == 200, \
            f"GET request to {rest_api} returned status code {response.status_code}"
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time first pass taken: {total_time} seconds")

    start_time = time.time()
    for rest_api, resp, one_hot in dataset.sample_all_rest_api():
        response = mock_rest.request(rest_api, "GET")
        assert response.status_code == 200, \
            f"GET request to {rest_api} returned status code {response.status_code}"
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time second pass taken: {total_time} seconds")


def test_all_one_hot_vectors(cmd):
    """
    Iterate over all REST API in the dataset, send mock requests to the server,
    and assert that all GET requests return a status code of 200.
    :return:
    """

    model, tokenizer, _ = IgcLllModule.load_llm_embeddings_model(cmd)
    dataset = JSONDataset(
        raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
        dataset_dir=f"datasets",
        verbose=True, tokenizer=tokenizer)

    env = RestApiEnv(
        args=args,
        model=model,
        tokenizer=tokenizer,
        discovered_rest_api=dataset,
        max_episode=100000000)

    _state, _ = env.reset()
    start_time = time.time()
    sent_one_hot_vectors = set()

    one_hot_get = RestApiEnv.encode_rest_api_method("GET")
    for rest_api, resp, one_hot in dataset.sample_all_rest_api():

        if one_hot in sent_one_hot_vectors:
            print(f"Same one-hot vector sent for REST API: {rest_api}")

        step_start_time = time.time()
        next_state, reward, done, terminated, info = env.step(
            RestApiEnv.concat_rest_api_method(one_hot, one_hot_get)
        )
        assert not done, f"done flag is True for REST API: {rest_api}"
        assert not terminated, f"terminated flag is True for REST API: {rest_api}"
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        print(f"Time taken for {rest_api} step: {step_time} seconds")
        sent_one_hot_vectors.add(one_hot)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time first pass taken: {total_time} seconds")


def debug_print_step_result(next_states, rewards, terminated, truncated, infos):
    """

    :param next_states:
    :param rewards:
    :param terminated:
    :param truncated:
    :param infos:
    :return:
    """
    print("Type of next_states:", type(next_states))
    print("Size of next_states:", next_states.size())
    print("Shape of next_states:", next_states.shape)
    print("Type of rewards:", type(rewards))
    print("Size of rewards:", rewards.size())
    print("Shape of rewards:", rewards.shape)
    print("Type of terminated:", type(terminated))
    print("Size of terminated:", terminated.size())
    print("Shape of terminated:", terminated.shape)
    print("Type of truncated:", type(truncated))
    print("Size of truncated:", truncated.size())
    print("Shape of truncated:", truncated.shape)
    print("Type of infos:", type(infos))
    print("Length of infos:", len(infos))


def batch_simulate_error_500(cmd):
    """ Batch of action one positive and one negative
    :param cmd:
    :return:
    """
    model, tokenizer, _ = IgcLllModule.load_llm_embeddings_model(cmd)
    dataset = JSONDataset(
        raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
        dataset_dir=f"datasets",
        verbose=True, tokenizer=tokenizer)

    # max episode 3
    env = VectorizedRestApiEnv(
        args=args,
        model=model,
        tokenizer=tokenizer,
        discovered_rest_api=dataset,
        max_episode=3,
        num_envs=2)

    # (batch_size, observation_shape)
    observation, info = env.reset()
    rest_apis, supported_methods, one_hot_vectors = dataset.sample_batch_of_rest_api(2)
    http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
    action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

    # first step we receive error , second step we ask mock to return 500
    # after that both batch should be terminated.
    for i in range(0, 3):
        # two GET should give us 0.1 on each batch, tensor([0.1000, 0.1000])
        next_states, rewards, terminated, truncated, infos = env.step(action_vector)
        # debug_print_step_result(next_states, rewards, terminated, truncated, infos)
        print(f"step {i} rewards", rewards)
        print(f"step {i} terminated", terminated)
        print(f"step {i} truncated", truncated)
        env.mock_server().set_simulate_http_500_error(True)
        if i == 1:
            env.mock_server().set_simulate_http_500_error(False)


def batch_simulate_error_500_check_termination(cmd):
    """ Batch of action one positive and one negative
    :param cmd:
    :return:
    """
    model, tokenizer, _ = IgcLllModule.load_llm_embeddings_model(cmd)
    dataset = JSONDataset(
        raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
        dataset_dir=f"datasets",
        verbose=True, tokenizer=tokenizer)

    # max episode 3
    env = VectorizedRestApiEnv(
        args=args,
        model=model,
        tokenizer=tokenizer,
        discovered_rest_api=dataset,
        max_episode=3,
        num_envs=2)

    # (batch_size, observation_shape)
    observation, info = env.reset()
    rest_apis, supported_methods, one_hot_vectors = dataset.sample_batch_of_rest_api(2)
    http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
    action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

    # first step we receive error , second step we ask mock to return 500
    # after that both batch should be terminated.
    for i in range(0, 3):
        # two GET should give us 0.1 on each batch, tensor([0.1000, 0.1000])
        next_states, rewards, terminated, truncated, infos = env.step(action_vector)
        # debug_print_step_result(next_states, rewards, terminated, truncated, infos)
        print(f"step {i} rewards", rewards)
        print(f"step {i} terminated", terminated)
        print(f"step {i} truncated", truncated)
        env.mock_server().set_simulate_http_500_error(True)
        if i == 1:
            env.mock_server().set_simulate_http_500_error(False)


class EnvChecker:
    def __init__(self, cmd):
        """
        """
        self._env = None
        start_time = time.time()
        self.model, self.tokenizer, _ = IgcLllModule.load_llm_embeddings_model(cmd)
        elapsed_time = time.time() - start_time
        print(f"Model loading time: {elapsed_time} seconds")

        start_time = time.time()
        self.dataset = JSONDataset(
            raw_json_directory_path=os.path.expanduser(args.raw_data_dir),
            dataset_dir=f"datasets",
            verbose=True, tokenizer=self.tokenizer)
        elapsed_time = time.time() - start_time
        print(f"Dataset loading time: {elapsed_time} seconds")

    def batch_simulate_error_500_check_termination(self, cmd):
        """ Batch of action one positive and one negative
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=3,
            num_envs=2)

        # (batch_size, observation_shape)
        observation, info = env.reset()
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(2)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while True:
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            env.mock_server().set_simulate_http_500_error(True)
            if i == 1:
                env.mock_server().set_simulate_http_500_error(False)

            if torch.any(terminated):
                break

        expected_rewards = torch.tensor([-0.4, -0.4])
        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        # print("Simulate 500 Error, Total rewards per trajectory:", rewards_sum_per_trajectory)
        assert torch.allclose(rewards_sum_per_trajectory, expected_rewards), "Unexpected rewards per trajectory"

    def batch_simulate_error_500_check_termination_step_0(self, cmd):
        """ Simulate 500 at step 0
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=3,
            num_envs=2)

        # (batch_size, observation_shape)
        observation, info = env.reset()
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(2)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while True:
            # we set mock respond to 500
            env.mock_server().set_simulate_http_500_error(True)
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            print(f"step {i} rewards", rewards)
            print(f"step {i} terminated", terminated)
            print(f"step {i} truncated", truncated)

            env.mock_server().set_simulate_http_500_error(True)
            if i == 1:
                env.mock_server().set_simulate_http_500_error(False)

            if any(terminated):
                break

        expected_rewards = torch.tensor([-0.5, -0.5])
        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        # print("Simulate 500 Error, Total rewards per trajectory:", rewards_sum_per_trajectory)
        assert torch.allclose(rewards_sum_per_trajectory, expected_rewards), "Unexpected rewards per trajectory"

    def batch_simulate_goal_reached(self, cmd, max_episode=4, num_envs=2):
        """ Batch of action one positive and one negative

        at step 0 r 0.1 and 0.1
        at step 1 we get 0.1 and 0.1
        at i = 1 we simulate reward for batch idx 0
        at step 2 we get 0.1 and 1.0

        Total 0.3 and 1.2

        :param num_envs:
        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=4,
            num_envs=2
        )

        # (batch_size, observation_shape)
        observation, info = env.reset()
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(2)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while True:
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            if i == 1:
                env.simulate_goal_reached(batch_id=0)
            if any(terminated):
                break
            i += 1

        expected_rewards = torch.tensor([1.2, 0.3])
        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        # print("Total rewards per trajectory:", rewards_sum_per_trajectory)
        assert torch.allclose(rewards_sum_per_trajectory, expected_rewards), "Unexpected rewards per trajectory"

    def batch_vectorized_env_positive_negative(self, cmd):
        """ Tests Batch of action one positive and one negative

        One trajectory 0.1 , 0.1 , -1.0 end of trajectory
        Second trajectory -0.2 -0.2, -1.0 end of trajectory

        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=3,
            num_envs=2)

        observation, info = env.reset()
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(2)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 2)
        http_methods_one_hot[1] = torch.tensor([0., 0., 0., 0., 0., 1.])
        # set one HTTP method invalid.

        # # torch.Size([batch_size, 2472])
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while True:
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            if any(terminated):
                break
            i += 1

        #
        expected_rewards = torch.tensor([-0.8, -1.4])
        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        assert torch.allclose(rewards_sum_per_trajectory, expected_rewards), "Unexpected rewards per trajectory"

    def batch_vectorized_4_envs(self, cmd):
        """Vectorized batch of 4 envs, normal gets and junk
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)

        # (batch_size, observation_shape)
        observation, info = env.reset()
        # one_hot_vectors (batch_size, observation_shape), # dim torch.Size([batch_size, 2466])
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        # # dim torch.Size([batch_size, 6])
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        http_methods_one_hot[1] = torch.tensor([0., 0., 0., 0., 0., 1.])

        # # torch.Size([batch_size, 2472])
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while True:
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            if any(terminated):
                break
            i += 1

        print(rewards_per_trajectory)

    def sample_goal_test(self, cmd):
        """

        :param cmd:
        :return:
        """
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)

        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()

        assert goal_observation.shape[0] == env.num_envs
        assert goal_observation.shape[0] == env.observation_space.shape[0]
        assert goal_observation.shape[1] == env.observation_space.shape[1]
        assert goal_observation.shape[2] == env.observation_space.shape[2]

        assert goal_observation.shape[2] == env.observation_space.shape[2]
        assert goal_action_vector.shape[0] == env.action_space.shape[0]
        assert goal_action_vector.shape[1] == env.action_space.shape[1]

        assert torch.allclose(goal_observation, goal_observation[0])
        assert torch.allclose(goal_action_vector, goal_action_vector[0])

    def goal_reward_state_goal_set_no_reward(self, cmd, max_episode: int = 10):
        """We set goal and check that we didn't get any reward.
        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=max_episode,
            num_envs=4)

        # reset and sample goal
        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        while not any(terminated) and i < max_episode:
            next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        print(rewards_sum_per_trajectory)

    def goal_reward_state_goal_set_get_reward(self, cmd, max_episode: int = 10):
        """

        We set goal and check execute a couple of steps and then pass action
        that should lead to goal state and get reward.

        :param max_episode:
        :param cmd:
        :return:
        """
        start_time = time.time()
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)
        elapsed_time = time.time() - start_time
        print(f"Environment creation time: {elapsed_time} seconds")

        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        # actions
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        terminated = [False] * env.num_envs
        truncated = [False] * env.num_envs

        while (not any(terminated) or not any(truncated)) and i < max_episode:
            if i == 2:
                print(f"goal action vector {action_vector.dtype}")
                next_states, rewards, terminated, truncated, infos = env.step(goal_action_vector)
            else:
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0)
        print(f"Stacked {rewards_sum_per_trajectory}")
        rewards_sum_per_trajectory = rewards_sum_per_trajectory.sum(dim=0, keepdim=True)
        print(f"sum {rewards_sum_per_trajectory}")

    def goal_reward_state_goal_single_trajectory(self, cmd, max_episode: int = 10):
        """

        We set goal and check execute a couple of steps and then pass action
        that should lead to goal state and get reward.

        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        start_time = time.time()
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)
        elapsed_time = time.time() - start_time
        print(f"Environment creation time: {elapsed_time} seconds")

        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        # actions
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        terminated = [False] * env.num_envs
        truncated = [False] * env.num_envs
        while (not any(terminated) or not any(truncated)) and i < max_episode:
            if i == 2:
                # Set goal action vector for one trajectory in the batch
                action_vector[1] = goal_action_vector[1]
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            else:
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0)
        rewards_sum_per_trajectory = rewards_sum_per_trajectory.sum(dim=0, keepdim=True)
        expected_sum = torch.tensor([[-0.1000, 1.2000, -0.1000, -0.1000]])
        assert torch.allclose(rewards_sum_per_trajectory,
                              expected_sum), "Sum of rewards does not match the expected value."

    def goal_two_trajectory_reward_state_goal_single_trajectory(self, cmd, max_episode: int = 10):
        """Two trajectory should reach goal.
        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)

        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        # actions
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        terminated = [False] * env.num_envs
        truncated = [False] * env.num_envs

        while (not any(terminated) or not any(truncated)) and i < max_episode:
            if i == 2:
                # Set goal action vector for one trajectory in the batch
                action_vector[1] = goal_action_vector[1]
                action_vector[2] = goal_action_vector[2]
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            else:
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0)
        rewards_sum_per_trajectory = rewards_sum_per_trajectory.sum(dim=0, keepdim=True)
        expected_sum = torch.tensor([[-0.1000, 1.2000, -1.2000, -0.1000]])
        print("Rewards {rewards_sum_per_trajectory}")
        # assert torch.allclose(rewards_sum_per_trajectory,
        #                       expected_sum), "Sum of rewards does not match the expected value."

    def goal_all_4_trajectory_reward_state_goal_single_trajectory(self, cmd, max_episode: int = 10):
        """All 4  trajectory should reach goal.
        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)

        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        # actions
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        terminated = [False] * env.num_envs
        truncated = [False] * env.num_envs

        while (not any(terminated) or not any(truncated)) and i < max_episode:
            if i == 2:
                # Set goal action vector for one trajectory in the batch
                action_vector[0] = goal_action_vector[0]
                action_vector[1] = goal_action_vector[1]
                action_vector[2] = goal_action_vector[2]
                action_vector[3] = goal_action_vector[3]
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            else:
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        print(rewards_sum_per_trajectory)

    def goal_two_trajectory_reward_two_terminated(self, cmd, max_episode: int = 10):
        """
        Two trajectory reached goal , two terminated.

        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=10,
            num_envs=4)

        observation, info = env.reset()
        goal_observation, goal_action_vector, rest_apis, supported_methods = env.sample_same_goal()
        env.add_goal_state(goal_observation)

        # actions
        rest_apis, supported_methods, one_hot_vectors = self.dataset.sample_batch_of_rest_api(4)
        http_methods_one_hot = RestApiBaseEnv.encode_batched_rest_api_method("GET", 4)
        action_vector = RestApiBaseEnv.concat_batch_rest_api_method(
            one_hot_vectors, http_methods_one_hot)

        i = 0
        rewards_per_trajectory = []
        terminated = [False] * env.num_envs
        truncated = [False] * env.num_envs

        while (not any(terminated) or not any(truncated)) and i < max_episode:
            if i == 2:
                # Set goal action vector for one trajectory in the batch
                action_vector[0] = goal_action_vector[0]
                action_vector[1] = goal_action_vector[1]
                action_vector[2] = goal_action_vector[2]
                action_vector[3] = goal_action_vector[3]
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            else:
                next_states, rewards, terminated, truncated, infos = env.step(action_vector)
            rewards_per_trajectory.append(rewards)
            i += 1

        rewards_sum_per_trajectory = torch.stack(rewards_per_trajectory, dim=0).sum(dim=0)
        print(rewards_sum_per_trajectory)

    def goal_for_registered_callback(self, cmd, max_episode: int = 10):
        """
         We set goal and check execute a couple of steps and then pass action
         that should lead to goal state and get reward.

        :param max_episode:
        :param cmd:
        :return:
        """
        # max episode 3
        env = VectorizedRestApiEnv(
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            discovered_rest_api=self.dataset,
            max_episode=self.max_episode_len,
            num_envs=4)

        env.mock_server().register_callback(
            "/redfish/v1/Systems/System.Embedded.1/SecureBoot", "PATCH", enable_secure_boot_callback)

        json_data = '{"SecureBootCurrentBoot": "Enabled"}'
        response = env.mock_server().request(
            "/redfish/v1/Systems/System.Embedded.1/SecureBoot", "PATCH", json_data)

        response = env.mock_server().request("/redfish/v1/Systems/System.Embedded.1/SecureBoot", "GET")
        print(response.json_data)


def main(cmd):
    """
    :return:
    """
    # batch_vectorized_env(cmd)
    # batch_simulate_error_500(cmd)
    # batch_simulate_error_500_check_termination(cmd)
    # mock_test_all_rest_api(cmd)
    # test_all_one_hot_vectors(cmd)

    env_checker = EnvChecker(cmd)
    # env_checker.batch_simulate_goal_reached(cmd)
    # env_checker.batch_simulate_error_500_check_termination(cmd)
    # env_checker.batch_simulate_error_500_check_termination_step_0(cmd)
    # env_checker.batch_vectorized_env_positive_negative(cmd)
    # env_checker.batch_vectorized_4_envs(cmd)

    # env_checker.goal_reward_state_goal_set_no_reward(cmd)
    # env_checker.goal_reward_state_goal_set_get_reward(cmd)
    env_checker.goal_reward_state_goal_single_trajectory(cmd)
    env_checker.goal_two_trajectory_reward_state_goal_single_trajectory(cmd)
    # env_checker.goal_all_4_trajectory_reward_state_goal_single_trajectory(cmd)
    # env_checker.goal_two_trajectory_reward_two_terminated(cmd)
    # env_checker.goal_for_registered_callback(cmd)


if __name__ == '__main__':
    args = shared_main()
    main(args)
