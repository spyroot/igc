import json
import os

import torch
import unittest

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_gym_env import RestApiEnv, HttpMethod
from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main
from .test_utils import register_reset_goal


class TestRestApiEnv(unittest.TestCase):
    # def setUp(self):
    #     args = shared_main()
    #     package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #
    #     model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
    #     json_directory_path = os.path.expanduser(args.raw_data_dir)
    #     self.dataset = JSONDataset(
    #         raw_json_directory_path=json_directory_path,
    #         dataset_dir=f"{package_dir}/datasets",
    #         verbose=True, tokenizer=tokenizer)
    #
    #     self.env = RestApiEnv(
    #         args=args, model=model,
    #         tokenizer=tokenizer,
    #         discovered_rest_api=self.dataset
    #     )

    def test_extract_action_method(self):
        """
        Test the extract_action_method method of RestApiEnv.
        """
        rest_api_one_hot = torch.tensor([1.0, 2.0, 3.0, 4.0])

        for method in RestApiEnv.METHOD_MAPPING:
            http_method_one_hot = RestApiEnv.encode_rest_api_method(method)
            merged = RestApiEnv.concat_rest_api_method(rest_api_one_hot, http_method_one_hot)
            one_hot_rest_out, one_hot_method_out = RestApiEnv.extract_action_method(merged)
            out_method_str = RestApiEnv.one_hot_to_method_string(one_hot_method_out)
            self.assertEqual(method, out_method_str)
            self.assertTrue(torch.allclose(rest_api_one_hot, one_hot_rest_out))
            self.assertTrue(torch.allclose(http_method_one_hot, one_hot_method_out))

    def test_extract_action_method_batched(self):
        """
        Test one hot encoding for rest api and http method for batch version.
        """
        rest_api_one_hot = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                         [4.0, 3.0, 2.0, 1.0]])

        batch_size = rest_api_one_hot.shape[0]
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        http_method_one_hot_batched = http_method_one_hot.unsqueeze(0).expand(batch_size, -1)
        batch_merged = RestApiEnv.concat_rest_api_method(rest_api_one_hot, http_method_one_hot_batched)
        one_hot_rest_out, one_hot_method_out = RestApiEnv.extract_action_method(batch_merged)
        one_hot_method_out_string = RestApiEnv.batch_one_hot_to_method_string(one_hot_method_out)
        self.assertTrue(torch.allclose(one_hot_rest_out, rest_api_one_hot))
        expected_method = "GET"
        for method_str in one_hot_method_out_string:
            self.assertEqual(method_str, expected_method)

    def test_extract_action_method(self):
        """
        Test the extract_action_method method of RestApiEnv.
        """
        rest_api_one_hot = torch.tensor([1.0, 2.0, 3.0, 4.0])

        for method in RestApiEnv.METHOD_MAPPING:
            http_method_one_hot = RestApiEnv.encode_rest_api_method(method)
            merged = RestApiEnv.concat_rest_api_method(rest_api_one_hot, http_method_one_hot)
            one_hot_rest_out, one_hot_method_out = RestApiEnv.extract_action_method(merged)
            out_method_str = RestApiEnv.one_hot_to_method_string(one_hot_method_out)
            self.assertEqual(method, out_method_str)
            self.assertTrue(torch.allclose(rest_api_one_hot, one_hot_rest_out))
            self.assertTrue(torch.allclose(http_method_one_hot, one_hot_method_out))

    def test_create_env(self):
        """
        Test one hot encoding for rest api and http method for batch version.
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

    def test_env_initial_state(self):
        """Test that we have can do basic root request to API
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset)

        rest_api, one_hot_vector = dataset.entry_rest_api()
        response = env.mock_server().request(rest_api, HttpMethod.GET.value)
        self.assertEqual(response.status_code, 200, "Response status code is not success")
        response_data = json.loads(response.json_data)
        expected_keys = [
            '@odata.context',
            '@odata.id',
            '@odata.type'
        ]
        for key in expected_keys:
            self.assertIn(key, response_data, f"Response data is missing key: {key}")
        self.assertIsInstance(response_data, dict, "Response data is not a valid JSON object")

        # emb
        # return torch.Size([1, 1024, 768])

        embeddings = env.encoder.encode(response.json_data)
        expected_shape = env.observation_space.shape
        self.assertEqual(embeddings.shape, expected_shape, "Embeddings shape mismatch")
        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings is not of type torch.Tensor")

        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings is not of type torch.Tensor")
        self.assertTrue(embeddings.dtype in [torch.float16, torch.float32],
                        "Embeddings data type is not float16 or float32")
        self.assertFalse(torch.isnan(embeddings).any(), "Embeddings contain NaN values")

    def test_mock_server_error(self):
        """Generate error request and check that mock return json error.
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        error_response = env.mock_server().generate_error_response()
        # Make a request that returns the error response
        response = env.mock_server().request('/wrong-endpoint', HttpMethod.GET.value)
        # Check that the response status code is not success
        self.assertNotEqual(response.status_code, 200, "Response status code is not an error")
        # Check that the response contains the error message
        self.assertEqual(error_response, response.json_data, "Response does not contain the expected error message")

    def test_reset_env(self):
        """
        Test reset env to initial state , where initial state root observation for rest api
        for redfish it /redfish/v1/
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

        # get observation for initial rest
        rest_api, one_hot_vector = dataset.entry_rest_api()
        response = env.mock_server().request(rest_api, HttpMethod.GET.value)

        self.assertEqual(response.status_code, 200, "Response status code is not success")
        response_data = json.loads(response.json_data)
        expected_keys = [
            '@odata.context',
            '@odata.id',
            '@odata.type'
        ]
        for key in expected_keys:
            self.assertIn(key, response_data, f"Response data is missing key: {key}")
        self.assertIsInstance(response_data, dict, "Response data is not a valid JSON object")

        embeddings = env.encoder.encode(response.json_data)
        expected_shape = env.observation_space.shape
        self.assertEqual(embeddings.shape, expected_shape, "Embeddings shape mismatch")
        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings is not of type torch.Tensor")
        self.assertIsInstance(embeddings, torch.Tensor, "Embeddings is not of type torch.Tensor")
        self.assertTrue(embeddings.dtype in [torch.float16, torch.float32],
                        "Embeddings data type is not float16 or float32")
        self.assertFalse(torch.isnan(embeddings).any(), "Embeddings contain NaN values")

        obs, info = env.reset()
        self.assertEqual(env.step_count, 0, "Step count mismatch")
        self.assertEqual(embeddings.shape, obs.shape, "Embedding shape does not match observation shape")
        self.assertTrue(torch.allclose(embeddings, obs), "Embeddings do not match observation")

    def test_basic_step(self):
        """
        Test basic step
        :return:
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

        obs, info = env.reset()
        rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
        next_observation, reward, done, terminated, info = env.step(action)

        #
        print(f"executing rest api {rest_api}")
        self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, False, "single get should not set done")
        self.assertEqual(terminated, False, "single get should not set terminated")
        self.assertEqual(env.step_count, 1, "step count increase")
        print(f"reward {0.1}")
        print(f"next_observation {next_observation}")

    def test_basic_n_step(self):
        """
        Test basic step
        :return:
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

        obs, info = env.reset()

        # seq of get ( normally that should fail )
        for n in range(0, 2):
            rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
            http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
            action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
            next_observation, reward, done, terminated, info = env.step(action)

            self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
            self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
            self.assertEqual(next_observation.shape, obs.shape,
                             "next observation shape does not match observation shape")
            self.assertEqual(done, False, "single get should not set done")
            self.assertEqual(terminated, False, "single get should not set terminated")
            self.assertEqual(env.step_count, n + 1, "Step count does not increase")

    def test_basic_step_reward_then_negative(self):
        """
        Test basic step
        :return:
        """
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset
        )

        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

        obs, info = env.reset()

        # execute and collect reward for get
        # seq of get ( normally that should fail )
        total_reward = 0.0

        for n in range(0, 2):
            rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
            http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
            action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
            next_observation, reward, done, terminated, info = env.step(action)
            total_reward += reward
            self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
            self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
            self.assertEqual(next_observation.shape, obs.shape,
                             "next observation shape does not match observation shape")
            self.assertEqual(done, False, "single get should not set done")
            self.assertEqual(terminated, False, "single get should not set terminated")
            self.assertEqual(env.step_count, n + 1, "Step count does not increase")

        # generate action for negative reward
        rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        # select something API doesn't support.
        unsupported_method = next(method for method in RestApiEnv.METHOD_MAPPING if method not in supported_method)
        print(f"supported method {supported_method} selected {unsupported_method}")
        http_method_one_hot = RestApiEnv.encode_rest_api_method(unsupported_method)
        action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

        next_observation, reward, done, terminated, info = env.step(action)
        # we should get -0.2 reward, and we don't terminate
        self.assertEqual(reward, -0.2, "Reward is not equal to -0.2")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must be the same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, False, "done should be True")
        self.assertEqual(terminated, False, "terminated should be False")
        self.assertEqual(env.step_count, 3, "Step count does not increase")

        # choose another wrong
        next_observation, reward, done, terminated, info = env.step(action)
        # we should get -0.2 reward, and we don't terminate
        self.assertEqual(reward, -0.2, "Reward is not equal to -0.2")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must be the same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, False, "done should be True")
        self.assertEqual(terminated, False, "terminated should be False")
        self.assertEqual(env.step_count, 4, "Step count does not increase")

    def test_basic_passive_reward_to_max_traj(self):
        """
        Test basic step
        :return:
        """

        max_episode_len = 5

        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset,
            max_episode=max_episode_len
        )

        self.assertEqual(env.max_steps, max_episode_len, "Number of actions mismatch")
        expected_num_actions = dataset.num_actions + len(RestApiEnv.METHOD_MAPPING)
        self.assertEqual(env.action_space.shape[0], expected_num_actions, "Number of actions mismatch")
        self.assertEqual(len(env.mock_server().responses), dataset.num_actions, "Number of REST APIs mismatch")

        obs, info = env.reset()

        # execute and collect reward for get
        # seq of get ( normally that should fail )
        total_reward = 0.0

        for n in range(0, max_episode_len - 1):
            rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
            http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
            action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
            next_observation, reward, done, terminated, info = env.step(action)
            total_reward += reward
            self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
            self.assertTrue(torch.allclose(env.last_observation, next_observation),
                            "last obs must same as next")
            self.assertEqual(next_observation.shape, obs.shape,
                             "next observation shape does not match observation shape")
            self.assertEqual(done, False, "single get should not set done")
            self.assertEqual(terminated, False, "single get should not set terminated")
            self.assertEqual(env.step_count, n + 1, "Step count does not increase")

        # Last episode
        rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
        next_observation, reward, done, terminated, info = env.step(action)
        self.assertEqual(reward, -1.0, "Last reward should be -1.0")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, True, "last episode should set done to True")
        self.assertEqual(terminated, True, "last episode should set terminated to True")
        self.assertEqual(env.step_count, max_episode_len, "Step count does not increase")

    def test_goal_reached(self):
        """Test execute two action last action is goal action.
        :return:
        """

        max_episode_len = 5
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        # sample some random goal and choose method
        # this our goal
        rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        goal_action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset,
            max_episode=max_episode_len,
            goal=goal_action
        )

        self.assertTrue(torch.allclose(env.goal_action, goal_action), "last obs must same as next")
        obs, info = env.reset()

        # execute and collect reward for get
        # seq of get ( normally that should fail )
        total_reward = 0.0

        for n in range(0, 2):
            rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
            http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
            action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
            next_observation, reward, done, terminated, info = env.step(action)
            total_reward += reward
            self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
            self.assertTrue(torch.allclose(env.last_observation, next_observation),
                            "last obs must same as next")
            self.assertEqual(next_observation.shape, obs.shape,
                             "next observation shape does not match observation shape")
            self.assertEqual(done, False, "single get should not set done")
            self.assertEqual(terminated, False, "single get should not set terminated")
            self.assertEqual(env.step_count, n + 1, "Step count does not increase")

        # execute action with goal for this task episode
        next_observation, reward, done, terminated, info = env.step(goal_action)
        self.assertEqual(reward, 1.0, "Last reward should be 1.0")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, True, "last episode should set done to True")
        self.assertEqual(terminated, False, "last episode should set terminated to True")
        self.assertEqual(env.step_count, 3, "Step count does not increase")

    def test_reset_goal_reached(self):
        """Test reset env with goal and execute two action last action is goal action.
        Expectation agent should receive reward 1.0
        :return:
        """

        max_episode_len = 5
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        # sample some random goal and choose method
        # this our goal
        rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        goal_action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset,
            max_episode=max_episode_len)

        obs, info = env.reset(goal=goal_action)
        self.assertTrue(torch.allclose(env.goal_action, goal_action), "last obs must same as next")

        total_reward = 0.0
        for n in range(0, 2):
            rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
            http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
            action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
            next_observation, reward, done, terminated, info = env.step(action)
            total_reward += reward
            self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
            self.assertTrue(torch.allclose(env.last_observation, next_observation),
                            "last obs must same as next")
            self.assertEqual(next_observation.shape, obs.shape,
                             "next observation shape does not match observation shape")
            self.assertEqual(done, False, "single get should not set done")
            self.assertEqual(terminated, False, "single get should not set terminated")
            self.assertEqual(env.step_count, n + 1, "Step count does not increase")

        # execute action with goal for this task episode
        next_observation, reward, done, terminated, info = env.step(goal_action)
        self.assertEqual(reward, 1.0, "Last reward should be 1.0")
        self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
        self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        self.assertEqual(done, True, "last episode should set done to True")
        self.assertEqual(terminated, False, "last episode should set terminated to True")
        self.assertEqual(env.step_count, 3, "Step count does not increase")

    def test_mutate_state(self):
        """Test reset env with goal and execute action last action is goal action.
        Expectation agent should receive reward 1.0
        :return:
        """

        max_episode_len = 5
        args = shared_main()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args, only_tokenizer=False)
        json_directory_path = os.path.expanduser(args.raw_data_dir)
        dataset = JSONDataset(
            raw_json_directory_path=json_directory_path,
            dataset_dir=f"{package_dir}/datasets",
            verbose=True, tokenizer=tokenizer)

        # sample some random goal and choose method
        # this our goal
        # rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        # http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        # goal_action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)

        env = RestApiEnv(
            args=args, model=model,
            tokenizer=tokenizer,
            discovered_rest_api=dataset,
            max_episode=max_episode_len)

        # register handler, which is our final goal state that we mutate.
        goal_reset_api = register_reset_goal(env.mock_server())
        # rest_action = dataset.action_to_rest[goal_reset_api]
        # print(rest_action)

        for a in dataset.action_space:
            if "ComputerSystem.Reset" in a:
                print(f"a {a}")
                print(f"a val {dataset.action_space[a]}")

        for a in dataset.action_to_rest:
            if "ComputerSystem.Reset" in a:
                print(f"a {a}")
                print(f"a val {dataset.action_to_rest[a]}")

        # one_hot_action = dataset.action(goal_reset_api)
        # print(f"one hot action {one_hot_action}")

        # obs, info = env.reset(goal=goal_action)
        #
        # self.assertTrue(torch.allclose(env.goal_action, goal_action), "last obs must same as next")
        #
        # total_reward = 0.0
        # for n in range(0, 2):
        #     rest_api, supported_method, one_hot_action = dataset.sample_rest_api()
        #     http_method_one_hot = RestApiEnv.encode_rest_api_method("GET")
        #     action = RestApiEnv.concat_rest_api_method(one_hot_action, http_method_one_hot)
        #     next_observation, reward, done, terminated, info = env.step(action)
        #     total_reward += reward
        #     self.assertEqual(reward, 0.1, "Reward is not equal to 0.1")
        #     self.assertTrue(torch.allclose(env.last_observation, next_observation),
        #                     "last obs must same as next")
        #     self.assertEqual(next_observation.shape, obs.shape,
        #                      "next observation shape does not match observation shape")
        #     self.assertEqual(done, False, "single get should not set done")
        #     self.assertEqual(terminated, False, "single get should not set terminated")
        #     self.assertEqual(env.step_count, n + 1, "Step count does not increase")
        #
        # # execute action with goal for this task episode
        # next_observation, reward, done, terminated, info = env.step(goal_action)
        # self.assertEqual(reward, 1.0, "Last reward should be 1.0")
        # self.assertTrue(torch.allclose(env.last_observation, next_observation), "last obs must same as next")
        # self.assertEqual(next_observation.shape, obs.shape, "next observation shape does not match observation shape")
        # self.assertEqual(done, True, "last episode should set done to True")
        # self.assertEqual(terminated, False, "last episode should set terminated to True")
        # self.assertEqual(env.step_count, 3, "Step count does not increase")


if __name__ == "__main__":
    unittest.main()
