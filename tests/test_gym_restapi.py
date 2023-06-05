import torch
import unittest

from igc.envs.rest_gym_env import RestApiEnv


class TestRestApiEnv(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
