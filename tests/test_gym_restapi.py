import torch
import unittest

from igc.envs.rest_gym_env import RestApiEnv


class TestRestApiEnv(unittest.TestCase):
    def test_extract_action_method(self):
        """

        :return:
        """
        input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0])
        rest_api_one_hot, method = RestApiEnv.extract_action_method(input_tensor)
        # Test the extracted values
        expected_rest_api_one_hot = torch.tensor([0.1, 0.2, 0.3, 0.4])
        expected_method = "POST"

        self.assertTrue(torch.allclose(rest_api_one_hot, expected_rest_api_one_hot))
        self.assertEqual(method, expected_method)


if __name__ == "__main__":
    unittest.main()
