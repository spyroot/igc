import math
import unittest

from igc.modules.base.igc_llm_base_module import LlmBaseModule
from igc.modules.base.igc_llm_metrics_type import MetricType
from igc.modules.base.prompt_types import PromptType


class TestExactMatch(unittest.TestCase):

    def test_exact_match(self):
        predictions = ["This is a test",
                       "The sky is blue",
                       "Movie matrix"]
        targets = ["This is a test",
                   "The sky is blue",
                   "Movie matrix"]

        prompt_types = PromptType.get_all_types()
        expected_results = [1.0, 1.0, 1.0, 1.0, 1.0]

        for prompt_type, expected_result in zip(prompt_types, expected_results):
            result = LlmBaseModule.performance_metric(predictions, targets, MetricType.EXACT_MATCH, prompt_type)
            error_msg = f"Prompt type: {prompt_type.value}, Expected: {expected_result}, Actual: {result}"
            self.assertEqual(expected_result, result, error_msg)

    def test_performance_metric_question(self):
        """
        Test case for performance metric with prompt type "question".

        :return:
        """
        # 3 prediction space before , after and A prefix
        predictions = [
            "A: France",
            "A: eight planets, 146 moons ",
            "A:Jane Austen"
        ]

        expected_answers = [
            "France",
            "eight planets, 146 moons",
            "Jane Austen"
        ]

        expected_result = 1.0
        result = LlmBaseModule.performance_metric(predictions,
                                                  expected_answers,
                                                  MetricType.EXACT_MATCH,
                                                  PromptType.QUESTION,
                                                  prefix_to_remove="A:")

        error_msg = f"Prompt type: {PromptType.QUESTION.value}, Expected: {expected_result}, Actual: {result}"
        self.assertEqual(expected_result, result, error_msg)

    def test_question_normalization(self):
        """

        :return:
        """
        predictions = [
            "Q: What is the capital of France?",
            "Q: How many planets are there in the solar system?",
            "Q: Who wrote the novel 'Pride and Prejudice'?"
        ]
        expected_normalized_questions = [
            ": What is the capital of France?",
            ": How many planets are there in the solar system?",
            ": Who wrote the novel 'Pride and Prejudice'?"
        ]

        normalized_questions = [
            LlmBaseModule._normalize(p, "Q") for p in predictions
        ]

        for q, expected_q in zip(normalized_questions, expected_normalized_questions):
            self.assertEqual(expected_q, q)

    def test_rouge_metric_negative(self):
        """

        :return:
        """
        predictions = ["This is a test",
                       "The",
                       "z"]
        targets = ["123",
                   "456",
                   "review"]

        expected_result = 0.0
        result = LlmBaseModule.performance_metric(predictions,
                                                  targets,
                                                  MetricType.ROUGE,
                                                  PromptType.CUSTOM,
                                                  callback=LlmBaseModule.compute_rouge_metric)

        error_msg = f"Prompt type: {PromptType.CUSTOM.value}, Expected: {expected_result}, Actual: {result}"
        self.assertEqual(expected_result, result, error_msg)

    def test_rouge_metric_positive(self):
        """

        :return:
        """

        predictions = ["The cat is on the mat"]
        targets = ["The cat is sitting on the mat"]

        expected_result = 0.92
        result = LlmBaseModule.performance_metric(predictions,
                                                  targets,
                                                  MetricType.ROUGE,
                                                  PromptType.CUSTOM,
                                                  callback=LlmBaseModule.compute_rouge_metric)

        error_msg = f"Prompt type: {PromptType.CUSTOM.value}, Expected: {expected_result}, Actual: {result}"
        self.assertTrue(math.isclose(expected_result, result, rel_tol=1e-2), error_msg)


if __name__ == "__main__":
    unittest.main()
