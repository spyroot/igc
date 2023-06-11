import unittest
import torch

from backups.llm_trainer import LLmTrainer


class LLmTrainerCollateTest(unittest.TestCase):
    def test_collate_random_span_fn(self):
        """

        :return:
        """
        # Initialize your class or create an instance of the class that contains the collate_random_span_fn function
        your_object = LLmTrainer(None)

        # Create sample data for testing
        batch = [
            {'input_ids': torch.tensor([[1, 2, 3, 4]]),
             'attention_mask': torch.tensor([[1, 1, 1, 1]])},

            {'input_ids': torch.tensor([[5, 6, 7, 8]]),
             'attention_mask': torch.tensor([[1, 1, 1, 1]])},

            {'input_ids': torch.tensor([[9, 10, 11, 12]]),
             'attention_mask': torch.tensor([[1, 1, 1, 1]])},
        ]

        # Call the collate_random_span_fn function
        result = your_object.collate_random_span_fn(batch)

        # Assert the output against your expected output
        self.assertIsInstance(result, dict)
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('labels', result)
        self.assertIsInstance(result['input_ids'], torch.Tensor)
        self.assertIsInstance(result['attention_mask'], torch.Tensor)
        self.assertIsInstance(result['labels'], torch.Tensor)
        # Add more assertions as needed to validate the output


if __name__ == '__main__':
    unittest.main()
