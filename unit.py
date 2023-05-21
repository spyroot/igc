import os
import unittest
from transformers import GPT2Tokenizer

from llm_trainer import JSONDataset


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        # set up the tokenizer and dataset here
        directory_path = os.path.expanduser("~/.json_responses")
        self.dataset = JSONDataset(directory_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_tokenizer(self):
        # iterate through your dataset
        for sample in self.dataset.data:
            # tokenize and then decode the tokenized output
            inputs = self.tokenizer(sample['respond'], return_tensors='pt')
            decoded_output = self.tokenizer.decode(inputs['input_ids'][0])
            # check if the decoded output matches the original input
            self.assertEqual(decoded_output, sample['respond'])


if __name__ == '__main__':
    unittest.main()
