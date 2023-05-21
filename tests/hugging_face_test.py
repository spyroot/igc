import unittest
import transformers
from typing import Any, Type

from huggingface_utils import model_hf_name, model_and_tokenizer


# Add your model_hf_name and model_and_tokenizer functions here...

class TestModelAndTokenizer(unittest.TestCase):
    def test_model_hf_name(self):
        self.assertEqual(model_hf_name('gpt2', 'small'), 'gpt2')
        self.assertEqual(model_hf_name('bert', 'large'), 'bert-large-uncased')
        self.assertEqual(model_hf_name('neo', 'large'), 'EleutherAI/gpt-neo-2.7B')
        with self.assertRaises(ValueError):
            model_hf_name('unknown', 'small')
        with self.assertRaises(ValueError):
            model_hf_name('gpt2', 'unknown')

    def test_model_and_tokenizer(self):
        model, tokenizer = model_and_tokenizer('gpt2', 'small', transformers.AutoModelForCausalLM)
        self.assertIsInstance(model, transformers.AutoModelForCausalLM)
        self.assertIsInstance(tokenizer, transformers.PreTrainedTokenizer)
        with self.assertRaises(ValueError):
            model_and_tokenizer('unknown', 'small', transformers.AutoModelForCausalLM)
        with self.assertRaises(ValueError):
            model_and_tokenizer('gpt2', 'unknown', transformers.AutoModelForCausalLM)
        with self.assertRaises(ValueError):
            model_and_tokenizer('gpt2', 'small', int)  # int is not a valid model class

if __name__ == '__main__':
    unittest.main()
