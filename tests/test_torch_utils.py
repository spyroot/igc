import unittest
import torch
from transformers import GPT2Tokenizer
from igc.shared.shared_torch_utils import shift_and_mask


class YourClassTestCase(unittest.TestCase):

    def test_1d_shifts_and_mask(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        pad_token_id = tokenizer.eos_token_id

        # 1D
        input_ids = torch.arange(1, 11).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        input_ids[:, -1] = 0
        attention_mask[:, -1] = 0

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        print("input", input_ids.shape, input_ids)
        print("mask", attention_mask.shape, attention_mask)
        input_ids, masks, labels = shift_and_mask(batch, pad_token_id)

        print("shift")
        print("input", input_ids.shape, input_ids)
        print("mask", masks.shape, masks)
        print("label", labels.shape, labels)

    def test_1d_shifts_eos_and_mask(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        pad_token_id = tokenizer.eos_token_id

        # 1D
        input_ids = torch.arange(1, 11).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        input_ids[:, -1] = pad_token_id
        attention_mask[:, -1] = 0

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        print("input", input_ids.shape, input_ids)
        print("mask", attention_mask.shape, attention_mask)
        input_ids, masks, labels = shift_and_mask(batch, pad_token_id)

        print("shift")
        print("input", input_ids.shape, input_ids)
        print("mask", masks.shape, masks)
        print("label", labels.shape, labels)

    def test_2d_shifts_and_mask(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        pad_token_id = tokenizer.eos_token_id

        # 1D
        input_ids = torch.arange(1, 21).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        input_ids[:, -1] = 0
        attention_mask[:, -1] = 0
        input_ids = input_ids.reshape(2, 10)
        attention_mask = attention_mask.reshape(2, 10)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        print("input", input_ids.shape, input_ids)
        print("mask", attention_mask.shape, attention_mask)
        input_ids, masks, labels = shift_and_mask(batch, pad_token_id)

        print("shift")
        print("input", input_ids.shape, input_ids)
        print("mask", masks.shape, masks)
        print("label", labels.shape, labels)

    def test_2d_narrow_shifts_and_mask(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        pad_token_id = tokenizer.eos_token_id

        # 1D
        input_ids = torch.arange(1, 5).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        input_ids[:, -1] = 0
        attention_mask[:, -1] = 0
        input_ids = input_ids.reshape(2, 2)
        attention_mask = attention_mask.reshape(2, 2)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        print("input", input_ids)
        print("mask", attention_mask)
        input_ids, masks, labels = shift_and_mask(batch, pad_token_id)

        print("shift")
        print(f"input {input_ids.shape} {input_ids}")
        print(f"mask  {masks.shape} {masks}")
        print(f"label  {labels.shape} {labels}")


if __name__ == '__main__':
    unittest.main()
