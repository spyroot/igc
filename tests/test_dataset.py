import os
import unittest
import torch
from transformers import GPT2Tokenizer
from ds.redfish_dataset import JSONDataset


class YourClassTestCase(unittest.TestCase):

    def test_create_chunks(self, model_name='gpt2'):
        """

        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        directory_path = os.path.expanduser("~/.json_responses")
        json_dataset = JSONDataset(
            directory_path,
            verbose=False,
            max_len=10,
            tokenizer=tokenizer,
            overlap=0,
            skip_creation=True)

        self.assertEqual(json_dataset._max_len, 10)
        self.assertEqual(json_dataset.is_empty(), True)
        self.assertEqual(json_dataset.chunk_overlap_size(), 0)

        #  input tensors, base case no chunks
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 1)
        chunk_input_ids, chunk_attention_mask = chunks[0]
        self.assertEqual(chunk_input_ids.shape, torch.Size([1, 5]))
        self.assertEqual(chunk_attention_mask.shape, torch.Size([1, 5]))

        # larger
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 2)

        chunk_input_ids_1, chunk_attention_mask_1 = chunks[0]
        self.assertEqual(chunk_input_ids_1.shape, torch.Size([1, 10]))
        self.assertEqual(chunk_attention_mask_1.shape, torch.Size([1, 10]))
        print("first chunk", chunk_input_ids_1)
        print("first chunk", attention_mask)

        chunk_input_ids_2, chunk_attention_mask_2 = chunks[1]
        self.assertEqual(torch.Size([1, 10]), chunk_input_ids_2.shape)
        self.assertEqual(torch.Size([1, 10]), chunk_attention_mask_2.shape)
        print("chunk_input_ids_2", chunk_input_ids_2)
        print("chunk_attention_mask_2", chunk_attention_mask_2)

    def test_create_50_percent_overlap(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        directory_path = os.path.expanduser("~/.json_responses")
        json_dataset = JSONDataset(
            directory_path,
            verbose=False,
            max_len=10,
            tokenizer=tokenizer,
            overlap=5,
            skip_creation=True)

        self.assertEqual(json_dataset._max_len, 10)
        self.assertEqual(json_dataset.is_empty(), True)
        self.assertEqual(json_dataset.chunk_overlap_size(), 5)

        #  base case input tensors
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 1)
        chunk_input_ids, chunk_attention_mask = chunks[0]
        self.assertEqual(chunk_input_ids.shape, torch.Size([1, 5]))
        self.assertEqual(chunk_attention_mask.shape, torch.Size([1, 5]))

        # larger then 11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 2)

        chunk_input_ids_1, chunk_attention_mask_1 = chunks[0]
        self.assertEqual(chunk_input_ids_1.shape, torch.Size([1, 10]))
        self.assertEqual(chunk_attention_mask_1.shape, torch.Size([1, 10]))
        print(f"{chunk_input_ids_1}, {chunk_attention_mask_1}")

        chunk_input_ids_2, chunk_attention_mask_2 = chunks[1]
        self.assertEqual(torch.Size([1, 10]), chunk_input_ids_2.shape)
        self.assertEqual(torch.Size([1, 10]), chunk_attention_mask_2.shape)
        print(f"{chunk_input_ids_2}, {chunk_attention_mask_2}")

        # tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[16, 17, 18, 19, 20, 21, 50256, 50256, 50256, 50256]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

        # larger then 11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(4, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)
            print(f"{chunk_input}, {chunk_mask}")

        # this should create
        # tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[26, 27, 28, 29, 30, 31, 50256, 50256, 50256, 50256]]), tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

        input_ids = torch.arange(1, 32).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(6, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)
            print(f"{chunk_input}, {chunk_mask}")

    def test_create_small_overlap(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        directory_path = os.path.expanduser("~/.json_responses")
        json_dataset = JSONDataset(
            directory_path,
            verbose=False,
            max_len=10,
            tokenizer=tokenizer,
            overlap=3,
            skip_creation=True)

        self.assertEqual(json_dataset._max_len, 10)
        self.assertEqual(json_dataset.is_empty(), True)
        self.assertEqual(json_dataset.chunk_overlap_size(), 3)

        #  base case input tensors
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 1)
        chunk_input_ids, chunk_attention_mask = chunks[0]
        self.assertEqual(chunk_input_ids.shape, torch.Size([1, 5]))
        self.assertEqual(chunk_attention_mask.shape, torch.Size([1, 5]))

        # larger then 11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 2)

        #
        chunk_input_ids_1, chunk_attention_mask_1 = chunks[0]
        self.assertEqual(chunk_input_ids_1.shape, torch.Size([1, 10]))
        self.assertEqual(chunk_attention_mask_1.shape, torch.Size([1, 10]))

        chunk_input_ids_2, chunk_attention_mask_2 = chunks[1]
        self.assertEqual(torch.Size([1, 10]), chunk_input_ids_2.shape)
        self.assertEqual(torch.Size([1, 10]), chunk_attention_mask_2.shape)
        print(f"{chunk_input_ids_2}, {chunk_attention_mask_2}")
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            print(f"{chunk_input}, {chunk_mask}")

        # larger then 11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(3, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)

        # this should create
        # tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[29, 30, 31, 50256, 50256, 50256, 50256, 50256, 50256, 50256]]), tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])

        input_ids = torch.arange(1, 32).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(5, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)

    def test_create_single_token_overlap(self, model_name='gpt2'):
        """
        :return:
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        directory_path = os.path.expanduser("~/.json_responses")
        json_dataset = JSONDataset(
            directory_path,
            verbose=False,
            max_len=10,
            tokenizer=tokenizer,
            overlap=1,
            skip_creation=True)

        self.assertEqual(json_dataset._max_len, 10)
        self.assertEqual(json_dataset.is_empty(), True)
        self.assertEqual(json_dataset.chunk_overlap_size(), 1)

        #  base case input tensors
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 1)
        chunk_input_ids, chunk_attention_mask = chunks[0]
        self.assertEqual(chunk_input_ids.shape, torch.Size([1, 5]))
        self.assertEqual(chunk_attention_mask.shape, torch.Size([1, 5]))

        # larger then 11
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(len(chunks), 2)
        chunk_input_ids_1, chunk_attention_mask_1 = chunks[0]
        self.assertEqual(chunk_input_ids_1.shape, torch.Size([1, 10]))
        self.assertEqual(chunk_attention_mask_1.shape, torch.Size([1, 10]))
        # tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[10, 11, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]]), [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]

        # larger then 11
        input_ids = torch.arange(1, 22).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(3, len(chunks))
        # tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[   19,    20,    21, 50256, 50256, 50256, 50256, 50256, 50256, 50256]]), tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0]])
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)

        input_ids = torch.arange(1, 32).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(4, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)

        # This should be
        # tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # tensor([[   28,    29,    30,    31,    32,    33,    34,    35,    36, 50256]]), tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
        input_ids = torch.arange(1, 37).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(4, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)

        # on boundary.
        input_ids = torch.arange(1, 38).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        chunks = json_dataset.create_chunks(input_ids, attention_mask)
        self.assertEqual(5, len(chunks))
        for i in range(len(chunks)):
            chunk_input, chunk_mask = chunks[i]
            self.assertEqual(torch.Size([1, 10]), chunk_input.shape)
            self.assertEqual(torch.Size([1, 10]), chunk_mask.shape)
            print(f"{chunk_input}, {chunk_mask}")


if __name__ == '__main__':
    unittest.main()
