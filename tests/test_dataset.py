import os
import tempfile
import unittest
import torch
from transformers import GPT2Tokenizer
from igc.ds.redfish_dataset import JSONDataset

j_data = {
    "@odata.context": "/redfish/v1/$metadata#MetricReportDefinitionCollection.MetricReportDefinitionCollection",
    "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions",
    "@odata.type": "#MetricReportDefinitionCollection.MetricReportDefinitionCollection",
    "Name": "MetricReportDefinitions",
    "Members": [
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/AggregationMetrics"
        },
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/CPUMemMetrics"
        },
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/CPURegisters"
        },
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/CPUSensor"
        },
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/FCPortStatistics"
        },
        {
            "@odata.id": "/redfish/v1/TelemetryService/MetricReportDefinitions/FCSensor"
        }
    ]
}


class DatasetTest(unittest.TestCase):

    def test_create_custom_dir(self):
        """Test that we can create custom dir and all files and downloaded.
        :return:
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            real_tmp_dir = os.path.realpath(temp_dir)
            json_dataset = JSONDataset(
                real_tmp_dir,
                skip_creation=True,
                skip_download=True,
            )
            self.assertEqual(json_dataset._max_len, 1024)
            self.assertEqual(json_dataset._overlap, 256)
            self.assertEqual(json_dataset._verbose, False)
            self.assertEqual(json_dataset._recreate_dataset, False)

            self.assertEqual(real_tmp_dir, json_dataset.root_dir())
            self.assertEqual(f"{real_tmp_dir}/raw", json_dataset._default_raw_dir)
            self.assertEqual(f"{real_tmp_dir}/orig", json_dataset._default_original_dir)
            self.assertEqual(f"{real_tmp_dir}/raw", json_dataset.raw_dir())
            self.assertEqual(f"{real_tmp_dir}/orig", json_dataset.orig_dir())
            self.assertEqual(f"{real_tmp_dir}/tokenizer", json_dataset.tokenizer_dir())

            self.assertEqual(f"{real_tmp_dir}/raw/processed_dataset_gpt2.pt",
                             json_dataset._dataset_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/processed_masked_dataset_gpt2.pt",
                             json_dataset._dataset_masked_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/rest_api_to_method_gpt2.pt",
                             json_dataset._rest_api_to_method_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/rest_api_to_respond_gpt2.pt",
                             json_dataset._rest_api_to_respond_file_name)
            self.assertEqual(f"{real_tmp_dir}/igc.tar.gz",
                             json_dataset._dataset_tarball_name)
            self.assertEqual(f"{real_tmp_dir}/json_data.tar.gz",
                             json_dataset._dataset_json_tarball_name)
            self.assertEqual(f"{real_tmp_dir}/tokenizer.tar.gz",
                             json_dataset._dataset_tokenizer_tarball_name)

            self.assertEqual(json_dataset._mirrors, [
                {"spec": f'http://192.168.254.78/ds/dataset.json'},
                {"train_dataset": f'http://192.168.254.78/ds/igc.tar.gz'},
                {"json_data": f'http://192.168.254.78/ds/json_data.tar.gz'},
                {"tokenizer": f'http://192.168.254.78/ds/tokenizer.tar.gz'},
            ])

            self.assertEqual(json_dataset._resources, [
                ("dataset.json", "", "spec"),
                ("igc.tar.gz", "", "train_dataset"),
                ("json_data.tar.gz", "", "json_data"),
                ("tokenizer.tar.gz", "", "tokenizer"),
            ])

    def test_create_custom_json_folder(self):
        """Test that we can create custom dir and all files and downloaded.
        :return:
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            real_tmp_dir = os.path.realpath(temp_dir)
            with tempfile.TemporaryDirectory() as json_temp_dir:
                json_real_dir = os.path.realpath(json_temp_dir)
                json_dataset = JSONDataset(
                    real_tmp_dir,
                    skip_creation=True,
                    skip_download=True,
                    raw_json_directory_path=json_real_dir
                )

            self.assertEqual(json_dataset._max_len, 1024)
            self.assertEqual(json_dataset._overlap, 256)
            self.assertEqual(json_dataset._verbose, False)
            self.assertEqual(json_dataset._recreate_dataset, False)

            self.assertEqual(real_tmp_dir, json_dataset.root_dir())
            self.assertEqual(json_real_dir, json_dataset._unprocessed)

            self.assertEqual(f"{real_tmp_dir}/raw", json_dataset._default_raw_dir)
            self.assertEqual(f"{real_tmp_dir}/orig", json_dataset._default_original_dir)
            self.assertEqual(f"{real_tmp_dir}/raw", json_dataset.raw_dir())
            self.assertEqual(f"{real_tmp_dir}/orig", json_dataset.orig_dir())
            self.assertEqual(f"{real_tmp_dir}/tokenizer", json_dataset.tokenizer_dir())

            self.assertEqual(f"{real_tmp_dir}/raw/processed_dataset_gpt2.pt",
                             json_dataset._dataset_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/processed_masked_dataset_gpt2.pt",
                             json_dataset._dataset_masked_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/rest_api_to_method_gpt2.pt",
                             json_dataset._rest_api_to_method_file_name)
            self.assertEqual(f"{real_tmp_dir}/raw/rest_api_to_respond_gpt2.pt",
                             json_dataset._rest_api_to_respond_file_name)
            self.assertEqual(f"{real_tmp_dir}/igc.tar.gz",
                             json_dataset._dataset_tarball_name)
            self.assertEqual(f"{real_tmp_dir}/json_data.tar.gz",
                             json_dataset._dataset_json_tarball_name)
            self.assertEqual(f"{real_tmp_dir}/tokenizer.tar.gz",
                             json_dataset._dataset_tokenizer_tarball_name)

            self.assertEqual(json_dataset._mirrors, [
                {"spec": f'http://192.168.254.78/ds/dataset.json'},
                {"train_dataset": f'http://192.168.254.78/ds/igc.tar.gz'},
                {"json_data": f'http://192.168.254.78/ds/json_data.tar.gz'},
                {"tokenizer": f'http://192.168.254.78/ds/tokenizer.tar.gz'},
            ])

            self.assertEqual(json_dataset._resources, [
                ("dataset.json", "", "spec"),
                ("igc.tar.gz", "", "train_dataset"),
                ("json_data.tar.gz", "", "json_data"),
                ("tokenizer.tar.gz", "", "tokenizer"),
            ])

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

    def test_create_single_token_overlap(self, model_name='gpt2'):
        # Set up the necessary data
        target_key = "@odata.id"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # json_lines = json.dumps(j_data)
        # attention_mask = mask_specific_key_and_value(json_lines, target_key, tokenizer=tokenizer, debug=True)
        # print("Modified attention_mask:", attention_mask)


if __name__ == '__main__':
    unittest.main()
