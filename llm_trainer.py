import os
import json

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import hashlib
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel


def model2hfname(model: str) -> str:
    return {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-med': 'prajjwal1/bert-medium',
        'small': 'gpt2',
        'med': 'gpt2-medium',
        'large': 'gpt2-large',
        'full': 'gpt2-xl',
        'gpt2-sm': 'gpt2',
        'gpt2-med': 'gpt2-medium',
        'gpt2-lg': 'gpt2-large',
        'gpt2': 'gpt2-xl',
        'neo': 'EleutherAI/gpt-neo-2.7B',
    }[model]


class JSONDataset(Dataset):
    def __init__(self, directory_path):
        """

        """
        self.data = []
        self.directory_path = directory_path
        self.load_json_files()

    @staticmethod
    def convert_file_name(file_name):
        converted_name = file_name.replace("_", "/")
        return converted_name

    @staticmethod
    def calculate_hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def load_json_files(self):
        """
        """
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.directory_path, file_name)
                with open(file_path, "r") as json_file:
                    rest_api = self.convert_file_name(file_name)
                    json_lines = json_file.read()
                    hash_value = self.calculate_hash(rest_api)
                    self.data.append({
                        "request_hash": hash_value,
                        "request": rest_api,
                        "respond": json_lines,
                    })

    @staticmethod
    def preprocess_sample(sample):
        return sample

    @staticmethod
    def preprocess_json_data(json_data):
        """
        """
        preprocessed_data = []
        for item in json_data:
            preprocessed_sample = JSONDataset.preprocess_sample(item)
            preprocessed_data.append(preprocessed_sample)
        return preprocessed_data

    def convert_to_one_hot(self, text):
        """
        """
        hash_value = hashlib.sha256(text.encode()).hexdigest()
        print(f"hash_value: {hash_value}")
        binary_value = bin(int(hash_value, 16))[2:]  # Convert hash to binary string
        one_hot_tensor = torch.tensor(list(binary_value)).float()
        return one_hot_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#
# json_responses_dir = "json_responses"
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
#
# training_data = []
#
# # Iterate over the JSON files in the directory
# for filename in os.listdir(json_responses_dir):
#     if filename.endswith(".json"):
#         filepath = os.path.join(json_responses_dir, filename)
#         with open(filepath, "r") as file:
#             json_data = json.load(file)
#             # Process the JSON data as needed and append to the training data list
#
# # Tokenize and encode the training data
# encoded_inputs = tokenizer(training_data, truncation=True, padding=True, return_tensors="pt")
#
# # Train the GPT model with the encoded inputs
# model.train_model(encoded_inputs["input_ids"])
#
# def train_gpt():
#     pass
#
#
# def build_dataset(tokenizer, json_dir):
#     """
#     """
#     preprocessed_data = []
#
#     for file_name in os.listdir(json_responses_dir):
#         if file_name.endswith(".json"):
#             file_path = os.path.join(json_responses_dir, file_name)
#             with open(filepath, "r") as json_file:
#                     json_data = json.load(json_file)
#
#     tokenized_data = tokenizer(preprocessed_data, truncation=True, padding=True)
#     dataset = Dataset.from_dict(
#         {
#             "input_ids": tokenized_data["input_ids"],
#             "attention_mask": tokenized_data["attention_mask"],
#         }
#     )
#
#     dataset = dataset.map(
#         lambda example: {
#             "input_ids": Value("int64"),
#             "attention_mask": Value("int64"),
#         },
#         features=Features(),
#     )
#
#

def get_model_and_tokenizer(model: str, Cls, **model_kwargs):
    """

    :param model:
    :param Cls:
    :param model_kwargs:
    :return:
    """
    hf_model_name = model2hfname(model)
    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.pad_token = '[PAD]'
    return m, tok


def main():
    """
    """
    # ses_dir = "json_responses"
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt3")
    # model = GPT2LMHeadModel.from_pretrained("gpt3")

    directory_path = os.path.expanduser("~/.json_responses")
    dataset = JSONDataset(directory_path)

    # tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer([sample['respond'] for sample in dataset.data],
                       truncation=True, padding=True, return_tensors='pt')

    print(inputs)
    labels = inputs.input_ids.detach().clone()
    model = GPT2Model.from_pretrained('gpt2-xl')

    print(f"Tokens: {inputs['input_ids'][0]}")

    # Decode the first sample back to text
    print(f"Decoded: {tokenizer.decode(inputs['input_ids'][0])}")

    #
    # training_args = TrainingArguments(
    #     output_dir="./results",             # The output directory
    #     num_train_epochs=3,                 # Total number of training epochs
    #     per_device_train_batch_size=16,     # Batch size per device during training
    #     warmup_steps=500,                   # Number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,                  # Strength of weight decay
    #     logging_dir='./logs',               # Directory for storing logs
    #     logging_steps=10,                   # How often to print logs
    # )
    #
    # trainer = Trainer(
    #     model=model,                         # The instantiated, transformers model to be trained
    #     args=training_args,                  # Training arguments, defined above
    #     train_dataset=inputs,                # Training dataset
    #     eval_dataset=inputs,                 # Evaluation dataset
    #     data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
    #                                 'attention_mask': torch.stack([f[1] for f in data]),
    #                                 'labels': torch.stack([f[0] for f in data])}
    # )
    #
    # # Train the model
    # trainer.train()


if __name__ == '__main__':
    main()
