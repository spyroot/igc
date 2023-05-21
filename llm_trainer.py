"""

we pre-train the model using the shift generation method:

In this step, we train the model to predict the next token in
the sequence by shifting the input sequence and using a masked language modeling objective.
This helps the model learn the language patterns and dependencies.

Fine-tune the pretrained model using the random span method: After pre-training, you can further fine-tune the model
using the random span method. In this case, you replace random spans of text with a single mask token and train the
model
to predict the original text. This helps the model learn to fill in missing information and generate coherent text.

By combining these two methods, you can benefit from both the language modeling capabilities learned through shift
generation and the ability to generate missing text using the random span method. This two-step process
allows the model to capture a broader range of language understanding and generation capabilities.
"""
import os
import numpy as np
import torch
import argparse
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from ds.RedfishDataset import JSONDataset
from torch_utils import print_gpu_utilization
import evaluate
from torch.utils.data import random_split

#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set to the index of the GPU you want to use

torch.cuda.empty_cache()


class LLmTrainer:
    def __init__(self, cmd):
        """

        :param cmd:
        """
        super().__init__()
        self.cmd = cmd
        self.pad_token_id = None
        self.collate_fn = self.collate_input_shift_fn
        self.directory_path = os.path.expanduser("~/.json_responses")


    @staticmethod
    def dataset_checker():
        """
        :return:
        """
        directory_path = os.path.expanduser("~/.json_responses")
        dataset = JSONDataset(directory_path, verbose=False)
        for data_point in dataset:
            rest_call = dataset.action(data_point["label"])
            print("Rest recovered:", rest_call)
            print("Rest original:", data_point["rest_api"])
            print("Rest original:", data_point["label"])

    def collate_random_span_fn(self, batch):
        """
        :param batch:
        :return:
        """
        input_ids = torch.cat([item['input_ids'].squeeze(1) for item in batch])
        attention_mask = torch.cat([item['attention_mask'].squeeze(1) for item in batch])
        labels = input_ids.clone()  # Make a copy of input_ids as labels

        # Mask a random span of text in each input
        for i in range(len(batch)):
            input_length = input_ids[i].size(0)
            # Randomly choose start position for masking
            mask_start = torch.randint(1, input_length - 1, (1,)).item()
            # Randomly choose end position for masking
            mask_end = mask_start + torch.randint(1, input_length - mask_start, (1,)).item()

            # Replace the selected span with pad_token_id
            input_ids[i, mask_start:mask_end] = self.pad_token_id
            # Set the labels to the original span
            labels[i, mask_start:mask_end] = input_ids[i, mask_start:mask_end]

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    @staticmethod
    def print_batch_shapes(input_ids, attention_mask, labels):
        """
        :param input_ids:
        :param attention_mask:
        :param labels:
        :return:
        """
        print(f"shapes "
              f"input:{input_ids.shape} "
              f"mask:{attention_mask.shape} "
              f"label:{labels.shape}")

    def collate_input_shift_fn(self, batch):
        """
        :param batch:
        :return:
        """
        input_ids = torch.cat(
            [item['input_ids'].squeeze(1) for item in batch]
        )

        attention_mask = torch.cat(
            [item['attention_mask'].squeeze(1) for item in batch]
        )

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # shifting
        labels = input_ids[:, 1:].clone()
        labels[:, -1] = -100  # ignore index
        mask = torch.tensor(input_ids == self.pad_token_id)
        labels = labels.masked_fill(mask, -100)

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    @staticmethod
    def compute_metrics(eval_prediction, ):
        """
        :param eval_prediction:
        :return:
        """
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def run(self):
        """
        :return:
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        directory_path = os.path.expanduser("~/.json_responses")
        dataset = JSONDataset(directory_path, default_tokenize=self.cmd.model_type, verbose=False)
        pad_token_id = dataset.tokenizer.pad_token

        train_size = int(len(dataset) * 0.8)
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        # labels = inputs.input_ids.detach().clone()
        # model = GPT2Model.from_pretrained('gpt2-xl').to(device)
        model = GPT2LMHeadModel.from_pretrained(self.cmd.model_type).to(device)
        print_gpu_utilization()

        # model.to(device)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        num_train_epochs = 2
        default_args = {
            "output_dir": "tmp",
            "evaluation_strategy": "steps",
            "num_train_epochs": 10,
            "log_level": "error",
            "report_to": "none",
            "do_train": True,
        }

        # the Trainer to evaluate during training by setting evaluation_
        # strategy to either "steps" (evaluate every eval_steps)
        # or "epoch" (evaluate at the end of each epoch).

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            fp16=True,
            **default_args,
            local_rank=cmd.local_rank)

        print(training_args)

        # Traineraloader = DataLoader(dataset, batch_size=4, collate_fn=my_collate_fn)
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=self.collate_fn,
                          compute_metrics=self.metrics)

        result = trainer.train()
        print(result)

        # callbacks=[LossMonitorCallback(logging_steps=10)]


def main(cmd):
    """

    :param cmd:
    :return:
    """
    llm_trainer = LLmTrainer(cmd)


def parse_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")

    parser.add_argument("--model_type",
                        type=str, default="gpt2-xl", choices=['gpt2-xl', 'gpt2-large', 'gpt2-medium'],
                        help="Model type.")

    parser.add_argument("--per_device_eval_batch_size",
                        type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")

    parser.add_argument("--learning_rate",
                        type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")

    parser.add_argument("--weight_decay",
                        type=float, default=0.0,
                        help="Weight decay to use.")

    parser.add_argument("--num_train_epochs",
                        type=int, default=3,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_train_steps",
                        type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")

    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_warmup_steps",
                        type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--output_dir",
                        type=str, default=None,
                        help="Where to store the model.")

    parser.add_argument("--seed",
                        type=int, default=None,
                        help="A seed for reproducible training.")

    parser.add_argument("--local-rank",
                        type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # deepspeed.init_distributed()

    print(args)
    main(args)
