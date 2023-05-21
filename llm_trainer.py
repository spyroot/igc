"""

we pre-train the model using the shift generation method:

In this step, we train the model to predict the next token in
the sequence by shifting the input sequence and using a masked language modeling objective.
This helps the model learn the language patterns and dependencies.

Fine-tune the pretrained model using the random span method:
After pre-training, you can further fine-tune the model
using the random span method. In this case, you replace random spans of
text with a single mask token and train the
model to predict the original text. This helps the model learn to fill in missing
information and generate coherent text.

By combining these two methods, you can benefit from both the language modeling capabilities learned through shift
generation and the ability to generate missing text using the random span method. This two-step process
allows the model to capture a broader range of language understanding and generation capabilities.
"""
import argparse
import os

import evaluate
import numpy as np
import torch
from transformers import (
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer
)
from ds.redfish_dataset import JSONDataset
from huggingface_utils import LossMonitorCallback
from shared_main import shared_main
from torch.utils.data import random_split

from shared_torch_utils import cuda_memory

#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set to the index of the GPU you want to use

torch.cuda.empty_cache()


class LLmTrainer:
    def __init__(self, cmd: argparse.Namespace):
        """
        :param cmd:
        """
        super().__init__()

        self.cmd = cmd
        self.collate_fn = self.collate_input_shift_fn
        self.metrics_fn = self.compute_metrics
        # datasset
        self.directory_path = os.path.expanduser("~/.json_responses")
        self.dataset = JSONDataset(
            self.directory_path, verbose=False)
        self.pad_token_id = self.dataset.tokenizer.pad_token

        self.train_dataset = self.dataset
        self.eval_dataset = self.dataset

        # split dataset
        self.split_dataset()

    def split_dataset(self, ratio: float = 0.8):
        """
        :param ratio:
        :return:
        """
        train_size = int(len(self.dataset) * ratio)
        eval_size = len(self.dataset) - train_size
        self.train_dataset, self.eval_dataset = random_split(
            self.dataset, [train_size, eval_size])

    @staticmethod
    def dataset_checker(self):
        """Dataset checker
        :return:
        """
        for data_point in self.dataset:
            rest_call = self.dataset.action(data_point["label"])
            print("rest recovered:", rest_call)
            print("rest original:", data_point["rest_api"])
            print("rest original:", data_point["label"])

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
            # randomly choose start position for masking
            mask_start = torch.randint(1, input_length - 1, (1,)).item()
            # randomly choose end position for masking
            mask_end = mask_start + torch.randint(1, input_length - mask_start, (1,)).item()
            # replace the selected span with pad_token_id
            input_ids[i, mask_start:mask_end] = self.pad_token_id
            # set the labels to the original span
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
    def compute_metrics(eval_prediction):
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
        # labels = inputs.input_ids.detach().clone()
        # model = GPT2Model.from_pretrained('gpt2-xl').to(device)
        print(self.cmd)
        print(self.cmd)

        model = GPT2LMHeadModel.from_pretrained(
            self.cmd.model_type).to(self.cmd.device)
        if torch.cuda.is_available():
            cuda_memory()

        print(f"Dataset size train {len(self.train_dataset)} {len(self.eval_dataset)}")
        model.to(self.cmd.device)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        default_args = {
            #
            "output_dir": self.cmd.output_dir,
            "overwrite_output_dir": False,

            # batch
            "auto_find_batch_size": self.cmd.auto_batch,
            "per_device_train_batch_size": self.cmd.per_device_train_batch_size,
            "per_device_eval_batch_size": self.cmd.per_device_eval_batch_size,

            # eval and train steps
            # "evaluation_strategy": self.cmd.evaluation_strategy,
            "num_train_epochs": self.cmd.num_train_epochs,
            "max_steps": self.cmd.max_steps,
            "report_to": self.cmd.report_to,

            # train and eval
            "do_train": self.cmd.train,
            "do_eval": self.cmd.eval,

            # mpc
            "use_mps_device": True,

            # data types
            # "bf16": self.cmd.bf16,
            "fp16": torch.cuda.is_available() and self.cmd.fp16 or False,
            "fp16_opt_level": torch.cuda.is_available() and self.cmd.fp16_opt_level or False,
            "half_precision_backend": torch.cuda.is_available() and self.cmd.half_precision_backend or False,
            "bf16_full_eval": torch.cuda.is_available() and self.cmd.bf16_full_eval or False,
            "fp16_full_eval": torch.cuda.is_available() and self.cmd.fp16_full_eval or False,
            "tf32": torch.cuda.is_available() and self.cmd.tf32 or False,

            # saving
            "save_strategy": self.cmd.save_strategy,
            "save_steps": self.cmd.save_steps,
            "save_total_limit": self.cmd.save_total_limit,
            "save_on_each_node": self.cmd.save_on_each_node,

            # seed
            "seed": self.cmd.seed,
            "data_seed": self.cmd.data_seed,

            # logging
            "log_level": self.cmd.llm_log_level,
            "logging_steps": self.cmd.log_steps,
            "log_on_each_node": self.cmd.log_on_each_node,
            "logging_dir": self.cmd.log_dir,

            # # distribute
            # "ddp_backend": self.cmd.ddp_backend if self.cmd.local_rank != -1 else None,
            # "sharded_ddp": self.cmd.sharded_ddp if self.cmd.local_rank != -1 else None,

            # deepspeed
            "deepspeed": self.cmd.deepspeed,

            "dataloader_pin_memory": self.cmd.dataloader_pin_memory,

            # optimizer, gradient_checkpointing
            "gradient_checkpointing": self.cmd.gradient_checkpointing,
            "gradient_accumulation_steps": self.cmd.gradient_accumulation_steps,
        }

        # the Trainer to evaluate during training by setting evaluation_
        # strategy to either "steps" (evaluate every eval_steps)
        # or "epoch" (evaluate at the end of each epoch).

        training_args = TrainingArguments(
            **default_args,
            local_rank=self.cmd.local_rank)

        print(training_args)

        # Traineraloader = DataLoader(dataset, batch_size=4, collate_fn=my_collate_fn)
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=self.train_dataset,
                          eval_dataset=self.eval_dataset,
                          data_collator=self.collate_fn,
                          compute_metrics=self.metrics_fn,
                          callbacks=[LossMonitorCallback(logging_steps=10)])

        result = trainer.train(use_cache=False)
        print(result)
        trainer.save_model(self.cmd.output_dir)


def main():
    """
    :param cmd:
    :return:
    """
    args = shared_main()
    llm_trainer = LLmTrainer(args)
    llm_trainer.run()


if __name__ == '__main__':
    main()
