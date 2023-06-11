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
from typing import Optional

import nltk
import torch
from transformers import (
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer, GPT2Tokenizer
)
from igc.ds.redfish_dataset import JSONDataset
from igc.shared.huggingface_utils import LossMonitorCallback
from backups.llm_base_trainer import LLmBaseTrainer
from igc.shared.shared_main import shared_main

from igc.shared.shared_torch_utils import cuda_memory

#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set to the index of the GPU you want to use

torch.cuda.empty_cache()


class LLmTrainer(LLmBaseTrainer):
    def __init__(self, cmd: argparse.Namespace,
                 default_tokenize: Optional[str] = "gpt2-xl"):
        """
        :param cmd:
        """
        super().__init__()

        self.cmd = cmd
        self.collate_fn = self.collate_input_shift_fn
        self.metrics_fn = self.compute_metrics

        self.model = None
        self.tokenizer = GPT2Tokenizer.from_pretrained(default_tokenize)

        # dataset
        self.directory_path = os.path.expanduser("~/.json_responses")
        self.dataset = JSONDataset(
            self.directory_path, verbose=False)

        self.train_dataset = self.dataset
        self.eval_dataset = self.dataset

        # split dataset
        self.split_dataset()

    def run(self):
        """
        :return:
        """
        # labels = inputs.input_ids.detach().clone()
        # model = GPT2Model.from_pretrained('gpt2-xl').to(device)
        if self.cmd.gradient_checkpointing:
            self.model = GPT2LMHeadModel.from_pretrained(
                self.cmd.model_type, use_cache=False).to(self.cmd.device)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(
                self.cmd.model_type, use_cache=False).to(self.cmd.device)

        if torch.cuda.is_available():
            cuda_memory()

        print(f"Dataset size train {len(self.train_dataset)} {len(self.eval_dataset)}")
        self.model.to(self.cmd.device)
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
        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=self.train_dataset,
                          eval_dataset=self.eval_dataset,
                          data_collator=self.collate_fn,
                          compute_metrics=self.metrics_fn,
                          callbacks=[LossMonitorCallback(logging_steps=self.cmd.log_steps)])

        result = trainer.train()
        print(result)
        trainer.save_model(self.cmd.output_dir)

    # def train_extract_goal(self):
    #     """
    #
    #     :return:
    #     """
    #     # Initialize tokenizer and model
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #     model = GPT2ForSequenceClassification.from_pretrained('gpt2')
    #
    #     # Tokenize input data
    #     input_text = ["Change Raid to RAID0", ...]  # list of your sentences
    #     labels = ["RAID0", ...]  # list of your goals
    #     encodings = tokenizer(input_text, truncation=True, padding=True)
    #     labels_enc = tokenizer(labels, truncation=True, padding=True)
    #
    #     # Prepare optimizer
    #     optimizer = Adam(model.parameters())
    #
    #     # Training loop
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #
    #         # Forward pass
    #         outputs = model(**encodings)
    #
    #         # Compute loss
    #         loss = CrossEntropyLoss()(outputs.logits, labels_enc['input_ids'])
    #
    #         # Backward pass
    #         loss.backward()
    #         optimizer.step()

    def map_verb_to_http_method(verb):
        """
        :return:
        """
        if verb in ["update", "change", "modify"]:
            return "PUT"
        elif verb in ["create", "add"]:
            return "POST"
        elif verb in ["remove", "delete"]:
            return "DELETE"
        else:
            return "GET"  # default

    @staticmethod
    def extract_verb(input_text):
        """"Tokenize the sentence into words,
            perform part-of-speech tagging, filter out the word(s) tagged as 'VB' (verb, base form)
            return the first verb, or None if no verb was found
        """
        words = nltk.word_tokenize(input_text)
        tagged_words = nltk.pos_tag(words)
        verbs = [word for word, tag in tagged_words if tag.startswith('VB')]
        return verbs[0] if verbs else None

    # Prompt: "Goal: Update the RAID configuration to RAID0, Parameters: RAID Type: RAID0, Disk Count: 4"
    def emit_goal_prompt(self, goal, parameters):
        """
        :param goal:
        :param parameters:
        :return:
        """
        # Generate the prompt based on the goal and parameters
        return f"Goal: {goal}\nParameters: {parameters}\n"

    @torch.inference_mode()
    def do_sample(self,
                  input_ids: torch.Tensor,
                  stop_tokens: torch.Tensor,
                  max_tokens: int, debug=False) -> torch.Tensor:
        """Sample from the model using

        :param input_ids:
        :param stop_tokens:  A list of token ids that indicates when should stop.
        :param max_tokens:  Stop sampling if we've sampled this many tokens
        :param debug:
        :return:
        """

        initial_shape = input_ids.shape[1]
        for i in range(0, max_tokens):
            # (batch_size, sequence_length, config.vocab_size)
            next_tokens = self.model(input_ids).logits[:, -1, :].argmax(dim=-1)
            if stop_tokens is not None:
                condition = (next_tokens == stop_tokens)
                if torch.any(condition):
                    if debug:
                        print("stopping criteria condition", condition, input_ids.shape)
                    break

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        ids_squeezed = input_ids.squeeze()
        new_ids = ids_squeezed[initial_shape:]
        return new_ids

    def train_extact_goal(self):
        """

        :return:
        """
        for epoch in range(epochs):
            total_loss = 0

            for data in self._data["train_data"]:
                optimizer.zero_grad()

                # Encode the input and output sentences
                input_text = data["request"]
                verb = extract_verb(input_text)  # You need to write the extract_verb function
                http_method = map_verb_to_http_method(verb)

                # Generate the expected output sentence
                output_text = f"HTTP method: {http_method}"

                # Tokenize input and output
                inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
                labels = tokenizer(output_text, return_tensors='pt', truncation=True, padding=True).input_ids

                # Forward pass
                outputs = model(**inputs, labels=labels)

                # Compute loss
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}, Loss: {total_loss}")


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
