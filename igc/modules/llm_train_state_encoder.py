"""
This class is used to train a goal extractor from input query.

Given input text provided by the user, or external system.
The goal is to extract a goal for the agent and parameters
that agent need used.

For example given input text: "Update raid with raid0"
The goal here update raid configuration and the
parameter is raid0.

In downstream task the goal encoded as one hot vector.
This what used to train RL agent.

Parameters just passed to agent. i.e. we don't train on parameters.

Author:Mus mbayramo@stanford.edu
"""
import argparse
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from igc.ds.redfish_dataset import JSONDataset
from igc.modules.base.igc_llm_base_module import LlmBaseModule
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.shared.shared_torch_builder import TorchBuilder
from accelerate import Accelerator
from torch.quantization import convert

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class LlmEmbeddingsTrainer(LlmBaseModule):
    """
    """
    def __init__(self,
                 name: str,
                 spec: argparse.Namespace,
                 ds: JSONDataset,
                 metric_logger: MetricLogger,
                 llm_model,
                 llm_tokenizer):
        """
        
        :param spec:
        :param ds: 
        :param metric_logger: 
        :param llm_model: 
        :param llm_tokenizer: 
        """
        super().__init__(name, spec, ds, metric_logger, llm_model, llm_tokenizer)

        self.is_quantize = False
        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size

        self.batch_log = 10
        self.shuffle = True
        self.num_workers = spec.num_workers
        self._default_mask_token = "@odata.id"

        self.optimizer = TorchBuilder.create_optimizer(
            spec.llm_optimizer,
            self.model,
            spec.llm_learning_rate,
            spec.llm_weight_decay,
            **vars(spec)
        )

        print(f"Rank {self.rank} creating "
              f"LlmEmbeddingsTrainer num epochs {self.num_epochs} "
              f"batch_size {self.batch_size} "
              f"dataset size {len(self.dataset)} "
              f"is overfit {self._overfit} "
              )

        self._mask_probability = 0.15
        self._best_validation_metric = float('-inf')

    @staticmethod
    def custom_collate_fn(samples):
        """Collate data before we pass to the model.
        :param samples:
        :return:
        """
        included_keys = ['input_ids', 'attention_mask']
        batch = {key: torch.stack([s[key] for s in samples]) for key in included_keys}
        return batch

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    @staticmethod
    def get_batch(src: Tensor, idx: int, chunk_size=35) -> Tuple[Tensor, Tensor]:
        """
        :param src: [full_seq_len, batch_size]
        :param idx
        :param chunk_size:
        :return: tuple (data, target),  shape [seq_len, batch_size], [seq_len * batch_size]
        """
        seq_len = min(chunk_size, len(src) - 1 - idx)
        data = src[idx:idx + seq_len]
        target = src[idx + 1:idx + 1 + seq_len].reshape(-1)
        return data, target

    def dataset_sampler(self):
        """
        :return:
        """
        if 'random_sampler_enabled' in self._trainer_args:
            sampler = RandomSampler(self.dataset) if self._trainer_args.random_sampler_enabled else None
            return sampler
        return None

    def validate(self, validation_dataset, accelerator: Accelerator):
        """ Perform validation on the emb llm model.

        :param accelerator:
        :param validation_dataset: Dataset for validation
        :return: Accuracy on the validation dataset
        """
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i, batch in enumerate(validation_dataset):
                labels = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.pad_token_id)
                labels = labels.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': labels.to(self.device)
                }

                masked_input_ids = batch["input_ids"].clone().detach()
                mask_indices = torch.rand(batch["input_ids"].shape) < self._mask_probability
                masked_input_ids[mask_indices] = -100

                outputs = self.model(**batch_inputs)
                predicted_tokens = torch.argmax(outputs.logits, dim=-1)

                predicted_masked_tokens = predicted_tokens[mask_indices]
                predicted_masked_tokens = predicted_masked_tokens.to(self.device)

                # compare predicted masked tokens with original tokens
                original_tokens = batch["input_ids"][mask_indices].to(self.device)
                correct_predictions += torch.sum(
                    torch.tensor(predicted_masked_tokens == original_tokens, dtype=torch.int)).item()

                total_predictions += original_tokens.numel()

        accuracy = correct_predictions / total_predictions * 100.0
        return accuracy

    def is_distributed(self):
        """
        :return:
        """
        return self.rank != -1

    def is_rank_zero(self):
        """
        :return:
        """
        return self.rank == -1 or self.rank == 0

    def train(self, overfit: Optional[bool] = True):
        """Train LLM model to map high level goal to redfish actions.

        For example
                "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"
        :param overfit:
        :return:
        """
        accelerator = Accelerator(device_placement=True)
        # if accelerator.is_main_process:
        #     time.sleep(2)
        # else:
        #     print("I'm waiting for the main process to finish its sleep...")
        # accelerator.wait_for_everyone()

        self.device = accelerator.device
        validation_accuracy = float('-inf')
        self.model.to(self.device)

        if self.checkpoint_dir is not None:
            last_epoch = self.load_checkpoint(self.checkpoint_dir)
        else:
            last_epoch = 0

        torch.cuda.empty_cache()

        self.model.train()
        train_dataset, eval_dataset = self.split_slice_dataset()

        sampler = self.dataset_sampler()
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      sampler=sampler,
                                      num_workers=self.num_workers,
                                      shuffle=self.shuffle,
                                      collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=self.batch_size,
                                     sampler=sampler,
                                     num_workers=self.num_workers,
                                     shuffle=False,
                                     collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        # self.model, self.optimizer, train_dataloader, eval_dataset = accelerator.prepare(
        #     [self.model, self.optimizer, train_dataloader, eval_dataset],
        #     device_placement=[True])

        self.model, self.optimizer, train_dataloader, eval_dataset = accelerator.prepare(
            [self.model, self.optimizer, train_dataloader, eval_dataset],
            device_placement=[True])

        if self.is_quantize:
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        total_batches = len(train_dataloader)
        dataset_size = len(train_dataset)
        calculated_total_batches = dataset_size // self.batch_size
        batch_log_frequency = round(32 * 0.2)

        if total_batches == calculated_total_batches:
            print(f"Staring training total_batches: {total_batches} "
                  f"train dataset size: {dataset_size} "
                  f"batch stats freq: {batch_log_frequency}.")

        for epoch in range(last_epoch, self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            batch_losses = np.zeros(total_batches)
            for i, batch in enumerate(train_dataloader):
                labels = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.pad_token_id)
                labels = labels.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                if epoch % 10 == 0:
                    for j in range(batch_inputs['input_ids'].size(0)):
                        batch_inputs["attention_mask"] = JSONDataset.mask_json_key_and_value(
                            batch_inputs, self._default_mask_token, self.tokenizer)

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                outputs = self.model(**batch_inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                accelerator.backward(loss)
                self.optimizer.step()

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()

                if self.is_quantize:
                    self.model.apply(torch.quantization.propagate_qconfig_)
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

                # calculate the progress percentage
                progress_percentage = int(round((num_batches + 1) / total_batches * 100))
                if (num_batches % batch_log_frequency == 0) or (num_batches == total_batches - 1):
                    print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Batch "
                          f"{num_batches + 1}/{total_batches} "
                          f"- Progress: {progress_percentage:.2f}% - Batch Loss mean: {batch_losses.mean():.4f}")
                    self.metric_logger.log_metric("llm_emb_batch_loss", batch_losses.mean(), epoch)

                num_batches += 1

            # validation
            if (epoch + 1) % 10 == 0:
                validation_accuracy = self.validate(eval_dataloader, accelerator)
                if self.rank == 0 or self.rank == -1:
                    self.metric_logger.log_metric("llm_emb_accuracy", validation_accuracy, epoch)
                print(f"Rank {self.rank} Epoch {epoch + 1} - Validation Accuracy: "
                      f"{validation_accuracy} Best: {self._best_validation_metric}")

            if num_batches > 0:
                average_loss = total_loss / num_batches
                if self.rank == 0 or self.rank == -1:
                    self.metric_logger.log_metric("llm_emb_epoch_loss", average_loss, epoch)
                print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

            # save best checkpoint
            if validation_accuracy > self._best_validation_metric and self.is_rank_zero():
                self._best_validation_metric = validation_accuracy
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(self.checkpoint_dir, epoch + 1)

        if self.is_quantize:
            self.model = convert(self.model)

        self.save_model(self.checkpoint_dir)

        del train_dataloader
        del eval_dataloader
        del self.optimizer
        del accelerator

        print("Embedding extractor training complete.")

