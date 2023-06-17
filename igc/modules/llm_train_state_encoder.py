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
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.quantization import convert
from torch.utils.data import DataLoader, RandomSampler

from igc.ds.redfish_masked_dataset import MaskingOption, MaskedJSONDataset
from igc.modules.base.igc_llm_base_module import LlmBaseModule
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.shared.shared_torch_builder import TorchBuilder


class LlmEmbeddingsTrainer(LlmBaseModule):
    """
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model=None,
                 llm_tokenizer=None,
                 dataset: Union[MaskedJSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference=False,
                 device=None):
        """
        :param llm_model:
        :param llm_tokenizer:

        :param spec: specs for llm trainer
        :param dataset: Union[JSONDataset, MaskedJSONDataset].
        :param metric_logger:  a metric logger used to log training progress
        """
        super().__init__(
            module_name,
            spec,
            llm_model,
            llm_tokenizer,
            ds=dataset,
            metric_logger=metric_logger,
            is_inference=is_inference,
            device=device
        )

        self.is_quantize = False
        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size

        self._batch_log = 10
        self._eval_freq = 10
        self._masked_freq = 5

        self._eval_freq = 8
        self._is_shuffle = True
        self._num_workers = spec.num_workers
        self._default_mask_token = "@odata.id"
        self._lr = spec.llm_learning_rate

        self.optimizer = TorchBuilder.create_optimizer(
            spec.llm_optimizer,
            self.model,
            spec.llm_learning_rate,
            spec.llm_weight_decay,
            **vars(spec)
        )

        self.logger.info(
            f"Rank {self.rank} creating llm trainer, num epochs {self.num_epochs} "
            f"batch_size {self.batch_size} "
            f"dataset size {len(self.dataset)} "
            f"is overfit {self._overfit} "
        )
        self._mask_probability = 1.0
        self._best_validation_metric = float('-inf')
        self.dataset = dataset

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

    @staticmethod
    def compute_perplexity(logits, labels):
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = torch.log(probabilities)
        loss = F.nll_loss(log_probabilities.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
        return perplexity

    def validate(self, validation_dataset):
        """ Perform validation on the emb llm model.

        :param validation_dataset: Dataset for validation
        :return: Accuracy on the validation dataset
        """

        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i, batch in enumerate(validation_dataset):
                labels = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.tokenizer.pad_token_id)
                labels = labels.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': labels.to(self.device)
                }

                mask_indices = mask[:, :-1].to(self.device)
                outputs = self.model(**batch_inputs)
                predicted_tokens = torch.argmax(outputs.logits, dim=-1)
                predicted_masked_tokens = predicted_tokens[mask_indices]
                predicted_masked_tokens = predicted_masked_tokens.to(self.device)
                batch["input_ids"] = batch["input_ids"].to(self.device)

                # masked tokens with original tokens
                original_tokens = batch["input_ids"][mask_indices].to(self.device)
                accuracy_bool = predicted_masked_tokens == original_tokens
                correct_predictions += accuracy_bool.sum().item()
                total_predictions += original_tokens.numel()

                # # perplexity
                # logits = outputs.logits
                # logits = logits.to(self.device)
                # loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                #                        labels.view(-1).to(self.device), ignore_index=-100)
                # total_loss += loss.item() * labels.numel()
                # total_predictions = total_predictions + labels.numel()

            # # masked_input_ids = batch["input_ids"].clone().detach()
            # mask_indices = torch.rand(batch["input_ids"].shape) < self._mask_probability
            # # masked_input_ids[mask_indices] = -100
            #
            # outputs = self.model(**batch_inputs)
            # predicted_tokens = torch.argmax(outputs.logits, dim=-1)
            # predicted_masked_tokens = predicted_tokens[mask_indices]
            # predicted_masked_tokens = predicted_masked_tokens.to(self.device)
            #
            # # compare predicted masked tokens with original tokens
            # original_tokens = batch["input_ids"][mask_indices].to(self.device)
            #
            # print(f"predicted {predicted_tokens.shape}, {original_tokens.shape}")
            #
            # accuracy_bool = predicted_masked_tokens == original_tokens
            # correct_predictions += accuracy_bool.sum().item()
            # total_predictions += original_tokens.numel()

        accuracy = correct_predictions / total_predictions * 100.0
        # perplexity = torch.exp(total_loss / torch.tensor(total_predictions)).item()
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

    @staticmethod
    def enable_masking_method(dataset, mask_type):
        """
        :param dataset:
        :param mask_type:
        :return:
        """
        if mask_type == MaskingOption.TARGET:
            dataset.mask_targets()
        elif mask_type == MaskingOption.ALLOWED_VALUE:
            dataset.mask_allowed_value()
        elif mask_type == MaskingOption.ODATA_ID:
            dataset.mask_odata_id()
        elif mask_type == MaskingOption.TARGET_KEY:
            dataset.mask_targets_key()
        elif mask_type == MaskingOption.JSON_OBJECT:
            dataset.mask_objects()
        elif mask_type == MaskingOption.JSON_ARRAY:
            dataset.mask_arrays()
        elif mask_type == MaskingOption.MASK_API_PREFIX:
            dataset.mask_api_prefix()
        elif mask_type == MaskingOption.MASK_NEW_TOKENS:
            dataset.mask_new_tokens(is_enabled=True)
        else:
            raise ValueError("Unknown masking type")

    def train(self, overfit: Optional[bool] = True, is_full_mask=None):
        """Train LLM model to map high level goal to redfish actions.

        For example
                "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"
        :param is_full_mask:
        :param overfit:
        :return:
        """
        self.model.resize_token_embeddings(len(self.dataset.tokenizer))
        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        torch.cuda.empty_cache()
        validation_accuracy = float('-inf')
        self.logger.info(f"Uploading model from {self.model.device} "
                         f"to device {self.device}, "
                         f"using accelerate: {self.is_accelerator}")
        self.model.to(self.device)

        if self.module_checkpoint_dir is not None:
            last_epoch = self.load_checkpoint(self.module_checkpoint_dir)
        else:
            last_epoch = 0

        self.model.train()
        train_dataset, eval_dataset = self.split_dataset()
        sampler = self.dataset_sampler()

        masking_methods = [
            MaskingOption.MASK_API_PREFIX,
            MaskingOption.MASK_NEW_TOKENS,
            MaskingOption.JSON_ARRAY,
            MaskingOption.JSON_OBJECT,
            MaskingOption.ODATA_ID
        ]

        self.logger.info(f"Creating dataloader {self.batch_size} num worker {self._num_workers}")
        train_data, eval_data = self.split_dataset()

        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            shuffle=self._is_shuffle,
            collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        eval_dataloader = DataLoader(
            eval_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        if self.is_accelerator:
            self.model, self.optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                [self.model, self.optimizer, train_dataloader, eval_dataloader],
                device_placement=[True])

        if self.is_quantize:
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        total_batches = len(train_dataloader)
        dataset_size = len(train_dataset)
        calculated_total_batches = dataset_size // self.batch_size
        batch_log_frequency = round(64 * 0.2)

        if total_batches == calculated_total_batches:
            print(f"Staring training total_batches: {total_batches} "
                  f"train dataset size: {dataset_size} "
                  f"batch_size {self.batch_size} "
                  f"lr {self._lr} "
                  f"batch stats freq: {batch_log_frequency}.")

        current_method_idx = 0
        for epoch in range(last_epoch, self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            batch_losses = np.zeros(total_batches)
            for i, batch in enumerate(train_dataloader):
                if (epoch + 1) % self._masked_freq == 0:
                    # switch to masking
                    self.dataset.enable_masking()
                    self.dataset.mask_new_tokens(True)
                    current_method = masking_methods[current_method_idx]
                    self.enable_masking_method(self.dataset, current_method)
                    current_method_idx = (current_method_idx + 1) % len(masking_methods)
                else:
                    self.dataset.disable_masking()

                labels = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.tokenizer.pad_token_id)
                labels = labels.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                input_ids = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)

                batch_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': masks
                }

                # if epoch % self._masked_freq == 0:
                #     for j in range(batch_inputs['input_ids'].size(0)):
                #         batch_inputs["attention_mask"] = JSONDataset.mask_json_key_and_value(
                #             batch_inputs, self._default_mask_token, self.tokenizer)
                #
                # batch_inputs = {
                #     'input_ids': batch['input_ids'].to(self.device),
                #     'attention_mask': batch['attention_mask'].to(self.device)
                # }

                labels = labels.to(self.device)
                outputs = self.model(**batch_inputs, labels=labels)
                loss = outputs.loss

                self.optimizer.zero_grad()
                if self.is_accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()
                #
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

                # # validation on epoch or freq
                # if self.on_epoch_eval or ((epoch + 1) % 32 == 0):
                #     validation_accuracy, perplexity = self.validate(eval_dataloader)
                #     if self.is_rank_zero():
                #         self.metric_logger.log_metric("llm_emb_accuracy", validation_accuracy, epoch)
                #         self.metric_logger.log_metric("llm_emb_perplexity", perplexity, epoch)

                num_batches += 1

            # validation on epoch or freq
            if self.on_epoch_eval or ((epoch + 1) % self._eval_freq == 0):
                validation_accuracy = self.validate(eval_dataloader)
                if self.is_rank_zero():
                    self.metric_logger.log_metric("llm_emb_accuracy", validation_accuracy, epoch)
                    # self.metric_logger.log_metric("llm_emb_perplexity", perplexity, epoch)

                print(f"Rank {self.rank} Epoch {epoch + 1} - Validation Accuracy: "
                      f"{validation_accuracy} Best: {self._best_validation_metric}")

            if num_batches > 0:
                average_loss = total_loss / num_batches
                if self.is_rank_zero():
                    self.metric_logger.log_metric("llm_emb_epoch_loss", average_loss, epoch)
                print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

            # save best checkpoint
            if self.is_rank_zero():
                if validation_accuracy > self._best_validation_metric or (epoch + 1) % 10 == 0:
                    self._best_validation_metric = validation_accuracy
                    if self.module_checkpoint_dir is not None:
                        self.save_checkpoint(self.module_checkpoint_dir, epoch + 1)

        if self.is_quantize:
            self.model = convert(self.model)

        self.save_model(self.module_checkpoint_dir)
        self.save_finetuned()

        del train_dataloader
        del eval_dataloader
        del self.optimizer

        print("Embedding extractor training complete.")
