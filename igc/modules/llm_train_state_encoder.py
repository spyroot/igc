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
import time
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.quantization import convert
from torch.utils.data import DataLoader, RandomSampler
from transformers import PreTrainedModel, PreTrainedTokenizer

from igc.modules.base.igc_llm_base_module import LlmModule
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.shared.shared_torch_builder import TorchBuilder

from igc.ds.redfish_masked_dataset import (
    MaskingOption,
    MaskedJSONDataset,
    MaskingType
)


class LlmEmbeddingsTrainer(LlmModule):
    """
    Large language model trainer. Its main job train a language
    model for fine-tuning a latent representation
    """

    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model: PreTrainedModel = None,
                 llm_tokenizer: PreTrainedTokenizer = None,
                 dataset: Union[MaskedJSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference=False,
                 device=None):
        """
        Note that tokenizer must IGC since we extend it to support masking , JSON etc

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

        self._is_quantize = False
        self.num_epochs = spec.num_train_epochs
        self.batch_size = spec.per_device_train_batch_size

        self._batch_log = 10
        # number of mask passes before we switch it off
        self._num_mask_passed = 10
        self._masked_freq = spec.llm_mask_freq

        self._eval_freq = 16
        self._save_freq = 16

        self._is_shuffle = True
        self._pin_memory = False if "pin_memory" not in spec else spec.pin_memory
        self._reset_lr = False if "reset_lr" not in spec else spec.reset_lr
        self._num_workers = spec.num_workers
        self._lr = spec.llm_learning_rate

        self.optimizer = TorchBuilder.create_optimizer(
            spec.llm_optimizer,
            self.model,
            spec.llm_learning_rate,
            spec.llm_weight_decay,
            **vars(spec)
        )

        self._mask_probability = 1.0
        self._best_validation_metric = float('-inf')
        self.dataset = dataset

        self.logger.info(
            f"Rank {self.rank} creating llm trainer, num epochs {self.num_epochs} "
            f"batch_size {self.batch_size} "
            f"dataset size {len(self.dataset)} "
            f"is overfit {self._overfit} "
        )

        if spec.masking_type == MaskingType.NO_MASK:
            self.masking_methods = [
                spec.masking_type
            ]

        if spec.masking_type == MaskingType.MASK_SECTION or MaskingType.MASK_NEW_TOKENS:
            self.masking_methods = [
                spec.masking_type
            ]

        if spec.masking_type == MaskingType.MASK_JSON_KV:
            self.masking_methods = [
                spec.masking_option
            ]

        # what method we're using for masking. i.e. we can stack
        self._current_mask_method_counter = 0
        self._current_mask_method_idx = 0
        self._masking_method_dispatcher = LlmEmbeddingsTrainer.create_masking_method(dataset)

    def get_model(self) -> PreTrainedModel:
        """Return module model.
        :return:
        """
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return module model.
        :return:
        """
        if self.dataset.tokenizer is None:
            self.dataset.load_tokenizer()

        return self.dataset.tokenizer

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
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i, batch in enumerate(validation_dataset):
                batch_size = batch['input_ids'].size(0)
                target_ids = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.tokenizer.pad_token_id)
                mask = mask.to(self.device)
                target_ids = target_ids.to(self.device)

                target_ids = target_ids.masked_fill(mask[:, 1:], -100)
                original_mask = batch['attention_mask']
                shifted_input = batch['input_ids'][:, :-1].to(self.device)
                shifter_mask = batch['attention_mask'][:, :-1].to(self.device)
                target_ids = target_ids.to(self.device)

                batch_inputs = {
                    'input_ids': shifted_input,
                    'attention_mask': shifter_mask,
                }

                outputs = self.model(**batch_inputs)
                compute_accuracy = self.compute_accuracy(
                    outputs.logits, target_ids, original_mask
                )
                correct_predictions += compute_accuracy * batch_size
                total_predictions += batch_size

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

    @staticmethod
    def create_masking_method(dataset: MaskedJSONDataset):
        """
        Return dict that store all callable masking methods.
        :return:
        """
        masking_methods = {
            MaskingType.MASK_SECTION: dataset.mask_section,
            MaskingType.MASK_NEW_TOKENS: dataset.mask_new_tokens,
            MaskingType.MASK_JSON_KV: dataset.enable_masking,
            MaskingType.NO_MASK: dataset.disable_masking,

            # options
            MaskingOption.TARGET: dataset.mask_targets,
            MaskingOption.ALLOWED_VALUE: dataset.mask_allowed_value,
            MaskingOption.ODATA_ID: dataset.mask_odata_id,
            MaskingOption.TARGET_KEY: dataset.mask_targets_key,
            MaskingOption.JSON_OBJECT: dataset.mask_objects,
            MaskingOption.JSON_ARRAY: dataset.mask_arrays,
            MaskingOption.MASK_API_PREFIX: dataset.mask_api_prefix,
        }
        return masking_methods

    def enable_masking_method(
            self, mask_type: Union[MaskingOption, MaskingType]
    ):
        """
        Receive mask enum and dispatch to its callback.
        :param mask_type:
        """
        if mask_type in self.masking_methods:
            print(f"Got mask type {mask_type}")
            callback = self.masking_methods[mask_type]
            callback()
        else:
            raise ValueError("Unknown masking type")

    def swap_masking_method(
            self, epoch: int,
            mask_type: List[Union[MaskingOption, MaskingType]] = None
    ):
        """
        Switch to masking method to next masking method after every epoch freq
        dictated by self._masked_freq.

        For example if we have 2 masking method let say mask freq is 2
        then we will switch to first method , on next epoch we will switch to off
        on third epoch we will switch to second method.

        So we cycle between methods.

        :return:
        """
        if (epoch + 1) % self._masked_freq == 0 and self._current_mask_method_counter == 0:
            # switch to masking pass, mask freq how fast i.e after epoch
            # or after 10 epoch etc.
            _current_method = mask_type[self._current_mask_method_idx]
            self.enable_masking_method(_current_method)
            self._current_mask_method_idx = (self._current_mask_method_idx + 1) % len(mask_type)
        else:
            # reset back
            if self._current_mask_method_counter == self._num_mask_passed:
                self.dataset.disable_masking()
                self._current_mask_method_counter = 0
                self.dataset.disable_masking()

    def _train(
            self,
            mask_type: List[Union[MaskingOption, MaskingType]] = None
    ):
        """Train LLM model to map high level goal to redfish actions.

        :return:
        """

        self.model.resize_token_embeddings(
            len(self.dataset.tokenizer)
        )

        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        torch.cuda.empty_cache()

        if not self.is_accelerator:
            self.model.to(self.device)

        lr = self._lr if self._reset_lr else None
        map_location = None if self.is_accelerator else self.device

        checkpoint_state = self.load_checkpoint(
            self._module_checkpoint_dir,
            map_location=map_location,
            lr=lr)

        print(self.optimizer)

        self.logger.info(f"Rank {self.rank}: "
                         f"Uploading model from {self.model.device} "
                         f"to device {self.device}, "
                         f"using accelerate: {'yes' if self.is_accelerator else 'no'}, "
                         f"current mem {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        sampler = self.dataset_sampler()

        self.logger.info(f"Creating dataloader "
                         f"batch size {self.batch_size} "
                         f"num worker {self._num_workers}")

        train_data, eval_data = self.split_dataset()

        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            shuffle=self._is_shuffle,
            pin_memory=self._pin_memory,
            collate_fn=LlmEmbeddingsTrainer.custom_collate_fn
        )

        eval_dataloader = DataLoader(
            eval_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=self._pin_memory,
            collate_fn=LlmEmbeddingsTrainer.custom_collate_fn,
        )

        last_epoch = checkpoint_state.last_epoch
        validation_accuracy = checkpoint_state.last_epoch
        self._trainer_args.epochs = self.num_epochs - last_epoch
        self._trainer_args.steps_per_epoch = len(train_dataloader)

        self.logger.info(
            f"Rank {self.rank}: Data loader created: "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        self.model = self.model.to(self.device)

        self.scheduler = TorchBuilder.create_scheduler(
            self._trainer_args.llm_scheduler,
            optimizer=self.optimizer,
            **vars(self._trainer_args)
        )

        self.logger.info(f"Rank {self.rank}: "
                         f"Memory utilization after we loaded model: "
                         f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # we update the scheduler state here.
        if checkpoint_state.scheduler_state is not None:
            self.scheduler.load_state_dict(checkpoint_state.scheduler_state)

        if self.is_accelerator:
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler
            )

        torch.cuda.empty_cache()
        self.logger.info(
            f"Rank {self.rank}: Memory utilization after we prepared : "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        if self._is_quantize:
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        total_batches = len(train_dataloader)
        dataset_size = len(train_data)
        calculated_total_batches = dataset_size // self.batch_size
        batch_log_frequency = round(64 * 0.2)

        if total_batches == calculated_total_batches:
            print(f"Rank {self.rank}, Staring training, "
                  f"total_batches: {total_batches} "
                  f"train dataset size: {dataset_size} "
                  f"batch_size {self.batch_size} "
                  f"current lr {self._lr} "
                  f"batch stats freq: {batch_log_frequency}.")

        for epoch in range(last_epoch, self.num_epochs):
            self.model.train()

            for i, param_group in enumerate(self.optimizer.param_groups):
                print(f'Learning rate of parameter group {i}: {param_group[i]["lr"]}')

            total_loss = 0.0
            num_batches = 0
            batch_losses = np.zeros(total_batches)

            # swap mask and pass a batch
            self.swap_masking_method(epoch, mask_type)

            for _, batch in enumerate(train_dataloader):

                attention_mask = batch['attention_mask'].to(self.device)
                mask_inverted = ~attention_mask.bool()

                input_ids = batch["input_ids"].to(self.device)
                input_ids = input_ids.masked_fill(mask_inverted == 1, -100).contiguous()

                target_ids = input_ids[:, 1:].clone().detach()
                target_ids = target_ids.to(self.device)

                mask = (batch["input_ids"] == self.tokenizer.pad_token_id)
                target_ids = target_ids.masked_fill(mask[:, 1:], -100)

                batch['input_ids'] = batch['input_ids'][:, :-1]
                batch['attention_mask'] = batch['attention_mask'][:, :-1]

                input_ids = batch['input_ids'].to(self.device)
                masks = batch['attention_mask'].to(self.device)

                batch_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': masks,
                    'labels': target_ids
                }

                outputs = self.model(**batch_inputs)
                loss = outputs.loss

                self.optimizer.zero_grad()
                if self.is_accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                self.optimizer.step()

                if self.is_last_worker():
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.metric_logger.log_metric("learning_rate", current_lr, epoch)

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()
                torch.cuda.empty_cache()

                #
                if self._is_quantize:
                    self.model.apply(torch.quantization.propagate_qconfig_)
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

                # calculate the progress percentage
                progress_percentage = int(round((num_batches + 1) / total_batches * 100))
                if (num_batches % batch_log_frequency == 0) or (num_batches == total_batches - 1):
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Batch "
                          f"{num_batches + 1}/{total_batches} "
                          f"- Progress: {progress_percentage:.2f}% - "
                          f"Batch Loss mean: {batch_losses.mean():.4f} - lr: {lr:.5f}")
                    self.metric_logger.log_metric("llm_emb_batch_loss", batch_losses.mean(), epoch)

                num_batches += 1
                self._current_mask_method_counter += 1

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
                is_best_accuracy = validation_accuracy > self._best_validation_metric
                if is_best_accuracy or (epoch + 1) % self._save_freq == 0:
                    self._best_validation_metric = validation_accuracy
                    if self._module_checkpoint_dir is not None:
                        if self.is_accelerator:
                            model = self.accelerator.unwrap_model(self.model)
                            opt = self.accelerator.unwrap_model(self.optimizer)
                            shed = self.accelerator.unwrap_model(self.scheduler)
                            self.save_checkpoint(
                                self._module_checkpoint_dir,
                                epoch + 1,
                                model=model,
                                optimizer=opt,
                                scheduler=shed,
                                last_accuracy=validation_accuracy,
                                initial_lr=self._lr,
                                is_best_accuracy=is_best_accuracy,
                            )
                        else:
                            self.save_checkpoint(
                                self._module_checkpoint_dir,
                                epoch + 1,
                                last_accuracy=validation_accuracy,
                                initial_lr=self._lr,
                                is_best_accuracy=is_best_accuracy,
                            )

        if self._is_quantize:
            self.model = convert(self.model)

        self.model = self.accelerator.unwrap_model(self.model)
        self.save_model(self._module_checkpoint_dir)
        self.save_finetuned()

        del train_dataloader
        del eval_dataloader
        del self.optimizer

        print("Embedding extractor training complete.")

    def train(self):
        """Train loop for the fine running.
        :return:
        """

        self._train(
            mask_type=self.masking_methods
        )

    def decode_masked_output(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ):
        """
        :param input_ids:
        :param attention_mask:
        :return:
        """
        decoded_batch = []
        for batch_idx in range(input_ids.shape[0]):
            unmasked_tokens = []
            for token in input_ids[batch_idx]:
                if token.item() != self.tokenizer.pad_token_id:
                    unmasked_tokens.append(token.item())

            decoded_tokens = self.dataset.tokenizer.decode(unmasked_tokens)
            decoded_batch.append(decoded_tokens)

        return decoded_batch

    @staticmethod
    def custom_loss(
            logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        """
        if logits.dim() == 2:
            return F.cross_entropy(logits, targets, ignore_index=-100)
        elif logits.dim() == 3:
            shifted_step_logits = logits[..., :-1, :].contiguous()
            shift_step_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shifted_step_logits.view(-1, shifted_step_logits.size(-1)),
                shift_step_labels.view(-1).long(), ignore_index=-100)
            return loss
        else:
            raise ValueError("Invalid shape")

    @staticmethod
    def compute_accuracy(
            logits: torch.Tensor,
            targets: torch.Tensor,
            original_mask: torch.Tensor
    ):
        """
        Computes  accuracy for either sequence classification or generation.

        :param logits:
        :param targets:
        :param original_mask:
        :return:
        """
        if logits.dim() == 2:
            y = torch.argmax(logits, dim=-1) == targets
            y = y.type(torch.float)
            return torch.mean(y).item()
        elif logits.dim() == 3:
            shifted_step_logits = logits[..., :-1, :].contiguous()
            shift_step_labels = targets[..., 1:].contiguous()
            masked_logits = shifted_step_logits.eq(-100)
            logits_inf = shifted_step_logits.masked_fill_(masked_logits, float("-Inf"))
            arg_maxed_idx = torch.argmax(logits_inf, dim=-1)
            r = shift_step_labels == arg_maxed_idx
            r = r.type(torch.float)
            return r.mean().item()
        else:
            raise ValueError("Invalid shape")

    def test_inference(self):
        """
          Does inference pass for entire dataset used for validation
          and report all metrics.

        :return:
        """
        train_data, eval_data = self.split_dataset()
        sampler = self.dataset_sampler()

        eval_dataloader = DataLoader(
            eval_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=LlmEmbeddingsTrainer.custom_collate_fn)

        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        total_predictions_no_pad = 0

        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                target_ids = batch["input_ids"][:, 1:].clone().detach()
                mask = (batch["input_ids"] == self.tokenizer.pad_token_id)
                mask = mask.to(self.device)
                target_ids = target_ids.to(self.device)

                target_ids = target_ids.masked_fill(mask[:, 1:], -100)
                original_mask = batch['attention_mask']
                shifted_input = batch['input_ids'][:, :-1].to(self.device)
                shifter_mask = batch['attention_mask'][:, :-1].to(self.device)
                target_ids = target_ids.to(self.device)

                batch_inputs = {
                    'input_ids': shifted_input,
                    'attention_mask': shifter_mask,
                }

                outputs = self.model(**batch_inputs)
                compute_accuracy = self.compute_accuracy(
                    outputs.logits, target_ids, original_mask
                )
                correct_predictions += compute_accuracy * batch_inputs['input_ids'].size(0)
                total_predictions += batch_inputs['input_ids'].size(0)

                non_pad_tokens = target_ids.numel() - target_ids.eq(-100).sum().item()
                total_predictions_no_pad += non_pad_tokens

                # perplexity
                logits = outputs.logits
                logits = logits.to(self.device)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       target_ids.view(-1).to(self.device),
                                       ignore_index=-100)
                total_loss += loss.item() * target_ids.numel()

        end_time = time.time()
        time_taken = end_time - start_time

        accuracy = correct_predictions / total_predictions * 100.0
        perplexity = torch.exp(total_loss / torch.tensor(total_predictions_no_pad)).item()

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Time taken: {time_taken:.2f} seconds")

        return accuracy, perplexity
