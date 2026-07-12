"""
Train the language-model state encoder on masked Redfish JSON examples.

The ``MaskedJSONDataset`` provided by ``igc.ds.redfish_masked_dataset`` supplies
tokenized REST response sequences and masking callbacks. This trainer applies a
next-token objective, cycles the configured masking methods, and saves
fine-tuned language-model checkpoints for downstream state encoding.

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
from igc.modules.shared.llm_shared import safe_resize_token_embeddings
from igc.shared.shared_accelerator import broadcast_flag
from igc.shared.shared_torch_builder import TorchBuilder

from igc.ds.redfish_masked_dataset import (
    MaskingOption,
    MaskedJSONDataset,
    MaskingType
)


def build_causal_lm_labels(input_ids, attention_mask, pad_token_id=None):
    """Build full-length next-token labels for a Hugging Face CausalLM.

    A HF causal LM shifts internally when given ``labels`` — it computes the loss of
    ``logits[:, :-1]`` against ``labels[:, 1:]`` — so labels must be the SAME length as
    ``input_ids`` and NOT pre-shifted. Pre-shifting (the old ``target_ids = input_ids[:,
    1:]`` with a truncated input) double-shifts and trains the model to predict two
    tokens ahead. Attention-padding and pad-token positions are set to ``-100`` so they
    are ignored in the loss; ``input_ids`` itself is never altered (``-100`` is not a
    valid embedding index).

    :param input_ids: ``[B, L]`` token ids.
    :param attention_mask: ``[B, L]`` mask (1 = attended, 0 = padding).
    :param pad_token_id: the tokenizer pad id to ignore, or ``None`` to skip that mask.
    :return: ``[B, L]`` labels with ``-100`` on ignored positions.
    """
    labels = input_ids.clone()
    labels = labels.masked_fill(~attention_mask.bool(), -100)
    if pad_token_id is not None:
        labels = labels.masked_fill(input_ids == pad_token_id, -100)
    return labels


def optimizer_steps_per_epoch(num_micro_batches: int, accum_steps: int) -> int:
    """Optimizer steps one epoch will actually take under gradient accumulation.

    Schedulers like OneCycleLR consume ``steps_per_epoch`` as OPTIMIZER steps;
    feeding them micro-batch counts under accumulation leaves most of the
    one-cycle schedule unexecuted.

    :param num_micro_batches: dataloader length for the epoch.
    :param accum_steps: gradient accumulation steps (coerced to >= 1).
    :return: ``ceil(num_micro_batches / accum_steps)``.
    """
    accum_steps = max(1, accum_steps)
    return -(-num_micro_batches // accum_steps)


def measure_grad_norm(model) -> float:
    """Measure (never clip) the current global gradient norm.

    Must run BEFORE ``optimizer.step()``/``zero_grad()`` — afterwards the
    gradients are cleared and the measurement is 0.0 forever, which is exactly
    the bug this helper exists to prevent regressing.

    :param model: the module whose parameter gradients are measured.
    :return: the global grad norm as a float.
    """
    import torch as _torch
    return _torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf")).item()


def is_accum_boundary(micro_step_index: int, accum_steps: int, total_batches: int) -> bool:
    """Whether a manually-accumulated optimizer step should fire on this micro-batch.

    Used on the plain (non-accelerator) path, which has no accumulation wrapper: fire every
    ``accum_steps`` micro-batches, and always on the final batch of the epoch so a trailing
    partial accumulation window is flushed rather than silently dropped. Over one epoch this
    yields ``ceil(total_batches / accum_steps)`` optimizer steps.

    :param micro_step_index: zero-based index of the current micro-batch within the epoch.
    :param accum_steps: gradient accumulation steps (coerced to ``>= 1``).
    :param total_batches: number of micro-batches in the epoch.
    :return: ``True`` to run optimizer.step()/scheduler.step()/zero_grad() now.
    """
    accum_steps = max(1, accum_steps)
    reached_window = (micro_step_index + 1) % accum_steps == 0
    is_last_batch = (micro_step_index + 1) == total_batches
    return reached_window or is_last_batch


def reached_max_steps(global_step: int, max_steps: Optional[int]) -> bool:
    """Whether training should stop because the optimizer-step cap is reached.

    ``--max_train_steps`` caps the number of OPTIMIZER updates (not micro-batches). A value of
    ``None`` or ``<= 0`` means no cap. Without this check ``_train`` loops ``num_train_epochs``
    (default 1000) and silently ignores the step cap.

    :param global_step: optimizer steps taken so far this run.
    :param max_steps: the configured cap, or ``None`` / ``<= 0`` for uncapped.
    :return: ``True`` when ``max_steps`` is a positive int and ``global_step >= max_steps``.
    """
    return max_steps is not None and max_steps > 0 and global_step >= max_steps


def unwrap_accelerate(wrapped, inner_attr: str):
    """Return the base optimizer/scheduler that accelerate wrapped, or ``wrapped``.

    ``Accelerator.unwrap_model`` is for ``nn.Module``s only — in accelerate>=1.14 it
    probes ``model._modules``, which an ``AcceleratedOptimizer`` / ``AcceleratedScheduler``
    does not have, raising ``AttributeError('_modules')`` and killing the checkpoint
    save on the multi-GPU (FSDP) path. accelerate exposes the inner object as
    ``AcceleratedOptimizer.optimizer`` / ``AcceleratedScheduler.scheduler``, so read that.

    :param wrapped: the (possibly accelerate-wrapped) optimizer or scheduler.
    :param inner_attr: ``"optimizer"`` or ``"scheduler"`` — the wrapper's inner attribute.
    :return: the unwrapped inner object, or ``wrapped`` if it is not wrapped.
    """
    return getattr(wrapped, inner_attr, wrapped)


class LlmEmbeddingsTrainer(LlmModule):
    """
    Train a language model on masked Redfish JSON sequences.

    The trainer fine-tunes the configured causal model with dataset-controlled
    masking methods and reports loss, accuracy, perplexity, and scheduler state.
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
        Initialize model, optimizer, masking schedule, and training controls.

        :param module_name: Stable module name used for logging and checkpoints.
        :param spec: Training configuration namespace parsed by the shared CLI.
        :param llm_model: Causal language model to fine-tune.
        :param llm_tokenizer: Tokenizer paired with ``llm_model`` and the dataset.
        :param dataset: Masked JSON dataset that owns masking callbacks.
        :param metric_logger: Optional metric sink for training and evaluation metrics.
        :param is_inference: Whether to skip training-only setup in the base module.
        :param device: Torch device used for model and batch tensors.
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
        # pin_memory enables overlapped host->device copies (helps only on GPU). The real
        # flag is --dataloader_pin_memory; the old "pin_memory" key never existed on spec,
        # so this was always False. Default it on when CUDA is present (the launcher does
        # not emit the flag) and honor an explicit --dataloader_pin_memory.
        self._pin_memory = bool(
            getattr(spec, "dataloader_pin_memory", False)) or torch.cuda.is_available()
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
        """Return the underlying language model.

        :return: The model managed by this trainer.
        """
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the dataset tokenizer, loading it on demand when needed.

        :return: Tokenizer used for Redfish JSON training examples.
        """
        if self.dataset.tokenizer is None:
            self.dataset.load_tokenizer()

        return self.dataset.tokenizer

    @staticmethod
    def custom_collate_fn(samples):
        """Stack tokenized samples into a model batch.

        :param samples: Dataset rows containing ``input_ids`` and ``attention_mask``.
        :return: Batch dictionary with tensor values stacked on the leading axis.
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
        Slice a language-model training window and its shifted target.

        :param src: [full_seq_len, batch_size]
        :param idx: Starting index in ``src``.
        :param chunk_size: Maximum sequence length for the returned window.
        :return: tuple (data, target),  shape [seq_len, batch_size], [seq_len * batch_size]
        """
        seq_len = min(chunk_size, len(src) - 1 - idx)
        data = src[idx:idx + seq_len]
        target = src[idx + 1:idx + 1 + seq_len].reshape(-1)
        return data, target

    def dataset_sampler(self):
        """
        Build the optional dataset sampler configured by trainer arguments.

        :return: ``RandomSampler`` when random sampling is enabled, otherwise ``None``.
        """
        if 'random_sampler_enabled' in self._trainer_args:
            sampler = RandomSampler(self.dataset) if self._trainer_args.random_sampler_enabled else None
            return sampler
        return None

    @staticmethod
    def compute_perplexity(logits, labels):
        """

        :param logits:
        :param labels:
        :return:
        """
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = torch.log(probabilities)
        loss = F.nll_loss(log_probabilities.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
        return perplexity

    def validate(self, validation_dataset):
        """Perform validation on the embedding language model.

        :param validation_dataset: Iterable validation dataloader of tokenized batches.
        :return: Accuracy percentage on the validation dataset.
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
        Check whether this trainer is running under distributed execution.

        :return: ``True`` when the local rank indicates distributed training.
        """
        return self.rank != -1

    def is_rank_zero(self):
        """
        Check whether the current process should emit rank-zero side effects.

        :return: ``True`` for single-process execution or distributed rank zero.
        """
        return self.rank == -1 or self.rank == 0

    @staticmethod
    def create_masking_method(dataset: MaskedJSONDataset):
        """
        Return dict that store all callable masking methods.

        :param dataset: Masked dataset that provides masking callbacks.
        :return: Mapping from masking enum values to dataset callback methods.
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

        :param mask_type: Masking enum to activate for subsequent dataset samples.
        """
        # dispatch through the method DICT — the enum list only orders the
        # curriculum; indexing it with an enum was a TypeError, so no masking
        # pass ever activated.
        if mask_type in self._masking_method_dispatcher:
            self.logger.info(f"Enabling masking method {mask_type}")
            callback = self._masking_method_dispatcher[mask_type]
            callback()
        else:
            raise ValueError(f"Unknown masking type {mask_type}")

    def swap_masking_method(
            self, epoch: int,
            mask_type: List[Union[MaskingOption, MaskingType]] = None
    ):
        """
        Enable the next masking method at the configured epoch boundary.

        A new method is activated only when ``(epoch + 1)`` is divisible by
        ``self._masked_freq`` and the batch counter is reset. After
        ``self._num_mask_passed`` batches, masking is disabled and the counter
        is reset for the next cycle.

        :param epoch: Zero-based epoch index used to decide when to swap masks.
        :param mask_type: Ordered masking methods to cycle through.
        """
        if not mask_type:
            return
        if (epoch + 1) % self._masked_freq == 0:
            # activate the next mask in the curriculum for the coming epoch. The old
            # `and counter == 0` guard compared against a per-micro-batch counter that
            # was never reset, so after epoch one no mask could ever activate.
            _current_method = mask_type[self._current_mask_method_idx]
            self.enable_masking_method(_current_method)
            self._current_mask_method_idx = (self._current_mask_method_idx + 1) % len(mask_type)
            self._current_mask_method_counter = 0
        elif self._current_mask_method_counter >= self._num_mask_passed:
            # enough masked batches seen since activation: back to plain batches.
            # (>= — the counter advances per micro-batch, an exact == is skipped over.)
            self.dataset.disable_masking()
            self._current_mask_method_counter = 0

    def _save_final_checkpoint(self):
        """Persist the trained model once at end-of-train, collective-safe.

        Under ZeRO-3/FSDP the state-dict gather is a COLLECTIVE: every rank must
        reach ``accelerator.get_state_dict`` together. The previous end-of-train
        path called ``save_model`` -> ``state_dict()`` behind ``is_rank_zero``, so
        rank 0 entered the gather alone and deadlocked while ranks 1..N ran ahead
        through ``save_finetuned`` and returned — the 4-GPU end-of-train hang where
        only 3 of 4 ranks print "training complete". Gather on ALL ranks first,
        hand the pre-gathered weights to the rank-0 writers, then barrier so no
        rank tears down its process group mid-write. Mirrors the per-epoch save.
        """
        final_state = None
        if self.is_accelerator:
            # Collective — must run on EVERY rank, before any is_rank_zero gate.
            # Gather on the still-wrapped model, exactly as the per-epoch save does.
            if self._module_checkpoint_dir is not None:
                final_state = self.accelerator.get_state_dict(self.model)
            self.model = self.accelerator.unwrap_model(self.model)

        # Rank-0 writers consume the pre-gathered dict (no second rank-0-only
        # state_dict() collective); the plain path passes None and saves locally.
        self.save_model(self._module_checkpoint_dir, model_state_dict=final_state)
        self.save_finetuned(model_state_dict=final_state)

        if self.is_accelerator:
            # No rank returns (and drops its NCCL group) until every rank is done.
            self.accelerator.wait_for_everyone()

    def _train(
            self,
            mask_type: List[Union[MaskingOption, MaskingType]] = None
    ):
        """Train the language model with shifted-token Redfish JSON targets.

        :param mask_type: Masking methods active during the training schedule.
        """

        safe_resize_token_embeddings(self.model, self.dataset.tokenizer)

        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        torch.cuda.empty_cache()

        if not self.is_accelerator:
            self.model = self.model.to(self.device)

        lr = self._lr if self._reset_lr else None
        map_location = None if self.is_accelerator else self.device

        checkpoint_state = self.load_checkpoint(
            self._module_checkpoint_dir,
            map_location=map_location,
            lr=lr)

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
            drop_last=True,  # equal batch count per rank -> no epoch-boundary save-collective deadlock
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
        # Seed the pre-loop accuracy with the restored best, not the epoch number: on epochs
        # where eval doesn't run, this value feeds `is_best_accuracy` below, so an epoch
        # integer here would spuriously beat _best_validation_metric and corrupt best-checkpoint tracking.
        validation_accuracy = checkpoint_state.best_accuracy
        self._trainer_args.epochs = self.num_epochs - last_epoch
        _accum = max(1, int(getattr(self._trainer_args, "gradient_accumulation_steps", 1)))
        self._trainer_args.steps_per_epoch = optimizer_steps_per_epoch(
            len(train_dataloader), _accum)

        self.logger.info(
            f"Rank {self.rank}: Data loader created: "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # Under accelerate, let accelerator.prepare() place the model — a manual .to(device)
        # here materializes the full (unsharded) model on one GPU before prepare, defeating
        # DeepSpeed ZeRO-3 / FSDP and OOM-ing a large backbone. Only place it on the plain path.
        if not self.is_accelerator:
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

        # opt-in torch.compile AFTER final placement/wrapping (FSDP wrapping must
        # precede compile); a safe no-op off-CUDA or when --compile is absent.
        self.model = TorchBuilder.maybe_compile(
            self.model, bool(getattr(self._trainer_args, "compile", False)))

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

        # Gradient accumulation. On the plain path we step every `accum` micro-batches; under
        # accelerate the prepared optimizer/scheduler gate themselves and accelerator.backward()
        # scales the loss, so `accum` here only drives the manual (non-accelerator) path.
        accum = max(1, int(getattr(self._trainer_args, "gradient_accumulation_steps", 1)))
        last_grad_norm = 0.0
        if self.is_rank_zero():
            self.logger.info(f"Gradient accumulation steps: {accum}")

        # Honor --max_train_steps: cap OPTIMIZER updates. Without this the loop runs
        # num_train_epochs (default 1000) and ignores the cap entirely.
        max_steps = getattr(self._trainer_args, "max_train_steps", None)
        global_opt_steps = 0

        # End-of-run report bookkeeping (emitted as report.json on rank zero).
        run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_started_clock = time.time()
        final_epoch_loss = None
        epochs_done = last_epoch

        if total_batches == calculated_total_batches:
            print(f"Rank {self.rank}, Staring training, "
                  f"total_batches: {total_batches} "
                  f"train dataset size: {dataset_size} "
                  f"batch_size {self.batch_size} "
                  f"current lr {self._lr} "
                  f"batch stats freq: {batch_log_frequency}.")

        for epoch in range(last_epoch, self.num_epochs):
            if reached_max_steps(global_opt_steps, max_steps):
                break
            self.model.train()

            total_loss = 0.0
            num_batches = 0
            batch_losses = np.zeros(total_batches)

            # swap mask and pass a batch
            self.swap_masking_method(epoch, mask_type)

            for _, batch in enumerate(train_dataloader):

                input_ids = batch['input_ids'].to(self.device, non_blocking=self._pin_memory)
                attention_mask = batch['attention_mask'].to(
                    self.device, non_blocking=self._pin_memory)

                # Full-length labels: a HF CausalLM shifts internally (loss over
                # logits[:-1] vs labels[1:]). Pre-shifting the target here as well
                # (the old target_ids = input_ids[:, 1:] + truncated input) double-shifts
                # and trains the model to predict two tokens ahead. build_causal_lm_labels
                # keeps the sequence full-length and only sets -100 on ignored positions.
                labels = build_causal_lm_labels(
                    input_ids, attention_mask, self.tokenizer.pad_token_id)

                batch_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }

                if self.is_accelerator:
                    # Accelerate owns accumulation: accelerator.backward() scales the loss by
                    # gradient_accumulation_steps (DeepSpeed scales internally), and the prepared
                    # optimizer/scheduler no-op on non-boundary micro-batches, so stepping every
                    # iteration inside accumulate() is correct.
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(**batch_inputs)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            last_grad_norm = measure_grad_norm(self.model)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    is_step = self.accelerator.sync_gradients
                else:
                    # Plain path has no wrapper: accumulate by hand. Scale the loss by accum,
                    # backward every micro-batch, and only step on the accumulation boundary so
                    # --gradient_accumulation_steps is actually honored (it was ignored before).
                    outputs = self.model(**batch_inputs)
                    loss = outputs.loss
                    (loss / accum).backward()
                    is_step = is_accum_boundary(num_batches, accum, total_batches)
                    if is_step:
                        last_grad_norm = measure_grad_norm(self.model)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                # Log once per real optimizer step so the curves track optimizer steps, not
                # micro-batches. grad_norm here is measure-only (max_norm=inf never clips).
                if is_step and self.is_rank_zero():
                    step = epoch * total_batches + num_batches
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.metric_logger.log_metric("train/loss", loss.item(), step)
                    self.metric_logger.log_metric("train/lr", current_lr, step)
                    self.metric_logger.log_metric("train/grad_norm", last_grad_norm, step)

                if is_step:
                    global_opt_steps += 1

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()

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

                num_batches += 1
                self._current_mask_method_counter += 1

                if reached_max_steps(global_opt_steps, max_steps):
                    break

            # one monotonic step at the epoch boundary for per-epoch metrics
            epoch_step = (epoch + 1) * total_batches - 1

            # validation on epoch or freq
            if self.on_epoch_eval or ((epoch + 1) % self._eval_freq == 0):
                validation_accuracy = self.validate(eval_dataloader)
                if self.is_rank_zero():
                    self.metric_logger.log_metric("eval/accuracy", validation_accuracy, epoch_step)

                print(f"Rank {self.rank} Epoch {epoch + 1} - Validation Accuracy: "
                      f"{validation_accuracy} Best: {self._best_validation_metric}")

            if num_batches > 0:
                average_loss = total_loss / num_batches
                final_epoch_loss = average_loss
                if self.is_rank_zero():
                    self.metric_logger.log_metric("train/epoch_loss", average_loss, epoch_step)
                print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")
            epochs_done = epoch + 1

            # save best checkpoint
            is_best_accuracy = validation_accuracy > self._best_validation_metric
            should_save = is_best_accuracy or (epoch + 1) % self._save_freq == 0
            gathered_state = None
            if self.is_accelerator and self._module_checkpoint_dir is not None:
                # rank 0's verdict must be uniform, and the state-dict gather is a
                # COLLECTIVE under ZeRO-3/FSDP — every rank participates or the
                # fleet deadlocks / rank 0 writes shards.
                should_save = broadcast_flag(self.accelerator, should_save)
                if should_save:
                    gathered_state = self.accelerator.get_state_dict(self.model)
            if self.is_rank_zero():
                if should_save:
                    self._best_validation_metric = validation_accuracy
                    if self._module_checkpoint_dir is not None:
                        if self.is_accelerator:
                            model = self.accelerator.unwrap_model(self.model)
                            # unwrap_model is for nn.Modules only; the optimizer/scheduler
                            # are accelerate wrappers (see unwrap_accelerate).
                            opt = unwrap_accelerate(self.optimizer, "optimizer")
                            shed = unwrap_accelerate(self.scheduler, "scheduler")
                            self.save_checkpoint(
                                self._module_checkpoint_dir,
                                epoch + 1,
                                model=model,
                                model_state_dict=gathered_state,
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

        self._save_final_checkpoint()

        # Emit the self-describing run report (report.json) next to the checkpoint so
        # every run is comparable through igc.modules.train.report.compare(). Emission
        # must never take down a finished training run, hence the broad guard.
        if self.is_rank_zero():
            try:
                from igc.modules.train.emit import build_run_bundle, emit_run_report
                manifest_fields = getattr(self.dataset, "run_manifest_fields", None)
                bundle = build_run_bundle(
                    vars(self._trainer_args),
                    training={
                        "final_epoch_loss": final_epoch_loss,
                        "epochs_done": epochs_done,
                        "optimizer_steps": global_opt_steps,
                        "best_eval": self._best_validation_metric,
                    },
                    metrics={"eval/accuracy": validation_accuracy},
                    dataset_fields=manifest_fields() if callable(manifest_fields) else None,
                    started_at=run_started_at,
                    ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    wall_clock_sec=round(time.time() - run_started_clock, 1),
                    checkpoint_path=str(self._module_checkpoint_dir or ""),
                )
                report_path = emit_run_report(bundle, str(self._module_checkpoint_dir or "."))
                self.logger.info(f"Run report written: {report_path}")
            except Exception as report_err:  # noqa: BLE001 — report loss < run loss
                self.logger.warning(f"run-report emission failed: {report_err}")

        del train_dataloader
        del eval_dataloader
        del self.optimizer

        print("Embedding extractor training complete.")

    def cpu_train_pass(
            self,
            mask_type: List[Union[MaskingOption, MaskingType]] = None
    ):
        """Train LLM model to map high level goal to redfish actions.

        :return:
        """

        safe_resize_token_embeddings(self.model, self.dataset.tokenizer)

        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        lr = self._lr if self._reset_lr else None
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
            drop_last=True,  # equal batch count per rank -> no epoch-boundary save-collective deadlock
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

        last_epoch = 0
        self._trainer_args.epochs = self.num_epochs - last_epoch
        self._trainer_args.steps_per_epoch = len(train_dataloader)

        print(self._trainer_args.epochs)
        print(self._trainer_args.steps_per_epoch)

        self.scheduler = TorchBuilder.create_scheduler(
            self._trainer_args.llm_scheduler,
            optimizer=self.optimizer,
            **vars(self._trainer_args)
        )

        self.logger.info(f"Rank {self.rank}: "
                         f"Memory utilization after we loaded model: "
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

            total_loss = 0.0
            num_batches = 0
            batch_losses = np.zeros(total_batches)

            # swap mask and pass a batch
            self.swap_masking_method(epoch, mask_type)

            for _, batch in enumerate(train_dataloader):

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # Full-length labels; the HF CausalLM does the single internal shift.
                # Pre-shifting here as well (the old target_ids = input_ids[:, 1:] +
                # truncated input) double-shifts and trains for two-tokens-ahead — the
                # same correction applied in _train via build_causal_lm_labels.
                labels = build_causal_lm_labels(
                    input_ids, attention_mask, self.tokenizer.pad_token_id)

                batch_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }

                outputs = self.model(**batch_inputs)
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.metric_logger.log_metric("learning_rate", current_lr, epoch)

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()

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

        del train_dataloader
        del eval_dataloader
        del self.optimizer

    def train(self):
        """Run the fine-tuning loop with the configured masking schedule.
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
        Decode non-padding tokens for each sequence in a masked batch.

        :param input_ids: Token id tensor shaped ``[batch, sequence]``.
        :param attention_mask: Attention mask for the same batch, kept for API symmetry.
        :return: List of decoded strings without padding tokens.
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
        Compute cross-entropy for classification or shifted token generation.

        :param logits: Model output logits.
        :param targets: Target labels aligned with ``logits``.
        :return: Cross-entropy loss with ignored padding labels.
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
        Compute raw class or shifted-token accuracy.

        The sequence path averages over all shifted positions produced by the
        current implementation. ``original_mask`` is accepted for API
        compatibility, but it is not applied inside this helper.

        :param logits: Model output logits.
        :param targets: Target ids or labels aligned with ``logits``.
        :param original_mask: Original attention mask for the unshifted batch.
        :return: Mean raw token or class accuracy.
        """
        if logits.dim() == 2:
            y = torch.argmax(logits, dim=-1) == targets
            y = y.type(torch.float)
            return torch.mean(y).item()
        elif logits.dim() == 3:
            # callers (validate/test_inference) already shift inputs vs targets by one,
            # so logits[t] predicts targets[t]; shifting again here measured a
            # two-tokens-ahead objective. The old -100 mask was applied to LOGIT VALUES
            # (a no-op), so pad positions counted as wrong — restrict to real labels.
            predictions = torch.argmax(logits, dim=-1)
            valid = targets.ne(-100)
            if valid.sum() == 0:
                return 0.0
            correct = (predictions == targets) & valid
            return (correct.sum().float() / valid.sum().float()).item()
        else:
            raise ValueError("Invalid shape")

    def test_inference(self):
        """
        Run inference over the validation split and report aggregate metrics.

        :return: Tuple of accuracy percentage and perplexity.
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
                # cross_entropy averaged over NON-PAD tokens; weight by the same
                # count so the final divide by total non-pad tokens is consistent.
                total_loss += loss.item() * non_pad_tokens

        end_time = time.time()
        time_taken = end_time - start_time

        accuracy = correct_predictions / total_predictions * 100.0
        perplexity = torch.exp(total_loss / torch.tensor(total_predictions_no_pad)).item()

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Time taken: {time_taken:.2f} seconds")

        return accuracy, perplexity
