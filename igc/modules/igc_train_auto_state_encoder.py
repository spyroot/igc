"""
Train the state autoencoder over backbone hidden states.

Tokenized Redfish batches are encoded by the configured Hugging Face backbone
module, then ``AutoStateEncoder`` reconstructs the resulting hidden-state
tensor. The trainer keeps backbone access shape-driven through
``igc.modules.encoders.backbone_utils``.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from igc.ds.redfish_dataset import JSONDataset
from .base.igc_base_module import IgcModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.encoders.backbone_utils import backbone_module, emb_shape
from igc.modules.llm.igc_autoencoder import AutoStateEncoder
from igc.shared.shared_torch_builder import TorchBuilder
import torch.nn.functional as F


class AutoencoderTrainer(IgcModule):
    """
    Train an autoencoder over backbone hidden states for compact state features.
    """
    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = False,
                 device: Optional[torch.device] = None):
        """
        Initialize the backbone sampler, autoencoder, and optimizer.

        :param module_name: Stable module name used for logging and checkpoints.
        :param spec: Training configuration namespace parsed by the shared CLI.
        :param llm_model: Backbone model that produces hidden states to reconstruct.
        :param llm_tokenizer: Tokenizer paired with ``llm_model``.
        :param ds: Redfish JSON dataset used for autoencoder batches.
        :param metric_logger: Optional metric sink for reconstruction metrics.
        :param is_inference: Whether to skip training-only setup in the base module.
        :param device: Torch device used for hidden states and autoencoder tensors.
        """
        super().__init__(
            module_name,
            spec,
            llm_model,
            llm_tokenizer,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=is_inference,
            device=device)

        self._input_dim = self.model.config.hidden_size
        self._latent_dim = self.model.config.hidden_size
        self._learning_rate = spec.auto_encoder_lr
        self._max_train_steps = getattr(spec, "auto_encoder_train_steps", None)

        self.logger.info(f"Creating auto-encoder input dim"
                         f" {self._input_dim} {self._latent_dim} batch_size: {self.batch_size}")

        # Backbone-agnostic base module + embedding dims via backbone_module()/emb_shape():
        # works for both the GPT-2 smoke backbone (m1_gpt2_smoke) and modern decoders (Qwen).
        # The old .transformer/.wpe access only worked for GPT-2 — kept agnostic on purpose.
        self._encoder_model = backbone_module(self.model)
        backbone_module(llm_model).config.is_decoder = False
        # self._llm_model.resize_token_embeddings(len(llm_tokenizer))
        input_shape = emb_shape(self.model)
        self.emb_shape = (input_shape[0] - 1, input_shape[1])

        # AutoStateEncoder reconstructs backbone hidden states shaped
        # (batch, seq_len, hidden_dim): hidden_dim is the backbone hidden size
        # (input_shape[1]) and seq_len is the dataset chunk length every sample is
        # padded to. Passing them keeps the encoder off the GPT-2 768/1024 defaults,
        # which mismatch any modern backbone (e.g. Qwen2.5-3B, hidden 2048).
        seq_len = self._resolve_seq_len(spec, input_shape)
        self.model_autoencoder = AutoStateEncoder(
            input_shape=input_shape,
            seq_len=seq_len,
            hidden_dim=input_shape[1],
        )

        self.logger.info(f"Creating optimizer {spec.auto_encoder_optimizer} "
                         f"lr: {spec.auto_encoder_lr} "
                         f"weight decay: {spec.auto_encoder_weight_decay}")

        self._learning_rate = spec.auto_encoder_lr
        self.optimizer = TorchBuilder.create_optimizer(
            spec.auto_encoder_optimizer,
            self.model_autoencoder,
            spec.auto_encoder_lr,
            spec.auto_encoder_weight_decay,
            **vars(spec)
        )

    def _resolve_seq_len(self, spec, input_shape):
        """Resolve the hidden-state sequence length the autoencoder reconstructs.

        Every dataset chunk is padded to ``max_len`` (built from ``--seq_len``), so
        the backbone emits hidden states of that length. Prefer the dataset's real
        ``max_len`` (the ground truth for the tensors fed to the autoencoder), fall
        back to the CLI ``seq_len``, then to the positional dimension for the legacy
        GPT-2 path.

        :param spec: training spec namespace (carries ``seq_len``).
        :param input_shape: ``(positions, hidden)`` backbone shape from ``emb_shape``.
        :return: the sequence length to build ``AutoStateEncoder`` with.
        """
        ds = getattr(self, "dataset", None)
        ds_max_len = getattr(ds, "_max_len", None)
        if isinstance(ds_max_len, int) and ds_max_len > 0:
            return ds_max_len
        spec_seq_len = getattr(spec, "seq_len", None)
        if isinstance(spec_seq_len, int) and spec_seq_len > 0:
            return spec_seq_len
        return int(input_shape[0])

    def _get_reconstruction_loss(self, batch):
        """
        Compute per-element reconstruction loss for a tuple-style tensor batch.

        :param batch: Tuple whose first item is the tensor to reconstruct.
        :return: Per-element mean squared reconstruction loss from ``self.model``.
        """
        x, _ = batch
        x_hat = self.model.forward(x)
        loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
        return loss

    def encode(self):
        """
        Attach a simple autoencoder encoder weight to the model embeddings.
        """
        input_dim = self.model.config.hidden_size
        latent_dim = 128

        gpt_encoder = self.model.get_input_embeddings()
        autoencoder = AutoStateEncoder(input_dim, latent_dim)

        # attach the autoencoder to the GPT model
        gpt_encoder.weight = nn.Parameter(autoencoder.encoder.weight)

    @staticmethod
    def custom_collate_fn(samples):
        """Stack tokenized samples into a model batch.

        :param samples: Dataset rows containing ``input_ids`` and ``attention_mask``.
        :return: Batch dictionary with tensor values stacked on the leading axis.
        """
        included_keys = [
            'input_ids',
            'attention_mask',
            'graph_node_count',
            'graph_edge_count',
            'action_candidate_count',
            'candidate_mask',
            'candidate_resource_type_id',
            'candidate_parent_type_id',
            'candidate_relation_name_id',
            'candidate_depth_bucket',
            'candidate_method_id',
            'candidate_has_action_target',
            'candidate_is_collection',
            'candidate_is_oem',
            'candidate_path_segment_hashes',
            'candidate_allowed_method_mask',
            'candidate_local_state_summary',
            'scope_mask',
            'scope_resource_type_id',
            'scope_parent_type_id',
            'scope_relation_name_id',
            'scope_depth_bucket',
            'scope_method_id',
            'scope_has_action_target',
            'scope_is_collection',
            'scope_is_oem',
            'scope_path_segment_hashes',
            'scope_allowed_method_mask',
            'scope_local_state_summary',
            'candidate_endpoint_scope_index',
        ]
        batch = {
            key: torch.stack([s[key] for s in samples])
            for key in included_keys
            if key in samples[0]
        }
        for key in ("state_fingerprint", "state_id"):
            if key in samples[0]:
                batch[key] = [s[key] for s in samples]
        return batch

    @torch.no_grad()
    def sample(self, batch):
        """
        Encode one tokenized batch into hidden states.

        :param batch: Tokenized batch accepted by the backbone model.
        :return: Last hidden state tensor on the trainer device.
        """
        with torch.no_grad():
            output = self._encoder_model(**self._model_inputs(batch))
        return output.last_hidden_state.to(self.device)

    @staticmethod
    def _model_inputs(batch):
        """Return only the Hugging Face model input tensors from a structured batch."""
        return {k: batch[k] for k in ("input_ids", "attention_mask") if k in batch}

    @torch.no_grad()
    def sample_all(self):
        """
        Encode the training split into a list of CPU hidden-state tensors.

        :return: List of detached hidden-state tensors sampled from the training split.
        """
        train_dataset, _ = self.split_slice_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=AutoencoderTrainer.custom_collate_fn)

        tensors = []
        with torch.no_grad():
            for batch in train_dataloader:
                hidden_state = self._encoder_model(**self._model_inputs(batch)).last_hidden_state
                tensors.append(hidden_state.detach().cpu())

        return tensors

    def measure_reconstruction(self, test_data):
        """
        Measure autoencoder reconstruction over hidden-state tensor batches.

        :param test_data: Iterable of tensors shaped for ``self.model_autoencoder``.
        :return: Average reconstruction loss across the provided batches.
        """
        self.model_autoencoder.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_data:
                batch = self._to_autoencoder_dtype(batch.to(self.device))
                reconstructed = self.model_autoencoder(batch)
                batch = batch.view(batch.shape[0], -1)
                loss = F.mse_loss(batch, reconstructed, reduction="mean")
                total_loss += loss.item()

        average_loss = total_loss / len(test_data)
        return average_loss

    def _to_autoencoder_dtype(self, tensor):
        """Cast backbone hidden states to the autoencoder's parameter dtype.

        A bf16 backbone (``--llm_torch_dtype bfloat16``) emits bf16 hidden
        states, but the autoencoder is fp32, so feeding them directly raises
        ``Input type (BFloat16) and bias type (float) should be the same``.

        :param tensor: hidden-state tensor from the backbone.
        :return: the tensor cast to the autoencoder's dtype.
        """
        param = next(self.model_autoencoder.parameters(), None)
        return tensor if param is None else tensor.to(param.dtype)

    def metric_name(self, name: str) -> str:
        """Return the purpose-qualified metric key for M2 autoencoder training."""
        prefix = getattr(self._trainer_args, "metric_prefix", None) or "m2/state_autoencoder"
        return f"{prefix}/{name}"

    @staticmethod
    def reached_max_steps(global_step: int, max_steps: Optional[int]) -> bool:
        """Whether the M2 optimizer-step cap has been reached."""
        return max_steps is not None and max_steps > 0 and global_step >= max_steps

    def train(self):
        """
        Train the autoencoder and save the resulting model.
        """
        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        if self._module_checkpoint_dir is not None:
            # load_checkpoint returns a CheckpointState namedtuple; the resume epoch
            # is its last_epoch field, not the state object itself.
            last_epoch = self.load_checkpoint(
                self._module_checkpoint_dir, model=self.model_autoencoder).last_epoch
        else:
            last_epoch = 0

        # torch.cuda.empty_cache()

        self.logger.info("Starting training")
        train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self._num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=AutoencoderTrainer.custom_collate_fn)

        self.model_autoencoder.eval()
        self.model_autoencoder.train()
        self.model_autoencoder.to(self.device)
        # the frozen backbone must sit on the same device as its inputs; it was
        # never placed, silently running the big forward on CPU on GPU nodes.
        self._encoder_model.to(self.device)

        total_batches = len(train_dataloader)
        batch_log_frequency = round(32 * 0.2)
        global_opt_steps = 0
        run_started_clock = time.time()

        for epoch in range(last_epoch, self.num_epochs):
            if self.reached_max_steps(global_opt_steps, self._max_train_steps):
                break
            total_loss = 0.0
            num_batches = 0

            batch_losses = np.zeros(total_batches)
            for batch in train_dataloader:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    output = self._encoder_model(**self._model_inputs(batch)).last_hidden_state.to(self.device)
                output = self._to_autoencoder_dtype(output)
                reconstructed = self.model_autoencoder(output)
                output = output.view(output.shape[0], -1)
                loss = F.mse_loss(output, reconstructed, reduction="none")
                loss = loss.mean()

                self.optimizer.zero_grad()
                # self.accelerator.backward(loss)
                loss.backward()
                self.optimizer.step()
                global_opt_steps += 1

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()

                # calculate the progress percentage
                progress_percentage = int(round((num_batches + 1) / total_batches * 100))
                if (num_batches % batch_log_frequency == 0) or (num_batches == total_batches - 1):
                    print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Batch "
                          f"{num_batches + 1}/{total_batches} "
                          f"- Progress: {progress_percentage:.2f}% - Batch Loss mean: {batch_losses.mean():.4f}")
                    if self.is_rank_zero():
                        step = epoch * total_batches + num_batches
                        graph_nodes = batch.get("graph_node_count")
                        graph_edges = batch.get("graph_edge_count")
                        action_candidates = batch.get("action_candidate_count")
                        candidate_mask = batch.get("candidate_mask")
                        self.metric_logger.log_metric(
                            self.metric_name("train/reconstruction_loss"),
                            batch_losses.mean(),
                            step,
                        )
                        self.metric_logger.log_metric(
                            self.metric_name("train/optimizer_step"),
                            float(global_opt_steps),
                            step,
                        )
                        if graph_nodes is not None:
                            self.metric_logger.log_metric(
                                self.metric_name("state/graph_nodes_per_batch"),
                                float(graph_nodes.float().mean().item()),
                                step,
                            )
                        if graph_edges is not None:
                            self.metric_logger.log_metric(
                                self.metric_name("state/graph_edges_per_batch"),
                                float(graph_edges.float().mean().item()),
                                step,
                            )
                        if action_candidates is not None:
                            self.metric_logger.log_metric(
                                self.metric_name("state/action_candidates_per_batch"),
                                float(action_candidates.float().mean().item()),
                                step,
                            )
                        if candidate_mask is not None:
                            self.metric_logger.log_metric(
                                self.metric_name("state/legal_candidate_slots"),
                                float(candidate_mask.float().sum(dim=1).mean().item()),
                                step,
                            )
                    self.metric_logger.log_metric("state_auto_encoder_batch", batch_losses.mean(), epoch)

                num_batches += 1
                if self.reached_max_steps(global_opt_steps, self._max_train_steps):
                    break

            # epoch end
            if num_batches > 0:
                average_loss = total_loss / num_batches
                if self.is_rank_zero():
                    epoch_step = (epoch + 1) * total_batches - 1
                    elapsed = max(time.time() - run_started_clock, 1e-9)
                    samples_seen = float(global_opt_steps * self.batch_size)
                    self.metric_logger.log_metric(
                        self.metric_name("train/epoch_reconstruction_loss"),
                        average_loss,
                        epoch_step,
                    )
                    self.metric_logger.log_metric(
                        self.metric_name("throughput/samples_per_sec"),
                        samples_seen / elapsed,
                        epoch_step,
                    )
                    self.metric_logger.log_metric(
                        self.metric_name("dataset/train_count"),
                        float(len(self.dataset)),
                        epoch_step,
                    )
                    self.metric_logger.log_metric(
                        self.metric_name("dataset/train_batches"),
                        float(total_batches),
                        epoch_step,
                    )
                    self.metric_logger.log_metric("state_auto_encoder_epoch", average_loss, epoch)
                print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

            # save best checkpoint
            if self.is_rank_zero() and epoch % 20 == 0:
                self.save_checkpoint(
                    self._module_checkpoint_dir, epoch + 1,
                    model=self.model_autoencoder, optimizer=self.optimizer)

        self.save_model(self._module_checkpoint_dir, model=self.model_autoencoder)
