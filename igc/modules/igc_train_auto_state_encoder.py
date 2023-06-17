"""
This class is used to train a state auto encoder.

Here is idea. We take REST API and pass to GPT
We take hidden representation and pass 1D conv layer with kernel size 2
That essentially create a polling layer with stride 2
We pass the two an encoder and reduce the dimension to fixed size block
Then do standard autoencoder procedure and reconstruct.

Author:Mus mbayramo@stanford.edu
"""
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from igc.ds.redfish_dataset import JSONDataset
from .base.igc_base_module import IgcBaseModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_autoencoder import AutoStateEncoder
from igc.shared.shared_torch_builder import TorchBuilder
import torch.nn.functional as F


class AutoencoderTrainer(IgcBaseModule):
    """

    Autoencoder trainer used to train the autoencoder to reduce
    state dimension of latent space llm outputs.

    """
    def __init__(self,
                 module_name: str,
                 spec: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = "False",
                 device: Optional[torch.device] = None):
        """

        :param module_name:
        :param spec:
        :param ds:
        :param metric_logger:
        :param llm_model:
        :param llm_tokenizer:
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

        self.logger.info(f"Creating auto-encoder input dim"
                         f" {self._input_dim} {self._latent_dim} batch_size: {self.batch_size}")

        self._encoder_model = self.model.transformer
        llm_model.transformer.config.is_decoder = False
        # self._llm_model.resize_token_embeddings(len(llm_tokenizer))
        input_shape = self._encoder_model.wpe.weight.shape
        self.emb_shape = (input_shape[0] - 1, input_shape[1])

        self.model_autoencoder = AutoStateEncoder(input_shape=input_shape)

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

    def _get_reconstruction_loss(self, batch):
        """
        Compute reconstruction loss

        :param batch:
        :return:
        """
        x, _ = batch
        x_hat = self.model.forward(x)
        loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
        return loss

    def encode(self):
        """
        :return:
        """
        input_dim = self.model.config.hidden_size
        latent_dim = 128

        gpt_encoder = self.model.get_input_embeddings()
        autoencoder = AutoStateEncoder(input_dim, latent_dim)

        # attach the autoencoder to the GPT model
        gpt_encoder.weight = nn.Parameter(autoencoder.encoder.weight)

    @staticmethod
    def custom_collate_fn(samples):
        """Collate data before we pass to the model.
        :param samples:
        :return:
        """
        included_keys = ['input_ids', 'attention_mask']
        batch = {key: torch.stack([s[key] for s in samples]) for key in included_keys}
        return batch

    @torch.no_grad()
    def sample(self, batch):
        with torch.no_grad():
            output = self._encoder_model(**batch)
        return output.last_hidden_state.to(self.device)

    @torch.no_grad()
    def sample_all(self):
        """
        :return:
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
                hidden_state = self._encoder_model(**batch).last_hidden_state
                tensors.append(hidden_state.detach().cpu())

        return tensors

    def measure_reconstruction(self, test_data):
        """
        Measure the reconstruction performance
        of the autoencoder on the test data.

        :param test_data: Test dataset
        :return: Average reconstruction loss
        """
        self.model_autoencoder.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_data:
                batch = batch.to(self.device)
                reconstructed = self.model_autoencoder(batch)
                batch = batch.view(batch.shape[0], -1)
                loss = F.mse_loss(batch, reconstructed, reduction="mean")
                total_loss += loss.item()

        average_loss = total_loss / len(test_data)
        return average_loss

    def train(self):
        """
        :return:
        """
        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        if self.module_checkpoint_dir is not None:
            last_epoch = self.load_checkpoint(self.module_checkpoint_dir)
        else:
            last_epoch = 0

        # torch.cuda.empty_cache()

        self.logger.info(f"Starting training")
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

        total_batches = len(train_dataloader)
        batch_log_frequency = round(32 * 0.2)

        for epoch in range(last_epoch, self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            batch_losses = np.zeros(total_batches)
            for batch in train_dataloader:
                with torch.no_grad():
                    output = self._encoder_model(**batch).last_hidden_state.to(self.device)
                reconstructed = self.model_autoencoder(output)
                output = output.view(output.shape[0], -1)
                loss = F.mse_loss(output, reconstructed, reduction="none")
                loss = loss.mean()

                self.optimizer.zero_grad()
                # self.accelerator.backward(loss)
                loss.backward()
                self.optimizer.step()

                batch_losses[num_batches] = loss.item()
                total_loss += loss.item()

                # calculate the progress percentage
                progress_percentage = int(round((num_batches + 1) / total_batches * 100))
                if (num_batches % batch_log_frequency == 0) or (num_batches == total_batches - 1):
                    print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Batch "
                          f"{num_batches + 1}/{total_batches} "
                          f"- Progress: {progress_percentage:.2f}% - Batch Loss mean: {batch_losses.mean():.4f}")
                    self.metric_logger.log_metric("state_auto_encoder_batch", batch_losses.mean(), epoch)

                num_batches += 1

            # epoch end
            if num_batches > 0:
                average_loss = total_loss / num_batches
                if self.is_rank_zero():
                    self.metric_logger.log_metric("state_auto_encoder_epoch", average_loss, epoch)
                print(f"Rank {self.rank} Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

            # save best checkpoint
            if self.is_rank_zero() and epoch % 20 == 0:
                self.save_checkpoint(self.module_checkpoint_dir, epoch + 1)

        self.save_model(self.module_checkpoint_dir)
