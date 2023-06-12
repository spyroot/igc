import argparse
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from igc.ds.redfish_dataset import JSONDataset
from .base.igc_base_module import IgcBaseModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_autoencoder import AutoStateEncoder
from igc.shared.shared_torch_builder import TorchBuilder
import torch.nn.functional as F
from accelerate import Accelerator


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
                 is_inference: Optional[bool] = "False"):
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
            is_inference=is_inference)

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
        self.model_autoencoder = AutoStateEncoder()

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
        return output.last_hidden_state

    def train(self):
        """
        :return:
        """
        accelerator = Accelerator(device_placement=True, split_batches=True)
        self.device = accelerator.device

        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        torch.cuda.empty_cache()

        # self._encoder_model.to(self.device)
        self._encoder_model.eval()
        self.model_autoencoder.to(self.device)
        self.model_autoencoder.train()

        if self.module_checkpoint_dir is not None:
            last_epoch = self.load_checkpoint(self.module_checkpoint_dir)
        else:
            last_epoch = 0

        num_epochs = 10
        self.logger.info(f"Starting training")

        train_dataset, _ = self.split_slice_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            collate_fn=AutoencoderTrainer.custom_collate_fn)

        self.model, self.optimizer, train_dataloader = accelerator.prepare(
            [self.model_autoencoder, self.optimizer, train_dataloader],
            device_placement=[True])

        # batch = {key: value.to(self.device) for key, value in batch.items()}
        # training loop
        for epoch in range(last_epoch, self.num_epochs):
            total_loss = 0.0
            for batch in train_dataloader:
                # with torch.no_grad():
                #     output = self._encoder_model(**batch)
                hidden_state = self.sample(batch)
                hidden_state = hidden_state.to(self.device)
                print("req grad", hidden_state.requires_grad)  # Check if requires_grad is False

                flat_input = hidden_state.view(hidden_state.shape[0], -1)
                latent_repr = self.model_autoencoder.encoder(flat_input)
                reconstructed = self.model_autoencoder.decoder(latent_repr)
                loss = F.mse_loss(flat_input, reconstructed, reduction="none")
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Loss {loss}")

                # Update the total loss
                total_loss += loss.item()

            # Print the average loss for the epoch
            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}")

        # Perform validation or evaluation steps if needed
        # self.save_model()
        # # Save the trained model if desired
        # torch.save(self.model.state_dict(), "trained_model.pth")
