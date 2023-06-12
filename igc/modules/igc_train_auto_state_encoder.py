import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch import autocast

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

    def train(self):
        """
        :return:
        """
        self.logger.info(
            f"Rank {self.rank} starting train, device {self.device}")

        torch.cuda.empty_cache()

        # self._encoder_model.to(self.device)
        self._encoder_model.eval()

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

        # train_dataloader, self.model_autoencoder, self.optimizer,  = self.accelerator.prepare(
        #     train_dataloader, self.model_autoencoder, self.optimizer)

        train_dataloader, self.model_autoencoder, self.optimizer = self.accelerator.prepare(
            [train_dataloader, self.model_autoencoder, self.optimizer],
            device_placement=[True])

        self.model_autoencoder.train()

        # self.model_autoencoder.train()
        self.model_autoencoder.to(self.device)
        # batch = {key: value.to(self.device) for key, value in batch.items()}
        # training loop

        for epoch in range(last_epoch, self.num_epochs):
            total_loss = 0.0
            for batch in train_dataloader:
                # with torch.no_grad():
                #     output = self._encoder_model(**batch)

                hidden_state = self.sample(batch)
                flat_input = hidden_state.view(hidden_state.shape[0], -1).to(self.device)
                latent_repr = self.model_autoencoder.encoder(flat_input).to(self.device)
                latent_repr = latent_repr.to(self.device).to(self.device)
                reconstructed = self.model_autoencoder.decoder(latent_repr).to(self.device)
                loss = F.mse_loss(flat_input, reconstructed, reduction="none")
                loss = loss.mean()

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                # loss.backward()
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
            num_workers=self.num_workers,
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

    def train_offline(self):
        """
        :return:
        """
        # tensors = self.sample_all()
        # self.logger.info(
        #     f"Rank {self.rank} starting train, device {self.device}")
        #
        # del self._encoder_model
        # del self.model
        #
        # print(self.model_autoencoder)

        # torch.cuda.empty_cache()
        # # self._encoder_model.to(self.device)
        # self._encoder_model.eval()
        #
        # if self.module_checkpoint_dir is not None:
        #     last_epoch = self.load_checkpoint(self.module_checkpoint_dir)
        # else:
        #     last_epoch = 0
        #
        # num_epochs = 10
        # self.logger.info(f"Starting training")
        #
        # train_dataset, _ = self.split_slice_dataset()
        # train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=self.batch_size,
        #     sampler=None,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     shuffle=True,
        #     collate_fn=AutoencoderTrainer.custom_collate_fn)
        #
        # # train_dataloader, self.model_autoencoder, self.optimizer,  = self.accelerator.prepare(
        # #     train_dataloader, self.model_autoencoder, self.optimizer)
        #
        # train_dataloader, self.model_autoencoder, self.optimizer = self.accelerator.prepare(
        #     [train_dataloader, self.model_autoencoder, self.optimizer],
        #     device_placement=[True])
        #
        # self.model_autoencoder.train()
        #


        train_dataset, _ = self.split_slice_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=AutoencoderTrainer.custom_collate_fn)

        self.model_autoencoder.eval()

        self.model_autoencoder.train()
        self.model_autoencoder.to(self.device)
        # # batch = {key: value.to(self.device) for key, value in batch.items()}
        # # training loop

        # batch = tensors[0]
        # batch = batch.to(self.device)

        for epoch in range(0, self.num_epochs):
            total_loss = 0.0
            for batch in train_dataset:
                with torch.no_grad():
                    output = self._encoder_model(**batch)
                    output.last_hidden_state.to(self.device)
                hidden_state = batch.to(self.device)
                reconstructed = self.model_autoencoder(hidden_state)
                batch = batch.view(batch.shape[0], -1)
                loss = F.mse_loss(batch, reconstructed, reduction="none")
                loss = loss.mean()
                #
                self.optimizer.zero_grad()
                # self.accelerator.backward(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        #
        #     # Print the average loss for the epoch
            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Average Loss: {average_loss}")
        #
        # # self.save_model()
        # # # Save the trained model if desired
        # # torch.save(self.model.state_dict(), "trained_model.pth")
