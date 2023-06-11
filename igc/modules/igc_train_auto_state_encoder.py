import argparse
from typing import Optional

import torch
import torch.nn as nn

from igc.ds.redfish_dataset import JSONDataset
from igc.shared.shared_torch_builder import TorchBuilder
from .base.igc_base_module import IgcBaseModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_autoencoder import AutoStateEncoder


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

        self.llm_model = llm_model
        self.train_dataloader = None

        self.input_dim = spec.llm_model.config.hidden_size
        self.latent_dim = spec.auto_encoder_latent_dim
        self.model_autoencoder = AutoStateEncoder(self.input_dim, self.latent_dim)

        # 0.001
        self.num_epochs = spec.spec.num_epoch_train
        self.learning_rate = spec.auto_encoder_lr

        self.optimizer = TorchBuilder.create_optimizer(
            spec.llm_optimizer,
            self.model,
            spec.auto_encoder_learning_rate,
            spec.auto_encoder_weight_decay,
            **vars(spec)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _get_reconstruction_loss(self, batch):
        """
        Compute reconstruction loss

        :param batch:
        :return:
        """
        x, _ = batch
        x_hat = self.model.forward(x)
        loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def encode(self):
        """
        :return:
        """
        input_dim = self.llm_model.config.hidden_size
        latent_dim = 128

        # here create instances of the GPT model and the Autoencoder
        gpt_encoder = self.llm_model.get_input_embeddings()
        autoencoder = AutoStateEncoder(input_dim, latent_dim)

        # attach the autoencoder to the GPT model
        gpt_encoder.weight = nn.Parameter(autoencoder.encoder.weight)

    def train(self):
        """
        :return:
        """
        # Set the training parameters
        num_epochs = 10
        learning_rate = 0.001

        # Define your optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            for input_data, target in self.train_dataloader:
                latent_repr = self.model.encoder(input_data)
                output = self.llm_model(input_data, latent_repr)

                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the total loss
                total_loss += loss.item()

            # Print the average loss for the epoch
            average_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}")

            # Perform validation or evaluation steps if needed

        # self.save_model()
        # # Save the trained model if desired
        # torch.save(self.model.state_dict(), "trained_model.pth")
