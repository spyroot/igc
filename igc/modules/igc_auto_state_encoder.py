import argparse

import torch
import torch.nn as nn
from transformers import GPT2Model

from igc.ds.redfish_dataset import JSONDataset
from igc.shared.shared_torch_builder import TorchBuilder
from .base.igc_base_module import IgcBaseModule
from .base.igc_metric_logger import MetricLogger
from .igc_autoencoder import AutoStateEncoder


class AutoencoderTrainer(IgcBaseModule):
    def __init__(self,
                 module_name: str,
                 input_dim,
                 latent_dim,
                 args: argparse.Namespace,
                 ds: JSONDataset,
                 metric_logger: MetricLogger,
                 spec: argparse.Namespace,
                 llm_model, llm_tokenizer):
        """
        :param input_dim:
        :param latent_dim:
        """
        super().__init__(module_name, spec, ds, metric_logger, llm_model, llm_tokenizer)

        self.llm_model = None
        self.auto_encoder = None
        self.train_dataloader = None

        num_epochs = 10
        learning_rate = 0.001

        self.optimizer = TorchBuilder.create_optimizer(
            args.llm_optimizer,
            self.model,
            args.auto_encoder_learning_rate,
            args.auto_encoder_weight_decay,
            **vars(args)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def _get_reconstruction_loss(self, batch):
        """
        :param batch:
        :return:
        """
        x, _ = batch
        x_hat = self.forward(x)
        loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def encode(self):
        """
        :return:
        """
        input_dim = self.llm_model.config.hidden_size
        latent_dim = 128

        # Create instances of the GPT model and the Autoencoder
        gpt_encoder = self.llm_model.get_input_embeddings()
        autoencoder = AutoStateEncoder(input_dim, latent_dim)

        # Attach the autoencoder to the GPT model
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

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            # Iterate over the training dataset
            for input_data, target in self.train_dataloader:
                latent_repr = self.auto_encoder.encoder(input_data)
                output = self.gpt_model(input_data, latent_repr)

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

        # Save the trained model if desired
        torch.save(self.model.state_dict(), "trained_model.pth")

        # Load the pre-trained GPT model
        gpt_model = GPT2Model.from_pretrained('gpt2')
        # Define the input dimensions and latent dimensions for the autoencoder
        input_dim = gpt_model.config.hidden_size
        latent_dim = 128

        # Create instances of the GPT model and the Autoencoder
        gpt_encoder = gpt_model.get_input_embeddings()
        autoencoder = AutoStateEncoder(input_dim, latent_dim)

        # Attach the autoencoder to the GPT model
        gpt_encoder.weight = nn.Parameter(autoencoder.encoder.weight)