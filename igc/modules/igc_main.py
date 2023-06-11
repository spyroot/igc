import argparse
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_llm_module import IgcLanguageModule
from .igc_rl_module import IgcRlModule
from ..ds.redfish_dataset import JSONDataset


def from_pretrained_default(args):
    """
    :param args:
    :return:
    """
    model = GPT2LMHeadModel.from_pretrained(args.model_type)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    return model, tokenizer


class IgcMain:
    """
    IGC main class

    """

    def __init__(self, specs: argparse.Namespace, from_pretrained=from_pretrained_default):
        """
        :param specs:
        """
        self._from_pretrained_fn = from_pretrained
        self._metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        self._directory_path = os.path.expanduser(specs.raw_data_dir)
        self._specs = specs

    def train(self):
        """
        Main igc trainer.

        :return:
        """
        print("Starting training.")

        if self._specs.train:
            model, tokenizer = self._from_pretrained_fn(self._specs)
            dataset = JSONDataset(
                self._directory_path, verbose=True, tokenizer=tokenizer)

            if (self._specs.train == "llm" or self._specs.train == "all") and self._specs.llm is not None:
                print("Starting RL training")
                llm_module = IgcLanguageModule(self._specs, self._metric_logger, dataset)
                llm_module.train()

            if (self._specs.train == "agent" or self._specs.train == "all") and self._specs.rl is not None:
                print("Starting RL training")
                rl_module = IgcRlModule(self._specs, self._metric_logger, dataset)
                rl_module.train()

    def load(self):
        pass
        # # Load the pre-trained GPT model
        # gpt_model = GPT2Model.from_pretrained('gpt2')
        # # Define the input dimensions and latent dimensions for the autoencoder
        # input_dim = gpt_model.config.hidden_size
        # latent_dim = 128
        #
        # # Create instances of the GPT model and the Autoencoder
        # gpt_encoder = gpt_model.get_input_embeddings()
        #
        # # Attach the autoencoder to the GPT model
        # gpt_encoder.weight = nn.Parameter(model_autoencoder.encoder.weight)

    def run(self):
        self.train()
