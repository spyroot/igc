import argparse
import logging
import os
from pathlib import Path

import torch

from .base.igc_metric_logger import MetricLogger
from .base.igc_state import IgcBaseState
from .igc_train_agent import IgcAgentTrainer
from .igc_train_auto_state_encoder import AutoencoderTrainer
from .llm.igc_llm_module import IgcLanguageModule
from ..ds.redfish_dataset import JSONDataset
from ..envs.rest_gym_batch_env import VectorizedRestApiEnv


class IgcRlModule(IgcBaseState):
    """
    """

    def __init__(
        self,
        module_name: str,
        spec: argparse.Namespace,
        metric_logger: MetricLogger,
        ds: JSONDataset,
        llm_model=None,
        device=None,
    ):
        """
        :param spec:
        """
        super().__init__(module_name, spec)

        if llm_model is None:
            llm_model, _, last_epoch = IgcLanguageModule.load(spec, module_names="state_encoder")
        else:
            llm_model = llm_model

        tokenizer = ds.tokenizer
        # encoder = BaseEncoder(model=llm_model, tokenizer=tokenizer, device=device)

        # create env
        env = VectorizedRestApiEnv(
            args=spec,
            model=llm_model,
            tokenizer=tokenizer,
            discovered_rest_api=ds,
            max_episode=spec.max_trajectory_length,
            num_envs=spec.rl_batch_size,
            device=device
            # encoder=encoder
        )

        directory_path = os.path.expanduser(spec.raw_data_dir)

        # module_name,
        # spec,
        # llm_model,
        # llm_tokenizer,
        # ds = ds,
        # metric_logger = metric_logger,
        # is_inference = is_inference,
        # device = device

        # def __init__(self,
        #              module_name: str,
        #              spec: argparse.Namespace,
        #              llm_model,
        #              llm_tokenizer,
        #              ds: Optional[JSONDataset] = None,
        #              metric_logger: Optional[MetricLogger] = None,
        #              is_inference: Optional[bool] = "False",
        #              device: Optional[torch.device] = None):

        self.spec = spec
        self.autoencoder = AutoencoderTrainer(
            "state_autoencoder",
            spec=spec,
            llm_model=llm_model,
            llm_tokenizer=tokenizer,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=True,
            device=device
        )

        self.rl_gent = IgcAgentTrainer(
            "rl_agent",
            spec=spec,
            llm_model=llm_model,
            llm_tokenizer=ds.tokenizer,
            env=env,
            ds=ds,
            metric_logger=metric_logger,
            is_inference=False,
            device=device
        )

    def train(self):
        """Main call to train all language models.
        :return:
        """
        self.rl_gent.train()

    @staticmethod
    def load_rl_model(
        args: argparse.Namespace,
        device: torch.device = "cpu"
    ):
        """
        Load RL Agent.
        i.e. agent will use this as encoder

        :param args: The command-line arguments.
        :param device: The device to load the model onto, defaults to "cpu".
        :return: The loaded model, tokenizer, and the last epoch from the checkpoint.
        """
        checkpoint_path_dir = Path(args.output_dir)
        checkpoint_path_dir = checkpoint_path_dir.resolve()
        checkpoint_path_dir = checkpoint_path_dir.absolute()

        logging.info(F"absolute {str(args.output_dir)}")
        if not checkpoint_path_dir.is_dir():
            raise ValueError("Invalid checkpoint directory.")

        checkpoint_files = [f for f in os.listdir(checkpoint_path_dir) if f.endswith('.pt')]
        checkpoint_files = [os.path.join(checkpoint_path_dir, f) for f in checkpoint_files]
        checkpoint_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if not checkpoint_files:
            raise ValueError(
                f"No checkpoint files found "
                f"in the checkpoint directory {checkpoint_files}.")

        # model = GPT2LMHeadModel.from_pretrained(args.model_type)
        # last_epoch = IgcLanguageModule.load(
        #     model, args)

        # TODO
        return None, 1
