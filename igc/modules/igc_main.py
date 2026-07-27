"""
This class is main class that encapsulate trainer logic.
The reason it's done this , I'm going some trainer logic to async io
since we can parallelize some of the training logic.

Author:Mus mbayramo@stanford.edu
"""
import argparse
import os
from pathlib import Path
from typing import Optional, Union, List, Dict

import torch

from .base.igc_base_module import IgcModule
from .base.igc_llm_base_module import LlmModule
from .base.igc_metric_logger import MetricLogger
from igc.modules.llm.igc_llm_module import IgcLanguageModule
from .igc_rl_module import IgcRlModule
from .train.sft import SFTTrainer
from .shared.llm_shared import safe_resize_token_embeddings
from .shared.llm_shared import (
    from_pretrained_default,
    load_pretrained_default,
    save_pretrained_default
)
from ..ds.redfish_masked_dataset import MaskedJSONDataset


class IgcMain:
    """
    IGC main class

    """

    def __init__(
        self,
        specs: argparse.Namespace,
        from_pretrained=from_pretrained_default,
        from_pretrained_load_fn=load_pretrained_default,
        from_pretrained_save_fn=save_pretrained_default,
    ):
        """
        from_pretrained_default creates initial model.
        if model saved in hugging face format we ue load_pretrained_default and save_pretrained_default
        to save and load model.

        :param specs: An `argparse.Namespace` object containing the specifications and arguments.
        :param from_pretrained: A function for loading the pretrained model and tokenizer.
        :param from_pretrained_load_fn: A function for loading a pretrained model from a directory.
        :param from_pretrained_save_fn: A function for saving the pretrained model and tokenizer.
        """
        self._dataset = None
        self._eval_dataset = None
        self._metric_logger = None
        self._from_pretrained_fn = from_pretrained
        self._from_pretrained_load_fn = from_pretrained_load_fn
        self._from_pretrained_save_fn = from_pretrained_save_fn

        self._metric_logger = MetricLogger(specs.metric_report, **vars(specs))
        self._directory_path = os.path.abspath(os.path.expanduser(specs.json_data_dir))
        self._dataset_dir = os.path.abspath(specs.dataset_dir)
        self._specs = specs

    @property
    def metric_logger(self):
        """
        :return:
        """
        if self._metric_logger is None:
            self._metric_logger = MetricLogger(
                self._specs.metric_report, **vars(self._specs))
        return self._metric_logger

    @property
    def dataset(self):
        """

        :return:
        """
        if self._dataset is None:
            task = None
            task_name = getattr(self._specs, "sft_task", "") or ""
            if task_name:
                from igc.modules.train.sft_tasks import resolve_sft_task

                task = resolve_sft_task(task_name)
                self._validate_sft_lineage(task)
                self._specs.task_spec_sha = task.spec_sha256
                self._specs.parent_role = task.parent_role
                self._specs.output_role = task.output_role
                self._specs.phase_number = task.phase
                if task.phase in (2, 3):
                    return self._labelled_sft_dataset(task)
            corpus_dir, eval_corpus_dir = self._corpus_dataset_dirs(
                strict_phase1=bool(task),
            )
            if corpus_dir:
                from igc.ds.corpus_dataset import CorpusJSONLDataset
                structural_loss_profile = getattr(
                    self._specs,
                    "phase1_structural_loss_profile",
                    "none",
                )
                structural_loss_seed = int(getattr(self._specs, "seed", 42))
                self._dataset = CorpusJSONLDataset(
                    corpus_dir,
                    default_tokenize=self._specs.model_type,
                    max_len=self._specs.seq_len,
                    objective=getattr(self._specs, "corpus_objective", "legacy"),
                    phase1_structural_loss_profile=structural_loss_profile,
                    phase1_structural_loss_mode="train",
                    phase1_structural_loss_seed=structural_loss_seed,
                )
                self._eval_dataset = CorpusJSONLDataset(
                    eval_corpus_dir,
                    default_tokenize=self._specs.model_type,
                    max_len=self._specs.seq_len,
                    tokenizer=self._dataset.tokenizer,
                    objective=getattr(self._specs, "corpus_objective", "legacy"),
                    phase1_structural_loss_profile=structural_loss_profile,
                    phase1_structural_loss_mode="evaluation",
                    phase1_structural_loss_seed=structural_loss_seed,
                )
                train_structural_loss_sha = getattr(
                    self._dataset,
                    "phase1_structural_loss_spec_sha",
                    "",
                )
                eval_structural_loss_sha = getattr(
                    self._eval_dataset,
                    "phase1_structural_loss_spec_sha",
                    "",
                )
                if train_structural_loss_sha != eval_structural_loss_sha:
                    raise ValueError(
                        "Phase 1 train and held-out structural-loss specs must match"
                    )
                self._specs.phase1_structural_loss_spec_sha = (
                    train_structural_loss_sha
                )
                self._dataset.set_eval_split_sha256(self._eval_dataset.data_sha256)
            elif task is not None:
                raise ValueError("--corpus_dir or --corpus_manifest is required for Phase 1 SFT")
            else:
                self._dataset = MaskedJSONDataset(
                    self._dataset_dir,
                    default_tokenize=self._specs.model_type,
                    max_len=self._specs.seq_len,
                    recreate_dataset=self._specs.recreate_dataset,
                    do_consistency_check=self._specs.do_consistency_check
                )
        return self._dataset

    @property
    def eval_dataset(self):
        """Return the explicit immutable held-out dataset for shared SFT."""
        if self._dataset is None:
            _ = self.dataset
        return self._eval_dataset

    def _labelled_sft_dataset(self, task):
        """Build the Phase 2/3 JSONL adapter selected by the YAML task."""
        from igc.ds.rest_goal_contract import render_phase2_sft, render_phase3_sft
        from igc.ds.sft_dataset import PromptCompletionJSONLDataset

        data_path = getattr(self._specs, "sft_data_path", "") or ""
        eval_data_path = getattr(self._specs, "sft_eval_data_path", "") or ""
        data_manifest = getattr(self._specs, "sft_data_manifest", "") or ""
        eval_manifest = getattr(self._specs, "sft_eval_manifest", "") or ""
        if not data_path:
            raise ValueError(f"--sft_data_path is required for SFT phase {task.phase}")
        if not eval_data_path:
            raise ValueError(
                f"--sft_eval_data_path is required for SFT phase {task.phase}"
            )
        if not data_manifest:
            raise ValueError(f"--sft_data_manifest is required for SFT phase {task.phase}")
        if not eval_manifest:
            raise ValueError(f"--sft_eval_manifest is required for SFT phase {task.phase}")
        renderers = {
            2: render_phase2_sft,
            3: render_phase3_sft,
        }
        self._dataset = PromptCompletionJSONLDataset(
            data_path,
            renderer=renderers[task.phase],
            metric_namespace=task.metric_namespace,
            max_len=self._specs.seq_len,
            tokenizer_name=self._specs.model_type,
            manifest_path=data_manifest,
            require_manifest=True,
            expected_dataset="D1",
        )
        self._eval_dataset = PromptCompletionJSONLDataset(
            eval_data_path,
            renderer=renderers[task.phase],
            metric_namespace=task.metric_namespace,
            max_len=self._specs.seq_len,
            tokenizer=self._dataset.tokenizer,
            manifest_path=eval_manifest,
            require_manifest=True,
            expected_dataset="D1",
        )
        self._dataset.bind_eval_dataset(self._eval_dataset)
        return self._dataset

    def _validate_sft_lineage(self, task) -> None:
        """Require each SFT profile to agree with its task and parent artifact."""
        weights_role = getattr(self._specs, "weights_role", "") or ""
        if weights_role != task.output_role:
            raise ValueError(
                f"weights_role {weights_role!r} must match task output_role "
                f"{task.output_role!r}"
            )
        parent_adapter = getattr(self._specs, "parent_adapter_dir", "") or ""
        parent_sha = getattr(self._specs, "parent_artifact_sha", "") or ""
        if task.phase == 1:
            if parent_adapter:
                raise ValueError("Phase 1 must initialize from the foundation model")
            return
        if not parent_adapter:
            raise ValueError(f"Phase {task.phase} requires --parent_adapter_dir")
        if not parent_sha:
            raise ValueError(f"Phase {task.phase} requires --parent_artifact_sha")

    def _corpus_dataset_dirs(self, *, strict_phase1: bool = False) -> tuple[str, str]:
        """Return immutable train/held-out corpus dirs, materializing when needed.

        ``--corpus_dir`` points directly at an existing ``examples.jsonl`` corpus.
        ``--corpus_manifest`` + ``--corpus_root`` point at redfish_ctl's materialized
        corpus manifest; this method composes those sources into the same written
        corpus format so the tokenizer bridge has one live input contract.

        :return: ``(train_dir, heldout_dir)`` or ``("", "")`` for the legacy path.
        :raises ValueError: when manifest inputs are incomplete or select no sources.
        """
        corpus_dir = getattr(self._specs, "corpus_dir", "") or ""
        if corpus_dir:
            eval_corpus_dir = getattr(self._specs, "corpus_eval_dir", "") or ""
            if not eval_corpus_dir:
                raise ValueError("--corpus_eval_dir is required with --corpus_dir")
            return corpus_dir, eval_corpus_dir

        manifest = getattr(self._specs, "corpus_manifest", "") or ""
        if not manifest:
            return "", ""
        if strict_phase1:
            raise ValueError(
                "Phase 1 SFT requires canonical D0 --corpus_dir and "
                "--corpus_eval_dir outputs from build_phase1_registry_corpus.py"
            )

        root = getattr(self._specs, "corpus_root", "") or ""
        if not root:
            raise ValueError("--corpus_root is required when --corpus_manifest is set")

        from igc.ds.sources import RedfishFixtureSource, TrustLevel
        from igc.ds.sources.corpus_io import write_corpus
        from igc.ds.sources.mixer import SourceMix
        from igc.ds.sources.training_object import normalize

        kind = getattr(self._specs, "corpus_kind", "dataset") or "dataset"
        sources = RedfishFixtureSource.from_redfish_ctl_manifest(
            manifest,
            root,
            trust_level=TrustLevel.REAL,
            kind=kind,
        )
        if not sources:
            raise ValueError(
                f"no redfish_ctl corpus sources selected from {manifest} "
                f"under {root} for kind={kind}"
            )

        out_root = Path(self._dataset_dir) / "redfish_ctl_corpus" / str(kind).lower()
        train_dir = out_root / "train"
        heldout_dir = out_root / "heldout"
        mix = SourceMix(sources)
        train_records, heldout_records = mix.split()
        if not train_records or not heldout_records:
            raise ValueError("corpus split must produce non-empty train and held-out sets")
        manifest_value = mix.manifest()
        write_corpus(normalize(train_records), manifest_value, str(train_dir))
        write_corpus(normalize(heldout_records), manifest_value, str(heldout_dir))
        return str(train_dir), str(heldout_dir)

    def train(self):
        """
        Main igc trainer.

        * Fine tune llm and save the model experiments/state_encoder
        * Training auto encoder and save the model experiments/state_auto_encoder
        Optional:
                * Use fine-tuned model and train goal extractor.
                * Use fined-tuned model and trains sub-goal and parameters extractor.

        * Use tune tuned model and load state_auto_encoder
        * Create vectorized rest api vectorized env and use auto encoder to compress state.
        * Train RL agent.

        :return:
        """
        if self._specs.train:

            if (self._specs.train == "llm" or self._specs.train == "all") and self._specs.llm is not None:
                llm_module = IgcLanguageModule(
                    self._specs,
                    self.metric_logger,
                    self.dataset,
                    eval_ds=self.eval_dataset,
                )
                llm_module.train()

            if (self._specs.train == "agent" or self._specs.train == "all") and self._specs.rl is not None:
                rl_module = IgcRlModule(
                    "rl_agent", self._specs, self.metric_logger, self.dataset,
                    device=self._specs.device)
                rl_module.train()

    def load(
        self,
        specs: argparse.Namespace,
        module_names: Optional[Union[str, List[str]]] = None,
        device: torch.device = "cpu"
    ) -> Dict[str, Union[LlmModule, IgcModule]]:
        """

        Load a module.

        :param module_names: 
        :param device:
        :param specs:
        :param module_names:
        :return:
        """

        modules = IgcLanguageModule.load(specs, device=device, module_names=module_names)
        # IgcRlModule.load(specs, device=device, module_names=module_names)
        return modules

    def run(self):
        """

        :return:
        """
        # Initialize the selected dataset once through the property so --corpus_dir
        # can choose the written corpus path instead of eagerly rebuilding the
        # legacy MaskedJSONDataset before train()/test/copy paths run.
        dataset = self.dataset

        # copy last checkpoint as last model with opt etc. so we can use it.
        if self._specs.copy_llm:
            model, _ = from_pretrained_default(self._specs, only_model=True)
            model.to(torch.device("cpu"))
            safe_resize_token_embeddings(model, dataset.tokenizer)
            # (BetterTransformer was removed in transformers 5.x; SDPA is native.)
            model, epoch, model_path = IgcModule.copy_checkpoint(self._specs, "state_encoder", model)
            print("Saved model to checkpoint file: ", model_path)
        elif self._specs.test_llm:
            modules = self.load(self._specs, "state_encoder")
            llm_trainer = SFTTrainer(
                "llm_trainer", self._specs,
                llm_model=modules["state_encoder"].model,
                llm_tokenizer=self.dataset.tokenizer,
                dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                metric_logger=self.metric_logger,
                is_inference=True
            )
            llm_trainer.test_inference()

        else:
            self.train()
