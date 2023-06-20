"""
Base env test
Author:Mus mbayramo@stanford.edu
"""

import argparse
import unittest

from igc.ds.base_ds import RandomDataset
from igc.ds.redfish_dataset import JSONDataset
from igc.modules.base.igc_base_module import IgcModule
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.modules.shared.llm_shared import from_pretrained_default, load_igc_tokenizer, igc_base_dir
from igc.shared.modules_typing import SaveStrategy


class TestRestApiEnv(unittest.TestCase):

    def test_read_spec(self):
        """
        Test the test_read_spec method of RestApiEnv.
        """
        models_data = IgcModule.read_model_specs()
        self.assertIn("mirrors", models_data)
        mirrors = models_data["mirrors"]
        for mirror_entries in mirrors.values():
            for entry in mirror_entries:
                self.assertIsInstance(entry, dict)
                if "files" in entry and "local_file" in entry:
                    self.assertIn("files", entry)
                    self.assertIn("local_file", entry)
                    files = entry["files"]
                    local_files = entry["local_file"]
                    self.assertIsInstance(files, list)
                    self.assertIsInstance(local_files, list)
                    self.assertEqual(len(files), len(local_files))

    def test_download(self):
        """
        Test the download module method.
        """
        IgcModule.download()

    def test_we_don_download(self):
        """
        Test the download module method.
        """
        IgcModule.download()
        IgcModule.download()

    def test_initialization(self):
        """

        :return:
        """
        module_name = "test_module"
        spec = argparse.Namespace(
            num_train_epochs=5,
            per_device_train_batch_size=8,
            eval_mode="on_epoch",
            overfit=False,
            metric_report="tensorboard"
        )

        llm_model, _ = from_pretrained_default("gpt2")
        llm_tokenizer = load_igc_tokenizer()
        ds = JSONDataset()
        _metric_logger = MetricLogger(spec.metric_report, **vars(spec))
        is_inference = False

        module = IgcModule(
            module_name=module_name,
            spec=spec,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            ds=ds,
            metric_logger=_metric_logger,
            is_inference=is_inference
        )

        self.assertEqual(module.module_name, module_name)
        self.assertEqual(module._trainer_args, spec)
        self.assertEqual(module.model, llm_model)
        self.assertEqual(module.tokenizer, llm_tokenizer)
        self.assertEqual(module.dataset, ds)
        self.assertEqual(module.metric_logger, _metric_logger)
        self.assertEqual(module._is_inference, is_inference)
        self.assertFalse(module._is_trained)
        self.assertEqual(module.num_epochs, spec.num_train_epochs)
        self.assertEqual(module.batch_size, spec.per_device_train_batch_size)
        self.assertEqual(module.on_epoch_eval, spec.eval_mode == "on_epoch")
        self.assertEqual(module.optimizer, None)
        self.assertEqual(module._save_strategy, SaveStrategy.EPOCH)

    def test_dirs(self):
        """

        :return:
        """
        module_name = "test_module"
        output_dir = f"{igc_base_dir()}/experiments"
        spec = argparse.Namespace(
            num_train_epochs=5,
            per_device_train_batch_size=8,
            eval_mode="on_epoch",
            overfit=False,
            metric_report="tensorboard",
            output_dir=output_dir
        )

        llm_model, _ = from_pretrained_default("gpt2")
        llm_tokenizer = load_igc_tokenizer()
        ds = RandomDataset(10, 1024)
        _metric_logger = MetricLogger(spec.metric_report, **vars(spec))
        is_inference = False

        module = IgcModule(
            module_name=module_name,
            spec=spec,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            ds=ds,
            metric_logger=_metric_logger,
            is_inference=is_inference
        )

        self.assertEqual(module.module_name, module_name)
        self.assertEqual(module._trainer_args, spec)
        self.assertEqual(module.model, llm_model)
        self.assertEqual(module.tokenizer, llm_tokenizer)
        self.assertEqual(module.dataset, ds)
        self.assertEqual(module.metric_logger, _metric_logger)
        self.assertEqual(module._is_inference, is_inference)
        self.assertFalse(module._is_trained)
        self.assertEqual(module.num_epochs, spec.num_train_epochs)
        self.assertEqual(module.batch_size, spec.per_device_train_batch_size)
        self.assertEqual(module.on_epoch_eval, spec.eval_mode == "on_epoch")
        self.assertEqual(module.optimizer, None)
        self.assertEqual(module._save_strategy, SaveStrategy.EPOCH)

        self.assertEqual(module._trainer_args.output_dir, output_dir)

