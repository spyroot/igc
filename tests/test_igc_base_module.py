"""
Base env test
Author:Mus mbayramo@stanford.edu
"""

import argparse
import os
import tempfile
import unittest

import torch

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

        expected_model_file = f"{output_dir}/{module_name}_last.pt"
        self.assertEqual(module._model_file(), expected_model_file)
        self.assertEqual(module.model_file(module._trainer_args.output_dir, module_name), expected_model_file)
        self.assertEqual(IgcModule.model_file(module._trainer_args.output_dir, module_name), expected_model_file)

        with self.assertRaises(ValueError):
            module._model_file(checkpoint_dir="wrong")

        with self.assertRaises(ValueError):
            module.model_file("wrong", module_name)

        with self.assertRaises(ValueError):
            IgcModule.model_file("wrong", module_name)

    def save(self):
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

        expected_model_file = f"{output_dir}/{module_name}_last.pt"
        self.assertEqual(module._model_file(), expected_model_file)
        self.assertEqual(module.model_file(module._trainer_args.output_dir, module_name), expected_model_file)
        self.assertEqual(IgcModule.model_file(module._trainer_args.output_dir, module_name), expected_model_file)

        with self.assertRaises(ValueError):
            module._model_file(checkpoint_dir="wrong")

        with self.assertRaises(ValueError):
            module.model_file("wrong", module_name)

        with self.assertRaises(ValueError):
            IgcModule.model_file("wrong", module_name)

    def test_save_and_load_model(self):
        """
        Test saving and loading the model.
        """
        module_name = "test_module"
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            output_dir = f"{checkpoint_dir}/experiments"

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

            module.save_model(checkpoint_dir)
            model_file = module._model_file(checkpoint_dir)
            self.assertTrue(os.path.exists(model_file))
            checkpoint = torch.load(model_file)
            expected_keys = ['model_state_dict', 'is_trained']
            for key in expected_keys:
                self.assertIn(key, checkpoint)

            success = module.load_model(checkpoint_dir)
            self.assertTrue(success)
            self.assertTrue(module._is_trained)

    def test_save_and_without_opt_sched(self):
        """
        Test saving and loading checkpoints with optimizer and scheduler.
        """
        module_name = "test_module"
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            spec = argparse.Namespace(
                num_train_epochs=5,
                per_device_train_batch_size=8,
                eval_mode="on_epoch",
                overfit=False,
                metric_report="tensorboard",
                output_dir=checkpoint_dir
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

            epoch = 5
            num_checkpoints_to_keep = 1
            saved_checkpoint = module.save_checkpoint(checkpoint_dir, epoch, num_checkpoints_to_keep)
            self.assertTrue(os.path.exists(saved_checkpoint))
            expected_checkpoint_file = f"{checkpoint_dir}/{module_name}_epoch_{epoch % num_checkpoints_to_keep}.pt"
            self.assertEqual(saved_checkpoint, expected_checkpoint_file)

    def test_save_and_with_opt_sched(self):
        """
        Test saving and loading checkpoints with optimizer and scheduler.
        """
        module_name = "test_module"
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            spec = argparse.Namespace(
                num_train_epochs=5,
                per_device_train_batch_size=8,
                eval_mode="on_epoch",
                overfit=False,
                metric_report="tensorboard",
                output_dir=checkpoint_dir
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

            optimizer = torch.optim.Adam(module.model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            module.optimizer = optimizer
            module.scheduler = scheduler

            epoch = 5
            num_checkpoints_to_keep = 1
            saved_checkpoint = module.save_checkpoint(checkpoint_dir, epoch, num_checkpoints_to_keep)
            self.assertTrue(os.path.exists(saved_checkpoint))
            expected_checkpoint_file = f"{checkpoint_dir}/{module_name}_epoch_{epoch % num_checkpoints_to_keep}.pt"
            self.assertEqual(saved_checkpoint, expected_checkpoint_file)

            self.assertIsNotNone(module.optimizer)
            self.assertIsNotNone(module.scheduler)

            loaded_checkpoint = torch.load(saved_checkpoint)
            expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'is_trained']
            for key in expected_keys:
                self.assertIn(key, loaded_checkpoint)

            if isinstance(module.scheduler, list):
                expected_sched_keys = ['scheduler_state_dicts']
            else:
                expected_sched_keys = ['scheduler_state_dict']

            for key in expected_sched_keys:
                self.assertIn(key, loaded_checkpoint)

    def test_save_and_load_and_with_opt_sched(self):
        """
        Test saving and loading checkpoints with optimizer and scheduler.
        """
        module_name = "test_module"
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            spec = argparse.Namespace(
                num_train_epochs=5,
                per_device_train_batch_size=8,
                eval_mode="on_epoch",
                overfit=False,
                metric_report="tensorboard",
                output_dir=checkpoint_dir
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

            optimizer = torch.optim.Adam(module.model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            module.optimizer = optimizer
            module.scheduler = scheduler

            epoch = 5
            num_checkpoints_to_keep = 1
            saved_checkpoint = module.save_checkpoint(checkpoint_dir, epoch, num_checkpoints_to_keep)
            self.assertTrue(os.path.exists(saved_checkpoint))
            expected_checkpoint_file = f"{checkpoint_dir}/{module_name}_epoch_{epoch % num_checkpoints_to_keep}.pt"
            self.assertEqual(saved_checkpoint, expected_checkpoint_file)

            self.assertIsNotNone(module.optimizer)
            self.assertIsNotNone(module.scheduler)

            num_epochs = module.load_checkpoint(saved_checkpoint)
            self.assertEqual(num_epochs, epoch)
            self.assertIsNotNone(module.optimizer)
            self.assertIsNotNone(module.scheduler)
