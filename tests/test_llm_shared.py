"""
Share utils unit test should be here.
Author:Mus mbayramo@stanford.edu
"""

import argparse
import os
from types import SimpleNamespace
import tempfile
import unittest

import pytest
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer
)
from igc.modules.shared.llm_shared import (
    from_pretrained_default,
    save_pretrained_default,
    load_pretrained_default,
    load_igc_tokenizer,
    safe_resize_token_embeddings,
)


class _DummyEmbeddings:
    def __init__(self, num_embeddings):
        self.num_embeddings = num_embeddings


class _DummyModel:
    def __init__(self, num_embeddings=100, name_or_path="demo/backbone"):
        self.config = SimpleNamespace(_name_or_path=name_or_path, vocab_size=num_embeddings)
        self.embeddings = _DummyEmbeddings(num_embeddings)
        self.resized_to = None

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, size):
        self.resized_to = size


class _DummyTokenizer:
    def __init__(self, size=100, name_or_path="demo/backbone"):
        self.size = size
        self.name_or_path = name_or_path

    def __len__(self):
        return self.size


class TestFromPretrainedDefault(unittest.TestCase):

    @pytest.mark.download  # from_pretrained_default("gpt2") pulls the model from HF
    def test_from_pretrained_default_with_string(self):
        """Test we can create from string
        :return:
        """
        args_str = "gpt2"
        only_tokenizer = False
        only_model = False
        add_padding = True
        device_map = "auto"

        model, tokenizer = from_pretrained_default(
            args_str,
            only_tokenizer=only_tokenizer,
            only_model=only_model,
            add_padding=add_padding,
            device_map=device_map
        )

        self.assertIsInstance(model, GPT2LMHeadModel)
        self.assertIsInstance(tokenizer, GPT2Tokenizer)

    @pytest.mark.download  # loads the gpt2 tokenizer from HF
    def test_from_pretrained_default_with_namespace(self):
        """Test we can create from argparse.Namespace
        :return:
        """
        args_ns = argparse.Namespace(model_type="gpt2")
        only_tokenizer = True
        only_model = False
        add_padding = False
        device_map = "auto"

        model, tokenizer = from_pretrained_default(
            args_ns,
            only_tokenizer=only_tokenizer,
            only_model=only_model,
            add_padding=add_padding,
            device_map=device_map
        )

        self.assertIsNone(model)
        self.assertIsInstance(tokenizer, GPT2Tokenizer)

    @pytest.mark.download  # GPT2LMHeadModel.from_pretrained("gpt2") pulls from HF
    def test_save_and_load_pretrained_default(self):
        """Test pre-trained model saving and loading
        :return:
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

            huggingface_dir = os.path.join(temp_dir, "huggingface")
            os.makedirs(huggingface_dir, exist_ok=True)
            save_result = save_pretrained_default(
                huggingface_dir,
                model,
                tokenizer,
                only_tokenizer=False
            )

            self.assertTrue(save_result)
            args = argparse.Namespace()
            loaded_model, loaded_tokenizer = load_pretrained_default(
                args,
                huggingface_dir
            )

            self.assertIsInstance(loaded_model, GPT2LMHeadModel)
            self.assertIsInstance(loaded_tokenizer, GPT2Tokenizer)
            self.assertEqual(model.state_dict().keys(), loaded_model.state_dict().keys())
            self.assertEqual(tokenizer.pad_token, loaded_tokenizer.pad_token)
            self.assertEqual(tokenizer.pad_token_id, loaded_tokenizer.pad_token_id)

    def test_load_igc_tokenizer(self):
        """
        :return:
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_dir = temp_dir
            os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(tokenizer_dir)

            loaded_tokenizer = load_igc_tokenizer(tokenizer_dir)
            self.assertIsInstance(loaded_tokenizer, GPT2Tokenizer)
            self.assertEqual(loaded_tokenizer.pad_token, loaded_tokenizer.eos_token)
            self.assertEqual(loaded_tokenizer.pad_token_id, loaded_tokenizer.eos_token_id)

    def test_load_igc_tokenizer_without_path(self):
        """
        :return:
        """
        loaded_tokenizer = load_igc_tokenizer()
        self.assertIsInstance(loaded_tokenizer, GPT2Tokenizer)
        self.assertEqual(loaded_tokenizer.pad_token, loaded_tokenizer.eos_token)
        self.assertEqual(loaded_tokenizer.pad_token_id, loaded_tokenizer.eos_token_id)

    def test_safe_resize_allows_same_backbone_padded_vocab_noop(self):
        """Model vocab padding above tokenizer length should not trigger a shrink."""
        model = _DummyModel(num_embeddings=128, name_or_path="vendor/model")
        tokenizer = _DummyTokenizer(size=120, name_or_path="vendor/model")

        safe_resize_token_embeddings(model, tokenizer)

        self.assertIsNone(model.resized_to)

    def test_safe_resize_rejects_obvious_wrong_tokenizer_shrink(self):
        """A much smaller tokenizer from another backbone still raises loudly."""
        model = _DummyModel(num_embeddings=128, name_or_path="vendor/model")
        tokenizer = _DummyTokenizer(size=32, name_or_path="other/model")

        with self.assertRaisesRegex(ValueError, "Refusing to shrink"):
            safe_resize_token_embeddings(model, tokenizer)
