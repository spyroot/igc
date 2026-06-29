"""Offline tests for the action-repr / launch-fix arg-parser changes.

Checks the new pointer-policy flags and the launch-blocker fixes: --action_repr
defaults to onehot, --raw_data_dir is defined (so the RL path stops crashing),
rl batch/buffer sizes are ints, and --model_type accepts a free-form backbone id.
Importable offline now that deepspeed is imported lazily.

Author:
Mus mbayramo@stanford.edu
"""
import sys

from igc.shared.shared_arg_parser import shared_arg_parser


def _parse(monkeypatch, argv):
    """Build the parser with a controlled argv and return the parsed namespace."""
    monkeypatch.setattr(sys, "argv", ["igc", *argv])
    args, _sections = shared_arg_parser()
    return args


def test_action_repr_defaults_onehot(monkeypatch):
    """--action_repr defaults to the legacy onehot path (safe default)."""
    args = _parse(monkeypatch, [])
    assert args.action_repr == "onehot"
    assert args.action_emb_dim == 256


def test_action_repr_pointer_selectable(monkeypatch):
    """--action_repr pointer selects the candidate-scoring path."""
    args = _parse(monkeypatch, ["--action_repr", "pointer"])
    assert args.action_repr == "pointer"


def test_raw_data_dir_is_defined(monkeypatch):
    """--raw_data_dir exists with a sensible default (fixes the AttributeError)."""
    args = _parse(monkeypatch, [])
    assert args.raw_data_dir == "~/.json_responses"


def test_rl_sizes_are_ints(monkeypatch):
    """rl batch/buffer sizes parse as ints, not floats."""
    args = _parse(monkeypatch, [])
    assert isinstance(args.rl_batch_size, int) and args.rl_batch_size == 8
    assert isinstance(args.rl_buffer_size, int)


def test_model_type_is_free_form(monkeypatch):
    """--model_type accepts an arbitrary HF repo id (no gpt2-only choices)."""
    args = _parse(monkeypatch, ["--model_type", "meta-llama/Meta-Llama-3-8B"])
    assert args.model_type == "meta-llama/Meta-Llama-3-8B"


# Author: Mus mbayramo@stanford.edu
