"""Offline tests for the GB300/Blackwell performance flags in the shared arg parser.

Pins the Blackwell-appropriate defaults: fp16 is OFF by default (bf16 is preferred on
GB300), and the TF32 / torch.compile opt-in flags exist and default off. No GPU needed.

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


def test_fp16_defaults_off(monkeypatch):
    """--fp16 defaults off; on Blackwell prefer bf16, not a forced-on fp16."""
    args = _parse(monkeypatch, [])
    assert args.fp16 is False


def test_fp16_still_opt_in(monkeypatch):
    """--fp16 can still be turned on explicitly."""
    args = _parse(monkeypatch, ["--fp16"])
    assert args.fp16 is True


def test_tf32_flag_defaults_off_and_selectable(monkeypatch):
    """--tf32 exists, defaults off, and flips on when passed."""
    assert _parse(monkeypatch, []).tf32 is False
    assert _parse(monkeypatch, ["--tf32"]).tf32 is True


def test_compile_flag_defaults_off_and_selectable(monkeypatch):
    """--compile exists, defaults off (compile pays a first-step cost), flips on when passed."""
    assert _parse(monkeypatch, []).compile is False
    assert _parse(monkeypatch, ["--compile"]).compile is True


# Author: Mus mbayramo@stanford.edu
