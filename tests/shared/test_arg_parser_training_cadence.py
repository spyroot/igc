"""Regression tests for profile-controlled evaluation and checkpoint cadence."""

import sys

from igc.shared.shared_arg_parser import shared_arg_parser


def _parse(monkeypatch, argv):
    """Parse a controlled command line through the shared training parser."""
    monkeypatch.setattr(sys, "argv", ["igc", *argv])
    args, _sections = shared_arg_parser()
    return args


def test_training_cadence_defaults_disable_periodic_work(monkeypatch):
    """Unprofiled runs do not silently evaluate or checkpoint at fixed intervals."""
    args = _parse(monkeypatch, [])

    assert args.eval_steps == 0
    assert args.save_steps == 0


def test_training_cadence_accepts_profile_values(monkeypatch):
    """A resolved profile may set both cadence values without parser conflicts."""
    args = _parse(
        monkeypatch,
        ["--eval_steps", "25", "--save_steps", "25"],
    )

    assert args.eval_steps == 25
    assert args.save_steps == 25
