"""Tests for shared module type names."""

from __future__ import annotations

import pytest

from igc.shared.modules_typing import IgcModuleType


def test_igc_module_type_accepts_argument_extractor() -> None:
    """Phase 3 uses argument_extractor as the active module role."""
    module_type = IgcModuleType.from_string("argument_extractor")

    assert module_type is IgcModuleType.ARGUMENT_EXTRACTOR
    assert str(module_type) == "argument_extractor"


def test_igc_module_type_rejects_retired_parameter_extractor() -> None:
    """parameter_extractor is a retired alias and must not parse as active."""
    with pytest.raises(ValueError, match="Invalid module type: parameter_extractor"):
        IgcModuleType.from_string("parameter_extractor")
