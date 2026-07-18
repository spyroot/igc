"""Offline tests for the merge-blocking goal-latent structural gate.

Structural checks are exercised with numpy fixtures (no model). The invariance
checks (changing a literal does not move z_*, etc.) are MODEL-ACCEPTANCE tests that
need the trained encoders and are BLOCKED while the GPU/model surface is off.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.gates.latent_contract import validate_goal_latent


def _valid_record(batch: int = 2, d_rest: int = 8, d_method: int = 6) -> dict:
    """A well-formed compiled goal-latent record (one active operation)."""
    return {
        "z_rest": np.zeros((batch, 3, d_rest), dtype=np.float32),
        "z_method": np.zeros((batch, 3, d_method), dtype=np.float32),
        "operation_mask": np.array([[True, False, False]] * batch, dtype=bool),
        "encoders": {
            "rest": {"id": "rest_encoder", "version": "v1"},
            "method": {"id": "method_encoder", "version": "v1"},
        },
        "raw_calls": [{"rest_api": "/redfish/v1/Systems/1/Bios/Settings", "method": "PATCH"}],
        "raw_arguments": [{"x": 1}],
    }


def test_valid_record_passes() -> None:
    """A structurally correct record has no violations."""
    assert validate_goal_latent(_valid_record()) == []


def test_wrong_rank_fails() -> None:
    """z_rest must be rank 3 [batch, 3, d_rest]."""
    rec = _valid_record()
    rec["z_rest"] = np.zeros((2, 8), dtype=np.float32)
    assert any("rank" in v for v in validate_goal_latent(rec))


def test_wrong_dtype_fails() -> None:
    """Latents must be float32, not float64."""
    rec = _valid_record()
    rec["z_method"] = rec["z_method"].astype(np.float64)
    assert any("dtype" in v for v in validate_goal_latent(rec))


def test_nan_or_inf_fails() -> None:
    """A NaN or infinity in a latent fails the gate."""
    rec = _valid_record()
    rec["z_rest"][0, 0, 0] = np.nan
    assert any("NaN or infinity" in v for v in validate_goal_latent(rec))


def test_wrong_operation_slot_fails() -> None:
    """The fixed operation dimension must be 3."""
    rec = _valid_record()
    rec["z_rest"] = np.zeros((2, 2, 8), dtype=np.float32)
    assert any("operation dim" in v for v in validate_goal_latent(rec))


def test_operation_mask_dtype_must_be_bool() -> None:
    """operation_mask must be bool."""
    rec = _valid_record()
    rec["operation_mask"] = np.zeros((2, 3), dtype=np.int64)
    assert any("operation_mask" in v and "dtype" in v for v in validate_goal_latent(rec))


def test_encoders_must_be_separate() -> None:
    """rest and method encoders must have distinct ids."""
    rec = _valid_record()
    rec["encoders"]["method"]["id"] = "rest_encoder"
    assert any("SEPARATE" in v for v in validate_goal_latent(rec))


def test_encoder_version_required() -> None:
    """Each encoder must carry a version."""
    rec = _valid_record()
    del rec["encoders"]["rest"]["version"]
    assert any("version" in v for v in validate_goal_latent(rec))


def test_raw_arguments_must_be_retained() -> None:
    """Raw argument bindings must be retained outside the latents."""
    rec = _valid_record()
    del rec["raw_arguments"]
    assert any("raw_arguments" in v for v in validate_goal_latent(rec))


def test_raw_calls_must_be_canonicalized() -> None:
    """An unsorted (non-canonical) call permutation fails."""
    rec = _valid_record()
    rec["raw_calls"] = [
        {"rest_api": "/redfish/v1/Systems/1/Bios", "method": "GET"},
        {"rest_api": "/redfish/v1/CertificateService/Certificates", "method": "GET"},
    ]  # Bios sorts AFTER Certificates -> not canonical
    assert any("canonical" in v for v in validate_goal_latent(rec))


def test_batch_dim_must_agree() -> None:
    """z_rest / z_method / operation_mask must share the batch dim."""
    rec = _valid_record(batch=2)
    rec["z_method"] = np.zeros((3, 3, 6), dtype=np.float32)
    assert any("batch dim mismatch" in v for v in validate_goal_latent(rec))


@pytest.mark.gpu
def test_model_acceptance_invariance_blocked() -> None:
    """Invariance (literal->same z, method->same z_rest, api->same z_method, no collapse)
    needs the trained RestEncoder/MethodEncoder. BLOCKED while the model surface is off."""
    pytest.skip("model-acceptance invariance needs trained encoders (GPU) — BLOCKED")


# Author: Mus mbayramo@stanford.edu
