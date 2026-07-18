"""Gate: latent.contract — merge-blocking structural checks for goal latents.

Phase 3 compiles a goal into two SEPARATE latents plus a mask:

* ``z_rest``   float32 ``[batch, 3, d_rest]``   (RestEncoder over rest_api + JSON context)
* ``z_method`` float32 ``[batch, 3, d_method]`` (MethodEncoder over http_method + operation
  + argument key/type structure)
* ``operation_mask`` bool ``[batch, 3]``        (which of the 3 operation slots are active)

Exact argument VALUES stay OUTSIDE both latents; the raw compiled calls and raw argument
bindings are retained for the executor/verifier. These are the MERGE-BLOCKING structural
checks (``configs/contracts/goal_latent.yaml``): tensor rank/dims/dtype, no NaN/inf, separate
encoder ids+versions, raw calls/args retained, and canonicalized call order. The invariance
properties (changing a literal does not move z_*, etc.) are MODEL-ACCEPTANCE tests that need
the trained encoders and are BLOCKED while the GPU/model surface is off — this gate does not
run a model, only validates the compiled record structure, so it is CI-safe.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

MAX_OPERATIONS = 3


def _as_array(value: Any) -> np.ndarray | None:
    """Best-effort convert a torch/numpy tensor to a numpy array (None on failure)."""
    try:
        return np.asarray(value)
    except Exception:  # non-array value -> reported by the caller
        return None


def _check_tensor(name: str, value: Any, *, rank: int, slot: int | None, dtype: str) -> list[str]:
    """Validate one latent/mask tensor's rank, fixed slot dim, and dtype."""
    arr = _as_array(value)
    if arr is None:
        return [f"{name}: not array-like"]
    violations: list[str] = []
    if arr.ndim != rank:
        violations.append(f"{name}: rank {arr.ndim}, expected {rank}")
    if slot is not None and arr.ndim >= 2 and arr.shape[1] != slot:
        violations.append(f"{name}: operation dim {arr.shape[1]}, expected {slot}")
    if str(arr.dtype) != dtype:
        violations.append(f"{name}: dtype {arr.dtype}, expected {dtype}")
    if dtype.startswith("float") and arr.size and not np.isfinite(arr).all():
        violations.append(f"{name}: contains NaN or infinity")
    return violations


def _check_encoders(encoders: Any) -> list[str]:
    """Require separate rest/method encoder ids + versions."""
    if not isinstance(encoders, Mapping):
        return ["encoders: missing or not a mapping"]
    violations: list[str] = []
    ids: list[str] = []
    for role in ("rest", "method"):
        spec = encoders.get(role)
        if not isinstance(spec, Mapping):
            violations.append(f"encoders.{role}: missing")
            continue
        enc_id = str(spec.get("id", "")).strip()
        version = str(spec.get("version", "")).strip()
        if not enc_id:
            violations.append(f"encoders.{role}.id: missing")
        if not version:
            violations.append(f"encoders.{role}.version: missing")
        ids.append(enc_id)
    if len(ids) == 2 and ids[0] and ids[0] == ids[1]:
        violations.append("encoders: rest and method must be SEPARATE (same id)")
    return violations


def _check_canonical_calls(raw_calls: Any) -> list[str]:
    """Require raw_calls present, non-empty, and in canonical (sorted) order.

    Canonicalizing the input call permutation means two orderings of the same calls
    reduce to one record; the gate asserts the stored order already equals the
    canonical order (sorted by ``rest_api`` then ``method``).
    """
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return ["raw_calls: missing or not a list"]
    if not raw_calls:
        return ["raw_calls: empty (compiled calls must be retained)"]
    keys = []
    for call in raw_calls:
        if not isinstance(call, Mapping):
            return ["raw_calls: entries must be mappings"]
        keys.append((str(call.get("rest_api", "")), str(call.get("method", ""))))
    if keys != sorted(keys):
        return ["raw_calls: input call permutation not canonicalized (unsorted)"]
    return []


def validate_goal_latent(record: Any) -> list[str]:
    """Validate one compiled goal-latent record against the v1 structural contract.

    :param record: the compiled goal record (z_rest/z_method/operation_mask + raw material).
    :return: list of human-readable violations; empty means the record passes the
        merge-blocking latent contract.
    """
    if not isinstance(record, Mapping):
        return [f"record must be a mapping, got {type(record).__name__}"]

    violations: list[str] = []
    violations += _check_tensor("z_rest", record.get("z_rest"), rank=3, slot=MAX_OPERATIONS, dtype="float32")
    violations += _check_tensor("z_method", record.get("z_method"), rank=3, slot=MAX_OPERATIONS, dtype="float32")
    violations += _check_tensor("operation_mask", record.get("operation_mask"), rank=2, slot=MAX_OPERATIONS, dtype="bool")

    # Batch dim must agree across z_rest / z_method / operation_mask.
    batches = set()
    for name in ("z_rest", "z_method", "operation_mask"):
        arr = _as_array(record.get(name))
        if arr is not None and arr.ndim >= 1:
            batches.add(arr.shape[0])
    if len(batches) > 1:
        violations.append(f"batch dim mismatch across tensors: {sorted(batches)}")

    violations += _check_encoders(record.get("encoders"))
    violations += _check_canonical_calls(record.get("raw_calls"))
    if "raw_arguments" not in record:
        violations.append("raw_arguments: missing (raw argument bindings must be retained)")
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Confirms the committed contract config is well-formed.

    The real gate validates compiled goal records emitted on the cluster; this CLI
    only sanity-checks the contract YAML (the record producer is the GPU path).
    """
    parser = argparse.ArgumentParser(description="Goal-latent structural contract gate.")
    parser.add_argument("--contract", default="configs/contracts/goal_latent.yaml")
    args = parser.parse_args(argv)

    contract = yaml.safe_load(Path(args.contract).read_text(encoding="utf-8"))
    problems = []
    if not isinstance(contract, dict) or contract.get("max_operations") != MAX_OPERATIONS:
        problems.append("contract max_operations must be 3")
    for key in ("tensor_shapes", "merge_blocking_checks", "retained_raw"):
        if key not in (contract or {}):
            problems.append(f"contract missing {key}")
    if problems:
        for p in problems:
            print(f"BLOCKER: {p}", file=sys.stderr)
        return 1
    print("OK: goal-latent contract config well-formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
