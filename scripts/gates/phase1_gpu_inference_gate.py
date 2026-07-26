"""Phase 1 adapter load-and-generate hard gate.

This script is intended to run inside the NV72/GB300 lab container, not on the
Mac laptop. It verifies that the Phase 1 base model is available from the lab
Hugging Face cache, loads the selected PEFT/rsLoRA adapter, runs deterministic
generation prompts, and writes sanitized JSON evidence for the run record.

It defaults to offline Hugging Face behavior. A missing cache, missing adapter
config, missing CUDA device, or empty/invalid generation is a hard failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class GateError(RuntimeError):
    """A hard-gate failure with a user-readable message."""


@dataclass(frozen=True)
class Paths:
    """Resolved model/cache inputs for the gate."""

    base_model: str
    cache_dir: Path
    adapter_dir: Path
    output_json: Path | None


@dataclass(frozen=True)
class GateSpec:
    """Validated inference gate spec."""

    path: Path
    name: str
    phase: str
    base_model: str
    cache_dir: Path
    adapter_dir: Path
    tokenizer: str
    output_json: Path | None
    prompts: list[dict[str, Any]]
    max_new_tokens: int
    seed: int
    torch_dtype: str
    device: str
    device_map: str
    require_cuda: bool
    trust_remote_code: bool
    adapter_sha256: str | None
    adapter_size_bytes: int | None
    adapter_method: str | None
    adapter_rank: int | None
    adapter_alpha: int | None


def blocker(message: str) -> None:
    """Print a blocker line and exit nonzero."""

    print(f"BLOCKER: {message}", file=sys.stderr)
    raise SystemExit(1)


def set_offline_env(cache_dir: Path) -> None:
    """Force offline cache lookup unless the caller already set stricter env."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    if cache_dir.name == "hub":
        os.environ.setdefault("HF_HOME", str(cache_dir.parent))


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is present in project env
        raise GateError(f"PyYAML is required to read inference spec: {exc}") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"cannot read spec {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise GateError(f"cannot parse YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError("inference spec must be a YAML mapping")
    return payload


def _mapping(payload: dict[str, Any], key: str, *, required: bool = False) -> dict[str, Any]:
    """Return a nested mapping from a spec."""

    value = payload.get(key)
    if value is None:
        if required:
            raise GateError(f"spec missing required section: {key}")
        return {}
    if not isinstance(value, dict):
        raise GateError(f"spec section {key} must be a mapping")
    return value


def _required_string(payload: dict[str, Any], key: str, *, context: str = "spec") -> str:
    """Return a required non-empty string field."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GateError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str, default: str = "") -> str:
    """Return an optional string field."""

    value = payload.get(key, default)
    if not isinstance(value, str):
        raise GateError(f"{key} must be a string")
    return value


def _optional_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    """Return an optional bool field."""

    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise GateError(f"{key} must be a boolean")
    return value


def _optional_int(payload: dict[str, Any], key: str, default: int | None = None) -> int | None:
    """Return an optional positive int field."""

    value = payload.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise GateError(f"{key} must be a non-negative integer")
    return value


def load_gate_spec(path: Path) -> GateSpec:
    """Load and validate an inference gate YAML spec."""

    raw = _load_yaml(path)
    unknown = set(raw) - {
        "version",
        "name",
        "phase",
        "base_model",
        "adapter",
        "runtime",
        "generation",
        "output",
    }
    if unknown:
        raise GateError(f"unknown top-level spec keys: {', '.join(sorted(unknown))}")
    if raw.get("version") != 1:
        raise GateError("spec.version must be 1")

    base = _mapping(raw, "base_model", required=True)
    adapter = _mapping(raw, "adapter", required=True)
    runtime = _mapping(raw, "runtime", required=True)
    generation = _mapping(raw, "generation", required=True)
    output = _mapping(raw, "output")

    prompts = generation.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise GateError("generation.prompts must be a non-empty list")

    max_new_tokens = _optional_int(generation, "max_new_tokens")
    if max_new_tokens is None or max_new_tokens < 1:
        raise GateError("generation.max_new_tokens must be a positive integer")
    seed = _optional_int(generation, "seed", 0)
    if seed is None:
        raise GateError("generation.seed must be a non-negative integer")

    torch_dtype = _optional_string(runtime, "torch_dtype", "bfloat16")
    if torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise GateError("runtime.torch_dtype must be one of auto|bfloat16|float16|float32")

    return GateSpec(
        path=path,
        name=_required_string(raw, "name"),
        phase=_optional_string(raw, "phase", "phase1"),
        base_model=_required_string(base, "id", context="base_model"),
        cache_dir=Path(_required_string(base, "cache_dir", context="base_model")),
        adapter_dir=Path(_required_string(adapter, "path", context="adapter")),
        tokenizer=_optional_string(base, "tokenizer", ""),
        output_json=Path(output["json"]) if isinstance(output.get("json"), str) else None,
        prompts=normalize_prompt_cases(prompts),
        max_new_tokens=max_new_tokens,
        seed=seed,
        torch_dtype=torch_dtype,
        device=_optional_string(runtime, "device", "cuda:0"),
        device_map=_optional_string(runtime, "device_map", "auto"),
        require_cuda=_optional_bool(runtime, "require_cuda", True),
        trust_remote_code=_optional_bool(runtime, "trust_remote_code", False),
        adapter_sha256=_optional_string(adapter, "sha256", "") or None,
        adapter_size_bytes=_optional_int(adapter, "size_bytes"),
        adapter_method=_optional_string(adapter, "method", "") or None,
        adapter_rank=_optional_int(adapter, "rank"),
        adapter_alpha=_optional_int(adapter, "alpha"),
    )


def repo_cache_dir(base_model: str, cache_dir: Path) -> Path | None:
    """Return the expected HF hub cache directory for a repo id, if applicable."""

    if "/" not in base_model or base_model.startswith(("/", ".")):
        return None
    return cache_dir / f"models--{base_model.replace('/', '--')}"


def preflight_inputs(paths: Paths, *, require_cuda: bool = True, skip_cache: bool = False) -> None:
    """Validate local cache/adapter structure without loading model weights."""

    if not paths.adapter_dir.is_dir():
        raise GateError(f"adapter directory does not exist: {paths.adapter_dir}")

    required = ("adapter_model.safetensors", "adapter_config.json")
    missing = [name for name in required if not (paths.adapter_dir / name).is_file()]
    if missing:
        raise GateError(f"adapter directory missing required PEFT file(s): {', '.join(missing)}")
    reject_lfs_pointer(paths.adapter_dir / "adapter_model.safetensors", "adapter weight")

    if paths.output_json is not None:
        paths.output_json.parent.mkdir(parents=True, exist_ok=True)

    if paths.base_model.startswith(("/", ".")):
        if not Path(paths.base_model).exists():
            raise GateError(f"base model path does not exist: {paths.base_model}")
    elif not skip_cache:
        expected = repo_cache_dir(paths.base_model, paths.cache_dir)
        if expected is not None and not expected.is_dir():
            raise GateError(
                f"base model cache missing: {expected}; refusing network download",
            )
        snapshots = expected / "snapshots" if expected is not None else None
        if snapshots is not None and not any(snapshots.glob("*")):
            raise GateError(f"base model cache has no snapshots: {snapshots}")
        if snapshots is not None:
            for weight in snapshots.glob("*/*.safetensors"):
                reject_lfs_pointer(weight, "base model weight")

    if require_cuda:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on lab image
            raise GateError(f"torch import failed before CUDA check: {exc}") from exc
        if not torch.cuda.is_available():
            raise GateError("CUDA is not available; pass --allow-cpu only for a local mock smoke")


def load_prompt_cases(path: Path) -> list[dict[str, Any]]:
    """Load prompt cases from JSON/JSONL."""

    if not path.is_file():
        raise GateError(f"prompt file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise GateError(f"prompt file is empty: {path}")
    if path.suffix.lower() == ".jsonl":
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        cases = payload["prompts"] if isinstance(payload, dict) and "prompts" in payload else payload
    if not isinstance(cases, list) or not cases:
        raise GateError("prompt file must contain a non-empty list of prompt cases")
    normalized = []
    seen: set[str] = set()
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise GateError(f"prompt case {idx} is not an object")
        prompt_id = str(case.get("id") or f"prompt_{idx}")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise GateError(f"prompt case {prompt_id} has no non-empty prompt")
        if prompt_id in seen:
            raise GateError(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        item = dict(case)
        item["id"] = prompt_id
        item["prompt"] = prompt
        item.setdefault("min_new_tokens", 1)
        item.setdefault("forbidden", ["Traceback"])
        normalized.append(item)
    return normalized


def normalize_prompt_cases(cases: list[Any]) -> list[dict[str, Any]]:
    """Validate prompt cases embedded in an inference YAML spec."""

    normalized = []
    seen: set[str] = set()
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise GateError(f"generation.prompts[{idx}] must be a mapping")
        prompt_id = str(case.get("id") or f"prompt_{idx}")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise GateError(f"prompt case {prompt_id} has no non-empty prompt")
        if prompt_id in seen:
            raise GateError(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        min_new_tokens = case.get("min_new_tokens")
        if not isinstance(min_new_tokens, int) or min_new_tokens < 1:
            raise GateError(f"prompt case {prompt_id} min_new_tokens must be a positive integer")
        item = dict(case)
        item["id"] = prompt_id
        item["prompt"] = prompt
        item["forbidden"] = _listify(item.get("forbidden"))
        item["must_contain"] = _listify(item.get("must_contain"))
        item["must_match"] = _listify(item.get("must_match"))
        normalized.append(item)
    return normalized


def reject_lfs_pointer(path: Path, label: str) -> None:
    """Fail if a supposed binary artifact is still a Git LFS pointer stub."""

    try:
        head = path.read_bytes()[:256]
    except OSError as exc:
        raise GateError(f"cannot read {label}: {path}: {exc}") from exc
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise GateError(f"{label} is a Git LFS pointer, not materialized weights: {path}")


def _listify(value: Any) -> list[str]:
    """Normalize string/list/None expectation fields."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise GateError(f"expectation field must be a string or list[str], got {value!r}")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return SHA256 for an artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_config(spec: GateSpec) -> dict[str, Any]:
    """Validate PEFT adapter metadata against the YAML model spec.

    :param spec: resolved gate spec with the base model and adapter metadata.
    :return: parsed ``adapter_config.json`` as a mapping.
    :raises GateError: when the adapter config is missing required PEFT metadata or
        contradicts the YAML spec.
    """

    path = spec.adapter_dir / "adapter_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"adapter_config.json is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise GateError("adapter_config.json must be a JSON object")
    if not config.get("peft_type"):
        raise GateError("adapter_config.json missing peft_type")
    base_name = config.get("base_model_name_or_path")
    if isinstance(base_name, str) and "/" in spec.base_model:
        cache_name = f"models--{spec.base_model.replace('/', '--')}"
        compatible = (
            base_name.rstrip("/") == spec.base_model.rstrip("/")
            or base_name.rstrip("/").endswith(spec.base_model.rstrip("/"))
            or cache_name in base_name
        )
        if not compatible and "/" in base_name:
            raise GateError(
                "adapter base_model_name_or_path does not match inference spec "
                f"({base_name!r} != {spec.base_model!r})",
            )
    if spec.adapter_rank is not None and config.get("r") is not None:
        if int(config["r"]) != spec.adapter_rank:
            raise GateError(f"adapter rank mismatch: spec={spec.adapter_rank}, config={config['r']}")
    if spec.adapter_alpha is not None and config.get("lora_alpha") is not None:
        if int(config["lora_alpha"]) != spec.adapter_alpha:
            raise GateError(
                f"adapter alpha mismatch: spec={spec.adapter_alpha}, config={config['lora_alpha']}",
            )
    return config


def ensure_generation_tokenizer_specials(tokenizer: Any) -> None:
    """Ensure generation has a pad token or a usable EOS fallback."""

    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token_id is None or eos_token is None:
        raise GateError(
            "tokenizer lacks pad_token_id and eos_token_id; configure a pad or EOS token",
        )
    tokenizer.pad_token = eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise GateError("tokenizer pad_token fallback did not produce pad_token_id")


def _dtype_from_name(name: str):
    """Map CLI dtype to torch dtype; imported lazily for offline tests."""

    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the base causal LM, tokenizer, and PEFT adapter from local files only.

    Inputs come from ``args.spec``. The returned tokenizer maps prompt text to
    ``input_ids``/``attention_mask`` tensors of shape ``[1, prompt_tokens]`` in
    :func:`run_generation`. The model returns generated token IDs of shape
    ``[1, prompt_tokens + new_tokens]``.
    """

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "cache_dir": args.spec.cache_dir,
        "local_files_only": True,
        "trust_remote_code": args.spec.trust_remote_code,
    }
    dtype = _dtype_from_name(args.spec.torch_dtype)
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    if args.spec.device_map != "none":
        load_kwargs["device_map"] = args.spec.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        args.spec.tokenizer or args.spec.base_model,
        cache_dir=args.spec.cache_dir,
        local_files_only=True,
        trust_remote_code=args.spec.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        ensure_generation_tokenizer_specials(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.spec.base_model, **load_kwargs)
    model = PeftModel.from_pretrained(
        model,
        args.spec.adapter_dir,
        local_files_only=True,
        is_trainable=False,
    )

    device = None
    if args.spec.device_map == "none":
        if torch.cuda.is_available():
            device = torch.device(args.spec.device)
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise GateError("CUDA is not available; pass --allow-cpu only for local mock smoke")
        model.to(device)
    else:
        device = next(model.parameters()).device
    model.eval()
    return model, tokenizer, device


def run_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    cases: Iterable[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run deterministic generation for each prompt case.

    Each prompt is tokenized as a single batch item. Inputs have shape
    ``[1, prompt_tokens]``; ``model.generate`` returns shape
    ``[1, prompt_tokens + generated_tokens]``. The output is a sanitized list of
    per-prompt dictionaries containing decoded text plus token counts; raw
    tensors are intentionally not serialized.
    """

    import torch

    results: list[dict[str, Any]] = []
    for case in cases:
        prompt = case["prompt"]
        encoded = tokenizer(case["prompt"], return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        else:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        input_len = int(encoded["input_ids"].shape[-1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(case.get("max_new_tokens", max_new_tokens)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_ids = generated[0]
        new_ids = full_ids[input_len:]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        completion = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append({
            "id": case["id"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "completion": completion,
            "full_text": full_text,
            "input_tokens": input_len,
            "new_tokens": int(new_ids.numel()),
            "max_new_tokens": int(case.get("max_new_tokens", max_new_tokens)),
            "min_new_tokens": int(case.get("min_new_tokens", 1)),
            "must_contain": _listify(case.get("must_contain")),
            "must_match": _listify(case.get("must_match")),
            "forbidden": _listify(case.get("forbidden")),
        })
    return results


def validate_results(results: Iterable[dict[str, Any]]) -> list[str]:
    """Return validation errors for generated outputs."""

    errors: list[str] = []
    for result in results:
        rid = result["id"]
        completion = str(result.get("completion", ""))
        full_text = str(result.get("full_text", ""))
        new_tokens = int(result.get("new_tokens", 0))
        min_new_tokens = int(result.get("min_new_tokens", 1))
        if new_tokens < min_new_tokens:
            errors.append(f"{rid}: generated {new_tokens} new tokens, expected >= {min_new_tokens}")
        if not completion.strip():
            errors.append(f"{rid}: empty completion")
        for needle in _listify(result.get("must_contain")):
            if needle not in full_text:
                errors.append(f"{rid}: missing required text {needle!r}")
        for pattern in _listify(result.get("must_match")):
            if re.search(pattern, full_text, flags=re.DOTALL) is None:
                errors.append(f"{rid}: output does not match regex {pattern!r}")
        for forbidden in _listify(result.get("forbidden")):
            if forbidden and forbidden in full_text:
                errors.append(f"{rid}: output contains forbidden text {forbidden!r}")
    return errors


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically enough for operator evidence files."""

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_payload(
    *,
    args: argparse.Namespace,
    paths: Paths,
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
    errors: list[str],
    elapsed_sec: float,
    adapter_sha256: str | None,
) -> dict[str, Any]:
    """Build sanitized gate evidence."""

    return {
        "schema": "igc.phase1_inference_gate.v1",
        "status": "pass" if not errors else "fail",
        "spec": str(args.spec.path),
        "name": args.spec.name,
        "phase": args.spec.phase,
        "base_model": paths.base_model,
        "adapter_dir": str(paths.adapter_dir),
        "cache_dir": str(paths.cache_dir),
        "torch_dtype": args.spec.torch_dtype,
        "max_new_tokens": args.spec.max_new_tokens,
        "seed": args.spec.seed,
        "device": args.spec.device if args.spec.device_map == "none" else f"device_map:{args.spec.device_map}",
        "local_files_only": True,
        "adapter_model_sha256": adapter_sha256,
        "adapter_method": args.spec.adapter_method,
        "adapter_rank": args.spec.adapter_rank,
        "adapter_alpha": args.spec.adapter_alpha,
        "prompt_count": len(prompts),
        "output_json": str(paths.output_json) if paths.output_json is not None else None,
        "elapsed_sec": round(elapsed_sec, 3),
        "errors": errors,
        "results": results,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute preflight, optional dry-run, and model generation from a YAML spec."""

    paths = Paths(
        base_model=args.spec.base_model,
        cache_dir=args.spec.cache_dir,
        adapter_dir=args.spec.adapter_dir,
        output_json=Path(args.output_json) if args.output_json else args.spec.output_json,
    )
    set_offline_env(paths.cache_dir)
    prompts = args.spec.prompts
    preflight_inputs(
        paths,
        require_cuda=args.spec.require_cuda and not args.allow_cpu and not args.dry_run,
        skip_cache=args.skip_cache_preflight,
    )
    validate_adapter_config(args.spec)

    adapter_sha = None
    if not args.skip_adapter_sha256:
        adapter_file = paths.adapter_dir / "adapter_model.safetensors"
        actual_size = adapter_file.stat().st_size
        if args.spec.adapter_size_bytes is not None and actual_size != args.spec.adapter_size_bytes:
            raise GateError(
                f"adapter size mismatch: expected {args.spec.adapter_size_bytes}, got {actual_size}",
            )
        adapter_sha = sha256_file(adapter_file)
        if args.spec.adapter_sha256 is not None and adapter_sha.lower() != args.spec.adapter_sha256.lower():
            raise GateError(
                f"adapter sha256 mismatch: expected {args.spec.adapter_sha256}, got {adapter_sha}",
            )

    started = time.time()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    if args.dry_run:
        results = [{
            "id": case["id"],
            "prompt_sha256": hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
            "completion": "",
            "full_text": case["prompt"],
            "input_tokens": None,
            "new_tokens": 0,
            "max_new_tokens": int(case.get("max_new_tokens", args.spec.max_new_tokens)),
            "min_new_tokens": 0,
            "must_contain": _listify(case.get("must_contain")),
            "must_match": _listify(case.get("must_match")),
            "forbidden": _listify(case.get("forbidden")),
            "dry_run": True,
        } for case in prompts]
    else:
        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(args.spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.spec.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model, tokenizer, device = load_model_and_tokenizer(args)
        results = run_generation(
            model,
            tokenizer,
            device,
            prompts,
            max_new_tokens=args.spec.max_new_tokens,
        )
        errors = validate_results(results)

    payload = build_payload(
        args=args,
        paths=paths,
        prompts=prompts,
        results=results,
        errors=errors,
        elapsed_sec=time.time() - started,
        adapter_sha256=adapter_sha,
    )
    if paths.output_json is not None:
        write_json(paths.output_json, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Inference gate YAML spec.")
    parser.add_argument("--output-json", default="", help="Write sanitized gate evidence JSON.")
    parser.add_argument("--allow-cpu", action="store_true", help="Local mock smoke only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/prompts; do not load model.")
    parser.add_argument("--skip-cache-preflight", action="store_true")
    parser.add_argument("--skip-adapter-sha256", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.spec = load_gate_spec(Path(args.spec))
    except GateError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    try:
        payload = run_gate(args)
    except GateError as exc:
        output_json = Path(args.output_json) if args.output_json else args.spec.output_json
        if output_json is not None:
            write_json(output_json, {
                "schema": "igc.phase1_inference_gate.v1",
                "status": "fail",
                "spec": str(args.spec.path),
                "name": args.spec.name,
                "phase": args.spec.phase,
                "error": str(exc),
            })
        blocker(str(exc))
    status = payload["status"]
    if status == "pass":
        print(
            "PHASE1_INFERENCE_GATE_PASS "
            f"base={payload['base_model']} prompts={payload['prompt_count']} "
            f"output={payload['output_json'] or '<stdout-only>'}",
            flush=True,
        )
        return 0
    for error in payload["errors"]:
        print(f"BLOCKER: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
