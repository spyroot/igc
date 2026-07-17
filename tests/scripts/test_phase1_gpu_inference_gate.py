"""Offline tests for the Phase 1 model_x inference hard gate."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script(name: str, relpath: str):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script("phase1_gpu_inference_gate", "scripts/phase1_gpu_inference_gate.py")
policy = _load_script("model_spec_policy_gate", "scripts/model_spec_policy_gate.py")


def _write_phase1_spec(tmp_path: Path, *, adapter_config: bool = True) -> Path:
    cache = tmp_path / "cache" / "hub"
    (cache / "models--local--tiny-model" / "snapshots" / "abc").mkdir(parents=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_bytes = b"tiny adapter bytes"
    (adapter / "adapter_model.safetensors").write_bytes(adapter_bytes)
    if adapter_config:
        (adapter / "adapter_config.json").write_text(
            '{"peft_type": "LORA", "base_model_name_or_path": "local/tiny-model", '
            '"r": 32, "lora_alpha": 64}',
            encoding="utf-8",
        )
    output = tmp_path / "gate.json"
    digest = hashlib.sha256(adapter_bytes).hexdigest()
    spec_path = tmp_path / "phase1-inference.yaml"
    spec_path.write_text(
        f"""
version: 1
name: local-phase1-gate
phase: phase1_inference
base_model:
  id: local/tiny-model
  cache_dir: {cache}
  tokenizer: local/tiny-model
adapter:
  path: {adapter}
  sha256: {digest}
  size_bytes: {len(adapter_bytes)}
  method: rslora
  rank: 32
  alpha: 64
runtime:
  torch_dtype: bfloat16
  device: cuda:0
  device_map: auto
  require_cuda: true
  trust_remote_code: false
generation:
  seed: 0
  max_new_tokens: 4
  prompts:
    - id: sample
      prompt: '{{"@odata.id":"/redfish/v1/Systems/1"}}'
      min_new_tokens: 1
      forbidden: ["Traceback"]
      must_contain: ["/redfish/v1/Systems/1"]
output:
  json: {output}
""",
        encoding="utf-8",
    )
    return spec_path


def test_phase1_gate_dry_run_reads_all_model_inputs_from_yaml(tmp_path):
    """Dry-run validates cache, adapter metadata, prompts, and writes evidence."""

    spec_path = _write_phase1_spec(tmp_path)
    args = gate.parse_args(["--spec", str(spec_path), "--dry-run", "--allow-cpu"])
    payload = gate.run_gate(args)

    assert payload["status"] == "pass"
    assert payload["base_model"] == "local/tiny-model"
    assert payload["max_new_tokens"] == 4
    assert payload["seed"] == 0
    assert payload["adapter_rank"] == 32
    assert payload["results"][0]["dry_run"] is True
    assert payload["results"][0]["max_new_tokens"] == 4
    assert "prompt" not in payload["results"][0]
    assert len(payload["results"][0]["prompt_sha256"]) == 64
    assert (tmp_path / "gate.json").is_file()


def test_phase1_gate_requires_peft_adapter_config(tmp_path):
    """The adapter gate fails before model loading if PEFT metadata is missing."""

    spec_path = _write_phase1_spec(tmp_path, adapter_config=False)
    args = gate.parse_args(["--spec", str(spec_path), "--dry-run", "--allow-cpu"])
    with pytest.raises(gate.GateError, match="adapter_config.json"):
        gate.run_gate(args)


def test_phase1_gate_rejects_lfs_pointer_weight(tmp_path):
    """The gate refuses adapter weights that are still Git LFS pointer stubs."""

    spec_path = _write_phase1_spec(tmp_path)
    args = gate.parse_args(["--spec", str(spec_path), "--dry-run", "--allow-cpu"])
    adapter_file = args.spec.adapter_dir / "adapter_model.safetensors"
    adapter_file.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "size 323014168\n",
        encoding="utf-8",
    )

    with pytest.raises(gate.GateError, match="Git LFS pointer"):
        gate.run_gate(args)


def test_phase1_gate_rejects_unknown_spec_keys(tmp_path):
    """Spec typos fail loudly instead of silently changing run behavior."""

    spec_path = _write_phase1_spec(tmp_path)
    text = spec_path.read_text(encoding="utf-8")
    spec_path.write_text(text + "\nunknown_section: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        gate.parse_args(["--spec", str(spec_path), "--dry-run"])


def test_phase1_gate_requires_pad_or_eos_token_for_generation():
    """Generation fails early if tokenizer specials cannot supply padding."""

    class NoPadOrEos:
        pad_token_id = None
        eos_token_id = None
        eos_token = None

    with pytest.raises(gate.GateError, match="pad_token_id and eos_token_id"):
        gate.ensure_generation_tokenizer_specials(NoPadOrEos())


def test_repo_phase1_inference_config_uses_whole_document_budget():
    """The committed Phase 1 spec must not truncate whole-document JSON gates."""

    root = Path(__file__).resolve().parents[2]
    spec = gate.load_gate_spec(
        root / "configs" / "inference" / "phase1_model_x_qwen2_5_7b_rslora.yaml")

    assert spec.max_new_tokens >= 1024
    assert spec.prompts
    assert any("max_new_tokens" not in case for case in spec.prompts)
    assert all(int(case.get("max_new_tokens", spec.max_new_tokens)) <= spec.max_new_tokens
               for case in spec.prompts)


def test_phase1_gate_validation_checks_generation_contract():
    """Generated text must be non-empty and satisfy per-prompt expectations."""

    errors = gate.validate_results([
        {
            "id": "bad",
            "completion": "",
            "full_text": "Traceback",
            "new_tokens": 0,
            "min_new_tokens": 2,
            "must_contain": ["/redfish/v1/Systems/1"],
            "forbidden": ["Traceback"],
        },
    ])
    assert any("generated 0 new tokens" in err for err in errors)
    assert any("empty completion" in err for err in errors)
    assert any("missing required text" in err for err in errors)
    assert any("forbidden" in err for err in errors)


def test_model_spec_policy_gate_passes_when_runtime_has_no_concrete_model(tmp_path, capsys):
    """Concrete model IDs belong in YAML specs, not runtime Python."""

    (tmp_path / "configs" / "training").mkdir(parents=True)
    (tmp_path / "configs" / "inference").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs" / "training" / "profiles.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "configs" / "inference" / "phase1.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "runner.py").write_text("MODEL = spec.base_model\n", encoding="utf-8")

    rc = policy.main(["--root", str(tmp_path), "--scan-path", "scripts/runner.py"])

    assert rc == 0
    assert "MODEL_SPEC_POLICY_PASS" in capsys.readouterr().out


def test_model_spec_policy_gate_fails_on_runtime_model_literal(tmp_path, capsys):
    """The policy gate reports hardcoded concrete model IDs in runtime code."""

    (tmp_path / "configs" / "training").mkdir(parents=True)
    (tmp_path / "configs" / "inference").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs" / "training" / "profiles.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "configs" / "inference" / "phase1.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "runner.py").write_text(
        'MODEL = "Qwen/Qwen2.5-7B-Instruct"\n',
        encoding="utf-8",
    )

    rc = policy.main(["--root", str(tmp_path), "--scan-path", "scripts/runner.py"])

    assert rc == 1
    assert "MODEL_SPEC_POLICY_FAIL" in capsys.readouterr().err
