"""Named M1 state-encoder training profiles + adapter specs (the run contract).

``docs/TRAINING_OPTIMIZATION_PLAN.md`` is the source of truth; this module makes its
profile/adapter matrix executable so every run resolves to an explicit, logged config
instead of carrying over GPT-2 / small-GPU defaults. A :class:`TrainingProfile` fully
determines a run (model, precision, batch, accumulation, lr, scheduler, warmup, sharding,
sequence length, and the :class:`AdapterSpec`); :func:`resolve_profile` applies overrides
and :func:`describe` yields the flat dict a launcher prints and logs to W&B config.

Pure standard library on purpose (no torch/peft): imported by the launcher and tests,
and must stay cheap/offline. :func:`apply_lora_kwargs` produces the exact kwargs for
:func:`igc.modules.llm.peft_lora.apply_lora` (which owns the actual PEFT construction).

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Qwen/Llama-family decoder linear projections — the plan's target-module list; also the
# default in igc.modules.llm.peft_lora.default_target_modules (kept in sync here).
DECODER_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Adapter-init name -> PEFT ``init_lora_weights`` value.
_INIT_MAP = {"default": True, "pissa": "pissa", "eva": "eva", "loftq": "loftq"}


@dataclass(frozen=True)
class AdapterSpec:
    """One adapter arm of the ablation matrix.

    :param method: ``lora`` | ``rslora`` | ``dora``.
    :param r: LoRA rank.
    :param alpha: LoRA scaling.
    :param dropout: LoRA dropout.
    :param init: adapter init family — ``default`` | ``pissa`` | ``eva`` | ``loftq``.
    :param target_modules: module names to adapt (defaults to the decoder projections).
    """

    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    init: str = "default"
    target_modules: tuple = DECODER_TARGETS

    def init_lora_weights(self):
        """The PEFT ``init_lora_weights`` value for this spec's :attr:`init`."""
        if self.init not in _INIT_MAP:
            raise ValueError(f"unknown adapter init {self.init!r}; use {sorted(_INIT_MAP)}")
        return _INIT_MAP[self.init]

    def to_dict(self) -> dict:
        return {
            "method": self.method, "r": self.r, "alpha": self.alpha,
            "dropout": self.dropout, "init": self.init,
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True)
class TrainingProfile:
    """A fully-resolved M1 run: model + precision + optimization + adapter.

    ``use_peft=False`` marks a full fine-tune (``adapter`` is ignored); large full FTs set
    ``sharding`` to ``zero3``/``fsdp``.
    """

    name: str
    model: str
    stage: str = "m1"
    llm_stage: str = "latent"
    implemented: bool = True
    purpose: str = ""
    use_peft: bool = True
    adapter: Optional[AdapterSpec] = field(default_factory=AdapterSpec)
    precision: str = "bf16"          # accelerate mixed_precision
    torch_dtype: str = "bfloat16"    # backbone load dtype
    batch_size: int = 8              # per-device train batch
    grad_accum: int = 1
    lr: float = 1e-4
    scheduler: str = "OneCycleLR"
    warmup_ratio: float = 0.03
    epochs: int = 3                  # used when max_steps is None
    max_steps: Optional[int] = None  # hard step cap (overrides epochs when set)
    sharding: str = "none"           # none | zero3 | fsdp
    seq_len: int = 1024
    num_workers: int = 8
    auto_encoder_lr: float = 1e-3
    auto_encoder_optimizer: str = "Adam"
    auto_encoder_weight_decay: float = 0.0
    metric_prefix: str = "m1/state_encoder"
    data_contract: str = "captured_redfish_json"
    live_redfish_allowed: bool = False
    nccl_mnnvl_enable: str = "0"
    nccl_cumem_enable: str = "1"
    source_path: Optional[str] = None

    def describe(self) -> dict:
        """Flat, log-safe dict of the resolved config (for stdout + W&B config)."""
        d = {
            "profile": self.name, "stage": self.stage, "llm_stage": self.llm_stage,
            "implemented": self.implemented, "purpose": self.purpose,
            "model": self.model, "use_peft": self.use_peft,
            "precision": self.precision, "torch_dtype": self.torch_dtype,
            "batch_size": self.batch_size, "grad_accum": self.grad_accum, "lr": self.lr,
            "scheduler": self.scheduler, "warmup_ratio": self.warmup_ratio,
            "epochs": self.epochs, "max_steps": self.max_steps, "sharding": self.sharding,
            "seq_len": self.seq_len, "num_workers": self.num_workers,
            "auto_encoder_lr": self.auto_encoder_lr,
            "auto_encoder_optimizer": self.auto_encoder_optimizer,
            "auto_encoder_weight_decay": self.auto_encoder_weight_decay,
            "metric_prefix": self.metric_prefix,
            "data_contract": self.data_contract,
            "live_redfish_allowed": self.live_redfish_allowed,
            "runtime_env": self.runtime_env(),
            "source_path": self.source_path,
        }
        if self.use_peft and self.adapter is not None:
            d["adapter"] = self.adapter.to_dict()
        else:
            d["adapter"] = {"method": "full_finetune"}
        return d

    def runtime_env(self) -> dict:
        """Public-safe runtime environment defaults for this profile."""
        return {
            "NCCL_MNNVL_ENABLE": self.nccl_mnnvl_enable,
            "NCCL_CUMEM_ENABLE": self.nccl_cumem_enable,
        }


# The named profiles of docs/TRAINING_OPTIMIZATION_PLAN.md §Target Training Profiles.
_LEGACY_PROFILES: Dict[str, TrainingProfile] = {
    "m1_gpt2_smoke": TrainingProfile(
        name="m1_gpt2_smoke", model="gpt2", use_peft=False, adapter=None,
        precision="no", torch_dtype="float32", batch_size=8, lr=5e-5,
        max_steps=50, seq_len=256, sharding="none",
    ),
    "m1_3b_lora": TrainingProfile(
        name="m1_3b_lora", model="Qwen/Qwen2.5-3B-Instruct",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=2, lr=1e-4, warmup_ratio=0.03,
    ),
    "m1_7b_lora": TrainingProfile(
        name="m1_7b_lora", model="Qwen/Qwen2.5-7B-Instruct",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "m1_7b_rslora_r32": TrainingProfile(
        name="m1_7b_rslora_r32", model="Qwen/Qwen2.5-7B-Instruct",
        adapter=AdapterSpec(method="rslora", r=32, alpha=64, init="default"),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "m1_local": TrainingProfile(
        # local weights dir from the environment (e.g. the staged DeepSeek-V4-Flash or
        # any node-local backbone) -- no path is baked into committed code.
        name="m1_local", model="$IGC_MODEL_DIR",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "m1_3b_full": TrainingProfile(
        name="m1_3b_full", model="Qwen/Qwen2.5-3B-Instruct", use_peft=False, adapter=None,
        batch_size=4, grad_accum=8, lr=2e-5, sharding="zero3", warmup_ratio=0.03,
    ),
    "m1_7b_full_zero3": TrainingProfile(
        name="m1_7b_full_zero3", model="Qwen/Qwen2.5-7B-Instruct", use_peft=False, adapter=None,
        batch_size=2, grad_accum=16, lr=1e-5, sharding="zero3", warmup_ratio=0.03,
    ),
}


def specs_dir() -> Path:
    """Directory containing committed YAML training profiles."""
    return Path("configs/training/profiles")


def _yaml_paths(root: Path | None = None) -> list[Path]:
    root = specs_dir() if root is None else root
    return sorted(root.glob("*.yaml")) if root.exists() else []


def _load_profile_file(path: Path) -> dict[str, TrainingProfile]:
    """Load one public-safe YAML profile file and any aliases."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: profile spec must be a YAML mapping")

    profile = _profile_from_mapping(raw, source_path=str(path))
    out = {profile.name: profile}
    for alias in raw.get("aliases", []) or []:
        out[str(alias)] = replace(profile, name=str(alias))
    return out


def _profile_from_mapping(raw: dict, source_path: str | None = None) -> TrainingProfile:
    name = str(raw.get("name") or "").strip()
    stage = str(raw.get("stage") or "").strip()
    if not name:
        raise ValueError(f"{source_path or '<profile>'}: missing profile name")
    if stage not in {"m1", "m2"}:
        raise ValueError(f"{name}: stage must be m1 or m2")

    implemented = bool(raw.get("implemented", False))
    if not implemented:
        raise ValueError(f"{name}: profile is marked unimplemented")

    backbone = _mapping(raw, "backbone", name)
    adapter = _mapping(raw, "adapter", name)
    optimizer = _mapping(raw, "optimizer", name)
    dataloader = _mapping(raw, "dataloader", name)
    training = _mapping(raw, "training", name)
    data = _mapping(raw, "data", name)
    runtime = _mapping(raw, "runtime", name)

    live_allowed = bool(data.get("live_redfish_allowed", False))
    if live_allowed:
        raise ValueError(f"{name}: live Redfish defaults are not allowed in training profiles")

    use_peft = bool(adapter.get("use_peft", False))
    adapter_spec = None
    if use_peft:
        adapter_spec = AdapterSpec(
            method=str(adapter.get("method", "lora")),
            r=int(adapter.get("r", 16)),
            alpha=int(adapter.get("alpha", 32)),
            dropout=float(adapter.get("dropout", 0.05)),
            init=str(adapter.get("init", "default")),
            target_modules=tuple(adapter.get("target_modules", DECODER_TARGETS)),
        )

    return TrainingProfile(
        name=name,
        model=str(backbone.get("model", "gpt2")),
        stage=stage,
        llm_stage=str(raw.get("llm_stage", "latent" if stage == "m1" else "encoder")),
        implemented=implemented,
        purpose=str(raw.get("purpose", "")),
        use_peft=use_peft,
        adapter=adapter_spec,
        precision=str(training.get("precision", "bf16")),
        torch_dtype=str(backbone.get("torch_dtype", "bfloat16")),
        batch_size=int(dataloader.get("batch_size", 8)),
        grad_accum=int(dataloader.get("grad_accum", 1)),
        lr=float(optimizer.get("llm_learning_rate", 1e-4)),
        scheduler=str(optimizer.get("llm_scheduler", "OneCycleLR")),
        epochs=int(training.get("epochs", 3)),
        max_steps=_optional_int(training.get("max_steps")),
        sharding=str(training.get("sharding", "none")),
        seq_len=int(dataloader.get("seq_len", 1024)),
        num_workers=int(dataloader.get("num_workers", 8)),
        auto_encoder_lr=float(optimizer.get("auto_encoder_lr", 1e-3)),
        auto_encoder_optimizer=str(optimizer.get("auto_encoder_optimizer", "Adam")),
        auto_encoder_weight_decay=float(optimizer.get("auto_encoder_weight_decay", 0.0)),
        metric_prefix=str(training.get("metric_prefix", "m1/state_encoder")),
        data_contract=str(data.get("contract", "captured_redfish_json")),
        live_redfish_allowed=live_allowed,
        nccl_mnnvl_enable=str(runtime.get("nccl_mnnvl_enable", "0")),
        nccl_cumem_enable=str(runtime.get("nccl_cumem_enable", "1")),
        source_path=source_path,
    )


def _mapping(raw: dict, key: str, profile_name: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{profile_name}: {key} must be a mapping")
    return value


def _optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


@lru_cache(maxsize=1)
def _profiles() -> Dict[str, TrainingProfile]:
    profiles = dict(_LEGACY_PROFILES)
    for path in _yaml_paths():
        profiles.update(_load_profile_file(path))
    return profiles


def profile_names() -> List[str]:
    """The registered profile names."""
    return list(_profiles())


def resolve_profile(name: str, **overrides) -> TrainingProfile:
    """Return the named profile with any top-level field overrides applied.

    :param name: a key of :data:`PROFILES`.
    :param overrides: :class:`TrainingProfile` field overrides (e.g. ``batch_size=16``,
        ``lr=2e-4``, ``max_steps=200``). Unknown keys raise, so a typo can't silently
        no-op.
    :return: the resolved (immutable) profile.
    """
    if str(name).endswith((".yaml", ".yml")):
        profiles = _load_profile_file(Path(name))
        base = next(iter(profiles.values()))
    else:
        profiles = _profiles()
        if name not in profiles:
            raise KeyError(f"unknown profile {name!r}; choose from {profile_names()}")
        base = profiles[name]

    if not base.implemented:
        raise ValueError(f"profile {name!r} is marked unimplemented")
    if base.live_redfish_allowed:
        raise ValueError(f"profile {name!r} enables live Redfish by default")

    if overrides:
        valid = set(base.__dataclass_fields__)
        bad = set(overrides) - valid
        if bad:
            raise ValueError(f"unknown profile override(s) {sorted(bad)}; valid: {sorted(valid)}")
        base = replace(base, **overrides)
    # expand $ENV_VAR model references (e.g. m1_local's $IGC_MODEL_DIR) at resolve
    # time so the env decides the weights dir, never committed code.
    if "$" in base.model:
        expanded = os.path.expandvars(base.model)
        if "$" in expanded:
            raise ValueError(
                f"profile {name!r} model {base.model!r} references an unset "
                f"environment variable; export it or override model=...")
        base = replace(base, model=expanded)
    return base


def apply_lora_kwargs(profile: TrainingProfile) -> dict:
    """The kwargs to pass to :func:`igc.modules.llm.peft_lora.apply_lora` for a profile.

    :param profile: a PEFT profile (``use_peft`` must be True).
    :return: ``r``/``alpha``/``dropout``/``target_modules``/``adapter_method``/
        ``init_lora_weights`` ready for ``apply_lora``.
    :raises ValueError: if the profile is a full fine-tune (no adapter).
    """
    if not profile.use_peft or profile.adapter is None:
        raise ValueError(f"profile {profile.name!r} is a full fine-tune; no LoRA kwargs")
    a = profile.adapter
    return {
        "r": a.r,
        "alpha": a.alpha,
        "dropout": a.dropout,
        "target_modules": list(a.target_modules),
        "adapter_method": a.method,
        "init_lora_weights": a.init_lora_weights(),
    }


# Author: Mus mbayramo@stanford.edu
