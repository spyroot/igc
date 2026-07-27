"""Named Phase 1/2/3 supervised-fine-tuning profiles and adapter specs.

``configs/training/profiles.yaml`` is the executable source of truth; this module loads the
shared SFT profile/adapter matrix so every run resolves to an explicit, logged config instead
of carrying over GPT-2 / small-GPU defaults. A
:class:`TrainingProfile` fully determines a run (phase, objective, model, precision, batch,
weight role, accumulation, lr, scheduler, warmup, sharding, sequence length, and the
:class:`AdapterSpec`); :func:`resolve_profile` applies overrides and :func:`describe` yields
the flat dict a launcher prints and logs to W&B config.

No torch/peft imports on purpose: imported by the launcher and tests, and must stay
cheap/offline. :func:`apply_lora_kwargs` produces the exact kwargs for
:func:`igc.modules.llm.peft_lora.apply_lora` (which owns the actual PEFT construction).

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Qwen/Llama-family decoder linear projections — the plan's target-module list; also the
# default in igc.modules.llm.peft_lora.default_target_modules (kept in sync here).
DECODER_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Adapter-init name -> PEFT ``init_lora_weights`` value.
_INIT_MAP = {"default": True, "pissa": "pissa", "eva": "eva", "loftq": "loftq"}
PROFILE_SPEC_PATH = Path(__file__).resolve().parents[3] / "configs" / "training" / "profiles.yaml"


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
    """A fully resolved SFT run: lineage + model + optimization + adapter.

    ``use_peft=False`` marks a full fine-tune (``adapter`` is ignored); large full FTs set
    ``sharding`` to ``zero3``/``fsdp``. ``llm_stage`` is always ``sft`` for
    Phase 1/2/3; the task spec selects the phase renderer and lineage.
    """

    name: str
    model: str
    task: str = "redfish_json_reconstruction"
    parent_adapter: str = ""
    parent_artifact_sha: str = ""
    foundation_model_sha: str = ""
    tokenizer_sha: str = ""
    phase: str = "phase1_finetune"
    weights_role: str = "model_x"
    llm_stage: str = "sft"
    corpus_objective: str = "phase1_pretrain"
    phase1_structural_loss_profile: str = "none"
    use_peft: bool = True
    adapter: Optional[AdapterSpec] = field(default_factory=AdapterSpec)
    precision: str = "bf16"          # accelerate mixed_precision
    torch_dtype: str = "bfloat16"    # backbone load dtype
    batch_size: int = 8              # per-device train batch
    grad_accum: int = 1
    lr: float = 1e-4
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    scheduler: str = "OneCycleLR"
    max_lr: float = 1e-4
    warmup_ratio: float = 0.03
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    cycle_momentum: bool = True
    anneal_strategy: str = "cos"
    epochs: int = 3                  # used when max_steps is None
    max_steps: Optional[int] = None  # hard step cap (overrides epochs when set)
    eval_steps: int = 100            # optimizer-step cadence; 0 means final only
    save_steps: int = 100            # optimizer-step cadence; best always saves
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.005
    sharding: str = "none"           # none | zero3 | fsdp
    seq_len: int = 1024
    num_workers: int = 8
    seed: int = 42

    def describe(self) -> dict:
        """Flat, log-safe dict of the resolved config (for stdout + W&B config)."""
        d = {
            "profile": self.name, "model": self.model, "use_peft": self.use_peft,
            "task": self.task,
            "parent_adapter": self.parent_adapter,
            "parent_artifact_sha": self.parent_artifact_sha,
            "foundation_model_sha": self.foundation_model_sha,
            "tokenizer_sha": self.tokenizer_sha,
            "phase": self.phase, "weights_role": self.weights_role, "llm_stage": self.llm_stage,
            "corpus_objective": self.corpus_objective,
            "phase1_structural_loss_profile": self.phase1_structural_loss_profile,
            "precision": self.precision, "torch_dtype": self.torch_dtype,
            "batch_size": self.batch_size, "grad_accum": self.grad_accum, "lr": self.lr,
            "optimizer": self.optimizer, "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "scheduler": self.scheduler, "max_lr": self.max_lr,
            "warmup_ratio": self.warmup_ratio, "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "cycle_momentum": self.cycle_momentum,
            "anneal_strategy": self.anneal_strategy,
            "epochs": self.epochs, "max_steps": self.max_steps,
            "eval_steps": self.eval_steps, "save_steps": self.save_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "sharding": self.sharding,
            "seq_len": self.seq_len, "num_workers": self.num_workers,
            "seed": self.seed,
        }
        if self.use_peft and self.adapter is not None:
            d["adapter"] = self.adapter.to_dict()
        else:
            d["adapter"] = {"method": "full_finetune"}
        return d


_OPTIONAL_PROFILE_FIELDS = {
    "foundation_model_sha",
    "tokenizer_sha",
    "phase1_structural_loss_profile",
}
_PROFILE_FIELDS = (
    set(TrainingProfile.__dataclass_fields__)
    - {"name", "adapter"}
    - _OPTIONAL_PROFILE_FIELDS
)
_ADAPTER_FIELDS = set(AdapterSpec.__dataclass_fields__)


def _load_yaml(path: Path) -> dict:
    """Load the training profile YAML spec."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read training profile spec {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse training profile spec {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("training profile spec must be a YAML mapping")
    return payload


def _adapter_from_raw(profile_name: str, raw: Optional[dict]) -> Optional[AdapterSpec]:
    """Build an adapter spec from YAML."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"profile {profile_name!r} adapter must be a mapping or null")
    missing = sorted(_ADAPTER_FIELDS - set(raw))
    unknown = sorted(set(raw) - _ADAPTER_FIELDS)
    if missing:
        raise ValueError(f"profile {profile_name!r} adapter missing keys: {missing}")
    if unknown:
        raise ValueError(f"profile {profile_name!r} adapter has unknown keys: {unknown}")
    target_modules = raw["target_modules"]
    if not isinstance(target_modules, list) or not all(isinstance(v, str) for v in target_modules):
        raise ValueError(f"profile {profile_name!r} adapter.target_modules must be list[str]")
    return AdapterSpec(
        method=str(raw["method"]),
        r=int(raw["r"]),
        alpha=int(raw["alpha"]),
        dropout=float(raw["dropout"]),
        init=str(raw["init"]),
        target_modules=tuple(target_modules),
    )


def _profile_from_raw(name: str, raw: dict) -> TrainingProfile:
    """Build a training profile from one YAML entry."""

    if not isinstance(raw, dict):
        raise ValueError(f"profile {name!r} must be a mapping")
    missing = sorted(_PROFILE_FIELDS - set(raw))
    unknown = sorted(
        set(raw) - (_PROFILE_FIELDS | _OPTIONAL_PROFILE_FIELDS | {"adapter"})
    )
    if missing:
        raise ValueError(f"profile {name!r} missing keys: {missing}")
    if unknown:
        raise ValueError(f"profile {name!r} has unknown keys: {unknown}")

    use_peft = bool(raw["use_peft"])
    adapter = _adapter_from_raw(name, raw.get("adapter"))
    if use_peft and adapter is None:
        raise ValueError(f"profile {name!r} has use_peft=true but adapter=null")
    if not use_peft and adapter is not None:
        raise ValueError(f"profile {name!r} has use_peft=false but adapter is set")

    from igc.modules.train.sft_tasks import resolve_sft_task

    task = resolve_sft_task(str(raw["task"]))
    if str(raw["weights_role"]) != task.output_role:
        raise ValueError(
            f"profile {name!r} weights_role must be {task.output_role!r}"
        )
    if str(raw["phase"]) != task.metric_namespace:
        raise ValueError(
            f"profile {name!r} phase must be {task.metric_namespace!r}"
        )
    if str(raw["llm_stage"]) != "sft":
        raise ValueError(f"profile {name!r} llm_stage must be 'sft'")
    if task.phase == 1 and str(raw["parent_adapter"]):
        raise ValueError(f"profile {name!r} Phase 1 parent_adapter must be empty")
    if task.phase > 1 and not str(raw["parent_adapter"]):
        raise ValueError(f"profile {name!r} requires parent_adapter")
    structural_loss_profile = str(raw.get("phase1_structural_loss_profile", "none"))
    from igc.ds.phase1_structural_loss import load_phase1_structural_loss_profile

    resolved_structural_loss = load_phase1_structural_loss_profile(
        structural_loss_profile
    )
    if task.phase > 1 and resolved_structural_loss.enabled:
        raise ValueError(
            f"profile {name!r} Phase {task.phase} cannot enable Phase 1 structural loss"
        )
    warmup_ratio = float(raw["warmup_ratio"])
    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError(f"profile {name!r} warmup_ratio must be between 0 and 1")
    if float(raw["lr"]) <= 0.0 or float(raw["max_lr"]) <= 0.0:
        raise ValueError(f"profile {name!r} lr and max_lr must be positive")
    if float(raw["weight_decay"]) < 0.0:
        raise ValueError(f"profile {name!r} weight_decay must be non-negative")
    if float(raw["max_grad_norm"]) <= 0.0:
        raise ValueError(f"profile {name!r} max_grad_norm must be positive")
    if float(raw["div_factor"]) <= 0.0 or float(raw["final_div_factor"]) <= 0.0:
        raise ValueError(f"profile {name!r} scheduler division factors must be positive")
    if str(raw["anneal_strategy"]) not in {"cos", "linear"}:
        raise ValueError(f"profile {name!r} anneal_strategy must be 'cos' or 'linear'")

    return TrainingProfile(
        name=name,
        model=str(raw["model"]),
        task=task.name,
        parent_adapter=str(raw["parent_adapter"]),
        parent_artifact_sha=str(raw["parent_artifact_sha"]),
        foundation_model_sha=str(raw.get("foundation_model_sha", "")),
        tokenizer_sha=str(raw.get("tokenizer_sha", "")),
        phase=str(raw["phase"]),
        weights_role=str(raw["weights_role"]),
        llm_stage=str(raw["llm_stage"]),
        corpus_objective=str(raw["corpus_objective"]),
        phase1_structural_loss_profile=structural_loss_profile,
        use_peft=use_peft,
        adapter=adapter,
        precision=str(raw["precision"]),
        torch_dtype=str(raw["torch_dtype"]),
        batch_size=int(raw["batch_size"]),
        grad_accum=int(raw["grad_accum"]),
        lr=float(raw["lr"]),
        optimizer=str(raw["optimizer"]),
        weight_decay=float(raw["weight_decay"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        gradient_checkpointing=bool(raw["gradient_checkpointing"]),
        scheduler=str(raw["scheduler"]),
        max_lr=float(raw["max_lr"]),
        warmup_ratio=warmup_ratio,
        div_factor=float(raw["div_factor"]),
        final_div_factor=float(raw["final_div_factor"]),
        cycle_momentum=bool(raw["cycle_momentum"]),
        anneal_strategy=str(raw["anneal_strategy"]),
        epochs=int(raw["epochs"]),
        max_steps=None if raw["max_steps"] is None else int(raw["max_steps"]),
        eval_steps=int(raw["eval_steps"]),
        save_steps=int(raw["save_steps"]),
        early_stopping_patience=int(raw["early_stopping_patience"]),
        early_stopping_min_delta=float(raw["early_stopping_min_delta"]),
        sharding=str(raw["sharding"]),
        seq_len=int(raw["seq_len"]),
        num_workers=int(raw["num_workers"]),
        seed=int(raw["seed"]),
    )


def load_profiles(path: Path = PROFILE_SPEC_PATH) -> Dict[str, TrainingProfile]:
    """Load named training profiles from the YAML spec."""

    payload = _load_yaml(path)
    if payload.get("version") != 1:
        raise ValueError("training profile spec version must be 1")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("training profile spec must contain non-empty profiles mapping")
    return {name: _profile_from_raw(name, raw) for name, raw in raw_profiles.items()}


PROFILES: Dict[str, TrainingProfile] = load_profiles()


def profile_names() -> List[str]:
    """The registered profile names."""
    return list(PROFILES)


def resolve_profile(name: str, **overrides) -> TrainingProfile:
    """Return the named profile with any top-level field overrides applied.

    :param name: a key of :data:`PROFILES`.
    :param overrides: :class:`TrainingProfile` field overrides (e.g. ``batch_size=16``,
        ``lr=2e-4``, ``max_steps=200``). Unknown keys raise, so a typo can't silently
        no-op.
    :return: the resolved (immutable) profile.
    """
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {profile_names()}")
    base = PROFILES[name]
    if overrides:
        valid = set(base.__dataclass_fields__)
        bad = set(overrides) - valid
        if bad:
            raise ValueError(f"unknown profile override(s) {sorted(bad)}; valid: {sorted(valid)}")
        base = replace(base, **overrides)
    expanded_values = {}
    for field_name in (
        "model",
        "parent_adapter",
        "parent_artifact_sha",
        "foundation_model_sha",
        "tokenizer_sha",
    ):
        value = getattr(base, field_name)
        if "$" not in value:
            continue
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"profile {name!r} {field_name} {value!r} references an unset "
                "environment variable; export it or override the field")
        expanded_values[field_name] = expanded
    if expanded_values:
        base = replace(base, **expanded_values)
    from igc.modules.train.promotion_evidence import is_sha256

    for field_name in (
        "parent_artifact_sha",
        "foundation_model_sha",
        "tokenizer_sha",
    ):
        value = getattr(base, field_name)
        if value and not is_sha256(value):
            raise ValueError(
                f"profile {name!r} {field_name} must be sha256:<64 hex>"
            )
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
