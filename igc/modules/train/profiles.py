"""Named Phase 1 Redfish JSON pretraining profiles + adapter specs.

``docs/P0_PHASE_WORKFLOW.md`` and ``docs/TRAINING_OPTIMIZATION_PLAN.md`` are the source of
truth; this module makes the Phase 1 profile/adapter matrix executable so every run resolves
to an explicit, logged config instead of carrying over GPT-2 / small-GPU defaults. A
:class:`TrainingProfile` fully determines a run (phase, objective, model, precision, batch,
accumulation, lr, scheduler, warmup, sharding, sequence length, and the
:class:`AdapterSpec`); :func:`resolve_profile` applies overrides and :func:`describe` yields
the flat dict a launcher prints and logs to W&B config.

Pure standard library on purpose (no torch/peft): imported by the launcher and tests,
and must stay cheap/offline. :func:`apply_lora_kwargs` produces the exact kwargs for
:func:`igc.modules.llm.peft_lora.apply_lora` (which owns the actual PEFT construction).

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

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
    """A fully-resolved Phase 1 run: model + precision + optimization + adapter.

    ``use_peft=False`` marks a full fine-tune (``adapter`` is ignored); large full FTs set
    ``sharding`` to ``zero3``/``fsdp``. ``llm_stage`` is the internal trainer route; Phase 1
    currently uses the latent/state-encoder trainer path while the profile/objective name
    records that the data task is Redfish JSON reconstruction.
    """

    name: str
    model: str
    phase: str = "phase1_finetune"
    llm_stage: str = "latent"
    corpus_objective: str = "phase1_pretrain"
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

    def describe(self) -> dict:
        """Flat, log-safe dict of the resolved config (for stdout + W&B config)."""
        d = {
            "profile": self.name, "model": self.model, "use_peft": self.use_peft,
            "phase": self.phase, "llm_stage": self.llm_stage,
            "corpus_objective": self.corpus_objective,
            "precision": self.precision, "torch_dtype": self.torch_dtype,
            "batch_size": self.batch_size, "grad_accum": self.grad_accum, "lr": self.lr,
            "scheduler": self.scheduler, "warmup_ratio": self.warmup_ratio,
            "epochs": self.epochs, "max_steps": self.max_steps, "sharding": self.sharding,
            "seq_len": self.seq_len, "num_workers": self.num_workers,
        }
        if self.use_peft and self.adapter is not None:
            d["adapter"] = self.adapter.to_dict()
        else:
            d["adapter"] = {"method": "full_finetune"}
        return d


# The named profiles of docs/P0_PHASE_WORKFLOW.md §Phase 1 and
# docs/TRAINING_OPTIMIZATION_PLAN.md §Target Training Profiles.
PROFILES: Dict[str, TrainingProfile] = {
    "phase1_gpt2_smoke": TrainingProfile(
        name="phase1_gpt2_smoke", model="gpt2", use_peft=False, adapter=None,
        precision="no", torch_dtype="float32", batch_size=8, lr=5e-5,
        max_steps=50, seq_len=256, sharding="none",
    ),
    "phase1_3b_lora": TrainingProfile(
        name="phase1_3b_lora", model="Qwen/Qwen2.5-3B-Instruct",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=2, lr=1e-4, warmup_ratio=0.03,
    ),
    "phase1_7b_lora": TrainingProfile(
        name="phase1_7b_lora", model="Qwen/Qwen2.5-7B-Instruct",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "phase1_7b_rslora_r32": TrainingProfile(
        name="phase1_7b_rslora_r32", model="Qwen/Qwen2.5-7B-Instruct",
        adapter=AdapterSpec(method="rslora", r=32, alpha=64, init="default"),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "phase1_local": TrainingProfile(
        # local weights dir from the environment (e.g. the staged DeepSeek-V4-Flash or
        # any node-local backbone) -- no path is baked into committed code.
        name="phase1_local", model="$IGC_MODEL_DIR",
        adapter=AdapterSpec(method="lora", r=16, alpha=32),
        batch_size=8, grad_accum=4, lr=1e-4, warmup_ratio=0.03,
    ),
    "phase1_3b_full": TrainingProfile(
        name="phase1_3b_full", model="Qwen/Qwen2.5-3B-Instruct", use_peft=False, adapter=None,
        batch_size=4, grad_accum=8, lr=2e-5, sharding="zero3", warmup_ratio=0.03,
    ),
    "phase1_7b_full_zero3": TrainingProfile(
        name="phase1_7b_full_zero3", model="Qwen/Qwen2.5-7B-Instruct", use_peft=False, adapter=None,
        batch_size=2, grad_accum=16, lr=1e-5, sharding="zero3", warmup_ratio=0.03,
    ),
}


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
    # expand $ENV_VAR model references (e.g. phase1_local's $IGC_MODEL_DIR) at resolve
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
