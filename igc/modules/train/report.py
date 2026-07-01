"""Result bundles + cross-arm comparison for the adapter ablation (the report contract).

``docs/TRAINING_OPTIMIZATION_PLAN.md`` requires every adapter arm to emit the same report
so promotion decisions are data-driven. This module makes that comparable and automatic:

* :class:`RunManifest` — the reproducibility header (profile, model, tokenizer, adapter +
  rank + init, data manifest, eval split, max steps, sequence length, tokens/sec, peak
  memory, git commit).
* :class:`ResultBundle` — one arm's manifest + metric table + artifact paths (plots dir,
  five best / five worst, known blockers), JSON round-trippable, written per run.
* :func:`compare` — reads many bundles and produces a **rows=arm x cols=metric** table,
  the delta vs. a baseline arm (the promotion view), and — crucially — the
  **fair-comparison check** (same model/tokenizer/data-manifest/eval-split/max-steps/
  seq-len across arms), so an unfair comparison is flagged, never silently reported.

Pure standard library (no torch): the metric *values* are computed by the eval harness;
this module only carries and compares them, offline, from artifacts on disk.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The manifest fields that MUST match for a comparison to be apples-to-apples
# (docs/TRAINING_OPTIMIZATION_PLAN.md: same backbone/tokenizer/split/steps/seq-len).
_FAIRNESS_KEYS = ("model", "tokenizer", "data_manifest", "eval_split", "max_steps", "seq_len")


@dataclass
class RunManifest:
    """Reproducibility header for one run (what was trained, on what, how far)."""

    run_id: str
    profile: str
    model: str
    tokenizer: str = ""
    adapter_method: str = "lora"
    adapter_rank: Optional[int] = None
    adapter_init: str = "default"
    data_manifest: str = ""          # id/hash of the exact dataset + source mix
    eval_split: str = ""             # id of the held-out split (source-separated)
    max_steps: Optional[int] = None
    seq_len: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    peak_mem_gb: Optional[float] = None
    git_commit: str = ""
    settings: dict = field(default_factory=dict)   # resolved profile / launcher settings

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class ResultBundle:
    """One arm's full report: manifest + metric table + artifact pointers."""

    manifest: RunManifest
    metrics: Dict[str, float] = field(default_factory=dict)
    plots_dir: str = ""
    best_examples: List = field(default_factory=list)
    worst_examples: List = field(default_factory=list)
    known_blockers: List[str] = field(default_factory=list)

    @property
    def arm(self) -> str:
        """Short arm label for the comparison rows, e.g. ``rslora-r32``."""
        m = self.manifest
        rank = "" if m.adapter_rank is None else f"-r{m.adapter_rank}"
        init = "" if m.adapter_init in ("", "default") else f"-{m.adapter_init}"
        return f"{m.adapter_method}{rank}{init}"

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "metrics": dict(self.metrics),
            "plots_dir": self.plots_dir,
            "best_examples": list(self.best_examples),
            "worst_examples": list(self.worst_examples),
            "known_blockers": list(self.known_blockers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResultBundle":
        return cls(
            manifest=RunManifest.from_dict(d.get("manifest") or {}),
            metrics=dict(d.get("metrics") or {}),
            plots_dir=d.get("plots_dir", ""),
            best_examples=list(d.get("best_examples") or []),
            worst_examples=list(d.get("worst_examples") or []),
            known_blockers=list(d.get("known_blockers") or []),
        )

    def write(self, path: str) -> None:
        """Serialize the bundle to ``path`` (the per-run ``report.json``)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)

    @classmethod
    def read(cls, path: str) -> "ResultBundle":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class ComparisonReport:
    """The cross-arm summary: one row per arm, one column per metric, + fairness + deltas."""

    arms: List[str]
    metrics: List[str]
    table: Dict[str, Dict[str, Optional[float]]]     # arm -> {metric: value}
    baseline: Optional[str]
    deltas: Dict[str, Dict[str, Optional[float]]]     # arm -> {metric: value - baseline}
    fairness_issues: List[str]

    def to_dict(self) -> dict:
        return {
            "arms": self.arms, "metrics": self.metrics, "table": self.table,
            "baseline": self.baseline, "deltas": self.deltas,
            "fairness_issues": self.fairness_issues,
        }

    def to_markdown(self) -> str:
        """A compact comparison table (+ fairness banner + baseline note) for a report."""
        lines = []
        if self.fairness_issues:
            lines.append("> ⚠️ NOT an apples-to-apples comparison:")
            lines.extend(f">  - {m}" for m in self.fairness_issues)
            lines.append("")
        header = "| arm | " + " | ".join(self.metrics) + " |"
        sep = "|---|" + "---|" * len(self.metrics)
        lines += [header, sep]
        for arm in self.arms:
            cells = []
            for metric in self.metrics:
                v = self.table[arm].get(metric)
                cells.append("—" if v is None else f"{v:.4g}")
            lines.append(f"| {arm}{' *(baseline)*' if arm == self.baseline else ''} | " + " | ".join(cells) + " |")
        if self.baseline:
            lines.append("")
            lines.append(f"_Deltas vs. baseline `{self.baseline}` are in the `deltas` field (positive = better for higher-is-better metrics)._")
        return "\n".join(lines)


def fairness_issues(bundles: List[ResultBundle]) -> List[str]:
    """List the manifest fields that differ across ``bundles`` (unfair-comparison flags).

    :param bundles: the arm bundles being compared.
    :return: human-readable messages for each :data:`_FAIRNESS_KEYS` field whose value is
        not identical across all arms (empty when the comparison is fair).
    """
    issues = []
    for key in _FAIRNESS_KEYS:
        values = {getattr(b.manifest, key) for b in bundles}
        if len(values) > 1:
            issues.append(f"{key} differs across arms: {sorted(map(str, values))}")
    return issues


def compare(bundles: List[ResultBundle], baseline: Optional[str] = "lora") -> ComparisonReport:
    """Aggregate arm bundles into a comparison table with fairness check + baseline deltas.

    :param bundles: the per-arm :class:`ResultBundle` objects (read from ``report.json``s).
    :param baseline: the arm label to diff against (matched by prefix, e.g. ``"lora"``
        matches ``"lora-r16"``); ``None`` to skip deltas.
    :return: the :class:`ComparisonReport`.
    """
    arms = [b.arm for b in bundles]
    metrics = sorted({m for b in bundles for m in b.metrics})
    table = {b.arm: {m: b.metrics.get(m) for m in metrics} for b in bundles}

    base_arm = None
    if baseline is not None:
        base_arm = next((a for a in arms if a == baseline or a.startswith(baseline)), None)
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    if base_arm is not None:
        base_row = table[base_arm]
        for arm in arms:
            deltas[arm] = {
                m: (None if table[arm][m] is None or base_row[m] is None
                    else table[arm][m] - base_row[m])
                for m in metrics
            }

    return ComparisonReport(
        arms=arms, metrics=metrics, table=table,
        baseline=base_arm, deltas=deltas, fairness_issues=fairness_issues(bundles),
    )


# Author: Mus mbayramo@stanford.edu
