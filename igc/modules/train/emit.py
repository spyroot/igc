"""
Run-report emission: turn a finished training run into a ResultBundle on disk.

Called only from ``LlmEmbeddingsTrainer._train`` (``igc/modules/llm_train_state_encoder.py``)
at end of run, on rank zero, inside a broad try/except so a report failure only warns and
never fails a finished training run. It exists so every run leaves a machine-readable
``report.json`` for the comparison tooling.

The trainer calls :func:`build_run_bundle` with its parsed spec and end-of-run stats,
then :func:`emit_run_report` writes ``report.json`` into the run's output directory —
making every run self-describing (model, adapter, data manifest, environment, final
metrics) and comparable through the existing ``compare()`` report tooling. Pure
functions, no trainer imports, so the contract is testable offline.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from igc.modules.train.report import ResultBundle, RunManifest, capture_environment

# spec keys copied into RunManifest.settings — the run knobs that matter for
# reproducing/comparing a run (never secrets; see _SENSITIVE_MARKERS).
_SETTINGS_KEYS = (
    "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs",
    "llm_learning_rate", "llm_scheduler", "llm_optimizer", "mixed_precision",
    "sharding", "use_accelerator", "use_peft", "lora_r", "lora_alpha", "lora_dropout",
    "masking_type", "num_workers", "seed",
)

# any spec key containing one of these is never emitted, whatever its value.
_SENSITIVE_MARKERS = ("token", "key", "password", "secret", "credential")


def _scrub(spec_vars: Dict) -> Dict:
    """Copy selected run settings from the spec, refusing sensitive keys.

    :param spec_vars: ``vars(spec)`` from the argument parser.
    :return: settings dict safe to persist in a report.
    """
    settings = {}
    for key in _SETTINGS_KEYS:
        if any(marker in key.lower() for marker in _SENSITIVE_MARKERS):
            continue
        if key in spec_vars:
            settings[key] = str(spec_vars[key])
    return settings


def build_run_bundle(spec_vars: Dict, *,
                     training: Dict,
                     metrics: Optional[Dict] = None,
                     dataset_fields: Optional[Dict] = None,
                     argv: Optional[List[str]] = None,
                     started_at: str = "",
                     ended_at: str = "",
                     wall_clock_sec: Optional[float] = None,
                     checkpoint_path: str = "") -> ResultBundle:
    """Assemble a ResultBundle for a finished run.

    :param spec_vars: ``vars(spec)`` — model/adapters/settings are read from here.
    :param training: end-of-run stats (e.g. ``final_epoch_loss``, ``epochs_done``,
        ``optimizer_steps``, ``best_eval``).
    :param metrics: headline metric values for the comparison table.
    :param dataset_fields: ``{"data_manifest", "eval_split"}`` from the corpus
        (``CorpusJSONLDataset.run_manifest_fields``), empty when training from raw captures.
    :param argv: the launch command (defaults to ``sys.argv``).
    :param started_at: ISO start timestamp.
    :param ended_at: ISO end timestamp.
    :param wall_clock_sec: run duration in seconds.
    :param checkpoint_path: where the run saved its model.
    :return: the assembled :class:`ResultBundle`.
    """
    dataset_fields = dataset_fields or {}
    model = str(spec_vars.get("model_type", ""))
    use_peft = bool(spec_vars.get("use_peft", False))
    started_tag = started_at.replace(":", "").replace("-", "") or "run"
    manifest = RunManifest(
        run_id=f"m1-{os.path.basename(model) or 'model'}-{started_tag}",
        profile=str(spec_vars.get("profile", "") or ""),
        model=model,
        tokenizer=model,
        adapter_method=str(spec_vars.get("adapter_method", "lora")) if use_peft else "none",
        adapter_rank=int(spec_vars["lora_r"]) if use_peft and "lora_r" in spec_vars else None,
        adapter_init=str(spec_vars.get("lora_init", "default") or "default"),
        data_manifest=str(dataset_fields.get("data_manifest", "")),
        eval_split=str(dataset_fields.get("eval_split", "")),
        max_steps=spec_vars.get("max_train_steps"),
        seq_len=spec_vars.get("seq_len"),
        settings=_scrub(spec_vars),
        argv=list(argv if argv is not None else sys.argv),
        started_at=started_at,
        ended_at=ended_at,
        wall_clock_sec=wall_clock_sec,
        checkpoint_path=str(checkpoint_path or ""),
        environment=capture_environment(),
        training=dict(training),
    )
    return ResultBundle(manifest=manifest, metrics=dict(metrics or {}))


def emit_run_report(bundle: ResultBundle, output_dir: str) -> str:
    """Write the bundle as ``report.json`` under ``output_dir``.

    :param bundle: the assembled run bundle.
    :param output_dir: run output directory (created if missing).
    :return: the written report path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.json")
    bundle.write(path)
    return path


# Author: Mus mbayramo@stanford.edu
