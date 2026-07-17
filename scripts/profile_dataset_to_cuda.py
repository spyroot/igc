"""Profile the real Redfish corpus -> tokenizer -> DataLoader -> CUDA train step.

This bounded profiler is intentionally separate from the trainer: it reuses the
same ``CorpusJSONLDataset`` bridge and ``LlmEmbeddingsTrainer.custom_collate_fn``
that Phase 1 training uses, then times the host-to-device copy, forward,
backward, and optimizer stages on a live CUDA device.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.corpus_dataset import CorpusJSONLDataset
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


STAGES = ("dataloader_collate", "host_to_device", "forward", "backward", "optimizer")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_json_member(tar: tarfile.TarFile, member_name: str) -> Mapping[str, Any] | None:
    try:
        fh = tar.extractfile(member_name)
    except KeyError:
        return None
    if fh is None:
        return None
    with fh:
        try:
            loaded = json.load(fh)
        except json.JSONDecodeError:
            return None
    return loaded if isinstance(loaded, Mapping) else None


def _load_rest_map(
        tar: tarfile.TarFile,
        member_name: str,
        trust_npy_pickle: bool) -> Mapping[str, Any]:
    if not trust_npy_pickle:
        raise RuntimeError(
            "rest_api_map.npy requires trusted pickle loading; rerun with "
            "--trust-npy-pickle only for approved redfish_ctl full-corpus tarballs")
    fh = tar.extractfile(member_name)
    if fh is None:
        return {}
    with fh:
        data = fh.read()
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=True).item()
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _candidate_maps(
        tar: tarfile.TarFile,
        trust_npy_pickle: bool) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith("rest_api_map.npy"):
            base = str(Path(member.name).parent)
            yield base, _load_rest_map(tar, member.name, trust_npy_pickle)


def _resolve_member_name(tar_names: set[str], base: str, file_name: Any) -> str | None:
    """Resolve relative and historical absolute corpus map file names."""
    raw = str(file_name)
    candidates = [
        f"{base}/{raw}",
        f"{base}/{Path(raw).name}",
        raw.lstrip("/"),
    ]
    for candidate in candidates:
        if candidate in tar_names:
            return candidate
    suffix = f"/{Path(raw).name}"
    for member_name in tar_names:
        if member_name.startswith(f"{base}/") and member_name.endswith(suffix):
            return member_name
    return None


def _sample_corpus_tar(
        corpus_tar: Path,
        out_dir: Path,
        sample_rows: int,
        trust_npy_pickle: bool) -> Path:
    """Write a small ``examples.jsonl`` corpus sampled from a Redfish tarball."""
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_path = out_dir / "examples.jsonl"
    rows = 0
    source_maps = 0

    with (
            tarfile.open(corpus_tar, "r:gz") as tar,
            examples_path.open("w", encoding="utf-8") as out):
        tar_names = {member.name for member in tar.getmembers() if member.isfile()}
        for base, rest_map in _candidate_maps(tar, trust_npy_pickle):
            source_maps += 1
            url_to_file = rest_map.get("url_file_mapping", {})
            method_map = rest_map.get("allowed_methods_mapping", {})
            if not isinstance(url_to_file, Mapping):
                continue
            for url, file_name in sorted(url_to_file.items()):
                if rows >= sample_rows:
                    break
                member_name = _resolve_member_name(tar_names, base, file_name)
                if member_name is None:
                    continue
                response = _read_json_member(tar, member_name)
                if response is None:
                    continue
                allowed = method_map.get(url, []) if isinstance(method_map, Mapping) else []
                if isinstance(allowed, str):
                    allowed = [allowed]
                method = str(next(iter(allowed), "GET")).upper()
                payload = {
                    "request_or_action": {"method": method, "url": str(url)},
                    "allowed_methods": list(allowed) if isinstance(allowed, list) else [],
                    "response": response,
                    "source": {
                        "corpus_tar": str(corpus_tar),
                        "rest_api_map": f"{base}/rest_api_map.npy",
                        "member": member_name,
                    },
                }
                out.write(json.dumps(payload, sort_keys=True) + "\n")
                rows += 1
            if rows >= sample_rows:
                break

    if rows == 0:
        raise RuntimeError(f"no usable Redfish JSON rows sampled from {corpus_tar}")
    _json_dump(
        out_dir / "manifest.json",
        {
            "source": "redfish_ctl full corpus tar sample",
            "corpus_tar": str(corpus_tar),
            "sample_rows": rows,
            "rest_api_maps_seen": source_maps,
            "trusted_npy_pickle": trust_npy_pickle,
        },
    )
    return out_dir


def _resolve_corpus(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.corpus_dir:
        corpus_dir = Path(args.corpus_dir).expanduser().resolve()
        if not (corpus_dir / "examples.jsonl").is_file():
            raise FileNotFoundError(f"{corpus_dir} has no examples.jsonl")
        return corpus_dir
    if args.corpus_tar:
        return _sample_corpus_tar(
            Path(args.corpus_tar).expanduser().resolve(),
            output_dir / "corpus_sample",
            args.sample_rows,
            args.trust_npy_pickle,
        )
    raise SystemExit("set either --corpus-dir or --corpus-tar")


def _build_model(model_type: str, seq_len: int) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(model_type, local_files_only=True)
    except Exception:
        config = AutoConfig.for_model(model_type)
    for attr in ("n_positions", "max_position_embeddings"):
        if hasattr(config, attr) and getattr(config, attr) < seq_len:
            setattr(config, attr, seq_len)
    return AutoModelForCausalLM.from_config(config)


def _load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    local_files_only = not args.allow_tokenizer_download
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "local cache or remote download"
        raise RuntimeError(f"could not load tokenizer {args.tokenizer!r} from {mode}") from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.device != "auto":
        return torch.device(args.device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.device_count() > 1 and not visible:
        raise RuntimeError(
            "multiple CUDA devices are visible; set CUDA_VISIBLE_DEVICES or pass --device")
    return torch.device("cuda:0")


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device, non_blocking: bool):
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def _labels(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    labels = batch.get("labels")
    if torch.is_tensor(labels) and labels.ne(-100).any():
        return labels
    return batch["input_ids"]


def _summarize_samples(samples: Mapping[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage, values in samples.items():
        summary[stage] = {
            "mean_ms": statistics.mean(values),
            "p50_ms": statistics.median(values),
            "max_ms": max(values),
            "min_ms": min(values),
        }
    return summary


def _nvidia_smi() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _time_stage(samples: dict[str, list[float]], name: str, use_cuda: bool):
    return _StageTimer(samples, name, use_cuda)


class _StageTimer:
    def __init__(self, samples: dict[str, list[float]], name: str, use_cuda: bool):
        self._samples = samples
        self._name = name
        self._use_cuda = use_cuda

    def __enter__(self):
        if self._use_cuda:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        if self._use_cuda:
            torch.cuda.synchronize()
        self._samples[self._name].append((time.perf_counter() - self._start) * 1e3)
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = _resolve_corpus(args, output_dir)
    tokenizer = _load_tokenizer(args)

    dataset_start = time.perf_counter()
    dataset = CorpusJSONLDataset(
        str(corpus_dir),
        default_tokenize=args.tokenizer,
        max_len=args.seq_len,
        tokenizer=tokenizer,
        objective=args.corpus_objective,
    )
    dataset_ms = (time.perf_counter() - dataset_start) * 1e3
    if len(dataset) < args.batch_size:
        raise RuntimeError(f"dataset has {len(dataset)} rows, need batch size {args.batch_size}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=LlmEmbeddingsTrainer.custom_collate_fn,
    )

    device = _resolve_device(args)
    use_cuda = device.type == "cuda"
    model = _build_model(args.model_type, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    autocast_ctx = (
        torch.autocast(
            "cuda",
            dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.precision],
        )
        if use_cuda and args.precision in {"bf16", "fp16"}
        else nullcontext()
    )

    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    iterator = iter(loader)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    profiler_ctx = nullcontext()
    prof = None
    if args.trace:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
        prof = profile(activities=activities, record_shapes=False, profile_memory=True)
        profiler_ctx = prof

    nvidia_smi_before = _nvidia_smi()
    with profiler_ctx:
        for step in range(args.warmup + args.steps):
            try:
                with _time_stage(samples, "dataloader_collate", use_cuda=False):
                    host_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                with _time_stage(samples, "dataloader_collate", use_cuda=False):
                    host_batch = next(iterator)

            with _time_stage(samples, "host_to_device", use_cuda=use_cuda):
                batch = _move_batch(host_batch, device, args.non_blocking)
                batch["labels"] = _labels(batch)

            with autocast_ctx:
                with _time_stage(samples, "forward", use_cuda=use_cuda):
                    loss = model(**batch).loss
            with _time_stage(samples, "backward", use_cuda=use_cuda):
                loss.backward()
            with _time_stage(samples, "optimizer", use_cuda=use_cuda):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if prof is not None:
                prof.step()

            if step == args.warmup - 1:
                samples = {stage: [] for stage in STAGES}
                wall_start = time.perf_counter()
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats(device)

    wall_ms = (time.perf_counter() - wall_start) * 1e3
    stage_summary = _summarize_samples(samples)
    step_ms = wall_ms / args.steps
    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else None
    )

    summary = {
        "status": "ok",
        "corpus_dir": str(corpus_dir),
        "corpus_source": str(args.corpus_tar or args.corpus_dir),
        "dataset_rows": len(dataset),
        "dataset_tokenize_cache_ms": dataset_ms,
        "dataset_tokenize_build_ms": dataset_ms,
        "model_type": args.model_type,
        "tokenizer": args.tokenizer,
        "allow_tokenizer_download": args.allow_tokenizer_download,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "non_blocking": args.non_blocking,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "precision": args.precision,
        "warmup_steps": args.warmup,
        "profile_steps": args.steps,
        "wall_ms": wall_ms,
        "step_ms": step_ms,
        "samples_per_sec": args.batch_size / (step_ms / 1e3),
        "stages": stage_summary,
        "peak_mem_gb": peak_mem_gb,
        "nvidia_smi_before_after": {
            "before": nvidia_smi_before,
            "after": _nvidia_smi(),
        },
    }
    _json_dump(output_dir / "profile_summary.json", summary)

    if prof is not None:
        sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_key, row_limit=args.top_ops)
        (output_dir / "top_ops.txt").write_text(table, encoding="utf-8")
        prof.export_chrome_trace(str(output_dir / "trace.json"))
        print(table)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-dir", default="", help="Directory with examples.jsonl")
    source.add_argument("--corpus-tar", default="", help="redfish_ctl full-corpus tar.gz")
    parser.add_argument("--sample-rows", type=int, default=64)
    parser.add_argument(
        "--trust-npy-pickle",
        action="store_true",
        help="Allow rest_api_map.npy pickle loading when sampling approved corpus tarballs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="gpt2")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow AutoTokenizer to download when the tokenizer is not already local.",
    )
    parser.add_argument(
        "--corpus-objective",
        default="legacy",
        choices=("legacy", "phase1_pretrain"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--top-ops", type=int, default=15)
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for debugging only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
