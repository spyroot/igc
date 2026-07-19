"""Hard gate for model/runtime configuration living in YAML specs.

Training and inference code may load model identifiers, adapter paths, runtime
dtypes, generation limits, and LoRA hyperparameters from YAML under
``configs/training`` or ``configs/inference``. It must not carry concrete model
IDs or lab artifact paths in runtime source files. This script validates the
YAML specs and fails when selected runtime files contain those concrete values.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_GLOBS = ("configs/training/*.yaml", "configs/inference/*.yaml")
DEFAULT_SCAN_PATHS = (
    "igc/modules/train",
    "scripts/phase1_inference_gate.py",
)
FORBIDDEN_PATTERNS = (
    re.compile(r"Qwen/Qwen[0-9A-Za-z./_-]*"),
    re.compile(r"DeepSeek|deepseek-v[0-9]", re.IGNORECASE),
    re.compile(r"/models/igc/[^\s'\"`]+"),
    re.compile(r"phase1-finetune-qwen[^\s'\"`]+"),
)
ALLOWED_SUFFIXES = {".py", ".sh"}


@dataclass(frozen=True)
class Finding:
    """One concrete model/config value found outside a YAML spec."""

    path: Path
    line: int
    text: str


def _load_yaml(path: Path) -> None:
    """Parse a YAML spec and require a top-level mapping with ``version``."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: spec must be a mapping")
    if "version" not in payload:
        raise ValueError(f"{path}: spec missing version")


def validate_specs(root: Path) -> list[Path]:
    """Validate configured YAML specs and return the files checked."""

    checked: list[Path] = []
    for pattern in SPEC_GLOBS:
        for path in sorted(root.glob(pattern)):
            _load_yaml(path)
            checked.append(path)
    if not checked:
        raise ValueError("no YAML specs found under configs/training or configs/inference")
    return checked


def iter_scan_files(root: Path, scan_paths: Iterable[str]) -> Iterable[Path]:
    """Yield runtime source files to scan for concrete model config values."""

    for raw in scan_paths:
        path = (root / raw).resolve()
        if not path.exists():
            raise ValueError(f"scan path does not exist: {raw}")
        if path.is_file():
            if path.suffix in ALLOWED_SUFFIXES:
                yield path
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in ALLOWED_SUFFIXES:
                yield child


def scan_runtime_files(root: Path, scan_paths: Iterable[str]) -> list[Finding]:
    """Return source locations with concrete model IDs or lab artifact paths."""

    findings: list[Finding] = []
    for path in iter_scan_files(root, scan_paths):
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(rel, line_no, line.strip()))
                    break
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--scan-path", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    root = Path(args.root).resolve()
    scan_paths = tuple(args.scan_path) if args.scan_path else DEFAULT_SCAN_PATHS
    try:
        specs = validate_specs(root)
        findings = scan_runtime_files(root, scan_paths)
    except ValueError as exc:
        print(f"MODEL_SPEC_POLICY_FAIL {exc}", file=sys.stderr)
        return 1
    if findings:
        print("MODEL_SPEC_POLICY_FAIL concrete model config found outside YAML specs", file=sys.stderr)
        for finding in findings:
            print(
                f"{finding.path}:{finding.line}: {finding.text}",
                file=sys.stderr,
            )
        return 1
    print(
        "MODEL_SPEC_POLICY_PASS "
        f"specs={len(specs)} scan_paths={','.join(scan_paths)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
