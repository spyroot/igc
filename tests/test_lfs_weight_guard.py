"""Offline guard: LFS weight objects must never auto-materialize on a clone.

Pins the repo-committed ``.lfsconfig`` (created by the LFS-weight-guard change)
that sets ``lfs.fetchexclude`` for model weight formats, so ``git lfs pull``,
``actions/checkout`` with ``lfs: true``, and laptop clones download pointer
metadata only — the bulky ``*.pt``/``*.safetensors`` objects stay on the LFS
remote/GB300 side unless a node explicitly overrides with
``git lfs pull --include=...``. Also pins that the weight formats remain
LFS-tracked in ``.gitattributes`` (the guard is meaningless if weights ever
become plain git blobs). Pure file reads; no network, GPU, or git-lfs binary.

Author:
Mus mbayramo@stanford.edu
"""

import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fetchexclude_patterns() -> list[str]:
    """Parse the committed .lfsconfig and return the fetchexclude patterns."""
    lfsconfig = REPO_ROOT / ".lfsconfig"
    assert lfsconfig.is_file(), ".lfsconfig missing — LFS weight guard removed?"
    parser = configparser.ConfigParser()
    parser.read_string(lfsconfig.read_text(encoding="utf-8"))
    raw = parser.get("lfs", "fetchexclude", fallback="")
    return [p.strip() for p in raw.split(",") if p.strip()]


def test_lfsconfig_excludes_model_weight_formats():
    """Checkpoints (*.pt) and adapters (*.safetensors) are fetch-excluded."""
    patterns = _fetchexclude_patterns()
    assert "*.pt" in patterns
    assert "*.safetensors" in patterns


def test_lfsconfig_does_not_exclude_corpus_tarballs():
    """Corpus/dataset tarballs stay fetchable for offline tests."""
    patterns = _fetchexclude_patterns()
    assert "*.tar.gz" not in patterns
    assert "*" not in patterns  # a blanket exclude would break corpus fetching


def test_weight_formats_remain_lfs_tracked():
    """.gitattributes keeps weights on LFS; the guard assumes pointer storage."""
    gitattributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.pt filter=lfs" in gitattributes
    assert "*.safetensors filter=lfs" in gitattributes


# Author: Mus mbayramo@stanford.edu
