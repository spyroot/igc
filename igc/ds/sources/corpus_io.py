"""
Persist a normalized training corpus to disk and read it back.

Writes :class:`TrainingExample` objects as JSONL (one ``to_dict()`` per line) with a
:class:`DataManifest` sidecar, and reads them back as dicts. Offline only — this is the seam
between the source/mixer/normalizer pipeline and a training run that consumes a fixed corpus.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Iterable, Iterator, List

from igc.ds.sources.mixer import DataManifest
from igc.ds.sources.training_object import TrainingExample


def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` when it is non-empty and missing.

    :param path: a file path whose parent directory should exist.
    :return: ``None``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_examples(examples: Iterable[TrainingExample], path: str) -> int:
    """Write training examples as JSONL (one ``to_dict()`` per line).

    :param examples: the examples to serialize.
    :param path: destination ``.jsonl`` path (parent dirs are created).
    :return: the number of examples written.
    """
    _ensure_parent(path)
    count = 0
    with open(path, "w") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict()))
            handle.write("\n")
            count += 1
    return count


def iter_examples(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one example dict per non-blank JSONL line (streams the file).

    :param path: a ``.jsonl`` file written by :func:`write_examples`.
    :return: an iterator of example dicts.
    """
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_examples(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL corpus fully into memory.

    :param path: a ``.jsonl`` file written by :func:`write_examples`.
    :return: the list of example dicts.
    """
    return list(iter_examples(path))


def write_manifest(manifest: DataManifest, path: str) -> None:
    """Write a :class:`DataManifest` as pretty, sorted JSON.

    :param manifest: the manifest to serialize.
    :param path: destination ``.json`` path (parent dirs are created).
    :return: ``None``.
    """
    _ensure_parent(path)
    with open(path, "w") as handle:
        json.dump(dataclasses.asdict(manifest), handle, indent=2, sort_keys=True)


def read_manifest(path: str) -> Dict[str, Any]:
    """Load a manifest JSON file written by :func:`write_manifest`.

    :param path: the manifest path.
    :return: the manifest as a dict.
    """
    with open(path, "r") as handle:
        return json.load(handle)


def write_corpus(examples: Iterable[TrainingExample], manifest: DataManifest,
                 out_dir: str) -> Dict[str, str]:
    """Write a corpus (``examples.jsonl`` + ``manifest.json``) into ``out_dir``.

    :param examples: the examples to serialize.
    :param manifest: the corpus manifest.
    :param out_dir: destination directory (created if missing).
    :return: mapping with ``examples`` / ``manifest`` paths and ``count`` (as a string).
    """
    os.makedirs(out_dir, exist_ok=True)
    examples_path = os.path.join(out_dir, "examples.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    count = write_examples(examples, examples_path)
    write_manifest(manifest, manifest_path)
    return {"examples": examples_path, "manifest": manifest_path, "count": str(count)}


# Author: Mus mbayramo@stanford.edu
