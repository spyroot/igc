"""Deterministic JSON-link extraction for captured REST resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def extract_links(
    body: Mapping[str, Any],
    *,
    known_uris: frozenset[str],
    source_uri: str,
) -> tuple[tuple[str, str], ...]:
    """Return known URI targets and the JSON pointers that expose them."""
    discovered: set[tuple[str, str]] = set()

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, str):
            if value in known_uris and value != source_uri:
                discovered.add((value, pointer or "/"))
            return
        if isinstance(value, Mapping):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                visit(value[key], f"{pointer}/{_pointer_token(key)}")
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")

    visit(body, "")
    return tuple(sorted(discovered, key=lambda item: (item[1], item[0])))
