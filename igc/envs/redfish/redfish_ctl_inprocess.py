"""In-process adapter for the provider-owned redfish_ctl simulator API.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib
from typing import Any, Mapping

from igc.envs.redfish.backend_types import (
    RedfishBackendCapabilities,
    RedfishBackendResponse,
    RedfishBackendStatus,
    RedfishProviderUnavailable,
    normalize_capabilities,
    normalize_response,
    normalize_status,
)


_BANNED_PROVIDER_FRAGMENTS = (
    "mock_bmc_server",
    "tests.",
    ".tests",
    "sandbox.",
    ".sandbox",
    "mutation_helpers",
)


class RedfishCtlInProcessBackend:
    """Wrap a provider simulator object without importing provider test helpers."""

    backend_kind = "redfish_ctl_inprocess"

    def __init__(
        self,
        provider: Any | None = None,
        module_name: str = "redfish_ctl.simulator",
        factory_name: str = "create_backend",
        **provider_kwargs: Any,
    ):
        """Initialize the in-process backend.

        :param provider: already-created provider simulator object for tests or
            provider-controlled construction.
        :param module_name: provider module exposing ``factory_name``.
        :param factory_name: factory function used when ``provider`` is omitted.
        :param provider_kwargs: keyword arguments passed to the provider factory.
        :raises RedfishProviderUnavailable: when the provider cannot be loaded.
        """
        if provider is not None:
            self._provider = provider
            return
        if any(fragment in module_name for fragment in _BANNED_PROVIDER_FRAGMENTS):
            raise RedfishProviderUnavailable(
                f"refusing to import non-contract redfish_ctl provider module: {module_name}"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RedfishProviderUnavailable(
                f"redfish_ctl in-process simulator provider is unavailable: {module_name}"
            ) from exc
        try:
            factory = getattr(module, factory_name)
        except AttributeError as exc:
            raise RedfishProviderUnavailable(
                f"redfish_ctl provider module {module_name!r} has no {factory_name!r} factory"
            ) from exc
        self._provider = factory(**provider_kwargs)

    def reset(self, seed: int | None = None) -> RedfishBackendStatus:
        """Reset provider state for a seeded episode."""
        return normalize_status(self._provider.reset(seed=seed), self.backend_kind)

    def request(
        self,
        method: str,
        path_or_capability: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RedfishBackendResponse:
        """Execute one provider request/capability."""
        raw = self._provider.request(
            method,
            path_or_capability,
            payload=dict(payload) if payload is not None else None,
            headers=dict(headers) if headers is not None else None,
        )
        return normalize_response(raw, self.backend_kind)

    def status(self) -> RedfishBackendStatus:
        """Return provider status."""
        return normalize_status(self._provider.status(), self.backend_kind)

    def capabilities(self) -> RedfishBackendCapabilities:
        """Return provider-declared simulator capabilities."""
        return normalize_capabilities(self._provider.capabilities())

    def close(self) -> None:
        """Close provider resources when the provider exposes ``close``."""
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


# Author: Mus mbayramo@stanford.edu
