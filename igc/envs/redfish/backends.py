"""Backend protocol and factory for Redfish simulator adapters.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from igc.envs.redfish.backend_types import (
    RedfishBackendCapabilities,
    RedfishBackendResponse,
    RedfishBackendStatus,
)


@runtime_checkable
class RedfishEnvironmentBackend(Protocol):
    """Protocol every Redfish simulator backend must satisfy."""

    def reset(self, seed: int | None = None) -> RedfishBackendStatus:
        """Reset provider state for one environment episode."""

    def request(
        self,
        method: str,
        path_or_capability: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RedfishBackendResponse:
        """Execute one simulator request or provider capability."""

    def status(self) -> RedfishBackendStatus:
        """Return provider health/session status."""

    def capabilities(self) -> RedfishBackendCapabilities:
        """Return provider-declared simulator capabilities."""

    def close(self) -> None:
        """Release provider resources."""


def make_backend(kind: str, **kwargs: Any) -> RedfishEnvironmentBackend:
    """Create a backend by config-friendly name.

    :param kind: one of ``redfish_ctl_inprocess``, ``redfish_ctl_http``, or
        ``legacy_mock``.
    :param kwargs: backend-specific constructor arguments.
    :return: backend instance.
    :raises ValueError: when ``kind`` is unsupported.
    """
    if kind == "redfish_ctl_inprocess":
        from igc.envs.redfish.redfish_ctl_inprocess import RedfishCtlInProcessBackend

        return RedfishCtlInProcessBackend(**kwargs)
    if kind == "redfish_ctl_http":
        from igc.envs.redfish.redfish_ctl_http import RedfishCtlHttpBackend

        return RedfishCtlHttpBackend(**kwargs)
    if kind == "legacy_mock":
        from igc.envs.redfish.legacy_mock import LegacyMockServerBackend

        return LegacyMockServerBackend(**kwargs)
    raise ValueError(f"unsupported Redfish backend kind: {kind}")


# Author: Mus mbayramo@stanford.edu
