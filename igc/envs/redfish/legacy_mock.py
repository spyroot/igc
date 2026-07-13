"""Compatibility adapter for the existing IGC MockServer.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from igc.envs.redfish.backend_types import (
    RedfishBackendCapabilities,
    RedfishBackendResponse,
    RedfishBackendStatus,
)


class LegacyMockServerBackend:
    """Expose ``MockServer`` through the new backend protocol."""

    backend_kind = "legacy_mock_server"

    def __init__(self, mock_server: Any):
        """Initialize the adapter.

        :param mock_server: object compatible with
            :meth:`igc.envs.rest_mock_server.MockServer.request`.
        """
        self._mock_server = mock_server

    def reset(self, seed: int | None = None) -> RedfishBackendStatus:
        """Return a compatibility status; legacy MockServer has no reset hook."""
        return RedfishBackendStatus(
            backend_kind=self.backend_kind,
            ready=True,
            seed=seed,
            provenance={"backend_kind": self.backend_kind, "legacy": True},
        )

    def request(
        self,
        method: str,
        path_or_capability: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RedfishBackendResponse:
        """Delegate to the legacy mock server and normalize its response."""
        json_data = None
        if payload is not None:
            json_data = json.dumps(dict(payload), sort_keys=True)
        accept = None
        if headers is not None:
            accept = headers.get("Accept") or headers.get("accept")
        response = self._mock_server.request(
            path_or_capability,
            method=method,
            json_data=json_data,
            accept_header=accept,
        )
        try:
            body = response.json()
        except (AttributeError, ValueError):
            body = getattr(response, "json_data", None)
        new_state = getattr(response, "new_state", None)
        if new_state is None and callable(getattr(response, "state", None)):
            new_state = response.state()
        mutation_metadata = {"new_state": new_state} if new_state is not None else {}
        return RedfishBackendResponse(
            status_code=int(getattr(response, "status_code", 200)),
            json_data=body,
            error=bool(getattr(response, "error", False)),
            mutation_metadata=mutation_metadata,
            provenance={"backend_kind": self.backend_kind, "legacy": True},
        )

    def status(self) -> RedfishBackendStatus:
        """Return a compatibility status."""
        return RedfishBackendStatus(
            backend_kind=self.backend_kind,
            ready=True,
            provenance={"backend_kind": self.backend_kind, "legacy": True},
        )

    def capabilities(self) -> RedfishBackendCapabilities:
        """Legacy mock has no provider-declared mutation capabilities."""
        return RedfishBackendCapabilities(
            contract_version="redfish-simulator/v1",
            provider_revision="igc-legacy-mock",
            action_capabilities=[],
            raw={"legacy": True},
        )

    def close(self) -> None:
        """Legacy mock owns no external resources."""


# Author: Mus mbayramo@stanford.edu
