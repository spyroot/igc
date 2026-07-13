"""HTTP adapter for a standalone redfish_ctl simulator.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from typing import Any, Mapping
from urllib import error, request

from igc.envs.redfish.backend_types import (
    RedfishBackendCapabilities,
    RedfishBackendError,
    RedfishBackendResponse,
    RedfishBackendStatus,
    normalize_capabilities,
    normalize_response,
    normalize_status,
)


class RedfishCtlHttpBackend:
    """Call a standalone provider simulator over a small JSON HTTP API."""

    backend_kind = "redfish_ctl_http"

    def __init__(self, base_url: str, timeout: float = 5.0):
        """Initialize the HTTP backend.

        :param base_url: base URL of the provider simulator.
        :param timeout: request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json_call(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a provider JSON endpoint and return a decoded dict."""
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                if not payload:
                    return {"status_code": resp.status, "json": None, "headers": dict(resp.headers)}
                decoded = json.loads(payload.decode("utf-8"))
                if isinstance(decoded, dict):
                    return decoded
                return {"status_code": resp.status, "json": decoded, "headers": dict(resp.headers)}
        except error.HTTPError as exc:
            payload = exc.read()
            try:
                decoded = json.loads(payload.decode("utf-8")) if payload else {}
            except json.JSONDecodeError:
                decoded = {"error": payload.decode("utf-8", errors="replace")}
            if isinstance(decoded, dict):
                decoded.setdefault("status_code", exc.code)
                decoded.setdefault("error", True)
                return decoded
            return {"status_code": exc.code, "json": decoded, "error": True}
        except OSError as exc:
            raise RedfishBackendError(f"redfish_ctl HTTP simulator unreachable: {exc}") from exc

    def reset(self, seed: int | None = None) -> RedfishBackendStatus:
        """Reset provider state for a seeded episode."""
        return normalize_status(self._json_call("POST", "/reset", {"seed": seed}), self.backend_kind)

    def request(
        self,
        method: str,
        path_or_capability: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RedfishBackendResponse:
        """Execute one provider request/capability."""
        body = {
            "method": method,
            "target": path_or_capability,
            "payload": dict(payload) if payload is not None else None,
            "headers": dict(headers) if headers is not None else None,
        }
        return normalize_response(self._json_call("POST", "/request", body), self.backend_kind)

    def status(self) -> RedfishBackendStatus:
        """Return provider status."""
        return normalize_status(self._json_call("GET", "/status"), self.backend_kind)

    def capabilities(self) -> RedfishBackendCapabilities:
        """Return provider-declared simulator capabilities."""
        return normalize_capabilities(self._json_call("GET", "/capabilities"))

    def close(self) -> None:
        """HTTP backend owns no persistent provider resources."""


# Author: Mus mbayramo@stanford.edu
