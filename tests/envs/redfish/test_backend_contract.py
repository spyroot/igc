"""Offline contract tests for redfish_ctl-backed Redfish environment backends.

The tests pin IGC's adapter boundary before the provider simulator lands in
full. They never import provider test helpers, never contact a live BMC, and
use only a local loopback HTTP server or tiny fake provider objects.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any

import pytest

from igc.envs.redfish.backend_types import (
    RedfishBackendError,
    RedfishBackendResponse,
    RedfishContractError,
    RedfishProviderUnavailable,
)
from igc.envs.redfish.backends import make_backend
from igc.envs.redfish.legacy_mock import LegacyMockServerBackend
from igc.envs.redfish.redfish_ctl_http import RedfishCtlHttpBackend
from igc.envs.redfish.redfish_ctl_inprocess import RedfishCtlInProcessBackend


class _FakeHttpResponse:
    """Context-manager response returned by a fake ``urlopen``."""

    def __init__(self, status: int, body: dict[str, Any]):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._payload = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        """Return encoded response bytes."""
        return self._payload


class _UrlopenRecorder:
    """Pure unit-test replacement for ``urllib.request.urlopen``."""

    def __init__(self, routes: dict[tuple[str, str], tuple[int, dict[str, Any]]]):
        self.routes = routes
        self.seen: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float) -> _FakeHttpResponse:
        """Record the request and return the configured response."""
        path = urlparse(req.full_url).path
        body = json.loads(req.data.decode("utf-8")) if req.data else {}
        method = req.get_method()
        self.seen.append({"method": method, "path": path, "body": body, "timeout": timeout})
        status, response_body = self.routes[(method, path)]
        return _FakeHttpResponse(status, response_body)


def _capabilities(contract_version: str = "redfish-simulator/v1") -> dict[str, Any]:
    """Return one minimal provider capability document."""
    return {
        "contract_version": contract_version,
        "provider_revision": "provider-abc123",
        "corpus_id": "fixture-full",
        "actions": [
            {
                "id": "systems.reset",
                "method": "POST",
                "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                "payload_schema": {"ResetType": ["GracefulRestart"]},
            }
        ],
    }


def test_http_backend_normalizes_request_response_and_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP simulator responses become one backend response shape."""
    routes = {
        ("GET", "/capabilities"): (200, _capabilities()),
        ("POST", "/request"): (
            200,
            {
                "status_code": 202,
                "json": {"Task": {"@odata.id": "/redfish/v1/TaskService/Tasks/1"}},
                "headers": {"Location": "/redfish/v1/TaskService/Tasks/1"},
                "provider_error_code": None,
                "mutation": {"changed": ["/redfish/v1/Systems/1"]},
                "task": {"state": "Running"},
            },
        ),
    }
    urlopen = _UrlopenRecorder(routes)
    monkeypatch.setattr("igc.envs.redfish.redfish_ctl_http.request.urlopen", urlopen)
    backend = RedfishCtlHttpBackend("http://simulator.invalid", timeout=2)
    caps = backend.capabilities()
    response = backend.request(
        "POST",
        "systems.reset",
        payload={"ResetType": "GracefulRestart"},
        headers={"If-Match": "etag-1"},
    )

    assert caps.contract_version == "redfish-simulator/v1"
    assert caps.provider_revision == "provider-abc123"
    assert caps.action_capabilities[0]["id"] == "systems.reset"
    assert response.status_code == 202
    assert response.json_data["Task"]["@odata.id"].endswith("/Tasks/1")
    assert response.headers["Location"].endswith("/Tasks/1")
    assert response.mutation_metadata == {"changed": ["/redfish/v1/Systems/1"]}
    assert response.task_metadata == {"state": "Running"}
    assert response.provenance["backend_kind"] == "redfish_ctl_http"
    assert urlopen.seen[-1]["body"] == {
        "method": "POST",
        "target": "systems.reset",
        "payload": {"ResetType": "GracefulRestart"},
        "headers": {"If-Match": "etag-1"},
    }
    assert urlopen.seen[-1]["timeout"] == 2


def test_http_backend_rejects_unknown_contract_major_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract version negotiation fails before any episode starts."""
    routes = {
        ("GET", "/capabilities"): (200, _capabilities("redfish-simulator/v99")),
    }
    urlopen = _UrlopenRecorder(routes)
    monkeypatch.setattr("igc.envs.redfish.redfish_ctl_http.request.urlopen", urlopen)
    backend = RedfishCtlHttpBackend("http://simulator.invalid", timeout=2)
    with pytest.raises(RedfishContractError, match="redfish-simulator/v99"):
        backend.capabilities()


def test_http_backend_reset_normalizes_seeded_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every reset returns provider status with seed and backend provenance."""
    routes = {
        ("POST", "/reset"): (
            200,
            {
                "backend_kind": "provider_http",
                "ready": True,
                "corpus_id": "fixture-full",
                "contract_version": "redfish-simulator/v1",
                "provider_revision": "provider-abc123",
                "seed": 123,
                "episode_id": "episode-123",
            },
        ),
    }
    urlopen = _UrlopenRecorder(routes)
    monkeypatch.setattr("igc.envs.redfish.redfish_ctl_http.request.urlopen", urlopen)
    status = RedfishCtlHttpBackend("http://simulator.invalid", timeout=2).reset(seed=123)

    assert status.backend_kind == "redfish_ctl_http"
    assert status.ready is True
    assert status.seed == 123
    assert status.episode_id == "episode-123"
    assert urlopen.seen[-1]["body"] == {"seed": 123}


def test_http_backend_unreachable_simulator_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection failures become backend errors with provider context."""

    def raise_os_error(req: Any, timeout: float) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr("igc.envs.redfish.redfish_ctl_http.request.urlopen", raise_os_error)

    with pytest.raises(RedfishBackendError, match="unreachable"):
        RedfishCtlHttpBackend("http://simulator.invalid", timeout=2).status()


class _FakeProvider:
    """Provider-like in-process simulator used only by tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.calls.append(("reset", seed))
        return {
            "ready": True,
            "contract_version": "redfish-simulator/v1",
            "provider_revision": "provider-xyz",
            "corpus_id": "inproc-fixture",
            "seed": seed,
            "episode_id": "inproc-7",
        }

    def request(
        self,
        method: str,
        target: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("request", method, target, payload, headers))
        return {"status_code": 200, "json": {"ok": True}, "headers": {"ETag": "1"}}

    def status(self) -> dict[str, Any]:
        self.calls.append(("status", None))
        return {"ready": True, "contract_version": "redfish-simulator/v1"}

    def capabilities(self) -> dict[str, Any]:
        self.calls.append(("capabilities", None))
        return _capabilities()

    def close(self) -> None:
        self.calls.append(("close", None))


def test_inprocess_backend_delegates_to_provider_contract() -> None:
    """The in-process adapter wraps provider reset/request/capabilities."""
    provider = _FakeProvider()
    backend = RedfishCtlInProcessBackend(provider=provider)

    status = backend.reset(seed=7)
    caps = backend.capabilities()
    response = backend.request("GET", "/redfish/v1/Systems/1")
    backend.close()

    assert status.backend_kind == "redfish_ctl_inprocess"
    assert status.seed == 7
    assert caps.provider_revision == "provider-abc123"
    assert response == RedfishBackendResponse(
        status_code=200,
        json_data={"ok": True},
        headers={"ETag": "1"},
        provenance={"backend_kind": "redfish_ctl_inprocess"},
    )
    assert provider.calls == [
        ("reset", 7),
        ("capabilities", None),
        ("request", "GET", "/redfish/v1/Systems/1", None, None),
        ("close", None),
    ]


def test_inprocess_backend_missing_provider_fails_clearly() -> None:
    """No provider dependency means a clear blocker, not a test-helper import."""
    with pytest.raises(RedfishProviderUnavailable, match="redfish_ctl"):
        RedfishCtlInProcessBackend(module_name="missing_redfish_ctl_provider_for_igc")


def test_inprocess_backend_refuses_test_or_sandbox_modules() -> None:
    """Provider imports are restricted to declared contract modules."""
    with pytest.raises(RedfishProviderUnavailable, match="non-contract"):
        RedfishCtlInProcessBackend(module_name="redfish_ctl.k8s.sandbox.mock_bmc_server")
    with pytest.raises(RedfishProviderUnavailable, match="non-contract"):
        RedfishCtlInProcessBackend(module_name="redfish_ctl.tests.vendor_corpus")


class _FakeMockResponse:
    """Small MockResponse-compatible object."""

    def __init__(self) -> None:
        self.status_code = 201
        self.error = False
        self.new_state = {"PowerState": "On"}

    def json(self) -> dict[str, Any]:
        return {"PowerState": "On"}


class _FakeMockServer:
    """MockServer-compatible object for the legacy adapter."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def request(
        self,
        url: str,
        method: str = "GET",
        json_data: str | None = None,
        accept_header: str | None = None,
    ) -> _FakeMockResponse:
        self.calls.append((url, method, json_data, accept_header))
        return _FakeMockResponse()


def test_legacy_mock_backend_preserves_mock_response_fields() -> None:
    """Legacy MockServer stays available behind the new protocol."""
    mock = _FakeMockServer()
    response = LegacyMockServerBackend(mock).request(
        "PATCH",
        "/redfish/v1/Systems/1",
        payload={"PowerState": "On"},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 201
    assert response.json_data == {"PowerState": "On"}
    assert response.mutation_metadata == {"new_state": {"PowerState": "On"}}
    assert response.error is False
    assert response.provenance["backend_kind"] == "legacy_mock_server"
    assert mock.calls == [
        (
            "/redfish/v1/Systems/1",
            "PATCH",
            json.dumps({"PowerState": "On"}, sort_keys=True),
            "application/json",
        )
    ]


def test_backend_factory_selects_supported_kinds() -> None:
    """Factory names are stable for config-driven environments."""
    mock_backend = make_backend("legacy_mock", mock_server=_FakeMockServer())

    assert isinstance(mock_backend, LegacyMockServerBackend)
    with pytest.raises(ValueError, match="unsupported Redfish backend"):
        make_backend("invented")


# Author: Mus mbayramo@stanford.edu
