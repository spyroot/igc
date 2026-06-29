"""Offline tests for MockServer request replay and callback behavior."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from igc.envs.rest_mock_server import MockResponse, MockServer
from igc.interfaces.rest_mapping_interface import RestMappingInterface


SYSTEM_URI = "/redfish/v1/Systems/1"
SYSTEM_PAYLOAD = {
    "@odata.id": SYSTEM_URI,
    "Name": "Tiny test system",
    "PowerState": "Off",
}


class TinyRestMapping(RestMappingInterface):
    """Small RestMappingInterface test double backed by local JSON files."""

    def __init__(self, mappings: dict[str, Path]) -> None:
        self._mappings = mappings

    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        """Return the local response file for a REST API."""
        path = self._mappings.get(rest_api)
        return "" if path is None else str(path)

    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        """Return the offline method advertised by this fixture."""
        return "GET" if rest_api in self._mappings else ""

    def get_rest_api_mappings(self) -> Iterator[tuple[str, str]]:
        """Yield REST API to local response-file mappings."""
        for rest_api, path in self._mappings.items():
            yield rest_api, str(path)

    def get_rest_api_methods(self) -> Iterator[tuple[str, str]]:
        """Yield the supported method for each REST API fixture."""
        for rest_api in self._mappings:
            yield rest_api, "GET"


@pytest.fixture
def mock_server(tmp_path: Path) -> MockServer:
    """Create a MockServer backed by one tiny local JSON response."""
    response_path = tmp_path / "system.json"
    response_path.write_text(json.dumps(SYSTEM_PAYLOAD), encoding="utf-8")

    args = argparse.Namespace(raw_data_dir=str(tmp_path))
    mapping = TinyRestMapping({SYSTEM_URI: response_path})
    return MockServer(args=args, rest_mapping=mapping)


def assert_error_response(response: MockResponse, expected_status: int) -> None:
    """Assert a synthesized Redfish-style error response."""
    assert response.status_code == expected_status
    assert response.error is True

    body = json.loads(response.json())
    assert body["error"]["code"] == "Base.1.2.GeneralError"
    assert "@Message.ExtendedInfo" in body["error"]


def test_mock_response_exposes_fields_and_helpers() -> None:
    """MockResponse keeps response data, status, error flag, and state."""
    new_state = {"PowerState": "On"}
    response = MockResponse({"ok": True}, 202, error=True, new_state=new_state)

    assert response.json_data == {"ok": True}
    assert response.status_code == 202
    assert response.error is True
    assert response.new_state == new_state
    assert response.json() == {"ok": True}
    assert response.state() == new_state
    assert "status_code=202" in str(response)
    assert "error=True" in str(response)


def test_request_get_replays_stored_response(mock_server: MockServer) -> None:
    """GET returns the stored response body and success status."""
    response = mock_server.request(SYSTEM_URI, method="GET")

    assert response.status_code == 200
    assert response.error is False
    assert json.loads(response.json()) == SYSTEM_PAYLOAD


@pytest.mark.parametrize(
    ("url", "method", "expected_status"),
    [
        (SYSTEM_URI, "POST", 405),
        (SYSTEM_URI, "PATCH", 405),
        ("/redfish/v1/Missing", "POST", 404),
        ("/redfish/v1/Missing", "GET", 404),
    ],
)
def test_request_without_registered_handler_returns_synthesized_errors(
    mock_server: MockServer,
    url: str,
    method: str,
    expected_status: int,
) -> None:
    """Missing callbacks return 405 for known URLs and 404 for unknown URLs."""
    response = mock_server.request(url, method=method, json_data=json.dumps({}))

    assert_error_response(response, expected_status)


def test_request_can_synthesize_http_500(mock_server: MockServer) -> None:
    """The HTTP 500 simulation flag overrides normal offline replay."""
    mock_server.set_simulate_http_500_error(True)

    response = mock_server.request(SYSTEM_URI, method="GET")

    assert_error_response(response, 500)

    mock_server.set_simulate_http_500_error(False)
    recovered = mock_server.request(SYSTEM_URI, method="GET")
    assert recovered.status_code == 200
    assert json.loads(recovered.json()) == SYSTEM_PAYLOAD


def test_registered_callback_dispatches_and_mutates_stored_get_view(
    mock_server: MockServer,
) -> None:
    """Callbacks can return a response and update replay state via new_state."""
    submitted_payload = {"PowerState": "On"}
    observed: dict[str, object] = {}

    def set_power_state(json_data: str, handler_view: dict[str, object]) -> MockResponse:
        observed["json_data"] = json_data
        observed["handler_view"] = handler_view

        current_state = json.loads(handler_view["json_data"])
        requested_state = json.loads(json_data)
        next_state = {**current_state, "PowerState": requested_state["PowerState"]}
        return MockResponse({"accepted": True}, 200, new_state=next_state)

    mock_server.register_callback(SYSTEM_URI, "PATCH", set_power_state)

    response = mock_server.request(
        SYSTEM_URI,
        method="PATCH",
        json_data=json.dumps(submitted_payload),
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert response.state() == {**SYSTEM_PAYLOAD, "PowerState": "On"}
    assert observed["json_data"] == json.dumps(submitted_payload)
    assert observed["handler_view"] is mock_server.responses[(SYSTEM_URI, "GET")]

    replayed = mock_server.request(SYSTEM_URI, method="GET")
    assert json.loads(replayed.json()) == {**SYSTEM_PAYLOAD, "PowerState": "On"}
