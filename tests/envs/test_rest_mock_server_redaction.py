"""Security regression: the mock REST server never logs live Redfish credentials.

Verifies MockServer._redact_headers masks Authorization / X-Auth-Token (the headers
that carry live credentials) while leaving non-sensitive headers intact. Required
before any live canary.

Author:
Mus mbayramo@stanford.edu
"""
from igc.envs.rest_mock_server import MockServer


def test_redact_headers_masks_credentials():
    """Authorization and X-Auth-Token values are masked; the secrets never appear."""
    redacted = MockServer._redact_headers(
        {
            "Authorization": "root:supersecret",
            "X-Auth-Token": "tok-abc123",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    assert redacted["Authorization"] == "<redacted>"
    assert redacted["X-Auth-Token"] == "<redacted>"
    assert redacted["Accept"] == "application/json"
    assert "supersecret" not in str(redacted)
    assert "tok-abc123" not in str(redacted)


def test_redact_headers_empty_and_nonsensitive_only():
    """Empty / non-sensitive header maps pass through unchanged."""
    assert MockServer._redact_headers({}) == {}
    assert MockServer._redact_headers({"Accept": "x"}) == {"Accept": "x"}


# Author: Mus mbayramo@stanford.edu
