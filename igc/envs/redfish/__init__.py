"""Redfish environment backend adapters.

This package holds the narrow adapter boundary between IGC's RL environments
and the provider-owned ``redfish_ctl`` simulator/corpus contracts. The adapters
are offline-safe by default and do not import provider test helpers.

Author:
Mus mbayramo@stanford.edu
"""

from igc.envs.redfish.backend_types import (
    RedfishBackendCapabilities,
    RedfishBackendError,
    RedfishBackendResponse,
    RedfishBackendStatus,
    RedfishContractError,
    RedfishProviderUnavailable,
)
from igc.envs.redfish.backends import RedfishEnvironmentBackend, make_backend

__all__ = [
    "RedfishBackendCapabilities",
    "RedfishBackendError",
    "RedfishBackendResponse",
    "RedfishBackendStatus",
    "RedfishContractError",
    "RedfishEnvironmentBackend",
    "RedfishProviderUnavailable",
    "make_backend",
]


# Author: Mus mbayramo@stanford.edu
