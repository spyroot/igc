"""Shared types for Redfish simulator backend adapters.

The provider-facing contract is intentionally small: adapters normalize
in-process and HTTP simulator responses into the same dataclasses before a Gym
environment or replay buffer sees them.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


SUPPORTED_SIMULATOR_CONTRACT = "redfish-simulator/v1"


class RedfishBackendError(RuntimeError):
    """Base class for Redfish backend adapter failures."""


class RedfishContractError(RedfishBackendError):
    """Raised when a provider contract or manifest version is unsupported."""


class RedfishProviderUnavailable(RedfishBackendError):
    """Raised when the requested provider backend cannot be imported or reached."""


@dataclass(frozen=True)
class RedfishBackendResponse:
    """Normalized simulator response returned by every backend.

    :param status_code: HTTP-like status code returned by the simulator.
    :param json_data: decoded JSON body, or ``None`` for empty responses.
    :param headers: response headers normalized to a plain dict.
    :param error: whether this response represents a provider or protocol error.
    :param provider_error_code: provider-specific error code when available.
    :param mutation_metadata: provider-declared mutation details, not policy input.
    :param task_metadata: provider-declared task/job details, not policy input.
    :param provenance: backend and corpus provenance suitable for Gym ``info``.
    """
    status_code: int
    json_data: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    error: bool = False
    provider_error_code: str | None = None
    mutation_metadata: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Return the decoded JSON body.

        :return: decoded response body.
        """
        return self.json_data


@dataclass(frozen=True)
class RedfishBackendStatus:
    """Normalized simulator status or reset result.

    :param backend_kind: IGC adapter kind, e.g. ``redfish_ctl_http``.
    :param ready: whether the provider reports itself ready.
    :param corpus_id: materialized corpus id when known.
    :param contract_version: simulator contract version when known.
    :param provider_revision: provider code/data revision when known.
    :param seed: reset seed for the current episode, when applicable.
    :param episode_id: provider episode/session id, when applicable.
    :param provenance: additional provider metadata.
    """
    backend_kind: str
    ready: bool = True
    corpus_id: str | None = None
    contract_version: str = ""
    provider_revision: str = ""
    seed: int | None = None
    episode_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RedfishBackendCapabilities:
    """Provider-declared simulator capabilities.

    :param contract_version: simulator contract version.
    :param provider_revision: provider code/data revision.
    :param corpus_id: materialized corpus id when known.
    :param action_capabilities: provider-declared simulatable actions.
    :param raw: raw provider document for diagnostics and future fields.
    """
    contract_version: str
    provider_revision: str = ""
    corpus_id: str | None = None
    action_capabilities: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def ensure_supported_contract(version: str) -> None:
    """Validate the simulator contract major version.

    :param version: provider contract string such as ``redfish-simulator/v1``.
    :raises RedfishContractError: when the version is missing or unsupported.
    """
    match = re.match(r"^redfish-simulator/v(?P<major>\d+)(?:$|[.\-])", version or "")
    if not match:
        raise RedfishContractError(f"unsupported Redfish simulator contract: {version!r}")
    if match.group("major") != "1":
        raise RedfishContractError(
            f"unsupported Redfish simulator contract: {version!r}; "
            f"expected {SUPPORTED_SIMULATOR_CONTRACT}"
        )


def normalize_response(raw: Any, backend_kind: str) -> RedfishBackendResponse:
    """Normalize provider response shapes into :class:`RedfishBackendResponse`.

    :param raw: provider dict, requests-like object, or already-normalized response.
    :param backend_kind: IGC adapter kind stamped into provenance.
    :return: normalized backend response.
    """
    if isinstance(raw, RedfishBackendResponse):
        provenance = dict(raw.provenance)
        provenance.setdefault("backend_kind", backend_kind)
        return RedfishBackendResponse(
            status_code=raw.status_code,
            json_data=raw.json_data,
            headers=dict(raw.headers),
            error=raw.error,
            provider_error_code=raw.provider_error_code,
            mutation_metadata=dict(raw.mutation_metadata),
            task_metadata=dict(raw.task_metadata),
            provenance=provenance,
        )

    if isinstance(raw, Mapping):
        headers = raw.get("headers") or {}
        json_data = raw.get("json", raw.get("json_data", raw.get("body")))
        if json_data is None and "status_code" not in raw:
            json_data = dict(raw)
        status_code = int(raw.get("status_code", raw.get("status", 200)))
        provenance = dict(raw.get("provenance") or {})
        provenance.setdefault("backend_kind", backend_kind)
        return RedfishBackendResponse(
            status_code=status_code,
            json_data=json_data,
            headers={str(k): str(v) for k, v in dict(headers).items()},
            error=bool(raw.get("error", status_code >= 400)),
            provider_error_code=raw.get("provider_error_code"),
            mutation_metadata=dict(raw.get("mutation") or raw.get("mutation_metadata") or {}),
            task_metadata=dict(raw.get("task") or raw.get("task_metadata") or {}),
            provenance=provenance,
        )

    status_code = int(getattr(raw, "status_code", 200))
    try:
        json_data = raw.json()
    except (AttributeError, ValueError):
        json_data = getattr(raw, "json_data", None)
    provenance = {"backend_kind": backend_kind}
    return RedfishBackendResponse(
        status_code=status_code,
        json_data=json_data,
        headers=dict(getattr(raw, "headers", {}) or {}),
        error=bool(getattr(raw, "error", status_code >= 400)),
        provenance=provenance,
    )


def normalize_status(raw: Any, backend_kind: str) -> RedfishBackendStatus:
    """Normalize provider status/reset documents.

    :param raw: provider status dict or already-normalized status.
    :param backend_kind: IGC adapter kind stamped into the result.
    :return: normalized backend status.
    """
    if isinstance(raw, RedfishBackendStatus):
        return RedfishBackendStatus(
            backend_kind=backend_kind,
            ready=raw.ready,
            corpus_id=raw.corpus_id,
            contract_version=raw.contract_version,
            provider_revision=raw.provider_revision,
            seed=raw.seed,
            episode_id=raw.episode_id,
            provenance=dict(raw.provenance),
        )
    data = dict(raw or {})
    version = str(data.get("contract_version") or data.get("simulator_contract") or "")
    if version:
        ensure_supported_contract(version)
    provenance = dict(data.get("provenance") or {})
    provenance.setdefault("backend_kind", backend_kind)
    return RedfishBackendStatus(
        backend_kind=backend_kind,
        ready=bool(data.get("ready", True)),
        corpus_id=data.get("corpus_id"),
        contract_version=version,
        provider_revision=str(data.get("provider_revision", "")),
        seed=data.get("seed"),
        episode_id=data.get("episode_id"),
        provenance=provenance,
    )


def normalize_capabilities(raw: Any) -> RedfishBackendCapabilities:
    """Normalize and validate provider capabilities.

    :param raw: provider capabilities dict or already-normalized capabilities.
    :return: normalized capabilities.
    """
    if isinstance(raw, RedfishBackendCapabilities):
        ensure_supported_contract(raw.contract_version)
        return raw
    data = dict(raw or {})
    version = str(data.get("contract_version") or data.get("simulator_contract") or "")
    ensure_supported_contract(version)
    actions = data.get("actions", data.get("action_capabilities", []))
    return RedfishBackendCapabilities(
        contract_version=version,
        provider_revision=str(data.get("provider_revision", "")),
        corpus_id=data.get("corpus_id"),
        action_capabilities=[dict(item) for item in actions],
        raw=data,
    )


# Author: Mus mbayramo@stanford.edu
