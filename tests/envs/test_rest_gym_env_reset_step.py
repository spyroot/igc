"""Offline RestApiEnv reset/step tests backed by local stubs.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import igc.envs.rest_gym_base as rest_gym_base
import igc.envs.rest_gym_env as rest_gym_env
import torch
from igc.envs.rest_gym_base import RestApiBaseEnv
from igc.envs.rest_gym_env import GoalTypeState, RestApiEnv
from igc.envs.rest_mock_server import MockServer
from igc.interfaces.rest_mapping_interface import RestMappingInterface


ROOT_URI = "/redfish/v1"
SYSTEM_URI = "/redfish/v1/Systems/1"

ROOT_PAYLOAD = {
    "@odata.id": ROOT_URI,
    "Systems": {"@odata.id": "/redfish/v1/Systems"},
}
SYSTEM_PAYLOAD = {
    "@odata.id": SYSTEM_URI,
    "Name": "Tiny offline system",
    "PowerState": "Off",
}


class StubEncoder:
    """Deterministic encoder with the shape RestApiEnv expects."""

    emb_shape = (2, 3)

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.calls: list[str] = []

    def encode(self, observation: str) -> torch.Tensor:
        self.calls.append(observation)
        seed = sum(observation.encode("utf-8")) % 97
        return torch.arange(6, dtype=torch.float32).reshape(self.emb_shape) + seed


class TinyRestMapping(RestMappingInterface):
    """Small RestMappingInterface double with action lookup helpers."""

    def __init__(self, mappings: dict[str, Path]) -> None:
        self._uris = list(mappings)
        self._mappings = mappings

    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        path = self._mappings.get(rest_api)
        return "" if path is None else str(path)

    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        return "GET" if rest_api in self._mappings else ""

    def get_rest_api_mappings(self) -> Iterator[tuple[str, str]]:
        for rest_api, path in self._mappings.items():
            yield rest_api, str(path)

    def get_rest_api_methods(self) -> Iterator[tuple[str, str]]:
        for rest_api in self._uris:
            yield rest_api, "GET"

    @property
    def num_actions(self) -> int:
        return len(self._uris)

    def entry_rest_api(self) -> tuple[str, torch.Tensor]:
        return ROOT_URI, self.action_for(ROOT_URI)

    def one_hot_vector_to_action(self, one_hot: torch.Tensor) -> str:
        return self._uris[int(torch.argmax(one_hot).item())]

    def action_for(self, rest_api: str) -> torch.Tensor:
        action = torch.zeros(len(self._uris), dtype=torch.float32)
        action[self._uris.index(rest_api)] = 1.0
        return action

    def sample_rest_api(self) -> tuple[str, str, torch.Tensor]:
        return SYSTEM_URI, "GET", self.action_for(SYSTEM_URI)


@pytest.fixture
def rest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RestApiEnv:
    """Create a RestApiEnv that only uses tmp JSON files and a StubEncoder."""
    root_path = tmp_path / "root.json"
    root_path.write_text(json.dumps(ROOT_PAYLOAD), encoding="utf-8")
    system_path = tmp_path / "system.json"
    system_path.write_text(json.dumps(SYSTEM_PAYLOAD), encoding="utf-8")

    mapping = TinyRestMapping({ROOT_URI: root_path, SYSTEM_URI: system_path})
    monkeypatch.setattr(rest_gym_env, "RestBaseEncoder", StubEncoder)

    args = argparse.Namespace(raw_data_dir=str(tmp_path))
    return RestApiEnv(
        args=args,
        model=None,
        tokenizer=None,
        discovered_rest_api=mapping,
        max_episode=5,
    )


def _action(mapping: TinyRestMapping, rest_api: str, method: str) -> torch.Tensor:
    return RestApiEnv.concat_rest_api_method(
        mapping.action_for(rest_api),
        RestApiEnv.encode_rest_api_method(method),
    )


def test_reset_returns_entry_observation_and_clears_step_count(rest_env: RestApiEnv):
    """reset encodes the entry REST resource and clears episode counters."""
    rest_env.step_count = 3

    observation, info = rest_env.reset()

    root_json = json.dumps(ROOT_PAYLOAD)
    assert rest_env.step_count == 0
    assert info == {"goal": None}
    assert rest_env.action_space.shape == (2 + len(RestApiEnv.METHOD_MAPPING),)
    assert rest_env.observation_space.shape == StubEncoder.emb_shape
    assert torch.equal(observation, rest_env.encoder.encode(root_json))
    assert torch.equal(rest_env.last_observation, observation)
    assert root_json in rest_env.encoder.calls


def test_step_get_known_resource_returns_success_observation(rest_env: RestApiEnv):
    """A valid GET action replays the stored response with a small reward."""
    _, info = rest_env.reset()
    action = _action(rest_env._discovered_rest_api, SYSTEM_URI, "GET")

    observation, reward, terminated, truncated, step_info = rest_env.step(action)

    assert info == {"goal": None}
    assert reward == pytest.approx(0.1)
    assert terminated is False
    assert truncated is False
    assert step_info == {}
    assert rest_env.step_count == 1
    assert torch.equal(observation, rest_env.encoder.encode(json.dumps(SYSTEM_PAYLOAD)))
    assert torch.equal(rest_env.last_observation, observation)


def test_sample_observation_uses_sampled_http_method(rest_env: RestApiEnv):
    """sample_observation passes the sampled method, not the one-hot tensor."""
    observation = rest_env.sample_observation()

    system_json = json.dumps(SYSTEM_PAYLOAD)
    assert torch.equal(observation, rest_env.encoder.encode(system_json))
    assert system_json in rest_env.encoder.calls


def test_base_sample_observation_uses_sampled_http_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The base env keeps the sampled method separate from the action vector."""
    system_path = tmp_path / "system.json"
    system_path.write_text(json.dumps(SYSTEM_PAYLOAD), encoding="utf-8")
    mapping = TinyRestMapping({SYSTEM_URI: system_path})
    monkeypatch.setattr(rest_gym_base, "RestBaseEncoder", StubEncoder)

    env = RestApiBaseEnv(
        args=argparse.Namespace(raw_data_dir=str(tmp_path)),
        model=None,
        tokenizer=None,
        discovered_rest_api=mapping,
        max_episode=5,
    )

    observation = env.sample_observation()

    system_json = json.dumps(SYSTEM_PAYLOAD)
    assert torch.equal(observation, env.encoder.encode(system_json))
    assert system_json in env.encoder.calls


def test_step_http_500_returns_terminal_error_observation(rest_env: RestApiEnv):
    """Mock HTTP 500 responses are terminal and use the synthesized error body."""
    rest_env.reset()
    rest_env.mock_server().set_simulate_http_500_error(True)
    action = _action(rest_env._discovered_rest_api, SYSTEM_URI, "GET")

    observation, reward, terminated, truncated, info = rest_env.step(action)

    error_json = MockServer.generate_error_response()
    assert reward == pytest.approx(-0.5)
    assert terminated is True  # dead-end error is a real terminal
    assert truncated is False
    assert info == {}
    assert torch.equal(observation, rest_env.encoder.encode(error_json))
    assert torch.equal(rest_env.last_observation, observation)


def test_reset_rejects_fixed_state_goal_with_wrong_shape(rest_env: RestApiEnv):
    """FixedState goals must match the encoder-backed observation shape."""
    with pytest.raises(ValueError, match="Fixed state goal dimensions"):
        rest_env.reset(goal=torch.zeros(1), goal_type=GoalTypeState.FixedState)


# Author: Mus mbayramo@stanford.edu


def test_step_limit_truncates_without_terminating(rest_env: RestApiEnv):
    """Hitting max_steps truncates (time-limit) and never marks terminal."""
    rest_env.reset()
    action = _action(rest_env._discovered_rest_api, SYSTEM_URI, "GET")
    rest_env.step_count = rest_env.max_steps - 1

    _, reward, terminated, truncated, _ = rest_env.step(action)

    assert truncated is True
    assert terminated is False
    assert reward == pytest.approx(-1.0)
