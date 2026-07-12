"""Offline boundary tests for ``VectorizedRestApiEnv`` batch alignment.

The vector env feeds replay and GPU batched encoders, so observation/reward flag
axes must stay aligned with ``num_envs``. These tests use tmp JSON fixtures and
a deterministic encoder; no network, GPU, or live Redfish host is involved.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.envs.rest_gym_env import HttpMethod
from igc.interfaces.rest_mapping_interface import RestMappingInterface


ROOT_URI = "/redfish/v1"
SYSTEM_URI = "/redfish/v1/Systems/1"


class StubEncoder:
    """Deterministic encoder with a stable vector observation shape."""

    emb_shape = (2, 3)

    def encode(self, observation: str) -> torch.Tensor:
        seed = sum(str(observation).encode("utf-8")) % 19
        return torch.arange(6, dtype=torch.float32).reshape(self.emb_shape) + seed

    def initialize(self) -> torch.Tensor:
        return torch.zeros(self.emb_shape, dtype=torch.float32)


class TinyRestMapping(RestMappingInterface):
    """Small RestMappingInterface double for vector env tests."""

    def __init__(self, mappings: dict[str, Path]) -> None:
        self._uris = list(mappings)
        self._mappings = mappings

    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        path = self._mappings.get(rest_api)
        return "" if path is None else str(path)

    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        return HttpMethod.GET.value if rest_api in self._mappings else ""

    def get_rest_api_mappings(self) -> Iterator[tuple[str, str]]:
        for rest_api, path in self._mappings.items():
            yield rest_api, str(path)

    def get_rest_api_methods(self) -> Iterator[tuple[str, str]]:
        for rest_api in self._uris:
            yield rest_api, HttpMethod.GET.value

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


@pytest.fixture
def vector_env(tmp_path: Path) -> VectorizedRestApiEnv:
    """Create a two-env vector REST environment backed by tmp JSON files."""
    root_path = tmp_path / "root.json"
    root_path.write_text(json.dumps({"@odata.id": ROOT_URI}), encoding="utf-8")
    system_path = tmp_path / "system.json"
    system_path.write_text(json.dumps({"@odata.id": SYSTEM_URI}), encoding="utf-8")

    mapping = TinyRestMapping({ROOT_URI: root_path, SYSTEM_URI: system_path})
    return VectorizedRestApiEnv(
        args=argparse.Namespace(raw_data_dir=str(tmp_path)),
        model=None,
        tokenizer=None,
        discovered_rest_api=mapping,
        max_episode=5,
        num_envs=2,
        encoder=StubEncoder(),
    )


def _action(env: VectorizedRestApiEnv, rest_api: str) -> torch.Tensor:
    """Build one vector-env action for ``rest_api`` and GET."""
    return VectorizedRestApiEnv.concat_rest_api_method(
        env._discovered_rest_api.action_for(rest_api),
        VectorizedRestApiEnv.encode_rest_api_method(HttpMethod.GET.value),
    )


def test_vector_reset_and_step_preserve_batch_axes(vector_env: VectorizedRestApiEnv):
    """A normal two-env step returns one observation/reward/flag row per env."""
    obs, info = vector_env.reset()
    actions = torch.stack([_action(vector_env, ROOT_URI), _action(vector_env, SYSTEM_URI)])

    next_obs, rewards, terminated, truncated, infos = vector_env.step(actions)

    assert info == {}
    assert obs.shape == (2, *StubEncoder.emb_shape)
    assert next_obs.shape == (2, *StubEncoder.emb_shape)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert len(infos) == 2


def test_vector_step_limit_sets_truncated_not_terminal(vector_env: VectorizedRestApiEnv):
    """Time-limit cuts keep bootstrap semantics: truncated true, terminal false."""
    vector_env.max_steps = 1
    vector_env.reset()
    actions = torch.stack([_action(vector_env, ROOT_URI), _action(vector_env, SYSTEM_URI)])

    next_obs, rewards, terminated, truncated, infos = vector_env.step(actions)

    assert next_obs.shape == (2, *StubEncoder.emb_shape)
    assert rewards.tolist() == [-1.0, -1.0]
    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [True, True]
    assert len(infos) == 2


def test_vector_http_500_sets_terminal_not_truncated(vector_env: VectorizedRestApiEnv):
    """Dead-end server errors are true terminals and should stop bootstrapping."""
    vector_env.reset()
    vector_env.mock_server().set_simulate_http_500_error(True)
    actions = torch.stack([_action(vector_env, ROOT_URI), _action(vector_env, SYSTEM_URI)])

    next_obs, rewards, terminated, truncated, infos = vector_env.step(actions)

    assert next_obs.shape == (2, *StubEncoder.emb_shape)
    assert rewards.tolist() == [-0.5, -0.5]
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert len(infos) == 2


@pytest.mark.xfail(
    reason="VectorizedRestApiEnv currently stacks only active responses after a prior done row.",
    strict=True,
)
def test_partial_done_step_preserves_num_envs_axis(vector_env: VectorizedRestApiEnv):
    """A skipped done sub-env must not shrink the observation batch axis."""
    obs, _ = vector_env.reset()
    vector_env.dones[0] = True
    vector_env.last_observation = obs
    actions = torch.stack([_action(vector_env, ROOT_URI), _action(vector_env, SYSTEM_URI)])

    next_obs, rewards, terminated, truncated, infos = vector_env.step(actions)

    assert next_obs.shape == (2, *StubEncoder.emb_shape)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert len(infos) == 2


# Author: Mus mbayramo@stanford.edu
