"""Offline regression for goal detection in ``RestApiEnv.is_goal_reached``.

``reset`` stores the ``GoalTypeState`` enum, but ``is_goal_reached`` compared
it against ``GoalTypeState.<member>.value`` — an enum never equals its value,
so every branch was False and the +1 goal reward / terminal was unreachable
for all three goal types in the single-env path. Uses the same tmp-JSON +
StubEncoder harness as the reset/step tests; CPU-only, no network.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import igc.envs.rest_gym_env as rest_gym_env
from igc.envs.rest_gym_env import GoalTypeState, RestApiEnv
from igc.interfaces.rest_mapping_interface import RestMappingInterface

ROOT_URI = "/redfish/v1"
ROOT_PAYLOAD = {"@odata.id": ROOT_URI, "Name": "root"}


class TinyMapping(RestMappingInterface):
    """One-resource RestMappingInterface double."""

    def __init__(self, mapping: dict[str, Path]) -> None:
        self._uris = list(mapping)
        self._mapping = mapping

    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        path = self._mapping.get(rest_api)
        return "" if path is None else str(path)

    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        return "GET" if rest_api in self._mapping else ""

    def get_rest_api_mappings(self) -> Iterator[tuple[str, str]]:
        for rest_api, path in self._mapping.items():
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


@pytest.fixture
def rest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_encoder_cls) -> RestApiEnv:
    """RestApiEnv over one tmp JSON resource with the shared StubEncoder."""
    root_path = tmp_path / "root.json"
    root_path.write_text(json.dumps(ROOT_PAYLOAD), encoding="utf-8")
    monkeypatch.setattr(rest_gym_env, "RestBaseEncoder", stub_encoder_cls)
    return RestApiEnv(
        args=argparse.Namespace(raw_data_dir=str(tmp_path)),
        model=None,
        tokenizer=None,
        discovered_rest_api=TinyMapping({ROOT_URI: root_path}),
        max_episode=5,
    )


def test_action_goal_is_reached_for_matching_action(rest_env: RestApiEnv):
    """An Action goal fires when the taken action equals the goal action."""
    goal = torch.zeros(1 + len(RestApiEnv.METHOD_MAPPING), dtype=torch.float32)
    goal[0] = 1.0
    rest_env.reset(goal=goal, goal_type=GoalTypeState.Action)

    assert rest_env.is_goal_reached(goal.clone()) is True


def test_action_goal_not_reached_for_different_action(rest_env: RestApiEnv):
    """A non-matching action does not fire the Action goal."""
    goal = torch.zeros(1 + len(RestApiEnv.METHOD_MAPPING), dtype=torch.float32)
    goal[0] = 1.0
    rest_env.reset(goal=goal, goal_type=GoalTypeState.Action)

    other = torch.zeros_like(goal)
    other[-1] = 1.0
    assert rest_env.is_goal_reached(other) is False


# Author: Mus mbayramo@stanford.edu
