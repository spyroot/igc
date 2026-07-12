"""Distributed RL environment rollout sanity for GB300.

Run this before a real multi-GPU RL training job. It builds a tiny offline
Redfish-like vector environment on each rank, steps it, stores transitions in
the replay buffer, moves rollout tensors to the rank device, and synchronizes
transition/shape counts across ranks.

The test is intentionally small: it proves the RL environment and replay path
are safe under ``torchrun`` without claiming policy learning or convergence.

Usage:
    torchrun --nproc_per_node=4 scripts/rl_env_sanity.py

Local CPU development only:
    python scripts/rl_env_sanity.py --allow-cpu --num-envs 2 --steps 2

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from igc.envs.rest_gym_base import HttpMethod
from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.interfaces.rest_mapping_interface import RestMappingInterface
from igc.modules.igc_experience_buffer import Buffer


ROOT_URI = "/redfish/v1"
SYSTEM_URI = "/redfish/v1/Systems/1"
CHASSIS_URI = "/redfish/v1/Chassis/1"


class SanityEncoder:
    """Deterministic encoder with a small fixed vector observation shape."""

    emb_shape = (2, 3)

    def encode(self, observation: object) -> torch.Tensor:
        """Encode a JSON-ish observation into a stable float32 tensor."""
        text = str(observation)
        seed = sum(text.encode("utf-8")) % 97
        return torch.arange(6, dtype=torch.float32).reshape(self.emb_shape) + seed

    def initialize(self) -> torch.Tensor:
        """Return the zero observation used for synthetic error states."""
        return torch.zeros(self.emb_shape, dtype=torch.float32)


class TinyRestMapping(RestMappingInterface):
    """Small in-memory REST mapping for the offline vector environment."""

    def __init__(self, mappings: dict[str, Path]) -> None:
        self._uris = list(mappings)
        self._mappings = mappings

    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        """Return the fixture path for ``rest_api`` or an empty miss."""
        path = self._mappings.get(rest_api)
        return "" if path is None else str(path)

    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        """Return the supported method for ``rest_api``."""
        return HttpMethod.GET.value if rest_api in self._mappings else ""

    def get_rest_api_mappings(self) -> Iterator[tuple[str, str]]:
        """Yield ``(uri, fixture_path)`` pairs for the mock server."""
        for rest_api, path in self._mappings.items():
            yield rest_api, str(path)

    def get_rest_api_methods(self) -> Iterator[tuple[str, str]]:
        """Yield ``(uri, method)`` pairs for action-space construction."""
        for rest_api in self._uris:
            yield rest_api, HttpMethod.GET.value

    @property
    def num_actions(self) -> int:
        """Number of REST URI actions."""
        return len(self._uris)

    def entry_rest_api(self) -> tuple[str, torch.Tensor]:
        """Return the reset entry URI and its one-hot action."""
        return ROOT_URI, self.action_for(ROOT_URI)

    def one_hot_vector_to_action(self, one_hot: torch.Tensor) -> str:
        """Decode one URI one-hot vector into its REST URI."""
        return self._uris[int(torch.argmax(one_hot).item())]

    def action_for(self, rest_api: str) -> torch.Tensor:
        """Encode ``rest_api`` as a one-hot URI vector."""
        action = torch.zeros(len(self._uris), dtype=torch.float32)
        action[self._uris.index(rest_api)] = 1.0
        return action


@dataclass(frozen=True)
class RolloutStats:
    """Small summary emitted by one rank's RL/env sanity rollout."""

    rank: int
    world: int
    num_envs: int
    steps: int
    transitions: int
    replay_len: int
    terminals: int
    truncated: int
    reward_sum: float
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    sample_state_shape: tuple[int, ...]
    sample_done_shape: tuple[int, ...]


def write_tiny_fixtures(root: Path, rank: int) -> TinyRestMapping:
    """Create tiny Redfish-like JSON fixtures and return their URI mapping."""
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        ROOT_URI: {"@odata.id": ROOT_URI, "rank": rank, "Systems": {"@odata.id": SYSTEM_URI}},
        SYSTEM_URI: {"@odata.id": SYSTEM_URI, "rank": rank, "PowerState": "On"},
        CHASSIS_URI: {"@odata.id": CHASSIS_URI, "rank": rank, "Thermal": "Nominal"},
    }
    paths: dict[str, Path] = {}
    for index, (uri, payload) in enumerate(payloads.items()):
        path = root / f"fixture_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[uri] = path
    return TinyRestMapping(paths)


def action_for(env: VectorizedRestApiEnv, rest_api: str) -> torch.Tensor:
    """Build one vector-env action for ``rest_api`` using GET."""
    return VectorizedRestApiEnv.concat_rest_api_method(
        env._discovered_rest_api.action_for(rest_api),
        VectorizedRestApiEnv.encode_rest_api_method(HttpMethod.GET.value),
    )


def rollout_actions(env: VectorizedRestApiEnv, step: int) -> torch.Tensor:
    """Return a deterministic action batch with one row per sub-environment."""
    uris = (ROOT_URI, SYSTEM_URI, CHASSIS_URI)
    actions = [
        action_for(env, uris[(step + env_index) % len(uris)])
        for env_index in range(env.num_envs)
    ]
    return torch.stack(actions, dim=0)


def build_env(base_dir: Path, rank: int, num_envs: int, max_episode: int) -> VectorizedRestApiEnv:
    """Build the tiny vector REST environment for one rank."""
    mapping = write_tiny_fixtures(base_dir, rank)
    return VectorizedRestApiEnv(
        args=argparse.Namespace(raw_data_dir=str(base_dir)),
        model=None,
        tokenizer=None,
        discovered_rest_api=mapping,
        max_episode=max_episode,
        num_envs=num_envs,
        encoder=SanityEncoder(),
    )


def run_rollout(
    *,
    num_envs: int,
    steps: int,
    device: torch.device,
    rank: int,
    world: int,
    base_dir: Path,
) -> RolloutStats:
    """Run one local vector-env rollout and return shape/count stats."""
    env = build_env(base_dir, rank, num_envs, max_episode=steps + 10)
    replay = Buffer(size=max(steps + 1, 4), sample_size=steps + 1)
    obs, _ = env.reset()

    transitions = terminals = truncated = 0
    reward_sum = 0.0
    action_shape: tuple[int, ...] | None = None

    for step in range(steps):
        actions = rollout_actions(env, step)
        action_shape = tuple(actions.shape)
        if step == steps - 1 and num_envs > 0:
            env.simulate_goal_reached(rank % num_envs)

        next_obs, rewards, terminated, time_limited, infos = env.step(actions)
        if next_obs.shape[0] != num_envs:
            raise RuntimeError(
                f"rank {rank}: observation batch shrank from {num_envs} to {next_obs.shape[0]}"
            )
        if rewards.shape != (num_envs,):
            raise RuntimeError(f"rank {rank}: reward shape mismatch {tuple(rewards.shape)}")
        if len(infos) != num_envs:
            raise RuntimeError(f"rank {rank}: info rows mismatch {len(infos)}")

        replay.add(obs, actions, rewards, next_obs, terminated)

        # Exercise the same device upload boundary that a GPU RL learner will hit.
        uploaded = (
            next_obs.to(device, non_blocking=True),
            actions.to(device, non_blocking=True),
            rewards.to(device, non_blocking=True),
            terminated.to(device, non_blocking=True),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del uploaded

        transitions += num_envs
        terminals += int(terminated.sum().item())
        truncated += int(time_limited.sum().item())
        reward_sum += float(rewards.sum().item())
        obs = next_obs

    state_batch, _, _, _, done_batch = replay.sample()
    return RolloutStats(
        rank=rank,
        world=world,
        num_envs=num_envs,
        steps=steps,
        transitions=transitions,
        replay_len=len(replay._buffer),
        terminals=terminals,
        truncated=truncated,
        reward_sum=reward_sum,
        obs_shape=tuple(obs.shape),
        action_shape=action_shape or (0,),
        sample_state_shape=tuple(state_batch.shape),
        sample_done_shape=tuple(done_batch.shape),
    )


def stats_tensor(stats: RolloutStats, device: torch.device) -> torch.Tensor:
    """Pack additive rollout stats for distributed reduction."""
    return torch.tensor(
        [
            float(stats.transitions),
            float(stats.replay_len),
            float(stats.terminals),
            float(stats.truncated),
            float(stats.reward_sum),
        ],
        dtype=torch.float64,
        device=device,
    )


def shape_tensor(stats: RolloutStats, device: torch.device) -> torch.Tensor:
    """Pack shape stats that must match across ranks."""
    obs_tail = stats.obs_shape[1:] if len(stats.obs_shape) > 1 else stats.obs_shape
    state_tail = stats.sample_state_shape[-2:] if len(stats.sample_state_shape) >= 2 else stats.sample_state_shape
    return torch.tensor(
        [
            stats.num_envs,
            *obs_tail,
            stats.action_shape[-1],
            *state_tail,
            stats.sample_done_shape[-1],
        ],
        dtype=torch.long,
        device=device,
    )


def resolve_runtime(allow_cpu: bool) -> tuple[int, int, int, torch.device, bool]:
    """Resolve distributed rank metadata and the rank device."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ or world > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise SystemExit("[rl-env] FAIL: CUDA not available; pass --allow-cpu only for local development")

    if distributed:
        import torch.distributed as dist

        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world, local_rank, device, distributed


def main(argv: list[str] | None = None) -> int:
    """Run the distributed RL/env sanity check."""
    parser = argparse.ArgumentParser(description="Distributed RL env rollout sanity")
    parser.add_argument("--num-envs", type=int, default=4, help="vector envs per rank")
    parser.add_argument("--steps", type=int, default=4, help="rollout steps per rank")
    parser.add_argument("--allow-cpu", action="store_true", help="permit CPU-only local development")
    args = parser.parse_args(argv)

    rank, world, local_rank, device, distributed = resolve_runtime(args.allow_cpu)
    if args.num_envs < 1 or args.steps < 1:
        raise SystemExit("[rl-env] FAIL: --num-envs and --steps must be positive")

    try:
        with tempfile.TemporaryDirectory(prefix=f"igc-rl-env-r{rank}-") as tmp:
            stats = run_rollout(
                num_envs=args.num_envs,
                steps=args.steps,
                device=device,
                rank=rank,
                world=world,
                base_dir=Path(tmp),
            )

        print(
            f"[rl-env] rank={rank}/{world} local={local_rank} device={device} "
            f"obs={stats.obs_shape} action={stats.action_shape} "
            f"transitions={stats.transitions} terminals={stats.terminals} "
            f"replay={stats.replay_len} upload=OK",
            flush=True,
        )

        if distributed:
            import torch.distributed as dist

            totals = stats_tensor(stats, device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

            shapes_min = shape_tensor(stats, device)
            shapes_max = shapes_min.clone()
            dist.all_reduce(shapes_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(shapes_max, op=dist.ReduceOp.MAX)
            if not torch.equal(shapes_min, shapes_max):
                raise SystemExit(
                    f"[rl-env] FAIL: shape mismatch across ranks min={shapes_min.tolist()} "
                    f"max={shapes_max.tolist()}"
                )
            dist.barrier()
            if rank == 0:
                print(
                    f"[rl-env] PASS world={world}: transitions={int(totals[0].item())} "
                    f"replay_items={int(totals[1].item())} terminals={int(totals[2].item())} "
                    f"truncated={int(totals[3].item())} reward_sum={totals[4].item():.2f} "
                    f"shapes=OK",
                    flush=True,
                )
            dist.destroy_process_group()
        else:
            print(
                f"[rl-env] PASS world=1: transitions={stats.transitions} "
                f"terminals={stats.terminals} truncated={stats.truncated} shapes=OK",
                flush=True,
            )
    except Exception as exc:
        if distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        raise SystemExit(f"[rl-env] FAIL rank={rank}: {exc}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
