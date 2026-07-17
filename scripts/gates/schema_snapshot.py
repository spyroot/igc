"""Gate: freeze the approved Phase 2/3 REST-goal JSON shape.

The snapshot stores field names and value types only. Values are fixture data and
are intentionally erased so the gate catches schema drift without committing
private captures or brittle examples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from igc.ds.rest_goal_contract import (  # noqa: E402
    RedfishContext,
    build_d1_rest_api_list_row,
    build_ordered_call_row,
    inference_target_calls_json,
    render_ordered_call_example,
    render_rest_api_list_example,
)

DEFAULT_SNAPSHOT = REPO_ROOT / "schemas/snapshots/rest_goal_contract.shape.json"
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BOOTSTRAP = 2


def _shape(value: Any) -> Any:
    """Return a stable shape-only representation for JSON-like values."""

    if isinstance(value, Mapping):
        return {key: _shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_merge_shapes(_shape(item) for item in value)] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def _merge_shapes(shapes: Any) -> Any:
    """Merge list item shapes into one representative shape."""

    shapes = list(shapes)
    if not shapes:
        return {}
    if all(isinstance(shape, dict) for shape in shapes):
        merged: dict[str, Any] = {}
        for shape in shapes:
            for key, value in shape.items():
                if key in merged:
                    merged[key] = _merge_shapes([merged[key], value])
                else:
                    merged[key] = value
        return merged
    if all(isinstance(shape, list) for shape in shapes):
        inner = [item for shape in shapes for item in shape]
        return [_merge_shapes(inner)] if inner else []
    first = shapes[0]
    return first if all(shape == first for shape in shapes) else sorted(
        {str(shape) for shape in shapes}
    )


def build_snapshot() -> dict[str, Any]:
    """Build the canonical Phase 2/3 schema snapshot."""

    read_ctx = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=("GET", "HEAD"),
        json={"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"},
    )
    write_ctx = RedfishContext(
        rest_api="/redfish/v1/Managers/1/EthernetInterfaces/1",
        allowed_methods=("GET", "PATCH"),
        json={"@odata.id": "/redfish/v1/Managers/1/EthernetInterfaces/1"},
    )
    phase2 = build_d1_rest_api_list_row(
        text="read system then set manager address",
        contexts=(read_ctx, write_ctx),
        rest_api_list=(read_ctx.rest_api, write_ctx.rest_api),
    )
    phase3 = build_ordered_call_row(
        text="read system then set manager address",
        contexts=(read_ctx, write_ctx),
        rest_api_list=(read_ctx.rest_api, write_ctx.rest_api),
        method_by_api={write_ctx.rest_api: "PATCH"},
        arguments_by_api={write_ctx.rest_api: {"Address": "192.168.1.1"}},
    )
    return {
        "schema": "igc.rest_goal_contract.shape.v1",
        "phase2_row": _shape(phase2),
        "phase2_rendered": _shape(render_rest_api_list_example(phase2).__dict__),
        "phase3_row": _shape(phase3),
        "phase3_rendered": _shape(render_ordered_call_example(phase3).__dict__),
        "phase3_inference": _shape(inference_target_calls_json(phase3)),
    }


def check(snapshot: Path = DEFAULT_SNAPSHOT) -> int:
    """Compare the current contract shape with the committed snapshot."""

    current = build_snapshot()
    if not snapshot.exists():
        return EXIT_BOOTSTRAP
    expected = json.loads(snapshot.read_text(encoding="utf-8"))
    return EXIT_OK if current == expected else EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    current = build_snapshot()
    if args.update:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"SCHEMA_SNAPSHOT_UPDATED {snapshot}")
        return EXIT_OK

    result = check(snapshot)
    if result == EXIT_OK:
        print("SCHEMA_SNAPSHOT_PASS")
    elif result == EXIT_BOOTSTRAP:
        print(f"SCHEMA_SNAPSHOT_BOOTSTRAP missing={snapshot}")
    else:
        print(f"SCHEMA_SNAPSHOT_FAIL drift={snapshot}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
