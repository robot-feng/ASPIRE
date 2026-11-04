"""Capture one read-only Open-RAIL AgiBot G1 RGB-D/state snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image

from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open-rail-root",
        type=Path,
        default=Path(os.environ.get("OPEN_RAIL_ROOT", "third_party/open-rail")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--maximum-skew-ms", type=float, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with OpenRailG1ObservationRuntime(
        open_rail_root=args.open_rail_root,
        maximum_rgb_depth_skew_ms=args.maximum_skew_ms,
    ) as runtime:
        snapshot = runtime.read_snapshot(timeout_s=args.timeout, require_depth=args.require_depth)

    Image.fromarray(snapshot.rgb).save(args.output / "head_rgb.png")
    if snapshot.depth_u16 is not None:
        import numpy as np

        np.save(args.output / "head_depth_u16.npy", snapshot.depth_u16)
    import numpy as np

    np.save(args.output / "robot_state.npy", snapshot.state)
    metadata = {
        **snapshot.metadata,
        "rgb_timestamp_ns": snapshot.rgb_timestamp_ns,
        "depth_timestamp_ns": snapshot.depth_timestamp_ns,
        "depth_skew_ns": snapshot.depth_skew_ns,
        "state_timestamps_ns": snapshot.state_timestamps_ns,
        "is_synchronized": snapshot.is_synchronized,
        "depth_error": snapshot.depth_error,
    }
    (args.output / "snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
