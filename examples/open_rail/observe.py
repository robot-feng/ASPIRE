"""Capture read-only Open-RAIL AgiBot G1 RGB-D/state snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from PIL import Image

from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationRuntime


def save_snapshot(snapshot, output: Path) -> dict:
    """Preserve decoded RGB, unscaled depth, raw state and source timestamps."""
    names = ("head_rgb.png", "head_depth_u16.npy", "robot_state.npy", "snapshot.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError(f"snapshot output already exists in {output}")
    Image.fromarray(snapshot.rgb).save(output / "head_rgb.png")
    if snapshot.depth_u16 is not None:
        np.save(output / "head_depth_u16.npy", snapshot.depth_u16)
    np.save(output / "robot_state.npy", snapshot.state)
    metadata = {
        **snapshot.metadata,
        "rgb_timestamp_ns": snapshot.rgb_timestamp_ns,
        "depth_timestamp_ns": snapshot.depth_timestamp_ns,
        "depth_skew_ns": snapshot.depth_skew_ns,
        "state_timestamps_ns": snapshot.state_timestamps_ns,
        "is_synchronized": snapshot.is_synchronized,
        "depth_error": snapshot.depth_error,
    }
    (output / "snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


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
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=2.5, help="seconds between samples")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval must be finite and nonnegative")

    args.output.mkdir(parents=True, exist_ok=True)
    with OpenRailG1ObservationRuntime(
        open_rail_root=args.open_rail_root,
        maximum_rgb_depth_skew_ms=args.maximum_skew_ms,
    ) as runtime:
        last_timestamp = None
        for index in range(1, args.samples + 1):
            if index > 1:
                time.sleep(args.interval)
            snapshot = runtime.read_snapshot(timeout_s=args.timeout, require_depth=args.require_depth)
            if last_timestamp is not None and snapshot.rgb_timestamp_ns <= last_timestamp:
                raise RuntimeError("RGB snapshot timestamp did not advance")
            last_timestamp = snapshot.rgb_timestamp_ns
            output = args.output if args.samples == 1 else args.output / f"sample_{index:04d}"
            if args.samples > 1:
                output.mkdir(exist_ok=False)
            metadata = save_snapshot(snapshot, output)
            if args.samples == 1:
                print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
            else:
                print(json.dumps({"sample": str(output), "rgb_timestamp_ns": last_timestamp}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
