"""Archive raw G1 head RGB and fisheye packets for an A4 ChArUco session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from cap_harness.open_rail.g1_calibration_capture import save_calibration_sample
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationRuntime


_FISHEYE_CHANNELS = ("head_left_fisheye", "head_right_fisheye")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open-rail-root",
        type=Path,
        default=Path(os.environ.get("OPEN_RAIL_ROOT", "third_party/open-rail")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fisheye-channel",
        choices=_FISHEYE_CHANNELS,
        action="append",
        required=True,
        help="configured fisheye channel; may be repeated to capture both sides",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="seconds between samples; 0 prompts before each sample after the first",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not (args.interval >= 0.0 and args.interval < float("inf")):
        parser.error("--interval must be finite and non-negative")
    if len(set(args.fisheye_channel)) != len(args.fisheye_channel):
        parser.error("--fisheye-channel values must be unique")

    with OpenRailG1ObservationRuntime(
        open_rail_root=args.open_rail_root,
        calibration_camera_keys=tuple(args.fisheye_channel),
        allow_physical_motion=False,
    ) as runtime:
        for index in range(1, args.samples + 1):
            if index > 1:
                if args.interval > 0.0:
                    time.sleep(args.interval)
                else:
                    input(
                        f"Move the A4 ChArUco board to pose {index}, keep cameras fixed, "
                        "then press Enter."
                    )
            packets = runtime.capture_calibration_packets(timeout_s=args.timeout)
            sample_dir = save_calibration_sample(args.output, index, packets)
            print(json.dumps({"sample": str(sample_dir), "camera_keys": [
                packet["metadata"]["camera_key"] for packet in packets
            ]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
