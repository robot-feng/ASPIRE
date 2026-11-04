# Open-RAIL G1 raw camera calibration capture

The read-only capture entry point stores the original packet payload and
metadata from the existing Open-RAIL camera receiver. It does not decode
images, move the robot, check ChArUco visibility, or solve camera calibration.
See [the A4 ChArUco board notes](open-rail-calibration.md) for print scale and
board dimensions.

The profile explicitly supports `head_left_fisheye` and `head_right_fisheye`,
mapped to the SDK channels of the same names. The head RGB channel is included
automatically. In the inspected `develop/fisheye_camera_conf.json`, the head
color, left fisheye, and right fisheye publishers are enabled. The head center
fisheye is not enabled by that profile/configuration and is not offered by this
entry point.

Example, with manual repositioning between samples:

```sh
PYTHONPATH=src python examples/open_rail/capture_charuco.py \
  --open-rail-root /data/open-rail \
  --output artifacts/open_rail/charuco_session \
  --fisheye-channel head_left_fisheye \
  --samples 20
```

For paced automatic captures, add `--interval 2.5`; move the board through a
new pose during each interval. The default interval is zero, which prompts
before each sample after the first. To capture both side cameras, repeat
`--fisheye-channel head_right_fisheye`.

Each `sample_NNNN` directory contains one raw `.packet` file per selected
camera and `sample.json`. The manifest records logical camera key, SDK channel,
resolved DDS topic, packet type, raw encoding and color format identifiers,
dimensions, reference and packet timestamps, timestamp delta, payload length,
and SHA-256 digest. Raw bytes remain unchanged. A false
`board_detection_performed` field makes clear that image visibility was not
checked by the capture path.

The example starts one read-only Open-RAIL proxy, whose G1 body creates one
Cosine camera receiver subscribed to head RGB, depth, and only the selected
fisheye channel(s). Fisheye packets are fetched only by the explicit capture
method; they are not appended to normal RGB-D observations. Do not run this
example alongside another owner of the G1 camera service. If a suitable
read-only runtime is already open with the selected channels enabled, call its
`capture_calibration_packets()` method and pass the returned packets to
`save_calibration_sample()` instead of starting another runtime.

The capture records per-camera timestamps and deltas but does not claim the
fish-eye and RGB images are synchronized. It also does not identify a camera's
physical serial number; the recorded identity is the configured SDK channel
and DDS topic. Calibration and camera-to-robot extrinsics remain separate
follow-up work.
