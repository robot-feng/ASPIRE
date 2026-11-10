# Open-RAIL camera calibration board

Print [this A4 ChArUco board](../examples/open_rail/charuco_A4_7x10_24mm.pdf)
at actual size (100%), without fitting or scaling, and mount it flat on a rigid
backing. The board uses OpenCV's `DICT_4X4_50`, 7 × 10 squares, 24 mm squares,
and 18 mm markers. Use these exact parameters when detecting it.

Verify the printed square size with a ruler or another verified length reference
before using the board for metric measurements. The dimensions in the PDF are
nominal; printing at 100% alone does not establish physical accuracy.

Move the board through the field of view at different angles and distances while
keeping the cameras fixed. Calibrate conventional RGB and fisheye images with
their respective camera models. The board can also support relative camera
calibration when cameras observe it simultaneously. Camera intrinsics alone do
not determine the camera-to-robot-base transform or the depth sensor's scale
and alignment; those require additional verified measurements.

For a camera mounted on the moving head, calibrate the fixed transform from the
camera to its mounting link, then compose it with the mounting link's forward
kinematics at the image timestamp. Recalibration is not required for every head
pose. The dynamic G1 conversion accepts an explicitly verified kinematics
provider and mount transform, but remains unavailable for the actual robot until
its model, camera mounting link and transform, and synchronized joint-state
timestamps are verified. The right gripper's flange-to-TCP transform also needs
independent verification.

The pattern was generated with OpenCV's ChArUco implementation. See the
[OpenCV calibration pattern documentation](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
and [ENPIRE's calibration-board guide](https://github.com/NVlabs/ENPIRE/blob/main/enpire/env/docs/CALIBRATION_BOARD.md).

## Offline RGB-D board diagnostics

The saved RGB-D samples from `examples/open_rail/observe.py --samples` can be
checked without opening a camera. Supply the RGB intrinsic fit, nominal or
physically measured square length, a **candidate** depth scale, and explicit
candidate invalid values:

```sh
python -m cap_harness.open_rail.rgbd_calibration_check \
  --samples /path/to/rgbd-samples --intrinsics /path/to/rgb/calibration.json \
  --output /path/to/new-report.json --square-mm 24 \
  --candidate-scale-m 0.001 --invalid-depth-values 0 65535
```

The report retains hashes, detected ChArUco corners, per-frame reprojection
error, raw RGB/depth times, board-plane distance and depth comparisons. These
are descriptive: print size, axial-Z depth semantics, and RGB/depth pixel
registration are assumptions, not verified facts. It never writes parameters
back to the robot. A physical length reference remains necessary for metric
scale.

Optionally provide `--paper-annotations annotations.json` to examine the
visible paper boundary and selected local RGB/depth discontinuities. The JSON
maps a sample directory name to four ordered, adjacent paper corners in RGB
pixels and their assumed corresponding paper edge lengths in millimetres:

```json
{
  "sample_0001": {
    "corners_px": [[180, 200], [460, 200], [460, 600], [180, 600]],
    "edge_lengths_mm": [297, 210, 297, 210]
  }
}
```

The corners and edge lengths above illustrate the format only and must be
replaced with measurements from the actual view. Paper corners must refer to the
visible edge of the same flat, untrimmed sheet and lie inside the image. The
local edge test samples only matching contrast transitions near those manually
marked edges, with exploratory thresholds; it cannot prove full-image
registration or replace inspection of the original images.

## Optional calibrated RGB-D undistortion

`G1RgbdRectification` may be passed explicitly to `G1ObservationControl` (or
`camera_observation_from_snapshot`) when the original RGB and depth are proven
to occupy the same *distorted RGB* grid and depth is proven to represent RGB
optical-axis Z. Supply verified raw RGB pinhole intrinsics, OpenCV Brown-Conrady
distortion coefficients and image dimensions. The normal camera calibration's
intrinsics are the output rectified pinhole K, and its rectification,
registration, depth, timing and camera-pose verification requirements still
apply. No candidate calibration is selected automatically.

The same OpenCV inverse map is applied to RGB using bilinear interpolation and
to raw Z16 using nearest-neighbour sampling. Source invalid values and
out-of-grid depth become invalid output, not interpolated measurements. The
source snapshot remains unchanged. Resampling itself can introduce a
subpixel RGB-depth discrepancy near discontinuities; independently verify that
the resulting alignment is adequate before using it for planning. This is not
a calibration or registration procedure, and does not validate metric scale.
