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
