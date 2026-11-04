# Offline ChArUco camera intrinsics

The offline tool calibrates one saved image stream per invocation. It uses the
provided A4 board (`charuco_A4_7x10_24mm.pdf`): 7×10 squares, 24 mm squares,
18 mm markers, `DICT_4X4_50`. The tool never opens a camera or robot connection.

From an environment with ASPIRE and OpenCV installed, pass a directory of saved
PNG/JPEG images. Use `pinhole` for the regular head RGB lens and `fisheye` for
each fisheye stream; choose the model explicitly from the physical lens.

```bash
python -m cap_harness.open_rail.offline_camera_calibration calibrate \
  --images /path/to/head-rgb-images \
  --output-dir outputs/camera-calibration/head-rgb \
  --camera-name head_rgb --model pinhole

python -m cap_harness.open_rail.offline_camera_calibration calibrate \
  --images /path/to/head-fisheye-images \
  --output-dir outputs/camera-calibration/head-fisheye \
  --camera-name head_fisheye --model fisheye
```

If OpenCV's fisheye initializer fails or the lens has a known approximate
starting point, pass a matching-resolution K/D JSON with
`--initial-intrinsics`. The tool accepts the older `{ "intrinsic": { "fu",
"fv", "pu", "pv", "k1".."k4" } }` format and this tool's `camera_matrix` /
`distortion_coefficients` output.

```bash
python -m cap_harness.open_rail.offline_camera_calibration calibrate \
  --images /path/to/head-fisheye-images \
  --output-dir outputs/camera-calibration/head-fisheye-seeded \
  --camera-name head_fisheye --model fisheye \
  --initial-intrinsics /path/to/fisheye-intrinsics.json
```

The file is an optimizer seed only: OpenCV starts from its K/D, then estimates
the parameters from the supplied images. The result records the seed values,
source path, and SHA-256. If the input declares an image size, it must match
the image stream. The tool does not automatically discard high-error views;
inspect per-frame residuals and the coverage warning before judging the set.

Use at least 8 usable frames with clearly different board positions and tilts;
15–25 well spread views are preferable. A frame needs at least 8 detected
ChArUco corners. A single image with 14 corners is a useful detection check,
but is not an intrinsic calibration set. Keep the board sharp, fill different
parts of the image, and include tilted views, especially for fisheye lenses.
A low reprojection error on a small central patch does not validate radial
distortion over the full fisheye field of view; spread the board into all four
outer regions as well.

The detector first uses the original image. If it finds fewer than 32 ChArUco
corners, it also tries a 2× cubic-interpolated detector image and keeps the
better detection. Detected pixels are mapped back to the original image before
calibration; neither the source images nor calibrated resolution are resized.
Use `--detect-upscale 1` to disable this fallback.

Each new output directory contains `calibration.json` with the camera model,
intrinsics, distortion coefficients, global RMS, and per-frame reprojection
errors, plus `samples.json` with copied accepted images, detected marker and
ChArUco counts/IDs/pixels, hashes, detection scale, and rejected-image reasons.
The calibration is intrinsic only; it does not produce camera-to-robot/base
extrinsics.

The PDF describes nominal printed geometry and says to print at 100%. The
workflow records the 24/18 mm dimensions as nominal and marks the print scale
unverified. Measure a printed square before using board translations as metric;
do not treat the resulting intrinsics as a camera-to-base calibration or as
validated for a different stream resolution, crop, or camera model.

Synthetic CPU-only checks:

```bash
PYTHONPATH=src pytest -q tests/test_offline_camera_calibration.py
```
