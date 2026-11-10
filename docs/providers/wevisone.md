# WeVisOne-VL provider

ASPIRE's `WeVisOneProvider` is a standard text-segmentation provider client. Its
service loads Tencent's released `WeVisOne-VL-2B` checkpoint in a provider-only
environment and uses the official `task="segmentation"` inference path. Model
weights stay outside Git under `third_party/models/wevisone-vl-2b`. With no
`--model` argument or `WEVISONE_MODEL_PATH`, the service selects that local
directory when it exists and verifies the required files and weight header
before loading. A present but incomplete snapshot fails locally instead of
falling back to a new Hub download. Local paths force Hugging Face offline mode.
Set `WEVISONE_MODEL_PATH` to select another local location or Hub ID;
`third_party/...` paths are relative to the ASPIRE checkout.

The client omits empty thresholded masks. The service reports the model's
pre-limit `total_instances` and a `truncated` flag. The client returns
`ErrorCode.TRUNCATED` whenever the service omitted instances, so a single
returned mask cannot be mistaken for a unique target.
The client thresholds the returned probabilities at `0.5` by default because
ASPIRE's `Segmentation` contract carries a binary mask. The official model does
not emit an instance confidence score, so the required ASPIRE score is a neutral
`0.5` and diagnostics explicitly state that it is not model confidence.
Point-prompted segmentation is currently unsupported by this provider.

For an explicitly enabled RGB-D pick workflow,
`cap_harness.providers.depth_refined.DepthRefinedSegmentationProvider` wraps the
text-segmentation provider. It rejects invalid depth, pixels far from the
masked median depth, and ambiguous depth-connected components before target
point-cloud construction. Configure all its depth and quality thresholds for
the verified sensor and task; there are no device defaults. It requires the
original unique WeVisOne result and matching calibrated `CameraObservation`;
timestamps and RGB, depth, intrinsics and pose hashes bind the result to that
observation. The original RGB-D image remains available for scene geometry.

An optional foreground/background depth check helps reject a target mask whose
valid pixels may all belong to a surface behind the target. Enable it only by
explicitly setting `background_ring_width_px`, `min_background_valid_pixels`,
`min_background_valid_fraction`, and `min_foreground_gap_m`. The ring excludes
the entire original mask, including pixels discarded by depth refinement. It
compares the refined target's median depth with valid surrounding depth from
the same camera snapshot. A clipped mask, inadequate surrounding depth, or an
insufficient gap returns `POINT_CLOUD_FAILED` with `depth_status=uncertain`;
it does not claim there is no object. The original mask and counts remain in
the trace. A pass does not prove the target identity, calibration, registration,
depth correspondence, or safety, and an apple close to the surrounding surface
may legitimately be inconclusive. No thresholds are inferred or enabled by
default; choose and validate them for the camera, object and workspace.

Wrap the combined provider in `TracedProvider(..., recorder, "wevisone")` and
pass the same `RunRecorder` to `CapApi`. Its provider trace serializes the
original and filtered masks as exact 0/1 uint8 arrays, alongside the thresholds
and each stage's pixel counts. This preserves evidence through the existing
`RunRecorder` payload format; without a traced recorder the caller must retain
the diagnostics itself. Filtering does not establish object identity, depth
registration or calibration and cannot remove background at the same depth.

Run through the environment supervisor after registering the `wevisone`
provider in `configs/environments.json`. The provider is separate from the
robot SDK environment. Its model ID, host, port, and device are explicit service
arguments; no policy runner is started by this service. Validate the local files
without loading model weights or using a GPU:

```bash
PYTHONPATH=src python -m cap_harness.providers.wevisone.snapshot
```

The standalone service uses the offline-first resolver:

```bash
WEVISONE_MODEL_PATH=third_party/models/wevisone-vl-2b \
PYTHONPATH=third_party/WeVisOne-VL:src \
  python -m cap_harness.providers.wevisone.service --port 8117
```

The official WeVisOne source pins PyTorch 2.11.0 and Transformers 5.12.1. This
target currently has JetPack R36.3 with NVIDIA PyTorch 2.4.0a0; Transformers
4.57.0 cannot read the release's tokenizer list or `rope_parameters` config
directly, while 5.12.1 rejects the installed PyTorch prerelease version. For
this specific local release snapshot, an explicit compatibility view adapts
only those two JSON files in a temporary directory and symlinks all other
snapshot files. The helper rejects unknown token lists and RoPE variants; it
does not edit the original checkpoint. This view can be requested with
`--compat-transformers-457`:

```bash
WEVISONE_MODEL_PATH=third_party/models/wevisone-vl-2b \
PYTHONPATH=third_party/WeVisOne-VL:src \
  python -m cap_harness.providers.wevisone.service \
    --port 8117 --device cpu --compat-transformers-457
```

This opt-in path was exercised with the installed Transformers 4.57.0 and
NVIDIA PyTorch 2.4.0a0, with CUDA disabled. A 128×128 synthetic red-square
image produced one parsed box and a finite 128×128 probability matte in about
60 seconds; peak RSS was about 14.23 GiB. This is a CPU compatibility smoke
check, not a quality or field-validation result. The model's generation call
also reports that `temperature`, `top_p`, and `top_k` may be ignored when
sampling is disabled. Live camera frames, depth association, and manipulation
remain unverified.

An additional local HTTP service/client smoke test on Orin GPU processed the
same historical apple image in about 5.7 seconds and returned one mask. It used
a saved RGB frame and did not access a camera or establish current scene state.
Its result and telemetry are in
`artifacts/open_rail/observe/wevisone/apple_gpu_20261104T123610Z/`.

To reproduce the CPU synthetic probe from the ASPIRE checkout:

```bash
source /data/agi-miniconda3/etc/profile.d/conda.sh
conda activate /data/conda-envs/open-rail
export LD_LIBRARY_PATH="/data/conda-envs/open-rail/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/src:$PWD/third_party/WeVisOne-VL"
export CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout --signal=TERM --kill-after=5s 180s python - <<'PY'
import pathlib
from PIL import Image, ImageDraw
import torch
from cap_harness.providers.wevisone.compat_457 import Transformers457SnapshotView
from wevisone_vl import WeVisOneVL

assert not torch.cuda.is_available()
view = Transformers457SnapshotView("third_party/models/wevisone-vl-2b")
try:
    engine = WeVisOneVL(str(view.path), device="cpu", backend="transformers",
                        dtype="float32", min_pixels=128**2, max_pixels=512**2)
    image = Image.new("RGB", (128, 128), (35, 35, 35))
    ImageDraw.Draw(image).rectangle((34, 32, 94, 96), fill=(220, 20, 20))
    result = engine.predict(image, task="segmentation", query="red square",
                            decode_matting=True, max_new_tokens=24)
    print(result.response, result.boxes, [mask.shape for mask in result.masks])
finally:
    view.cleanup()
PY
```
