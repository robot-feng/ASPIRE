# Open-RAIL apple grasp implementation status

Updated 2026-11-04 (Asia/Shanghai). This file is the current evidence index for
the first AgiBot G1 apple grasp integration.

## Repository baseline

| Component | Pinned/current revision | Working tree |
| --- | --- | --- |
| ASPIRE | `feature/open-rail-apple`; baseline `fa00f0e` | local integration changes in progress |
| Open-RAIL source reference | `18673e192bde176235ce949be57fe8f17d887224` | clean pinned submodule at `third_party/open-rail` |
| external `/data/open-rail` | same HEAD as pinned source | has unrelated operator edits; left untouched |
| WeVisOne-VL source | `9a6bcd85404505df0d05cfb5d97512ccacf35dee` | clean pinned submodule |
| GraspGen-X source | `b9429097728cb1c430dd78b92edf17ba318aad03` | clean pinned submodule |

`src/cap_harness/libero` and `examples/libero/object_swap_seed1.py` remain at the
ASPIRE baseline. New Open-RAIL code is kept in its sibling package.

## Device and model evidence

- The local CAP MCP tool graph is not available in this session, so no registered
  robot observation or control command can be issued here.
- `/run/a2d-control-manager/state.json` reports scene `develop` and `active: null`.
  The A2D mode-switch service is running. No StarVLA inference or Open-RAIL robot
  client process was found during the baseline check.
- G1 RGB camera parameters exist at `/data/parameters/head_intrinsic_params.json`;
  a head extrinsic file also exists at
  `/data/parameters/head_extrinsic_params.json`. Their association to the current
  RGB/depth stream, frame convention, and transform direction have not yet been
  verified. These files are not treated as a calibrated ASPIRE profile.
- The prior read-only capture is preserved at
  `artifacts/open_rail/observe/snapshot.json`, alongside its RGB image and state
  vector. It contains 1280×800 RGB and 20 state values, but no depth packet and
  is not a synchronized RGB-D observation.
- The original `develop` camera configuration published D457 `cam8` as
  `head_color` without `cam7`. The installed
  `/data/parameters/fisheye_camera_info.json` identifies `cam7` as the same
  D457's `head_depth`; a separate camera configuration enables that channel.
  The vendor scene manifests also start HAL and motion control, so changing
  scenes is not treated as a camera-only operation. Static inspection shows
  that the launcher's `RESTART_SCENE` path invokes `RestartAll`, including
  `motion_control` in the `develop` scene; no restart was performed.
- A candidate `cam7` edit was first restored after finding that the launcher's
  original `vr.json` argument differed from the running `develop` scene.
  Further inspection showed that scene switching updates the running app's
  configuration. The original configuration SHA-256 is
  `1344cd28dcbddaf33c4c9f7d1cee528034dde2ed9c6c0e10245668d31520ceef`;
  before/after copies are in `artifacts/open_rail/camera_enable_20261104/`.
  The `cam7` entry was then added and only `Cosine` was asked to restart. The
  command returned result `1` (failure); logs show the old process was stopped,
  an immediate launch was skipped, then the supervisor relaunched Cosine with
  `. develop`. The motion-control PID did not change. The modified configuration
  remains active and may be overwritten by a later mode switch.
- A new read-only capture at `artifacts/open_rail/observe_depth_20261104/`
  includes RGB and raw Z16 depth, both observed at 1280×800. The depth packet
  was 18.77 ms after RGB, 74.28% of raw depth values were nonzero, and head/waist
  state was about 24.9 ms older than RGB. No synchronization tolerance, metric
  scale, RGB-depth alignment, or camera-to-base transform is yet verified.
  No apple is visible on the work surface in that frame. The raw-value preview
  is not a metric point cloud.
- WeVisOne-VL and GraspGen-X files occupy 4.6 GB and 1.6 GB respectively under
  ignored `third_party/models/` paths. GraspGen-X generator and discriminator
  checkpoints have been decoded with `torch.load(..., map_location="cpu",
  weights_only=True)`; this is a file-integrity check, not model inference.
- The WeVisOne local snapshot has 626 tensors; its weight and decoder hashes
  match the download metadata. An isolated CPU-only probe using the installed
  NVIDIA Torch 2.4 prerelease and Transformers 4.57 loaded the weights and
  generated an instance box and finite 128×128 matte for a synthetic image.
  It used a temporary snapshot view with tokenizer and RoPE configuration
  translated for that version; the source files were unchanged. The probe
  took about 60 seconds and peaked at approximately 14.23 GiB RSS. A separate
  offline probe on a historical RGB frame located the visible apple, while a
  saved frame without a visible apple returned no instance. Inputs, responses,
  masks, overlays and versions are indexed in
  `artifacts/open_rail/observe/wevisone/apple_cpu_20261104T115426Z/manifest.json`.
  These two samples do not establish accuracy or the current scene state.
- The same historical apple image was also passed through the actual local
  WeVisOne HTTP service and ASPIRE client. The client returned one boolean mask
  matching the service probabilities at threshold 0.5. Service logs and
  results are stored under the above evidence directory's `service_client/`.
  The test service was stopped after the check.
- A candidate G1 OmniPicker URDF and meshes are present at
  `/data/dreamlabs/g1_demo/tcp_setup/arms/g1-omnipicker.urdf` (SHA-256
  `8e2721f121557dc41e961105f505fe8da1a949cb8321af9d1ef5903eb52f4d35`).
  All 30 referenced meshes are present in its accompanying SDK-limits ZIP.
  Its match to this robot and the actual gripper swept volume is not verified.
  A different vendor's similarly named G1 description is not a substitute.
- The candidate model places the gripper center 143.08 mm from the arm flange;
  its accompanying Dream TCP calibration reports 126 mm; the active controller
  has a generic 200 mm TCP. The calibration's full orientation and its
  correspondence to the active robot are unconfirmed. None of these values is
  used as a verified GraspGen-X-to-robot TCP transform.
- Open-RAIL uses a normalized gripper command with 0=open and 1=closed. The
  mapping from this value to the candidate URDF's gripper joint angle or actual
  opening width has not been established; it cannot yet support swept-volume
  or grasp-width checks.
- The existing cuRobo service has no G1 robot model or attached-object geometry;
  its `collision_aware` flag alone cannot establish collision clearance for a
  grasp or a lift carrying the apple.

## Offline regression evidence

- Open-RAIL adapter, shared API, runtime and provider contract checks pass with
  fakes; these do not establish real model inference or robot behavior.
- The original YAM simulator adapter and motion-interlock tests passed 32 cases
  in an isolated aarch64 test environment with MuJoCo 3.3.0 and mink 0.0.13.
  No physical YAM was connected.

## Active work split

- A: Open-RAIL/G1 observation and control adapter, robot profile, read-only
  synchronization evidence.
- B: WeVisOne-VL segmentation provider and mask/empty-result handling.
- C: GraspGen-X grasp provider and frame/descriptor audit.
- Main: provider topology and dependency integration, ASPIRE runner wiring,
  apple task orchestration, review and final evidence.

No physical motion has been issued for this implementation. Any later motion
requires a calibrated robot profile, a collision-aware plan, a current observation,
an available single control owner, and the operator's workspace-clear confirmation.
