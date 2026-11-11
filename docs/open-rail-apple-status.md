# Open-RAIL apple grasp implementation status

Updated 2026-11-11 (Asia/Shanghai). This file is the current evidence index for
the first AgiBot G1 apple grasp integration.

## Current RGB-D calibration status

- On 2026-11-11 the operator presented the printed board for 30 additional
  read-only RGB-D samples at `artifacts/open_rail/rgbd_board_65to100cm_20261111_ZTTne6/`.
  The offline detector found 28 board views; a corner-motion heuristic grouped
  these into approximately 10 poses, with frames 12–30 in one near-duplicate
  group. Across 27 views with enough raw depth near corners, the nominal
  median board depth ranged from about 0.327 to 0.620 m. Both pose depth and
  inferred scale still depend on the unverified printed size, RGB-D alignment,
  raw depth units and calibration; they are not a calibration certificate.
  Results are in `artifacts/open_rail/rgbd_board_65to100cm_analysis_20261111/`.
- Two further operator-assisted read-only collections preserved 15 and 40
  RGB-D frames under `artifacts/open_rail/rgbd_board_far_20261111_iBVFQz/` and
  `artifacts/open_rail/rgbd_board_fast_20261111_9mCkQr/`. The first had 14
  detected views in approximately 14 pose groups; the second had 34 detected
  views in approximately 18 groups, with its last 17 views near-duplicates.
  Nominal median board depth was 0.326–0.620 m and 0.302–0.513 m respectively,
  under candidate RGB intrinsics and the unmeasured nominal print size. The
  faster capture used a requested 0.2 s interval, but the median RGB timestamp
  interval was 0.733 s due to acquisition and saving overhead. Its RGB and
  depth source times differed by -14.28 to +14.66 ms; clock domains and an
  acceptable skew remain unverified. No new verified calibration is inferred.
  Reports are in `artifacts/open_rail/rgbd_board_far_analysis_20261111/` and
  `artifacts/open_rail/rgbd_board_fast_analysis_20261111/`.
- A 15-frame static-scene check of the new bounded buffered capture mode is in
  `artifacts/open_rail/rgbd_buffer_speed_20261111_gYUV06/`. At the unchanged
  1280×800 RGB resolution and a requested zero interval, the median RGB source
  timestamp spacing was 42.13 ms (about 24 Hz); all 15 RGB timestamps and 14
  depth timestamps differed. RGB–depth source differences ranged from -8.81
  to +12.83 ms. This is a short throughput observation, not a validated frame
  synchronization tolerance or a guarantee for sustained capture; no device
  action was sent. The worker again needed forced close and emitted the known
  shared-memory tracker warnings.
- An operator-assisted buffered sweep then captured 90 RGB frames in
  `artifacts/open_rail/rgbd_board_buffered_20261111_0mUJ5w/`, with 90 distinct
  RGB timestamps and 80 distinct depth timestamps. The median RGB interval was
  37.65 ms, and 67 images yielded at least 12 detected board corners. A pixel
  heuristic grouped the detected images into about 42 poses; 10 representative
  groups had nominal board depth at least 0.52 m, but these included only seven
  different depth frames. In one pair, the same depth frame was paired with RGB
  frames 25 ms apart whose PnP-estimated board depths differed by 26.9 mm. This
  is a warning about motion and RGB-D timing, not a metric measurement of board
  motion or a verified synchronization limit. All distances depend on candidate
  intrinsics and nominal print size. The source hashes and calculations are in
  `artifacts/open_rail/rgbd_board_buffered_analysis_20261111/`.
- After reprinting the board at the operator's 100% setting, two RGB-D sessions
  acquired 24 and 50 raw snapshots on 2026-11-09. They are under
  `artifacts/open_rail/rgbd_board_100pct_20261109/` and
  `artifacts/open_rail/rgbd_board_100pct_extra50_20261109/`. The robot was not
  commanded to move. The earlier Open-RAIL observation/inference session was
  stopped at the operator's request before the new capture; its control status
  was already false. Capture used one read-only runtime at a time.
- The combined RGB fit used 46 of 74 frames, with 0.535 px training RMS and
  aggregate detected-corner spans of 86.7% by 85.4% of the frame. Fitting the
  first session alone and estimating board poses in 27 second-session views
  produced per-view RMS of 0.483 px median, 0.793 px at the 90th percentile,
  and 1.913 px maximum. This is a held-out fit check, not a camera certificate.
  See `artifacts/open_rail/rgbd_board_combined_20261109/` for the inputs,
  fit, per-frame residuals and session holdout.
- The extra session contains 50 complete raw RGB, depth, state and metadata
  sets. A more conservative depth diagnostic finds 23 usable views, which
  reduce to 18 candidate pose groups under its near-duplicate heuristic. Most
  groups are near 0.30–0.40 m. Their agreement under assumed board size,
  depth scale and registration is useful evidence, not independent verification
  of those assumptions. Details are in
  `artifacts/open_rail/rgbd_board_100pct_analysis_20261109/grouped_results.json`.
- An exploratory full-sheet edge projection on the saved views produced at
  least four usable gradient comparisons for an edge in 8 of 43 detected
  frames. Against two manually marked paper corners, the predicted corners
  differed by roughly 1.4–8.0 px. This is comparable to the measured edge
  shifts, so the automatic projections cannot establish RGB-D alignment. The
  assumed PDF/A4 geometry and candidate RGB fit are unverified; the procedure
  is in `artifacts/open_rail/rgbd_projected_edge_audit_20261110/`.
- A separate exploratory depth-plane fit highlights a blind-zone risk: in
  `rgbd_board_100pct_extra50_20261109/sample_0017`, the dominant depth plane
  inside the projected board region had a 95th-percentile absolute fit residual
  of approximately 2.4 mm, yet differed from the RGB board plane by approximately
  76.9 degrees and 79 mm. This is consistent with a different/background plane,
  not proof of its cause. Both plane estimates assume the unverified RGB
  calibration, pixel alignment, depth semantics and 0.001 m/count scale; see
  `artifacts/open_rail/rgbd_plane_consistency_20261110/`. A coherent depth
  surface within a segmentation mask is not sufficient to prove it belongs to
  the object.
- Replaying the previously saved WeVisOne response against its exact historical
  RGB-D frame gave 6,397 source-mask pixels and 5,983 pixels in one filtered
  depth component. With exploratory, unverified 0.001 m/count scale and a
  six-pixel background ring, the median target and ring depths were 0.593 m and
  0.652 m respectively. This is a historical diagnostic of one view, not a
  validated threshold, current observation, registration check or grasp plan;
  files and assumptions are in `artifacts/open_rail/rgbd_foreground_audit_20261110/`.
- A paper-edge comparison in two first-session images gives candidate square
  widths of 23.93 and 24.08 mm, assuming the full visible sheet is flat,
  untrimmed A4. This is consistent with the print setting, but is not a physical
  size measurement. Under a nominal 24 mm square and assumed aligned axial-Z
  depth, 13 first-session views had a median inferred scale of 0.0009896 m per
  raw unit. These assumptions and the unverified RGB-depth time tolerance mean
  this is not yet an established device scale or registration.
- The 2026-11-04 ChArUco RGB images yielded a **candidate** intrinsic fit from
  17 views with 0.188 px reprojection RMS. Four held-out views had 0.15–0.24 px
  RMS after fitting their board poses with intrinsics fixed. The candidate and
  held-out results are in `artifacts/open_rail/charuco_session_20261104/`;
  neither establishes camera-to-robot extrinsics or depth calibration.
- On one saved RGB-D image of the printed board, 28 depth samples near detected
  ChArUco corners had median raw value 552. The RGB fit, using the PDF's nominal
  24 mm squares, placed the board at approximately 0.624 m; a hypothetical
  0.001 m/raw-unit depth scale would give 0.552 m. A single plane does not
  identify the cause of this discrepancy or prove registration. The actual
  printed scale is unmeasured and must not be assumed to equal the PDF scale.
- Independent estimates from two saved images, treating the visible paper as
  an untrimmed, flat 297 × 210 mm A4 sheet, put the printed square at roughly
  21.0 mm rather than the nominal 24 mm. Applying that hypothetical scale to
  the RGB plane estimate gives about 0.546 m, near 552 raw depth units under the
  hypothetical 1 mm/unit scale. Paper shape and the selected boundaries introduce
  unquantified systematic error; this is a possible explanation, not a metric
  calibration. Reproduction and annotated inputs are in
  `artifacts/open_rail/rgbd_scale_investigation_20261109/`.
- The former observation/inference client session was stopped at the operator's
  request before the November 9 capture. On November 10 a new Open-RAIL web
  process was present; its read-only status reported observing, inference and
  control all stopped. Process/status checks are snapshots and do not reserve
  control ownership for future use.
- The current task uses head RGB-D; fisheye calibration is not a prerequisite.
  No candidate calibration has been applied to the live robot configuration.

## Same-snapshot apple and plate detection

- `examples/open_rail/pick_apple.py` now localizes both `apple` and `plate` from
  the same captured `Observation`. The two text-prompted WeVisOne calls are
  sequential, but both consume the exact same RGB-D and robot-state snapshot;
  mismatched snapshot IDs/fingerprints or a missing plate stop the preview
  before grasp generation. The plate is contextual destination perception;
  placement and robot motion are still not implemented or verified.
- A read-only saved snapshot at
  `artifacts/open_rail/apple_after_calibration_20261111_lVytS3/` was segmented
  with WeVisOne-VL using `apple` and `plate` prompts against the same RGB
  image. It returned one apple mask (6,734 pixels) and one plate mask (16,425
  pixels). The respective masks covered 6,383 and 16,324 nonzero raw depth
  samples. Their raw Z16 p05/median/p95 values were 559/570/590 and
  628/673/708. `apple_plate_same_frame_overlay.png` shows both masks on that
  frame. These raw values are not metric distances: depth scale and RGB-depth
  registration are still unverified, and the associated robot-state timestamps
  are not bound to the state values. This demonstrates paired image-space
  detection only, not a calibrated apple-to-plate transform or placement plan.
- A later read-only capture at 16:37 CST on 2026-11-11 is saved under
  `artifacts/open_rail/g1_live_scene_20261111_JAWQ2g/`. In that frame WeVisOne-VL
  found one apple mask (5,980 pixels) and one plate mask (18,332 pixels).
  A GraspGen-X replay used 5,627 filtered apple-depth pixels and returned eight
  candidates in `graspgenx_report_UNVERIFIED.json`. The replay assumed candidate
  RGB intrinsics, RGB-depth registration, a 0.001 m/raw depth unit and a
  model-only gripper sweep. It did not apply a calibrated TCP transform or check
  scene collisions. The RGB-depth source timestamps differed by 4.48 ms and the
  state timestamps were not bound to the values. The capture worker needed forced
  termination on close. This historical frame and its candidates are not an
  executable plan or evidence of the current scene.

## ASPIRE task pattern and waypoint policy

- The simulated `examples/libero/object_swap_seed1.py` and
  `examples/robosuite/cube_lifting_seed1.py` use a small grasp sequence: move
  above the target, descend with a fixed gripper orientation, close, then lift.
  The Open-RAIL task follows ASPIRE's observation/localization, GraspGen-X,
  `plan_motion` and scene APIs; WeVisOne-VL supplies the apple and plate masks.
- For G1 contact work, keep the selected grasp orientation fixed and use only
  the necessary key poses: pregrasp, short approach, and lift. Reobserve before
  contact if the snapshot or object geometry has become stale. The count of
  target poses alone does not control the planner's intermediate TCP posture;
  the full timed joint path still needs validated FK, limits, collision and
  tracking checks before execution. The current chain preview is not executable
  and makes hypothetical post-stage-state assumptions.

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
- The Open-RAIL G1 driver has historical physical-control evidence: the local
  `/data/open-rail/docs/guides/agibot-g1-full20k-startup.zh-CN.md` records a
  5.01-second Control run followed by Pause and a 10-second observation in
  which right-arm peak-to-peak movement was about 0.000244 rad. The same guide
  also records an operator reporting robot motion. Those prior results establish
  that the Open-RAIL driver has operated the robot, not the behavior of this
  integration's right-arm-only commands, terminal feedback or in-flight cancel.
- The tested upstream G1 `control_robot` path passes the arm array to
  `RobotDds.move_arm`, which publishes the SDK's arm-joint name list and the
  supplied positions; its gripper call publishes to both hand endpoints. The
  ASPIRE single-right-arm path instead uses the SDK's WBC `right_arm` group.
  The historical full-body Open-RAIL run does not independently establish
  that separate path's group scoping or stop behavior on this robot.
- On 2026-11-11, passive ROS graph discovery with `ROS_LOCALHOST_ONLY=0` found
  `/wbc/joint_position_control` and `/wbc/motion_stop`, each with one publisher
  and one subscriber; their endpoint node names were unavailable. With
  `ROS_LOCALHOST_ONLY=1` the same local probe saw neither topic. Listening for
  eight seconds found no joint-position message. The installed forwarder binary
  contains `Pb2RosNode` type references for both command types and their topic
  names, supporting a candidate protobuf-to-ROS path. These are time-limited
  discovery and static-binary observations, not proof of control ownership,
  live message delivery, command receipt, or safe stopping. No command was
  published or service called by this probe. The discovery output and binary
  hashes are under `artifacts/open_rail/g1_ros_discovery_20261111_S8P9Tg/`.
- The same graph listed `/wbc/arm_go_pose`. A candidate implementation in the
  installed `vr_control_py/vr_arm_control.py` ignores the request's `block`
  field, returns success after starting timed interpolation, and publishes a
  14-joint arm command even when right arm is selected. A small offline mock
  of its callback reproduced immediate success for both `block` values and an
  invalid arm selector. Its running service binding was not established. This
  service has not been called and cannot establish right-arm-only motion,
  terminal feedback or cancellation from the available evidence.
- Three subsequent read-only RGB-D snapshots at about 18:18 CST are retained in
  `artifacts/open_rail/g1_bound_observe_20261111_8GqEZM/`. All three contain
  state values bound to their source-message timestamps and raw right-arm and
  gripper feedback from the same broadcaster. Depth-to-RGB timestamp offsets
  were +0.76, -6.49 and -16.18 ms; head/waist source offsets reached about
  -41.90 ms. No maximum acceptable skew or common-clock certification was
  supplied, so the snapshots remain marked unsynchronized. Across these three
  observations the largest reported 14-arm joint span was 0.000070 rad; this
  is a short historical observation, not evidence of current stillness or
  of gripper measurement freshness. No command was sent; the worker again
  needed forced termination during close.
- On 2026-11-11 a single read-only Open-RAIL capture at
  `artifacts/open_rail/live_readonly_20261111_GLY9dK/` received RGB, raw Z16
  and state. The frame shows a red apple to the right of a pink dish. Its RGB
  and depth timestamps differ by 18.35 ms; the synchronization tolerance,
  clock domains, depth calibration and camera-to-base transform remain
  unverified. This historical frame does not establish the present scene or
  workspace clearance. The worker again required termination after its close
  timeout and emitted shared-memory tracker warnings; no motion was commanded.
- On 2026-11-10, read-only G1 capture with a separate DDS subscriber node
  obtained bound values and timestamps for arm, head, waist and both grippers,
  alongside RGB and raw depth. Three saved snapshots passed a **configured,
  experimental** 50 ms timestamp-window check; the camera/depth difference
  ranged from -18.37 to +12.85 ms. The clock domains and an acceptable motion
  skew have not been independently verified. Two later samples preserve the raw
  arm velocity field, which has 7 values while the message contains 14 named arm
  positions. Its side, ordering and units are unverified, so it cannot populate
  the 14-velocity CAP state. No motion was commanded. See
  `artifacts/open_rail/bound_state_named_probe_20261110/` and
  `artifacts/open_rail/bound_state_raw_velocity_20261110/`. The SDK rejects a
  second subscription to the same topic on the original node; the failed
  attempts are retained under `bound_state_series_20261110` and
  `bound_state_node_probe_20261110`. The observation worker required forced
  termination after the normal close timeout and emitted shared-memory tracker
  warnings; no robot command was sent to recover it.
- The active `a2d_v01` configuration subscribes to separate
  `/hal/left_arm_data` and `/hal/right_arm_data` topics. A subsequent read-only
  capture retained each message's timestamp, seven motor IDs, positions and
  velocities without using them for motion or substituting them into CAP state.
  In one stationary snapshot both sides reported IDs 0–6; compared in array
  order, their positions differed from the 14-value joint message by at most
  0.000036 on the left and zero on the right. This does not establish joint-ID
  correspondence, velocity units or behavior during motion. See
  `artifacts/open_rail/bound_state_arm_sides_20261110/`.
- In that same stationary sample, nearest-value matching maps side motor IDs
  0–6 to combined-message indices 0–6 on each arm. The smallest difference
  between the best and second-best match is about 0.00785 on the left and
  0.15785 on the right. This is a useful candidate correspondence for those
  two message streams, not a dynamic test or proof that either maps to the
  physical URDF joints; no motor mapping is enabled on that basis alone.
  The source hashes and comparison are in
  `artifacts/open_rail/joint_feedback_match_20261110/`.
- A later read-only `--broadcast` probe collected three RGB-D frames through
  one Open-RAIL proxy and saved a right-arm seven-ID position/velocity sample
  and right-gripper sample from the parent feedback buffer. These are raw
  post-acquisition records, not command-completion evidence. The first frame
  may precede bound feedback; no command was sent. Source and feedback evidence
  are in `artifacts/open_rail/broadcast_feedback_probe_20261110/`. The
  upstream worker again needed termination after its normal close timeout and
  produced shared-memory tracker warnings.
- `/run/a2d-control-manager/state.json` reports scene `develop` and `active: null`.
  The A2D mode-switch service is running. No StarVLA inference or Open-RAIL robot
  client process was found during the baseline check.
- A 2026-11-11 offline audit of the installed A2D SDK and Open-RAIL proxy found
  that `MotionController.set_joint_position_control` creates a right-arm-only
  `JointPositionControl` with a lifetime and group mask, but does not populate
  its header. A pure in-memory fake publisher captured an unset protobuf header
  with zero timestamp before the native DDS publisher; the installed IDL says
  the timestamp is the send time. Whether the native publisher fills it or the
  controller enforces expiry remains unverified. The proxy's ordinary RPC
  timeout limits how long the caller waits; it does not invalidate that RPC
  queue entry. `pause_controls()` invalidates a separate action queue and does
  not cancel a queued scoped RPC. No real control connection was created.
- The scoped gripper publisher supplies only `JointState.position`, with no
  velocity, effort or current limit in its message. The installed `MotorState`
  definition exposes position, velocity, effort, current, status and error
  fields, but its opening/force/stall status labels are explicitly described
  for dexterous hands, not the installed OmniPicker. Neither its use nor any
  threshold or hold behavior is verified for this gripper. The `EndState` and
  `MotionControlStatus` definitions provide no per-command acknowledgement
  identifier, so a later state sample is not in itself proof that a particular
  command completed. `EndEmergStop` has no side selector in its service type;
  `EndControl` has one, but its current server, use and effects have not been
  established. No cancellation or gripper service was called. The audited
  paths, SHA-256 hashes and a reproducible in-memory packet probe are indexed at
  `artifacts/open_rail/g1_control_protocol_audit_20261111/`.
- A static inspection of the installed `/home/agi/app/bin/hal` aarch64 binary
  (SHA-256 `1ebf199163862611e78d5299becbb51db1147be5a862c253fc228b4ded290309`)
  found that the demangled OmniPicker `SetForce` and `SetVelocity` symbols share
  address `0x1f19c0`, where the entire machine-code body is `mov w0, #1; ret`.
  `GetVelocity` and `SetMode` likewise immediately return 1 without touching
  their arguments. Under the ordinary aarch64 ABI and these symbol identities,
  these disk-binary methods do not configure force or speed or fill velocity
  output; the meaning of return value 1 is not established. A running HAL
  process reports that pathname in its command line, but `/proc/<pid>/exe`
  access was denied, so the inode and executable digest could not be checked.
  This does not establish firmware behavior, actual gripper limits or other
  control paths. The local right-hand version file reports `omnipicker` and
  firmware `141`, which is not an independent online identity verification.
  The read-only symbol/disassembly reproduction and file hashes are in
  `artifacts/open_rail/omnipicker_hal_static_audit_20261111/`. No device or
  control service was opened.
- In the same disk binary, a `ParseMsgFrame` branch stores a byte from message
  offset 11 scaled by 30/255 into object offset 192. `GetForce` reads the same
  offset and stamps the returned value using `Clock::PTP::now()` when called,
  rather than copying the CAN-source timestamp. This is a static offset and ABI
  interpretation: the byte's meaning, units, source, update age and actual
  publication are unknown. The returned timestamp alone cannot establish when
  the sensor was sampled; neither a force sensor nor a safe force threshold is
  established by this finding. The disassembly is retained in the same audit.
- Static vtable and call-site analysis indicates `RightHandPublish` calls the
  OmniPicker joint, current, force and status getters with a shared timestamp
  destination. The status getter stamps that destination with PTP now and checks
  a roughly three-second internal last-message timeout. If this matches the live
  binary, repeated publisher timestamps can advance while the underlying cached
  measurement has not changed. `G1FeedbackCalibration` therefore requires
  independent verification that gripper timestamps identify actual new hardware
  measurements; a verified clock or advancing message header alone is not
  enough. This prerequisite is currently unverified and is not enabled for G1.
  The trace is a disk-binary/ABI inference, not a physical disconnect test.
- G1 RGB camera parameters exist at `/data/parameters/head_intrinsic_params.json`;
  a head extrinsic file also exists at
  `/data/parameters/head_extrinsic_params.json`. Their association to the current
  RGB/depth stream, frame convention, and transform direction have not yet been
  verified. These files are not treated as a calibrated ASPIRE profile.
- The local historical `/data/calib/head_d457/` record contains 12 original
  images, 15 poses, the board geometry, and an `eyeinhand` result. Its RGB
  intrinsic file identifies D457 serial `343622300358`, the same serial listed
  for `head_color` and `head_depth` in the local camera configuration; this is
  a configuration match, not a live device identity check. Its extrinsic JSON
  matches `/data/parameters/head_extrinsic_params.json` byte for byte.
  Re-detecting the ChArUco board in those 12 images and using the saved RGB
  intrinsics and distortion gives a fixed-board consistency check: under the
  unverified assumptions that numbered images correspond to the same pose
  indices, the board stayed fixed, and its recorded dimensions are accurate,
  saved-pose × extrinsic × camera-to-board has 1.07 mm median and 1.54 mm maximum
  positional deviation from the coordinate-wise median; replacing the extrinsic
  by its inverse gives 38.7 mm and 64.7 mm respectively. The same data were
  used for calibration, so this is an internal consistency check, not a held-out
  estimate of physical calibration accuracy.
- Comparing the same 15 saved poses with the installed `a2d_v01` URDF gives a
  candidate frame interpretation: the model's `base_link` to `head_pitch_link`
  transform, followed by a 180-degree rotation about the link's local Z, matches
  the first saved pose at zero joint values within 0.00242 mm and 0.00052°.
  Fitting hypothetical joint values to each of the 15 saved poses gives maximum
  residuals of 0.00250 mm and 0.00040°. These numerical residuals compare a
  saved pose with a model; the fitted angles are not measured historical joint
  positions. Taken together, the checks support using the JSON in the forward
  direction from that *candidate* rotated head frame to RGB optical frame.
  They do not establish that the installed model, camera mounting, pose source,
  current device identity or RGB-D registration still matches the historical
  calibration. No value is enabled for live robot control. Source hashes,
  assumptions, all pose comparisons and the offline reproduction script are
  in `artifacts/open_rail/vendor_head_handeye_audit_20261111/`.
- A further check applied that **candidate** model and head-camera transform to
  the historical saved apple frame. Its raw 20-value vector includes the
  source-labelled head yaw/pitch and waist pitch/lift entries; the recorded head
  and waist messages are approximately 13.50 ms later than RGB, without an
  established clock domain or acceptable skew. This is not a verified dynamic
  state mapping. Under the additional unverified assumptions that the published
  depth is RGB-registered RGB axial Z at 0.001 m per raw count and that the
  vendor RGB distortion model applies, a rectangular region selected by visual
  inspection on the saved tabletop fits a plane tilted approximately 0.77° from
  candidate base Z. On a fixed random sample of 5,000 nonzero ROI points, all
  lie within 10 mm of that fitted plane. The median WeVisOne apple-mask point
  lies approximately 64.8 mm above it. This shows no obvious geometric conflict
  in that one view, but the selected ROI, common assumptions and lack of an
  independent tabletop measurement make it unsuitable for certifying any
  calibration, reachability or collision clearance. The source hashes, exact
  ROI, assumptions and script are in
  `artifacts/open_rail/vendor_head_apple_scene_audit_20261111/`.
- An additional projection diagnostic used the 18:18 bound-state RGB snapshot,
  the candidate factory camera chain, and the candidate a2d_v01 robot model.
  Most known arm meshes were outside the image. Assuming the installed grippers
  match the official OmniPicker rigid-base model and its flange attachment, the
  projected bases overlapped manually chosen image rectangles more under the
  forward factory transform than with its inverse. The rectangles are coarse,
  occlusion is not modeled and the gripper assumption is unverified, so this
  helps reject some large frame-direction errors only; it is not a metric
  camera calibration or TCP measurement. Inputs, overlays and assumptions are
  in `artifacts/open_rail/g1_rgb_fk_projection_20261111/`.
- The historical factory file for `cam7` specifies depth intrinsics at
  1280×720, while this published raw Z16 frame is 1280×800, matching the RGB
  frame's dimensions. An intermediate conversion, resizing or registration
  might explain this discrepancy, but these records do not establish which, or
  that RGB and depth pixels are aligned. The depth scale and frame semantics
  remain unverified.
- Static inspection of the current `develop` configuration shows `cam7` set to
  1280×720. The observation adapter reads width and height from the published
  Z16 packet itself and checks its byte length before reshaping; the historical
  1280×800 array is therefore not caused solely by forcing it to the RGB shape
  in the adapter. The installed D457 camera library uses the V4L2/CSI path;
  inspection of the installed publishing function shows a buffer-dump and
  publish path, but does not rule out processing in the sensor, driver or an
  earlier stage. The source configuration and binary provenance remain separate
  from proof of what processing the historical packet underwent. The checked
  config SHA-256 is `3b4abc1e79e31a6d01d16866bba19909fb4d9dce5132cc9d5e9e5abb2ad25caf`;
  the publisher library SHA-256 is
  `ddd7d88f9c7afddd59735d5c63874dc03e0c8a92add681a33888668536d42e2b`.
- The manufacturer's [D457 datasheet, table 4-2](https://realsenseai.com/wp-content/uploads/2025/08/RealSense-D457-Datasheet-Aug_2025.pdf)
  lists 1280×720 Z16 depth and 1280×800 RGB as separate native GMSL stream
  formats. The historical 1280×800 depth packet therefore requires an explanation
  in the configured stream, processing chain, or packet metadata; matching the
  RGB dimensions does not establish registration. The [RealSense SDK projection
  documentation](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
  describes 1 mm per count as the D400 default, while explicitly permitting
  the depth-unit setting to change. This is independent support for the
  diagnostic's 0.001 m/count hypothesis, not a measurement or proof of this
  camera's configured scale, alignment, timing or current serial number.
- The same datasheet's table 4-4 lists a nominal minimum Z of 520 mm for the
  D450 module at native 1280×720 depth resolution; D457 contains that module.
  Many previously sampled board poses were nominally 300–400 mm from RGB, while
  the historical apple's assumed median raw depth corresponds to about 594 mm.
  The former are therefore particularly poor independent evidence of depth
  scale or surface identity under that native-mode specification. Actual
  stream processing, depth settings, pose and scale remain unverified: this
  nominal figure alone cannot classify individual recorded pixels as invalid.
- A local registration *diagnostic* compares the saved WeVisOne apple mask to
  thresholded raw depth in the same historical frame, searching translations
  within a 100-pixel margin. For raw intervals [570,615), [575,620) and
  [580,620), the best mask overlap IoU is approximately 0.93 with depth-image
  offsets (+1,+2), (0,+3) and (0,+3) pixels. These thresholds were chosen after
  inspecting this target and the comparison is a single object in one frame;
  the estimated offsets include mask boundary error, background selection and
  unverified RGB-depth time skew. It is not a metric-scale or full-image
  registration certificate. Exact inputs, hashes, other tested thresholds and
  a reproduction script are under
  `artifacts/open_rail/apple_depth_overlap_alignment_20261111/`.
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
  No apple is visible on the work surface in that frame. The raw-value and
  RGB/depth-boundary previews are qualitative; they do not establish metric
  scale, registration, or a metric point cloud.
- After the operator placed an apple, a further raw capture was saved under
  `artifacts/open_rail/observe_apple_20261104/`; RGB and depth timestamps
  differ by 3.95 ms, with no verified synchronization tolerance. The capture
  labels its RGB order as `RGB`, but the operator confirmed its blue-looking
  apple and purple-looking dish are actually red and pink. A channel-swapped
  diagnostic and the operator confirmation are recorded in `color_check.json`;
  the original frame is preserved. GPU inference on the color-corrected saved
  frame returned one apple mask; the uncorrected frame returned no target.
  Outputs are under `observe_apple_20261104/wevisone_gpu/`. The frame is not
  suitable for control.
- An offline diagnostic now combines that saved RGB frame, its actual WeVisOne
  mask and its same-capture raw depth. Of 6,397 mask pixels, 6,089 have raw
  depth other than the candidate invalid values 0 and 65535. The accompanying
  camera-frame PLY assumes the later candidate RGB intrinsics, 1 mm/count,
  RGB-depth registration and RGB axial-Z depth; none is established by this
  diagnostic. Source hashes, the assumptions, code and output are preserved in
  `artifacts/open_rail/apple_rgbd_diagnostic_20261110/`. No camera-to-base pose,
  physical grasp descriptor or executable trajectory is inferred.
- A subsequent live capture after the adapter's packet-aware RGB fix is under
  `artifacts/open_rail/observe_apple_rgbfixed_20261104/`. It shows the confirmed
  red apple and pink dish, identifies the RGB packet as JPEG/RGB, and matches
  its source timestamp exactly. A Z16 depth frame was also captured, 8.24 ms
  after RGB. Its scale, registration and synchronization tolerance remain
  unverified, so this capture is still raw evidence rather than a planning scene.
- A later depth-required capture at
  `artifacts/open_rail/observe_apple_final_20261104/` also shows the red apple
  and records JPEG/RGB plus uncompressed Z16 metadata. RGB and depth differ by
  18.98 ms. It is raw evidence, with the same outstanding calibration gaps.
- WeVisOne-VL and GraspGen-X files occupy 4.6 GB and 1.6 GB respectively under
  ignored `third_party/models/` paths. GraspGen-X generator and discriminator
  checkpoints have been decoded with `torch.load(..., map_location="cpu",
  weights_only=True)`; this is a file-integrity check, not model inference.
- An isolated GraspGen-X service/client GPU smoke test on an official generic
  sample returned five candidates. Its sample gripper descriptor is not the
  G1's, and no scene collision or robot execution was checked. Evidence is in
  `artifacts/open_rail/graspgenx_generic_smoke_20261104/`.
- On 2026-11-11, a second offline GPU inference used the saved historical
  apple frame, its saved WeVisOne mask refined to 5,983 candidate depth
  points, and a sweep descriptor derived from the candidate OmniPicker URDF
  and meshes. The successful replay returned eight gripper-base poses in the
  assumed camera frame in 5.58 seconds; no controller TCP transform or scene
  collision check was applied. The mesh descriptor models two hypothetical
  static finger configurations, assumes a closed linkage and uses manually
  chosen margins. Its frame, geometry and linkage are not verified against
  the installed hardware; the RGB-D scale and registration are also assumed.
  These are model-debugging candidates, not robot targets or evidence of a
  successful grasp. The first attempt wrote results but ended with a logging
  serialization error; a separate second replay completed normally. Inputs,
  source hashes, assumptions, scripts and candidate PLY are under
  `artifacts/open_rail/apple_graspgenx_model_diagnostic_20261111/`, with the
  model-only descriptor and geometry analysis under
  `artifacts/open_rail/omnipicker_sweep_candidate_20261111/`.
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
- [AGIBOT's pinned GenieSim G1 OmniPicker URDF](https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/1eb3b68b740f87fb369b0146ee53f3bb3da6b0d0/G1_omnipicker/G1_omnipicker.urdf)
  (SHA-256 `311389b09e8bcc49eae6189fb784b0114058209bd068d122ea2acde0ea6f229a`)
  agrees with the Dream model's right-gripper attachment and 143.08 mm center
  joint. The Dream 126 mm record's digest matches its model, but its tool did
  not connect to a robot or camera; that is file provenance, not a measurement
  of the installed tool. The official model is simulation evidence and does not
  verify the physical TCP or a collision model for this robot.
- An offline mesh and linkage audit found that the candidate URDF's finger
  length and opening directions, expressed in its flange frame, are roughly
  +Z and +X respectively; the Dream TCP annotation labels approach and jaw
  directions roughly +X and +Z. This discrepancy prevents using the annotation
  as a verified GraspGen-X sweep descriptor. A model-only half-open gap of
  approximately 78.2 mm is not a measurement of the installed gripper. The
  comparison and assumptions are in
  `artifacts/open_rail/omnipicker_model_audit_20261110/`.
- Open-RAIL uses a normalized gripper command with 0=open and 1=closed. The
  mapping from this value to the candidate URDF's gripper joint angle or actual
  opening width has not been established; it cannot yet support swept-volume
  or grasp-width checks.
- cuRobo can load a local G1 model for offline experiments. A candidate based
  on the installed `a2d_v01` URDF matched independent FK numerically, but a
  saved-state synthetic-scene planning probe failed with `start_self_collision`.
  The model, gripper geometry, TCP and locked head/waist posture are not
  verified against the robot, so its trajectories are not marked collision-aware
  for execution. Optional attached-object spheres do not by themselves establish
  collision clearance for a grasp or lift. Offline evidence is under
  `artifacts/open_rail/a2d_curobo_candidate_20261109/`.
- An offline comparison on the same saved state and a synthetic scene found
  `start_self_collision` when only adjacent links were excluded. Using the
  installed SRDF's `Adjacent`, `Never` and `Default` exclusions allowed the
  same 0.005 rad joint-space probe to return a trajectory. This isolates the
  effect of the exclusion list; it does not establish that those exclusions or
  the resulting trajectory are safe on the physical robot. Evidence is in
  `artifacts/open_rail/a2d_curobo_collision_audit_20261110/`.
- A later isolated cuRobo diagnostic used the historical apple frame, its first
  actual GraspGen-X candidate, candidate camera and gripper transforms and the
  candidate G1 model. The whole observed depth scene was retained, including
  the apple. The stock service rejected points beyond 5 m; an isolated 7 m
  experiment returned `no_result`, although its start-state check passed. This
  is not proof of unreachability: calibration, kinematics, gripper geometry and
  the cause of the planning failure are unresolved. No trajectory was returned
  or sent. Inputs and logs are in
  `artifacts/open_rail/apple_g1_planning_diagnostic_20261111/`.
- On Orin, the isolated cuRobo CUDA 12.2 environment successfully planned a
  generic Franka motion through the ASPIRE HTTP client, including a separate
  test with a synthetic attached sphere. Logs and scripts are in
  `artifacts/open_rail/curobo_generic_smoke_20261104/`. This combination uses
  NVIDIA Torch 2.4.0a0, below the upstream declared `torch>=2.5` requirement;
  it is an experimental compatibility result, not a supported G1 planner.

## Offline regression evidence

- Open-RAIL adapter, shared API, runtime and provider contract checks pass with
  fakes; these do not establish real model inference or robot behavior.
- The apple preview runner can optionally chain pregrasp, approach and lift
  against one scene and the preceding predicted joint endpoint. It checks the
  localization, grasp set and partition against the same snapshot and target
  mask, and requires an explicit hypothetical attachment and lift offset. Tests
  use simulated service responses; neither contact nor the attached apple is
  observed, the results are non-executable, and this chain has not been run
  through a real G1 planning service.
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
