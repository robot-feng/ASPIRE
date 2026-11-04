# G1 runtime limits and command evidence

`src/cap_harness/open_rail/g1_control.py` maps only the right arm and right
gripper. The arm request uses the A2D `right_arm` group in
`RobotController.set_joint_position_control`; the mapper does not fill or send
left-arm, head, waist, torso, or base values. The gripper request uses the
dedicated `/wbc/right_ee_command` endpoint with Open-RAIL's normalized
`0=open, 1=closed` value. `g1_body.py` rejects Open-RAIL's generic action,
manual-control, and robot-reset entry points so those cannot bypass the scope.

These mappings are not currently executable. The SDK methods only publish
commands. `RobotProxy`'s RPC return means the body method returned, not that the
robot reached its target. `MotionControlStatus` reports mode, error, frames,
and collisions, but it has no per-command completion identifier. The SDK exposes
`/wbc/motion_stop` only as a whole-controller stop; it has no right-arm scoped
cancel. The joint-control lease is sent in seconds, but the installed code does
not establish whether lease expiry stops or holds the selected arm. Also,
`RobotController.set_joint_position_control()` leaves the message header at its
default, while the installed control GUI stamps the message. The command
runtime requires verified command framing, feedback, stop/hold behavior, and
configured, verified seven-joint position limits before allowing arm dispatch.
The gripper endpoint has no command lease or scoped stop. Therefore the command
runtime defaults to disabled and returns `UNSUPPORTED` before calling a
publisher until measured feedback tolerances and stop/hold behavior are
verified. The current profile has no verified joint limits, so it cannot enable
arm dispatch even if feedback is supplied. A local timeout never claims
cancellation and never sends the global stop.

After a command publish attempt, a feedback exception or unconfirmed timeout
permanently latches that Open-RAIL proxy session as motion-uncertain. Every
later arm or gripper command is rejected across command facades. Reset, cancel,
and close do not clear the latch. The current SDK has no verified, read-only
quiescence source, so this runtime deliberately provides no unlock operation;
the session stays blocked until a separately validated control session is
established.

Construction does not publish an action. `RobotDds` creates state subscribers
and publishers in its constructor; `RobotController` creates motion-control
publishers/subscribers. The custom G1 body constructs the additional motion
controller lazily on the first authorized arm command. `OpenRailAdapter.reset`
is observational, while the G1 body's direct `reset_robot` RPC is rejected.

The read-only `OpenRailG1ObservationRuntime.health` view reports the selected
right arm, its configured joint names, allowed and forbidden command scopes,
camera calibration fields and source paths, the TCP transform field, and
worker/command readiness. Missing values remain `null` and are listed in
`calibration_gaps`; a source path is not treated as calibration data.

## Camera source and alignment

The static `develop/fisheye_camera_conf.json` publishes cam8 `head_color` at
1280x800 and has no `head_depth` entry. This accounts for the previous missing
depth frame: the A2D SDK maps `head_depth` to `/camera/head_depth`. The separate
`camera_copilot/d457_fisheye_camera_conf_53.json` file configures cam7
`head_depth` at 1280x720 and cam8 `head_color` at 1280x800, but it is not the
`develop` configuration and does not declare image encoding or calibration.
The camera receiver decodes a packet as uint16 `(H,W,1)` only when its format is
`RS2_FORMAT_Z16`; the config file alone does not prove that the cam7 publisher
uses that format.

The color intrinsic file has no image dimensions. The head extrinsic file has a
rotation and translation but no source/destination frame names. No depth scale,
depth intrinsics, or depth-to-color transform was found. The two camera topics
have different configured dimensions and the SDK performs no alignment between
them. The raw snapshot must remain raw; it cannot become `CameraObservation`, a
metric point cloud, or a camera-to-base target without verified calibration.

The G1 arm joint ordering and SDK group names are confirmed. The local DH file
does not identify units/frames and the installed Pinocchio helper defaults to a
different model; the active URDF/TCP/base transforms remain unverified. No
joint-space motion profile, position tolerance, velocity tolerance, gripper
range, or contact threshold is inferred from those files.
