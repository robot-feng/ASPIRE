# G1 runtime limits and command evidence

`src/cap_harness/open_rail/g1_control.py` maps only the right arm and right
gripper. The arm request uses the A2D `right_arm` group in
`RobotController.set_joint_position_control`; the mapper does not fill or send
left-arm, head, waist, torso, or base values. The gripper request uses the
dedicated `/wbc/right_ee_command` endpoint with Open-RAIL's normalized
`0=open, 1=closed` value. `g1_body.py` rejects Open-RAIL's generic action,
manual-control, and robot-reset entry points so those cannot bypass the scope.

ASPIRE's shared gripper contract uses `1=open, 0=closed`. A G1 control facade
must therefore convert ASPIRE commands at the G1 boundary. The command runtime
provides `set_cap_gripper()` for this explicit conversion; its existing
`set_gripper()` remains in the native Open-RAIL domain. The available normalized
feedback convention has not been verified physically and must not be converted
to an ASPIRE `RobotState` until its direction and normalization are established.
Action and trajectory gripper columns need the same conversion in the future
G1 facade, which does not yet exist; neither path is currently implemented.

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

`G1MeasuredCommandFeedback` can evaluate post-command DDS samples inside the
observation worker once the motor mapping, units, clock and tolerances are
verified. It requires a continuous window of fresh position and velocity
measurements and preserves terminal evidence. It is not yet connected to the
parent command facade: Open-RAIL's fixed-timeout RPC and its interaction with
concurrent observation, cancellation and trajectory timing remain unverified.
A gripper stopping short of its requested closed position may indicate contact;
the target-position check alone neither confirms a grasp nor provides a safe
contact-completion signal for a lift.

`G1SnapshotBroadcaster` can explicitly reserve the existing observation runtime's
single frame reader, start one background reader, and distribute snapshots to
the read-only observation facade and bound right-arm/gripper feedback to the
parent-process `G1MeasuredCommandFeedback`. It starts no thread on construction
and never opens a second camera or sends commands. Feedback waits use a local
condition variable and the original DDS sample timestamps, so repeated frames
do not count twice and a short feedback timeout does not wait on the camera or
a feedback RPC. Closing or a reader error wakes waiters as an error; neither
indicates that an in-flight command stopped. A blocked reader makes close raise
after the configured close timeout and keeps the underlying runtime open.

This is a feedback-only bridge, tested offline with a separate synthetic producer
process. The facade is still read-only and must not be treated as a complete G1
control implementation. The command **publish** itself still uses RobotProxy's
synchronous RPC with a fixed timeout; the feedback timeout starts after that RPC
returns and is not an end-to-end command deadline. Arm-specific cancellation,
lease expiry and contact/force detection remain unverified. A short gripper
travel remains an unknown contact/grasp state and latches command uncertainty.

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

The original `develop/fisheye_camera_conf.json` published cam8 `head_color` at
1280x800 without `head_depth`. This accounted for the previous missing depth
frame: the A2D SDK maps `head_depth` to `/camera/head_depth`. The separate
`camera_copilot/d457_fisheye_camera_conf_53.json` file configures cam7
`head_depth` at 1280x720 and cam8 `head_color` at 1280x800, but it is not the
`develop` configuration and does not declare image encoding or calibration.
The camera receiver decodes a packet as uint16 `(H,W,1)` only when its format is
`RS2_FORMAT_Z16`; the config file alone does not prove that the cam7 publisher
uses that format.

After adding the cam7 entry and a scoped camera-app restart attempt, the
supervisor relaunched Cosine and a raw 1280x800 Z16 packet was captured; see
`docs/open-rail-apple-status.md` for the restart response and evidence. This
does not establish the depth scale or alignment.

Static inspection of the installed camera library found a default D457
depth-to-color alignment path that reads device calibration tables and uses a
0.001 m/count projection constant. Its caller does not check the alignment
function's return value; some allocation failures can still publish the
untransformed buffer. The packet contains no alignment-success or depth-scale
metadata. Thus neither the alignment path nor its constant verifies this
particular frame's registration or physical depth scale.

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
