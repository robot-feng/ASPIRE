# GraspGenX provider

ASPIRE's `GraspGenXProvider` sends a segmented object point cloud and a
per-gripper sweep-volume-v2 descriptor to the isolated `graspgenx` service. The
service loads the official GraspGenX source and checkpoints and returns scored
poses through the existing ASPIRE `GraspProvider` contract.

The input point cloud must contain XYZ values in metres and declare its frame.
The output pose keeps that frame. Transform the cloud into the planner's base
frame before requesting grasps, or transform the returned poses with a verified
camera-to-base transform before IK.

The upstream `infer_object` contract returns each grasp as a 4×4 transform in
the input point cloud's frame. The transform describes the GraspGenX **gripper
base**. ASPIRE stores this frame in `GraspCandidate.pose` by default. When a
caller has a measured transform `T_gripper_base_tcp`, it can pass it as
`gripper_base_to_tcp`; the provider then composes
`T_cloud_tcp = T_cloud_gripper_base @ T_gripper_base_tcp`. The candidate remains
in the same point-cloud frame, but its pose now denotes the TCP. There is no
default transform: do not use this option until the physical gripper and TCP
have been measured and the planner's end-effector link is confirmed to use that
TCP.

The descriptor's `extents_open`, `offset_open`, `extents_mid`, and `offset_mid`
are meters in GraspGenX's gripper-base frame (+Z approach, +X closing). They
must describe the actual hardware. The provider validates their shape and
values, but cannot establish their physical provenance. Do not substitute an
unrelated vendor's `G1` descriptor. The provider uses `extents_open[0]` as the
candidate opening width, matching the upstream `SweepVolumeParams.jaw_width`
convention.

The frame helpers `gripper_base_pose_to_tcp()` and `tcp_pregrasp_pose()` are
model-independent. Both require the caller to pass the measured
`T_gripper_base_tcp`; `tcp_pregrasp_pose()` also requires an explicit positive
distance. GraspGenX's OBB candidate builder points local +Z into the object, so
the helper retreats along local -Z in gripper-base coordinates, then reapplies
the base-to-TCP transform. This remains correct when the TCP has a rotational
offset.

GraspGenX does not collision-check returned poses against the rest of the scene.
The provider marks this in result diagnostics; ASPIRE's planner remains
responsible for checking candidate approach and retreat trajectories against the
scene and robot geometry.

The upstream inference API and frame semantics are documented in
`third_party/GraspGenX/client-server/README.md`. Checkpoints remain local under
`third_party/models/graspgenx-checkpoints` and must include both the generator
and discriminator config and weights before the service can start.

At the time this provider was checked, the runtime contains both upstream
checkpoint files (`gen/epoch_736.pth`, 1,210,918,342 bytes;
`dis/epoch_1056.pth`, 483,889,478 bytes) and their configs under
`/data/ASPIRE/third_party/models/graspgenx-checkpoints/release`. These are local
files, not committed provider assets. Model inference has not been run as part
of the CPU-only checks. Their local SHA-256 digests are `8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44`
and `cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e`,
respectively. The checked upstream tree does not provide expected checkpoint
digests for comparison.

## Robot integration boundary

The installed Genie motion-control bundle contains a G1 description at
`/home/agi/app/bin/motion-control/share/robot_model/urdf/G1/model.urdf` with
7-joint left and right arms. ASPIRE's cuRobo service does not load that model;
it currently accepts `panda`, `dual_panda`, and `yam_real`. `dual_panda` is
built from two Franka Panda descriptions, so its other-arm collision checks do
not describe this G1.

The installed Omnipicker control profiles name `left_gripper_link` and
`right_gripper_link` and give one revolute joint with range `[0, 1]` each, but
do not provide gripper geometry or a base-to-TCP transform. The inspected G1
URDF instead has fixed `hand_left_base_link` and `hand_right_base_link` links
with A2D hand mesh and collision geometry; it does not contain the Omnipicker
end-frame names.
Do not use those A2D hand shapes as an Omnipicker model. A head-camera extrinsic
file exists under `/data/parameters`, but its direction and reference frame
must be confirmed before using it as a camera-to-planning-frame transform.

cuRobo requests carry an observed scene point cloud but no attached-object
geometry. The service checks that world scene and the configured robot's own
collision geometry; `collision_aware=True` does not establish collision
checking for an apple held in the gripper.

Before executing generated poses on an AgiBot G1, ASPIRE still needs a
cuRobo-compatible config based on the installed G1 arm description, a
consistent joint/link mapping and tool frame, the Omnipicker
collision geometry and sweep descriptor, a measured gripper-base-to-TCP
transform, confirmation of the camera extrinsic convention, and planner support
for the held apple plus the inactive G1 arm. Until those are wired and verified,
use the provider only to generate offline candidates; do not treat them as
executable robot targets.
