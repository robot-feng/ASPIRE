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
model's sweep-box aperture proxy, matching the upstream
`SweepVolumeParams.jaw_width` convention. The model does not predict a physical
opening for each returned candidate: `GraspCandidate.width_m` is `None`. The
conditioning value is recorded separately in candidate metadata as
`sweep_box_open_extent_x_m` and must not be used as a measured gripper command.

The frame helpers `gripper_base_pose_to_tcp()` and `tcp_pregrasp_pose()` are
model-independent. Both require the caller to pass the measured
`T_gripper_base_tcp`; `tcp_pregrasp_pose()` also requires an explicit positive
distance. GraspGenX's OBB candidate builder points local +Z into the object, so
the helper retreats along local -Z in gripper-base coordinates, then reapplies
the base-to-TCP transform. This remains correct when the TCP has a rotational
offset.

Each candidate's metadata also records `approach_direction_in_frame`, a unit
vector pointing along the grasp-base +Z approach axis in
`approach_direction_frame` (the candidate pose's parent frame). It is computed
before applying the TCP transform, so it is not assumed to equal the TCP's +Z
axis. A frame conversion must rotate the vector as well as the pose. This
geometric metadata does not verify the physical TCP or a collision-free path.

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
files, not committed provider assets. Their local SHA-256 digests are `8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44`
and `cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e`,
respectively. The checked upstream tree does not provide expected checkpoint
digests for comparison.

On this Jetson, `/data/conda-envs/cap-harness-graspgenx` is an isolated
dependency overlay using the existing NVIDIA Torch 2.4 / CUDA 12.2 build.
Source its `activate_graspgenx.sh` to configure local assets and caches. An
offline service/client smoke test on Orin used an official sample point cloud
and the generic sweep parameters from the upstream README: the service returned
five candidates in about 5.2 seconds. The request, transforms, scores and service
log are in `artifacts/open_rail/graspgenx_generic_smoke_20261104/`. That
descriptor is only a software test; it is not an AgiBot G1 gripper model, and
the candidates were not checked against a scene or executed.

## Robot integration boundary

The installed Genie motion-control bundle contains a G1 description at
`/home/agi/app/bin/motion-control/share/robot_model/urdf/G1/model.urdf` with
7-joint left and right arms. The cuRobo service accepts an `agibot_g1` model
only when the absolute path of an external cuRobo V2 YAML is provided in
`CAP_HARNESS_CUROBO_G1_CONFIG`. The YAML must describe the actual robot,
including both arms, the inactive arm and the fixed head/torso state, tool
frames, and conservative collision geometry. Both arm states and their actual
joint names are required. A plan or IK solution is refused if its inactive arm
differs from the observed state; synchronized G1 planning is not enabled.
Loading a YAML does not establish physical correspondence or geometric coverage:
the service marks G1's model unverified and the client labels its plans
`collision_aware=False`, preventing the Open-RAIL adapter from executing them.
This is an offline integration path until evidence-backed model verification,
including fixed joint state and collision geometry, is implemented. `dual_panda` is
built from two Franka Panda descriptions, so its other-arm collision checks do
not describe this G1.

The optional `CAP_HARNESS_CUROBO_G1_JOINT_MAP` points to an absolute JSON file
whose `joint_name_map` maps every SDK joint name to its model joint name. The
service checks complete, unique coverage of both arms, binds the map for a
request, and returns trajectory columns in SDK order. With no map, names must
match exactly; there is no automatic name substitution. A candidate mapping for
the locally running `--robot=a2d_v01` profile is in
`src/cap_harness/configs/open_rail/agibot_g1_a2d_v01_joint_map.candidate.json`.
Its SDK names and URDF names appear in the local SDK and motion-control
configuration respectively; their actual correspondence and the model remain
unverified. The inspected motion-control description is
`/home/agi/app/bin/motion-control/share/robot_model/urdf/a2d_v01/model.urdf`
(SHA-256 `8d667a4f7e9b0b87d17314363d9df51a97818fd2957e15884a0a5b1dd1dc8cb8`),
and its ordered arm joints are listed in
`/home/agi/app/bin/motion-control/configuration/robot/a2d_v01/model_info/default.yaml`.
The candidate map is not enabled by default.

The installed Omnipicker control profiles name `left_gripper_link` and
`right_gripper_link` and give one revolute joint with range `[0, 1]` each, but
do not provide gripper geometry or a base-to-TCP transform. The inspected G1
URDF instead has fixed `hand_left_base_link` and `hand_right_base_link` links
with A2D hand mesh and collision geometry; it does not contain the Omnipicker
end-frame names.
Do not use those A2D hand shapes as an Omnipicker model. A head-camera extrinsic
file exists under `/data/parameters`, but its direction and reference frame
must be confirmed before using it as a camera-to-planning-frame transform.
The installed G1 URDF sets the origin of prismatic joint `idx01_body_joint1`
to z=0.3485 m; the locally found Dream OmniPicker URDF uses z=0.6485 m for the
same joint, a 0.300 m difference. Their reference frames and lift-zero
conventions must be verified before combining their robot-base transforms.

cuRobo requests can carry caller-supplied attached-object collision spheres.
These spheres and their contact-link exceptions are acknowledged by the service,
but their coverage and the scene carve's effect on nearby surfaces remain
unverified; acknowledgement alone does not establish collision clearance for an
apple held in the gripper.

Before executing generated poses on an AgiBot G1, ASPIRE still needs a
cuRobo-compatible config based on the installed G1 arm description, a
consistent joint/link mapping and tool frame, the Omnipicker
collision geometry and sweep descriptor, a measured gripper-base-to-TCP
transform, confirmation of the camera extrinsic convention, and planner support
for the held apple plus the inactive G1 arm. Until those are wired and verified,
use the provider only to generate offline candidates; do not treat them as
executable robot targets.
