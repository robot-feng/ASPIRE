# Open-RAIL embodiment

ASPIRE's Open-RAIL integration lives in `src/cap_harness/open_rail`. It is an
additive embodiment boundary; the LIBERO benchmark adapter and its examples
remain unchanged. `OpenRailAdapter` adapts one typed Open-RAIL control runtime to
ASPIRE's `EnvironmentAdapter` contract. A robot-specific `OpenRailProfile`
supplies joint names, arm aliases, kinematic model, base transforms, and
end-effector links.

Perception and grasp generation stay in ASPIRE's provider layer. The adapter
does not select a VLA policy. A robot runtime implements `OpenRailControl` to
translate ASPIRE commands to Open-RAIL, return measured terminal state, and
provide calibrated RGB-D observations. This keeps the shared adapter independent
of any one Open-RAIL robot implementation.

The upstream Open-RAIL source is pinned at `third_party/open-rail`. This checkout
is a reference dependency; robot-specific profiles and ASPIRE integration code
belong in this repository.

## AgiBot G1

The generic 7-DoF arm contract is registered as `open_rail_7dof`. A usable G1
profile still needs the verified arm model and end-effector links, the actual
gripper descriptor, and a calibrated head RGB-D stream expressed in the robot
base frame. Those values are required by `OpenRailProfile` rather than guessed
from the robot name. `reset()` only adopts task metadata and observes; it never
homes or moves hardware. Physical commands remain interlocked unless a launcher
explicitly enables them.

The initial control scope is arm joints and grippers only. G1 head, waist, torso,
lift, and base values may appear in observations, but are not command channels.
Configure `controlled_arms=("right",)` for the initial G1 task. Both arm states
remain available to the planner; commands for the other arm are rejected before
they reach the Open-RAIL runtime.

`examples/open_rail/pick_apple.py` is a CAP program for the first planning
stage. Given an Open-RAIL CAP registry and explicit `open_rail_pick_apple` task
metadata, it reuses one observation to localize one apple and one plate, then
uses GraspGen-X and full-scene pregrasp planning for the apple. It returns a
preview only when the returned trajectory is marked collision-aware, and always
reports the full grasp task as incomplete. It sends no gripper or arm commands.

`cap_harness.open_rail.runner.run_apple_preview` composes that program with
WeVisOne, depth-mask refinement, GraspGen-X, cuRobo and the normal CAP runner
and recorder. It takes a caller-owned `OpenRailAdapter`, recorder, explicit
depth-filter thresholds, gripper sweep geometry, gripper-base-to-TCP transform
and control period. Service URLs can be supplied independently. The adapter
must be read-only and scoped to the right arm; the caller closes it and
finalizes the recorder. No model service or policy process is started by this
function. The inputs are not proof of physical calibration.

The same runner can optionally preview pregrasp, approach and lift for the
task's generated candidates. Set `preview_full_chain=True` and an explicit
positive `preview_scene_voxel_size_m` in the task metadata, and provide both
`chain_lift_offset_m` in the robot base frame and `chain_attachment` as an
`AttachedObjectGeometry` to the runner. The runner checks that the target
partition and grasp set use the same snapshot and mask. It tries candidates in
order, recording each planner response, and stops at the first complete chain.
The attachment and lift are assumptions: this preview does not execute motion,
verify contact, or establish that any trajectory is safe. G1 camera, TCP,
gripper, attachment and planner inputs remain subject to real verification.
`G1ObservationControl` connects a caller-owned read-only G1 runtime to the
adapter, converting the camera and both arms from the same raw snapshot. It
requires verified calibration, a verified common-clock reference and explicit
freshness limits; its command methods always refuse motion. The physical G1
calibration is not yet available. An experimental `G1ControlFacade` combines
the observation source with right-arm timed dispatch and measured feedback.
It requires a verified fresh-scene guard, independent base/head/waist checks,
bounded transport, and an in-flight monitor; no verified hardware transport or
monitor is supplied, so the existing G1 proxy cannot dispatch through it.
Stopping later waypoint dispatch does not cancel a waypoint already sent.
Contact, lift, holding verification and a complete execution launcher remain
to be integrated and verified on hardware.
When candidate generation succeeds, the recorder also writes
`media/grasp-candidates.ply` with the observed scene, framed candidate axes,
and approach directions. It is a visualization, not a collision or reachability
certificate, and does not depict a guessed physical gripper.

The read-only `examples/open_rail/observe.py` writes every frame immediately by
default. Optional `--buffered` snapshots are copied into memory and saved in
capture order after collection, including when collection raises an exception.
Buffering permits at most 200 frames and limits retained array and serialized
metadata sizes to 512 MiB by default; `--buffer-limit-mib` may set a positive
limit up to 1024 MiB. Python and runtime overhead are additional. Sudden process
termination or power loss can still lose frames not yet written, and this mode
does not establish a device sampling rate or timestamp synchronization.

For an observed target already attached to the tool, the Open-RAIL CAP API
can partition the target and environment using the target's pixels from the
same observation. Pass the resulting scene and partition together with the
attachment to a supporting cuRobo planner. The service verifies the transmitted
scene and avoids a second attachment-sphere carve; the robot-body carve may
still remove nearby obstacles, so the resulting plan is explicitly marked
non-executable. This is a preview capability and does not establish that
occluded space is free or that the segmentation and attachment are correct.

The CAP `plan_motion` option `preserve_observed_scene=True` also permits a
bound partition without an attachment for an approach preview. It retains all
observed background points and rejects scenes outside planner limits instead
of silently dropping them. Target contact and occluded space remain unverified,
so this option also produces only non-executable preview trajectories.

`HoldVerifier` compares a post-close baseline with timestamped observations
during and after the lift. It requires the object to rise with the tool and
remain at a consistent tool-relative position for at least three seconds.
Open-RAIL localization also carries a content fingerprint of the RGB-D image,
camera calibration and robot state; grasp generation and hold verification reject
a different observation even if it reuses the same timestamps.
The caller must establish that the initial object was on the table, validate
observation freshness, and supply thresholds and the vertical direction from
the calibrated robot setup. Missing detections, timestamp mismatches, or gaps
reset the evidence window; they do not count as a successful grasp.
Each hold sample must also carry the original target mask and point cloud. The
verifier reprojects the mask through that snapshot and recomputes its geometry;
inconsistent target evidence resets the window rather than proving a hold.
