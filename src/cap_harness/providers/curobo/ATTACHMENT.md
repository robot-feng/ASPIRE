# Explicit held-object geometry

`attachment.py` defines an optional held-object payload for the typed
`CuRoboProvider` and its service. With `attached_object` omitted, the existing
request, robot config, and scene paths remain unchanged. When present, the
service writes/caches a robot config with an explicit fixed object link and
collision spheres, gets its base pose from current-state FK, removes that
object's sphere-covered points from the world point cloud, and plans with both
the robot and object collision spheres. Attachment state is not stored in the
robot state; callers must provide it again for every held-object plan.

The caller must provide all of the following from calibration or a separately
validated model:

- the parent link and object link names;
- parent-to-object xyz and quaternion in `wxyz` order;
- local collision spheres in meters;
- only the gripper links that physically contact the held object and should be
  ignored for self-collision.

`merge_attachment_into_kinematics` copies an inline-sphere kinematics mapping,
adds an `extra_links` fixed joint, adds the object spheres under
`collision_spheres`, includes the object link in `tool_frames` and
`collision_link_names`, and records the explicit contact exclusions. The schema
is based on the pinned cuRobo source at `/data/ASPIRE/third_party/curobo`:
`KinematicsLoaderCfg.extra_links`, `LinkParams.create`,
`KinematicsLoaderCfg.extra_collision_spheres`, and the example attachment link
in `curobo/content/configs/robot/franka.yml`. The helper uses the fixed link's
seven-value transform and local sphere centers. A generic Franka service/client
smoke test with a synthetic attached sphere succeeded on Orin; this is not G1
or physical-scene verification. Service-side config
generation verifies the parent/contact names exist in the selected URDF and
that contact links are in the parent link's gripper subtree. It rejects
nonstandard `<loop_joint>` tags because the supported model is a tree. The
caller still owns confirming that the sphere union covers the whole object and
that the contact list contains no noncontact links.
Generated configs are keyed by source-config bytes, URDF bytes, and attachment
payload so distinct geometry does not reuse a planner cache entry.
The provider's existing planner cache retains planners by config path; unique
attachment transforms/geometry therefore create additional GPU planners for
the life of the service. This path is intended for a small number of stable
carry segments, not repeated high-variance grasp search until the service gains
bounded planner eviction.

`pose_in_base` composes a measured parent pose with the explicit rigid
attachment transform. `carve_attached_object_points` removes only points inside
the transformed supplied sphere union. This is suitable for removing observed
object points after attachment when the world scene still contains the object;
it may also remove tabletop or plate points wherever those points overlap the
sphere union plus padding. It does not prove that the support surface remains
intact, nor that the spheres cover the object's full collision shape. The
client diagnostics mark the geometry as unverified and expose ignored contact
links. A `collision_aware` result here means cuRobo checked the supplied sphere
model and filtered scene; it is not a claim of physical safety. It does not
replace target segmentation during approach. The optional request is available
through `CuRoboProvider.plan_to_pose`, `plan_to_joints`, and
`plan_synchronized_motion`. `CapApi.plan_motion` and `move_to_pose` accept an
explicit descriptor or mapping for a single-arm plan, reject planners without
attachment support, and reject simultaneous target-AABB exclusion. The
synchronized `CapApi` methods and `RobotState` do not carry attachment state.

## G1 candidate boundary

The local `/data/dreamlabs/g1_demo/tcp_setup/arms/g1-omnipicker.urdf` is a
candidate description, not a verified model for the active `a2d_v01` runtime.
Its corresponding mesh bundle is a separate ZIP, the URDF contains nonstandard
`loop_joint` elements, and the active motion-control configuration did not load
this URDF or its TCP JSON. The service rejects those loop tags before
generating an attachment config. The SDK's normalized gripper command is not a
verified mapping to every articulated/mimic joint in this URDF. The helper
therefore does not contain G1 joint defaults, a guessed TCP, or a ready-to-run
G1 robot config. The service model literal remains `panda`, `dual_panda`, or
`yam_real`; G1 cannot be selected through this path. No G1 hardware validation
was performed.

The existing target point cloud removal in `CapApi._exclude_target_from_scene`
uses an expanded target AABB. A local surface-distance alternative may preserve
more of a nearby table, but this branch deliberately does not change that
shared API path. The attachment carve implemented here is limited to the
caller-supplied held-object collision geometry.
