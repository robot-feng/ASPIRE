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

`HoldVerifier` compares a post-close baseline with timestamped observations
during and after the lift. It requires the object to rise with the tool and
remain at a consistent tool-relative position for at least three seconds.
The caller must establish that the initial object was on the table, validate
observation freshness, and supply thresholds and the vertical direction from
the calibrated robot setup. Missing detections, timestamp mismatches, or gaps
reset the evidence window; they do not count as a successful grasp.
