# Open-RAIL embodiment adapter

`src/cap_harness/open_rail/g1_runtime.py` is the G1-specific observation and
scoped-command runtime used alongside the generic Open-RAIL embodiment boundary.
It owns one `RobotProxy` and exposes raw RGB, raw Z16 depth when available,
source timestamps, and the configured state vector. The G1 observation profile
is `src/cap_harness/configs/open_rail/agibot_g1_observation.yaml`. Open-RAIL
remains an external dependency; its checkout is not modified.

The observation path is read-only. The separately gated command facade does
not yet implement ASPIRE's complete motion `EnvironmentAdapter` contract:
G1 gripper geometry, TCP, active
kinematic model, and camera/depth calibration are not verified. It keeps these
fields explicitly unset instead of deriving a `CameraObservation` or planning
scene from incomplete calibration.

## G1 evidence

Open-RAIL's committed G1 configuration and `RobotDds` establish this 20-value
state layout:

| Slice | Meaning | Units |
| --- | --- | --- |
| `0:7` | left arm | rad |
| `7:14` | right arm | rad |
| `14:16` | left/right grippers | Open-RAIL normalized feedback |
| `16:18` | head yaw/pitch, read-only | rad |
| `18:20` | waist pitch/lift, read-only | rad, m |

The ASPIRE control boundary accepts the 14 arm joints and gripper positions
only. It does not issue head, waist, torso, lift, or base commands.

`RobotDds` names the arm joints `Joint1_l` through `Joint7_l`, then
`Joint1_r` through `Joint7_r`. Open-RAIL maps normalized gripper commands as
`0=open`, `1=closed`; its body computes normalized feedback with
`clip(raw / 120, 0, 1)`. The configured URDF is only a candidate: the installed
Pinocchio helper defaults to a different robot description, and no G1 gripper
TCP is defined in the profile.

One read-only capture is stored under
`artifacts/open_rail/observe/snapshot.json`. It had
an RGB frame of 1280x800 and a state vector, but no depth packet near that RGB
timestamp. The camera reference configuration lists 1280x720 color and
640x360 depth. Existing head intrinsic/extrinsic files do not declare the
depth scale, depth-to-color registration, or extrinsic frame semantics. The
capture therefore remains raw evidence; it is not a calibrated
`CameraObservation` or a metric point cloud.

For the later live RGB-D capture, corrected packet-aware RGB decoding and depth
startup evidence, see `docs/open-rail-apple-status.md`. Calibration remains
incomplete.

The Open-RAIL worker did not exit during its five-second graceful-close window
after this capture; `RobotProxy.close()` terminated it. No robot control,
reset, home, or gripper call was made. The proxy version's shared-memory
tracking was extended for its `FramePacket` transport so worker termination
can still unlink frame slots.

## Read-only capture

From an ASPIRE checkout with the Open-RAIL/A2D runtime configured:

```bash
python -m examples.open_rail.observe \
  --open-rail-root third_party/open-rail \
  --output artifacts/open_rail/observe --require-depth
```

`--require-depth` skips initial RGB-only frames until the timeout. It does not
certify RGB-depth registration, metric scale, or synchronization.
