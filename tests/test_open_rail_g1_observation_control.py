from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import threading

import numpy as np
import pytest

from cap_harness.contracts import Pose, TaskContext
from cap_harness.open_rail import OpenRailAdapter, OpenRailProfile
from cap_harness.open_rail.g1_camera import G1CameraCalibration, G1DynamicCameraCalibration, G1HeadKinematics, G1RgbdRectification
from cap_harness.open_rail.g1_observation import G1ObservationControl
from cap_harness.open_rail.g1_broadcast import G1SnapshotBroadcaster
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailG1ObservationRuntime, OpenRailSnapshot
from cap_harness.open_rail.g1_state import G1StateKinematics


def _profile():
    names = {arm: tuple(f"{arm}_{i}" for i in range(7)) for arm in ("left", "right")}
    return OpenRailProfile(
        name="synthetic", robot_type="agibot_g1", model="test", base_frame="base",
        arms=("left", "right"), joint_names=names,
        end_effector_links={arm: f"{arm}_tcp" for arm in names},
        base_transforms={arm: np.eye(4) for arm in names}, arm_aliases={"primary": "right"},
    )


def _snapshot():
    names = _profile().joint_names
    return OpenRailSnapshot(
        rgb=np.full((3, 4, 3), 20, dtype=np.uint8),
        depth_u16=np.full((3, 4), 1000, dtype=np.uint16),
        state=np.array([*range(14), 0.2, 0.7, 0.0, 0.0, 0.0, 0.0]),
        arm_velocities_rad_s=np.arange(14) / 10,
        rgb_timestamp_ns=1_000_000_000, depth_timestamp_ns=1_000_100_000,
        state_timestamps_ns={name: 1_000_000_000 for name in ("arm", "head", "waist", "gripper_left", "gripper_right")},
        depth_error=None,
        metadata={
            "robot_type": "agibot_g1", "public_camera_key": "head", "rgb_order": "RGB",
            "rgb_packet_timestamp_ns": 1_000_000_000,
            "rgb_encoding_format": "UNCOMPRESSED", "rgb_color_format": "RGB", "rgb_decoder": "raw-rgb",
            "depth_encoding": "RS2_FORMAT_Z16", "state_timestamps_bind_state_values": True,
            "maximum_rgb_depth_skew_ms": 1.0,
            "arm_joint_names": [*names["left"], *names["right"]],
            "state_layout": {
                "left_arm_rad": [0, 7], "right_arm_rad": [7, 14],
                "left_gripper_normalized": [14, 15], "right_gripper_normalized": [15, 16],
                "head_yaw_pitch_rad": [16, 18], "waist_pitch_rad_lift_m": [18, 20],
            },
        },
    )


def _camera_calibration():
    return G1CameraCalibration(
        depth_scale_m_per_unit=0.001, depth_scale_verified=True,
        depth_registered_to_rgb=True, depth_registration_verified=True,
        intrinsics=np.array([[100, 0, 2], [0, 100, 1], [0, 0, 1]]),
        intrinsics_resolution_hw=(3, 4), intrinsics_verified=True,
        rgb_depth_rectified_for_pinhole=True, rectification_verified=True,
        base_from_camera=np.eye(4), camera_to_base_verified=True,
        public_camera_key="head", base_frame="base", camera_frame="head_optical",
        maximum_snapshot_skew_ms=1.0, synchronization_threshold_verified=True,
        calibration_head_yaw_pitch_rad=(0, 0), head_yaw_pitch_tolerance_rad=(0.01, 0.01),
        calibration_waist_pitch_lift_m=(0, 0), waist_pitch_lift_tolerance=(0.01, 0.01),
        mount_posture_verified=True, invalid_depth_values=(0,), invalid_depth_values_verified=True,
    )


def _kinematics():
    return G1StateKinematics(
        model="test", verified=True, joint_order_verified=True,
        position_velocity_units_verified=True, gripper_feedback_verified=True,
        synchronization_verified=True, timestamps_share_clock_verified=True,
        wire_joint_names=_profile().joint_names,
        end_effector_pose=lambda arm, link, joints, head, waist: Pose(
            [joints[0], 0, 0], [1, 0, 0, 0], "base"
        ),
    )


class _Runtime:
    allow_physical_motion = False

    def __init__(self, snapshot=None):
        self.snapshot = snapshot or _snapshot()
        self.reads = 0
        self.closed = False

    def read_snapshot(self, *, require_depth):
        assert require_depth is True
        self.reads += 1
        return self.snapshot

    def close(self):
        self.closed = True


def _control(runtime, **kwargs):
    options = dict(clock_now_ns=lambda: 1_001_000_000, clock_verified=True,
                   max_snapshot_age_ns=2_000_000)
    return G1ObservationControl(runtime, _profile(), _camera_calibration(), _kinematics(), **(options | kwargs))


def test_read_only_control_supplies_one_bound_snapshot_to_adapter():
    runtime = _Runtime()
    control = _control(runtime)
    task = TaskContext("open_rail", 0, "apple", "apple", "real")
    adapter = OpenRailAdapter(control, _profile(), task_context=task, controlled_arms=("right",))
    observation = adapter.reset()
    assert runtime.reads == 1
    assert observation.cameras["head"].timestamp_s == 1.0
    assert observation.robot_state.timestamp_s == 1.0
    np.testing.assert_array_equal(observation.cameras["head"].depth_m, np.ones((3, 4)))
    np.testing.assert_array_equal(observation.robot_state.joint_positions["right"], np.arange(7, 14))
    assert observation.robot_state.gripper_positions["right"] == pytest.approx(0.3)
    assert not control.set_gripper(1.0, arm="right").ok
    assert not control.execute_trajectory(None).ok
    assert runtime.reads == 1
    adapter.close()
    assert runtime.closed


def test_guard_input_uses_the_exact_same_validated_snapshot_without_a_second_read():
    runtime = _Runtime()
    observation, snapshot = _control(runtime).get_observation_with_snapshot()
    assert snapshot is runtime.snapshot and runtime.reads == 1
    assert observation.cameras["head"].timestamp_s == snapshot.rgb_timestamp_ns / 1e9
    np.testing.assert_array_equal(observation.robot_state.joint_positions["right"], snapshot.state[7:14])


def test_read_only_control_passes_explicit_rectification_without_mutating_snapshot():
    runtime = _Runtime()
    rectification = G1RgbdRectification(
        raw_intrinsics=_camera_calibration().intrinsics,
        distortion_coefficients=np.zeros(5),
        raw_intrinsics_resolution_hw=(3, 4), camera_key="head",
        distortion_model="opencv_pinhole_brown_conrady",
        raw_intrinsics_verified=True, distortion_verified=True,
        input_is_distorted_rgb_verified=True,
        depth_registered_to_distorted_rgb_verified=True,
        depth_is_rgb_axial_z_verified=True,
    )
    observation = _control(runtime, rectification=rectification).get_observation()
    np.testing.assert_array_equal(observation.cameras["head"].rgb, runtime.snapshot.rgb)
    np.testing.assert_array_equal(observation.cameras["head"].depth_m, np.ones((3, 4)))
    np.testing.assert_array_equal(runtime.snapshot.depth_u16, np.full((3, 4), 1000))
    assert runtime.reads == 1


def test_observation_control_accepts_broadcast_without_second_camera_reader(tmp_path, monkeypatch):
    class Proxy:
        def __init__(self):
            self.frames = Queue()
            self.closed = False

        def retrieve_observation(self):
            try:
                return self.frames.get_nowait()
            except Empty:
                return None

        def close(self):
            self.closed = True

    proxy = Proxy()
    runtime = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path, proxy=proxy, bound_state_sampling=True,
        maximum_rgb_depth_skew_ms=1,
    )
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=1, close_timeout_s=2)
    broker.start()
    adapter = OpenRailAdapter(_control(broker), _profile(), controlled_arms=("right",))
    waiting = threading.Event()
    original_wait = broker._condition.wait

    def signal_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(broker._condition, "wait", signal_wait)
    sample = _snapshot()
    payload = {
        "cam.head": sample.rgb, "cam.head_depth": sample.depth_u16,
        "obs.state": sample.state, "rgb_timestamp_ns": sample.rgb_timestamp_ns,
        "rgb_packet_timestamp_ns": sample.rgb_timestamp_ns, "rgb_order": "RGB",
        "rgb_encoding_format": "UNCOMPRESSED", "rgb_color_format": "RGB", "rgb_decoder": "raw-rgb",
        "depth_timestamp_ns": sample.depth_timestamp_ns, "depth_encoding": "RS2_FORMAT_Z16",
        "state_timestamps_ns": sample.state_timestamps_ns, "state_timestamps_bind_state_values": True,
        "bound_arm_sample": {"names": sample.metadata["arm_joint_names"],
                             "velocities_raw": sample.arm_velocities_rad_s, "sides": {}},
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(adapter.reset)
        assert waiting.wait(1)
        proxy.frames.put(payload)
        observation = result.result(timeout=1)
    np.testing.assert_array_equal(observation.robot_state.joint_positions["right"], np.arange(7, 14))
    assert observation.cameras["head"].timestamp_s == 1.0
    assert broker.feedback_samples.wait_next_right_gripper(0, 0)[0] == 1_000_000_000
    adapter.close()
    assert proxy.closed


@pytest.mark.parametrize("now", [999_000_000, 1_100_000_000])
def test_control_rejects_future_or_stale_snapshot(now):
    runtime = _Runtime()
    with pytest.raises(OpenRailG1ObservationError, match="stale or from the future"):
        _control(runtime, clock_now_ns=lambda: now).get_observation()


def test_control_rejects_unverified_calibration_and_unbound_values():
    runtime = _Runtime(replace(_snapshot(), metadata={**_snapshot().metadata, "state_timestamps_bind_state_values": False}))
    with pytest.raises(OpenRailG1ObservationError, match="not synchronized"):
        _control(runtime).get_observation()
    control = G1ObservationControl(
        _Runtime(), _profile(), replace(_camera_calibration(), depth_scale_verified=False), _kinematics(),
        clock_now_ns=lambda: 1_001_000_000, clock_verified=True, max_snapshot_age_ns=2_000_000,
    )
    with pytest.raises(OpenRailG1ObservationError, match="depth_scale_verified"):
        control.get_observation()


def test_control_checks_arm_skew_against_verified_camera_limit():
    snapshot = _snapshot()
    snapshot = replace(
        snapshot,
        state_timestamps_ns={**snapshot.state_timestamps_ns, "arm": 999_000_000},
        metadata={**snapshot.metadata, "maximum_rgb_depth_skew_ms": 50.0},
    )
    camera = replace(_camera_calibration(), maximum_snapshot_skew_ms=0.5)
    control = G1ObservationControl(
        _Runtime(snapshot), _profile(), camera, _kinematics(),
        clock_now_ns=lambda: 1_001_000_000, clock_verified=True, max_snapshot_age_ns=5_000_000,
    )
    with pytest.raises(OpenRailG1ObservationError, match="verified synchronization threshold"):
        control.get_observation()


def test_control_requires_read_only_runtime_and_verified_clock():
    runtime = _Runtime()
    runtime.allow_physical_motion = True
    with pytest.raises(ValueError, match="read-only"):
        _control(runtime)
    with pytest.raises(ValueError, match="verified common-clock"):
        _control(_Runtime(), clock_verified=False)


def test_control_composes_moving_head_camera_pose_from_the_same_snapshot():
    runtime = _Runtime()
    calibration = G1DynamicCameraCalibration(
        depth_scale_m_per_unit=0.001, depth_scale_verified=True,
        depth_registered_to_rgb=True, depth_registration_verified=True,
        intrinsics=np.eye(3), intrinsics_resolution_hw=(3, 4), intrinsics_verified=True,
        rgb_depth_rectified_for_pinhole=True, rectification_verified=True,
        mount_from_camera=np.eye(4), mount_from_camera_verified=True,
        public_camera_key="head", base_frame="base", mount_frame="head_mount",
        camera_frame="head_optical", maximum_snapshot_skew_ms=1,
        synchronization_threshold_verified=True, timestamps_share_clock_verified=True,
        state_layout_verified=True, state_units_verified=True, fk_model_id="test",
        invalid_depth_values=(0,), invalid_depth_values_verified=True,
    )

    def fk(head, waist):
        result = np.eye(4)
        result[0, 3] = head[0]
        return result

    head = G1HeadKinematics(
        base_frame="base", mount_frame="head_mount", model_id="test", verified=True,
        state_layout_verified=True, state_units_verified=True, base_from_mount=fk,
    )
    control = G1ObservationControl(
        runtime, _profile(), calibration, _kinematics(), head_kinematics=head,
        clock_now_ns=lambda: 1_001_000_000, clock_verified=True, max_snapshot_age_ns=2_000_000,
    )
    first = control.get_observation()
    values = runtime.snapshot.state.copy()
    values[16] = 0.2
    runtime.snapshot = replace(runtime.snapshot, state=values)
    second = control.get_observation()
    assert first.cameras["head"].camera_pose.position[0] == pytest.approx(0)
    assert second.cameras["head"].camera_pose.position[0] == pytest.approx(0.2)
    assert runtime.reads == 2
