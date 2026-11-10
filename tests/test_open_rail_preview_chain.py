from dataclasses import replace

import numpy as np
import pytest

from cap_harness.contracts import Pose, RobotPlanningContext, Trajectory
from cap_harness.open_rail.preview_chain import _History, _context_id, _endpoint, preview_pick_chain
from cap_harness.providers.curobo import service
from cap_harness.providers.curobo.client import CuRoboProvider, _validated_partition_payload
from cap_harness.providers.wire import decode_numpy, encode_numpy
from test_open_rail_scene_partition import _attachment, _fixture, _partition


class Session:
    def __init__(self, *, fail_stage=None, bad_start=False):
        self.calls = []
        self.fail_stage = fail_stage
        self.bad_start = bad_start

    def request(self, _method, _url, **kwargs):
        payload = kwargs['json']
        self.calls.append(payload)
        stage = len(self.calls)
        request = service.PlanRequest.model_validate(payload)
        if stage >= self.fail_stage if self.fail_stage is not None else False:
            return type('Response', (), {'status_code': 500, 'text': 'failure', 'json': lambda self: {}})()
        start = np.array(payload['joint_positions']['right'])
        if self.bad_start and stage == 2:
            start += 0.1
        end = start + 0.02
        data = {'joint_positions_base64': encode_numpy(np.stack([start, end])), 'dt_s': 0.1,
                'attachment_diagnostics': service._attachment_diagnostics(request),
                'scene_partition_diagnostics': service._scene_partition_diagnostics(request),
                'collision_diagnostics': {'collision_disabled': False, 'scene_supplied': True,
                    'world_collision_check_enabled': True, 'self_collision_check_enabled': True,
                    'robot_model_verified': True}}
        return type('Response', (), {'status_code': 200, 'text': '', 'json': lambda self: data})()


def setup():
    observation, localization = _fixture()
    partition = _partition(observation, localization)
    state = observation.robot_state
    context = RobotPlanningContext(state.embodiment, 'agibot_g1', state.joint_names,
                                   {'right': np.eye(4)}, {'right': 'tcp'})
    pose = Pose([0, 0, 1], [1, 0, 0, 0], 'base')
    return partition, context, pose


def run(session):
    partition, context, pose = setup()
    result = preview_pick_chain(CuRoboProvider(session=session), partition, context, arm='right',
                                pregrasp=pose, approach=pose, lift=pose, attached_object=_attachment())
    return partition, context, result


def test_three_stages_seeded_from_actual_prior_endpoint_and_never_executable():
    session = Session()
    partition, _, result = run(session)
    assert result.completed and [stage.name for stage in result.stages] == ['pregrasp', 'approach', 'lift']
    plans = result.stages
    assert all(not stage.diagnostics['collision_aware'] for stage in plans)
    assert all(not isinstance(stage, Trajectory) for stage in plans)
    for i in (1, 2):
        assert np.array_equal(session.calls[i]['joint_positions']['right'], plans[i-1].waypoints[-1])
        assert np.array_equal(plans[i].start_joints, plans[i-1].waypoints[-1])
        assert result.stages[i].diagnostics['planning_start_is_hypothetical']
    assert len(decode_numpy(session.calls[0]['scene_points_base64'])) == len(partition.scene.point_cloud.points) + len(partition.target.points)
    assert len(decode_numpy(session.calls[1]['scene_points_base64'])) == len(partition.scene.point_cloud.points)
    assert session.calls[1]['target_scene_partition'] == session.calls[2]['target_scene_partition']
    assert 'attached_object' not in session.calls[1] and 'attached_object' in session.calls[2]
    assert all(not entry.diagnostics['target_reobserved'] for entry in result.stages)
    assert any('grasp success is assumed' in item for item in result.assumptions)


def test_rejects_fake_start_changed_context_and_changed_partition():
    session = Session()
    partition, context, result = run(session)
    first_stage = result.stages[0]
    first = Trajectory(np.asarray(first_stage.waypoints), 0.1, first_stage.joint_names, 'test', False,
                       partition.source_observation.robot_state, arm='right',
                       gripper_positions=np.ones(len(first_stage.waypoints)), embodiment='open_rail_7dof')
    history = _History(partition.snapshot_fingerprint, partition.scene_fingerprint, _context_id(context), 'right', (first,))
    state = _endpoint(partition.source_observation.robot_state, first,
                      partition.source_observation.cameras[partition.target_camera], 'right')
    def verify(part, st, ctx, hist=history):
        return _validated_partition_payload(part, part.scene, st, None, preserve_observed_scene=True,
                                            preview_history=hist, context=ctx, arm='right')
    verify(partition, state, context)
    with pytest.raises(ValueError, match='previous trajectory endpoint'):
        verify(partition, replace(state, joint_positions={'right': np.ones(7)}), context)
    with pytest.raises(ValueError, match='context'):
        verify(partition, state, replace(context, model='other'))
    with pytest.raises(ValueError, match='source observation'):
        verify(replace(partition, scene_fingerprint='f' * 64), state, context)
    with pytest.raises(ValueError, match='preceding planning state'):
        verify(partition, state, context, replace(history, trajectories=(first, first)))


def test_failures_stop_chain_and_invalid_first_waypoint_is_rejected():
    session = Session(fail_stage=2)
    _, _, result = run(session)
    assert not result.completed and len(session.calls) >= 2 and not result.stages[-1].ok
    with pytest.raises(ValueError, match='first waypoint'):
        run(Session(bad_start=True))
