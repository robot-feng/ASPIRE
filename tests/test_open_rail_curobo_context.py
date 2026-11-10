"""Open-RAIL state must not be planned using another robot's joint mapping."""

from dataclasses import replace

import numpy as np
import pytest

from cap_harness.contracts import PlanningScene, PointCloud, Pose, RobotPlanningContext, RobotState
from cap_harness.errors import ErrorCode
from cap_harness.providers.curobo import CuRoboProvider


class _Session:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        raise AssertionError("invalid planning context reached the service")


def _state() -> RobotState:
    arms = ("left", "right")
    pose = Pose([0, 0, 0], [1, 0, 0, 0], frame="base")
    return RobotState(
        joint_positions={a: np.zeros(7) for a in arms},
        joint_velocities={a: np.zeros(7) for a in arms},
        end_effector_poses={a: pose for a in arms},
        gripper_positions={a: 0.5 for a in arms},
        joint_names={a: tuple(f"{a}_joint{i}" for i in range(7)) for a in arms},
        base_frame="base",
        embodiment="open_rail_7dof",
    )


def _context(state: RobotState) -> RobotPlanningContext:
    return RobotPlanningContext(
        embodiment=state.embodiment,
        model="agibot_g1",
        joint_names=state.joint_names,
        base_transforms={a: np.eye(4) for a in state.arms},
        end_effector_links={a: f"{a}_tcp" for a in state.arms},
    )


@pytest.mark.parametrize("defect", ["embodiment", "arm_set", "joint_order", "panda", "yam"])
def test_g1_context_mismatch_never_reaches_ik_or_planning(defect: str) -> None:
    state = _state()
    context = _context(state)
    if defect == "embodiment":
        context = replace(context, embodiment="robosuite")
    elif defect == "arm_set":
        context = replace(
            context,
            joint_names={"right": context.joint_names["right"]},
            base_transforms={"right": context.base_transforms["right"]},
            end_effector_links={"right": context.end_effector_links["right"]},
        )
    elif defect == "joint_order":
        context = replace(context, joint_names={**context.joint_names, "right": tuple(reversed(context.joint_names["right"]))})
    elif defect == "panda":
        context = replace(context, model="dual_panda")
    else:
        context = replace(context, model="yam_real")

    scene = PlanningScene(PointCloud([[0.1, 0.2, 0.3]], frame="base"))
    session = _Session()
    provider = CuRoboProvider(session=session)  # type: ignore[arg-type]
    target = state.end_effector_poses["right"]
    ik = provider.solve_ik_with_context(target, state, arm="right", context=context)
    plan = provider.plan_to_pose(state, target, arm="right", scene=scene, context=context)
    sync = provider.plan_synchronized_motion(state, state.end_effector_poses, scene=scene, context=context)

    expected = ErrorCode.UNSUPPORTED if defect in {"panda", "yam"} else ErrorCode.INVALID_REQUEST
    assert ik.error.code is expected
    assert plan.error.code is expected
    assert sync.error.code is expected
    assert session.calls == []


def test_g1_ik_without_context_does_not_fall_back_to_panda() -> None:
    state = _state()
    session = _Session()
    result = CuRoboProvider(session=session).solve_ik(  # type: ignore[arg-type]
        state.end_effector_poses["right"], state, arm="right"
    )

    assert result.error.code is ErrorCode.UNSUPPORTED
    assert session.calls == []


def test_explicit_g1_context_reaches_service_without_model_substitution() -> None:
    class _Response:
        status_code = 422
        text = '{"detail":"model not supported"}'

        def json(self):
            return {"detail": "model not supported"}

    class _Refusal(_Session):
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return _Response()

    state = _state()
    session = _Refusal()
    result = CuRoboProvider(session=session).solve_ik_with_context(  # type: ignore[arg-type]
        state.end_effector_poses["right"], state, arm="right", context=_context(state)
    )

    assert not result.ok
    assert result.error.code is ErrorCode.IK_FAILED
    assert len(session.calls) == 1
    assert session.calls[0][2]["json"]["model"] == "agibot_g1"
