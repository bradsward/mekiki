"""Tests for `mekiki.checks.idle`, built on trajectories with known idle spans."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.checks.idle import MotionChannel, analyze_idle_time
from mekiki.episode import Episode, EpisodeMetadata, Frame, Pose, Proprioception
from tests.conftest import CLEAN_ACTION_SPACE

DT = 0.1


def _frame(
    i: int,
    x: float,
    *,
    angle: float = 0.0,
    gripper: float = 0.0,
    joint: float = 0.0,
    pose_frame: str = "base",
    time: float | None = None,
) -> Frame:
    half = angle / 2.0
    orientation = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return Frame(
        timestamp=i * DT if time is None else time,
        proprioception=Proprioception(
            joint_positions=np.array([joint, 0.0]),
            joint_velocities=None,
            ee_poses={"ee": Pose(np.array([x, 0.0, 0.0]), orientation, pose_frame)},
            grippers={"ee": gripper},
            extra={"raw": np.array([x, 0.0])},
        ),
        action=np.zeros(2),
        images={},
        is_first=(i == 0),
        is_last=False,
    )


def _episode(frames: list[Frame]) -> Episode:
    meta = EpisodeMetadata("e", "synthetic", "x", CLEAN_ACTION_SPACE, "synthetic")
    return Episode(metadata=meta, frames=frames)


def _from_x(xs: list[float], **kw: list[float]) -> Episode:
    return _episode([_frame(i, x, **{k: v[i] for k, v in kw.items()}) for i, x in enumerate(xs)])


# 1 cm per step at 0.1 s = 0.1 m/s while moving; 0 while still
POS = MotionChannel("position", 0.01, end_effector="ee")


def _profile(lead: int, move: int, mid: int, move2: int, trail: int) -> list[float]:
    xs = [0.0] * lead
    x = 0.0
    for _ in range(move):
        x += 0.01
        xs.append(x)
    xs += [x] * mid
    for _ in range(move2):
        x += 0.01
        xs.append(x)
    xs += [x] * trail
    return xs


def test_finds_leading_interior_and_trailing_idle_with_known_durations() -> None:
    result = analyze_idle_time(_from_x(_profile(10, 20, 8, 20, 12)), (POS,), min_idle_seconds=0.3)
    kinds = [s.kind for s in result.segments]
    assert kinds == ["leading", "interior", "trailing"]
    lead, mid, trail = result.segments
    # 10 still frames after frame 0 -> the run spans frames 0..9 = 0.9 s
    assert (lead.start_index, lead.end_index) == (0, 9)
    assert lead.duration_seconds == pytest.approx(0.9)
    assert mid.duration_seconds == pytest.approx(0.8)
    assert trail.duration_seconds == pytest.approx(1.2)
    assert result.leading_seconds == pytest.approx(0.9)
    assert result.interior_seconds == pytest.approx(0.8)
    assert result.trailing_seconds == pytest.approx(1.2)
    assert result.recoverable_fraction == pytest.approx((0.9 + 1.2) / result.total_seconds)
    assert result.interior_fraction == pytest.approx(0.8 / result.total_seconds)


def test_a_moving_episode_has_no_idle_segments() -> None:
    result = analyze_idle_time(_from_x([0.01 * i for i in range(40)]), (POS,), min_idle_seconds=0.3)
    assert result.segments == ()
    assert result.recoverable_fraction == 0.0
    assert result.peak_speed["position:ee"] == pytest.approx(0.1)


def test_runs_shorter_than_the_minimum_are_ignored() -> None:
    xs = _profile(0, 10, 2, 10, 0)  # a 0.2 s pause in the middle
    assert analyze_idle_time(_from_x(xs), (POS,), min_idle_seconds=0.3).segments == ()
    kept = analyze_idle_time(_from_x(xs), (POS,), min_idle_seconds=0.15).segments
    assert [s.kind for s in kept] == ["interior"]


def test_an_episode_that_never_moves_is_a_single_whole_episode_segment() -> None:
    result = analyze_idle_time(_from_x([0.0] * 20), (POS,), min_idle_seconds=0.3)
    (segment,) = result.segments
    assert segment.kind == "whole_episode"
    assert result.recoverable_fraction == pytest.approx(1.0)
    assert result.peak_speed["position:ee"] == 0.0


def test_the_threshold_is_strict_and_decides_what_counts_as_moving() -> None:
    xs = _profile(10, 10, 0, 0, 10)  # 0.1 m/s while moving
    loose = analyze_idle_time(
        _from_x(xs), (MotionChannel("position", 0.5, end_effector="ee"),), min_idle_seconds=0.3
    )
    assert [s.kind for s in loose.segments] == ["whole_episode"]  # 0.1 <= 0.5, nothing moved
    tight = analyze_idle_time(_from_x(xs), (POS,), min_idle_seconds=0.3)
    assert [s.kind for s in tight.segments] == ["leading", "trailing"]


def test_a_gripper_moving_on_a_still_arm_is_not_idle_when_declared() -> None:
    xs = [0.0] * 20
    grip = [0.0] * 5 + [0.1 * k for k in range(1, 10)] + [0.9] * 6
    arm_only = analyze_idle_time(_from_x(xs, gripper=grip), (POS,), min_idle_seconds=0.3)
    assert [s.kind for s in arm_only.segments] == ["whole_episode"]
    both = analyze_idle_time(
        _from_x(xs, gripper=grip),
        (POS, MotionChannel("gripper", 0.05, end_effector="ee")),
        min_idle_seconds=0.3,
    )
    assert [s.kind for s in both.segments] == ["leading", "trailing"]
    assert both.peak_speed["gripper:ee"] == pytest.approx(1.0)


def test_orientation_channel_uses_angular_speed_in_radians_per_second() -> None:
    angles = [0.0] * 8 + [0.05 * k for k in range(1, 11)] + [0.5] * 8
    result = analyze_idle_time(
        _from_x([0.0] * 26, angle=angles),
        (MotionChannel("orientation", 0.1, end_effector="ee"),),
        min_idle_seconds=0.3,
    )
    assert result.peak_speed["orientation:ee"] == pytest.approx(0.5)  # 0.05 rad / 0.1 s
    assert [s.kind for s in result.segments] == ["leading", "trailing"]


def test_joint_and_extra_channels() -> None:
    joints = [0.0] * 6 + [0.02 * k for k in range(1, 9)] + [0.16] * 6
    n = len(joints)
    j = analyze_idle_time(
        _from_x([0.0] * n, joint=joints),
        (MotionChannel("joint", 0.05, joint_index=0),),
        min_idle_seconds=0.3,
    )
    assert [s.kind for s in j.segments] == ["leading", "trailing"]
    assert j.peak_speed["joint:0"] == pytest.approx(0.2)

    xs = _profile(8, 10, 0, 0, 8)
    e = analyze_idle_time(
        _from_x(xs), (MotionChannel("extra", 0.01, state_key="raw"),), min_idle_seconds=0.3
    )
    assert [s.kind for s in e.segments] == ["leading", "trailing"]


def test_channel_validation() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        MotionChannel("position", 0.0, end_effector="ee")
    with pytest.raises(ValueError, match="positive and finite"):
        MotionChannel("position", float("inf"), end_effector="ee")
    with pytest.raises(ValueError, match="needs end_effector"):
        MotionChannel("gripper", 0.1)
    with pytest.raises(ValueError, match="needs joint_index"):
        MotionChannel("joint", 0.1)
    with pytest.raises(ValueError, match="needs state_key"):
        MotionChannel("extra", 0.1)


def test_argument_validation() -> None:
    ep = _from_x([0.0] * 5)
    with pytest.raises(ValueError, match="at least one MotionChannel"):
        analyze_idle_time(ep, (), min_idle_seconds=0.3)
    with pytest.raises(ValueError, match="duplicate channels"):
        analyze_idle_time(_from_x([0.0] * 5), (POS, POS), min_idle_seconds=0.3)
    with pytest.raises(ValueError, match="min_idle_seconds"):
        analyze_idle_time(_from_x([0.0] * 5), (POS,), min_idle_seconds=0.0)


def test_non_increasing_timestamps_raise_instead_of_dividing_by_zero() -> None:
    frames = [_frame(0, 0.0), _frame(1, 0.0, time=0.0)]
    with pytest.raises(ValueError, match="doesn't increase"):
        analyze_idle_time(_episode(frames), (POS,), min_idle_seconds=0.3)


def test_pose_frame_change_raises() -> None:
    frames = [_frame(0, 0.0), _frame(1, 0.0, pose_frame="world")]
    with pytest.raises(ValueError, match="frame changed"):
        analyze_idle_time(_episode(frames), (POS,), min_idle_seconds=0.3)


def test_missing_data_for_a_channel_raises_with_what_is_available() -> None:
    ep = _from_x([0.0] * 4)
    with pytest.raises(ValueError, match="no pose"):
        analyze_idle_time(
            ep, (MotionChannel("position", 0.1, end_effector="nope"),), min_idle_seconds=0.3
        )
    with pytest.raises(ValueError, match="no gripper"):
        analyze_idle_time(
            _from_x([0.0] * 4),
            (MotionChannel("gripper", 0.1, end_effector="nope"),),
            min_idle_seconds=0.3,
        )
    with pytest.raises(ValueError, match=r"isn.t in Proprioception.extra"):
        analyze_idle_time(
            _from_x([0.0] * 4),
            (MotionChannel("extra", 0.1, state_key="nope"),),
            min_idle_seconds=0.3,
        )
    with pytest.raises(ValueError, match="out of range"):
        analyze_idle_time(
            _from_x([0.0] * 4),
            (MotionChannel("joint", 0.1, joint_index=9),),
            min_idle_seconds=0.3,
        )


def test_joint_channel_without_joint_positions_raises() -> None:
    def no_joints(i: int) -> Frame:
        f = _frame(i, 0.0)
        p = f.proprioception
        return Frame(
            timestamp=f.timestamp,
            proprioception=Proprioception(
                joint_positions=None,
                joint_velocities=None,
                ee_poses=p.ee_poses,
                grippers=p.grippers,
                extra=p.extra,
            ),
            action=f.action,
            images={},
            is_first=f.is_first,
            is_last=False,
        )

    with pytest.raises(ValueError, match=r"needs Proprioception.joint_positions"):
        analyze_idle_time(
            _episode([no_joints(0), no_joints(1)]),
            (MotionChannel("joint", 0.1, joint_index=0),),
            min_idle_seconds=0.3,
        )


def test_fewer_than_two_frames_has_zero_duration_and_no_segments() -> None:
    result = analyze_idle_time(_from_x([0.0]), (POS,), min_idle_seconds=0.3)
    assert result.n_frames == 1
    assert result.total_seconds == 0.0
    assert result.segments == ()
    assert result.recoverable_fraction == 0.0
    assert result.interior_fraction == 0.0
    assert analyze_idle_time(_episode([]), (POS,), min_idle_seconds=0.3).n_frames == 0
