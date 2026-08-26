"""Tests for the core Episode/Frame data model."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.episode import (
    ActionDimSpec,
    CameraFrame,
    Episode,
    Pose,
    Proprioception,
)
from tests.conftest import make_clean_episode, make_clean_frame


def test_clean_episode_iterates_all_frames() -> None:
    episode = make_clean_episode(n_frames=5)
    frames = list(episode)
    assert len(frames) == 5
    assert frames[0].is_first
    assert frames[-1].is_last
    assert not any(f.is_first for f in frames[1:])
    assert not any(f.is_last for f in frames[:-1])


def test_episode_is_iterable_only_via_its_frames_source() -> None:
    # frames may be a one-shot generator, per docs/episode.md — Episode must
    # not require a list and must not silently cache/replay it.
    def frame_stream() -> object:
        for i in range(3):
            yield make_clean_frame(timestamp=i / 10.0, is_first=(i == 0), is_last=(i == 2))

    episode = Episode(metadata=make_clean_episode(n_frames=1).metadata, frames=frame_stream())  # type: ignore[arg-type]
    first_pass = list(episode)
    assert len(first_pass) == 3
    # a generator is exhausted after one pass — Episode doesn't try to hide that
    assert list(episode) == []


def test_pose_rejects_wrong_position_shape() -> None:
    with pytest.raises(ValueError, match="position"):
        Pose(
            position=np.zeros(2, dtype=np.float64),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            frame="base_link",
        )


def test_pose_rejects_wrong_orientation_shape() -> None:
    with pytest.raises(ValueError, match="orientation"):
        Pose(
            position=np.zeros(3, dtype=np.float64),
            orientation=np.zeros(3, dtype=np.float64),
            frame="base_link",
        )


def test_proprioception_rejects_out_of_range_gripper() -> None:
    with pytest.raises(ValueError, match="gripper"):
        Proprioception(
            joint_positions=np.zeros(7, dtype=np.float64),
            joint_velocities=None,
            ee_poses={
                "ee": Pose(
                    position=np.zeros(3, dtype=np.float64),
                    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                    frame="base_link",
                )
            },
            grippers={"ee": 1.5},
            extra={},
        )


def test_proprioception_allows_missing_velocities() -> None:
    prop = Proprioception(
        joint_positions=np.zeros(7, dtype=np.float64),
        joint_velocities=None,
        ee_poses={
            "ee": Pose(
                position=np.zeros(3, dtype=np.float64),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                frame="base_link",
            )
        },
        grippers={"ee": 0.0},
        extra={},
    )
    assert prop.joint_velocities is None


def test_proprioception_allows_no_arm_shape_at_all() -> None:
    # e.g. PushT: a 2D pusher position, no joints/ee/gripper at all
    prop = Proprioception(
        joint_positions=None,
        joint_velocities=None,
        ee_poses={},
        grippers={},
        extra={"observation.state": np.array([222.0, 97.0], dtype=np.float64)},
    )
    assert prop.joint_positions is None
    assert prop.ee_poses == {}
    assert prop.grippers == {}
    assert prop.extra["observation.state"].shape == (2,)


def test_proprioception_supports_bimanual_ee_and_grippers() -> None:
    pose = Pose(
        position=np.zeros(3, dtype=np.float64),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        frame="base_link",
    )
    prop = Proprioception(
        joint_positions=np.zeros(14, dtype=np.float64),
        joint_velocities=None,
        ee_poses={"left": pose, "right": pose},
        grippers={"left": 1.0, "right": 0.0},
        extra={},
    )
    assert set(prop.ee_poses) == {"left", "right"}
    assert set(prop.grippers) == {"left", "right"}


def test_camera_frame_read_is_lazy() -> None:
    calls = 0

    def read() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    frame = CameraFrame(read=read, timestamp=0.0, resolution=(4, 4), timestamp_is_measured=True)
    assert calls == 0  # constructing the frame must not decode anything
    image = frame.read()
    assert calls == 1
    assert image.shape == (4, 4, 3)


def test_action_dim_spec_is_immutable() -> None:
    spec = ActionDimSpec("x", "delta", "m", "ee")
    with pytest.raises(AttributeError):
        spec.mode = "absolute"  # type: ignore[misc]


def test_clean_episode_action_space_matches_action_length() -> None:
    episode = make_clean_episode(n_frames=2)
    n_dims = len(episode.metadata.action_space)
    for frame in episode:
        assert frame.action.shape == (n_dims,)
