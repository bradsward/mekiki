"""Tests for `mekiki.state`."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.episode import CameraFrame, Episode, EpisodeMetadata, Frame, Proprioception
from mekiki.state import (
    StateField,
    reconstruct_episode_proprioception,
    reconstruct_proprioception,
)
from tests.conftest import CLEAN_ACTION_SPACE


def _proprio(
    *,
    extra: dict[str, np.ndarray] | None = None,
    grippers: dict[str, float] | None = None,
    joint_positions: np.ndarray | None = None,
) -> Proprioception:
    return Proprioception(
        joint_positions=joint_positions,
        joint_velocities=None,
        ee_poses={},
        grippers=grippers or {},
        extra=extra or {},
    )


def test_gripper_field_populates_grippers() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0, 0.7])})
    spec = {"observation.state": (None, None, StateField(kind="gripper", end_effector="ee"))}
    result = reconstruct_proprioception(proprio, spec)
    assert result.grippers["ee"] == pytest.approx(0.7)


def test_extra_is_never_removed_even_when_reconstructed() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0, 0.7])})
    spec = {"observation.state": (None, None, StateField(kind="gripper", end_effector="ee"))}
    result = reconstruct_proprioception(proprio, spec)
    assert "observation.state" in result.extra
    assert result.extra["observation.state"].tolist() == pytest.approx([1.0, 2.0, 0.7])


def test_gripper_reconstruction_preserves_other_existing_grippers() -> None:
    proprio = _proprio(
        extra={"observation.state": np.array([0.5])},
        grippers={"left": 0.2},
    )
    spec = {"observation.state": (StateField(kind="gripper", end_effector="right"),)}
    result = reconstruct_proprioception(proprio, spec)
    assert result.grippers == {"left": pytest.approx(0.2), "right": pytest.approx(0.5)}


def test_joint_field_populates_joint_positions() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.1, 2.2, 3.3])})
    spec = {
        "observation.state": (
            StateField(kind="joint", joint_index=0),
            StateField(kind="joint", joint_index=1),
            None,
        )
    }
    result = reconstruct_proprioception(proprio, spec)
    assert result.joint_positions is not None
    assert result.joint_positions.tolist() == pytest.approx([1.1, 2.2])


def test_joint_reconstruction_extends_existing_joint_positions() -> None:
    proprio = _proprio(
        extra={"observation.state": np.array([9.9])},
        joint_positions=np.array([0.1, 0.2]),
    )
    spec = {"observation.state": (StateField(kind="joint", joint_index=2),)}
    result = reconstruct_proprioception(proprio, spec)
    assert result.joint_positions is not None
    assert result.joint_positions.tolist() == pytest.approx([0.1, 0.2, 9.9])


def test_joint_reconstruction_overwrites_existing_index() -> None:
    proprio = _proprio(
        extra={"observation.state": np.array([9.9])},
        joint_positions=np.array([0.1, 0.2]),
    )
    spec = {"observation.state": (StateField(kind="joint", joint_index=0),)}
    result = reconstruct_proprioception(proprio, spec)
    assert result.joint_positions is not None
    assert result.joint_positions.tolist() == pytest.approx([9.9, 0.2])


def test_gripper_and_joint_together_in_one_array() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0, 3.0, 0.9])})
    spec = {
        "observation.state": (
            None,
            None,
            StateField(kind="joint", joint_index=0),
            StateField(kind="gripper", end_effector="ee"),
        )
    }
    result = reconstruct_proprioception(proprio, spec)
    assert result.grippers["ee"] == pytest.approx(0.9)
    assert result.joint_positions is not None
    assert result.joint_positions.tolist() == pytest.approx([3.0])


def test_all_none_spec_leaves_proprioception_unchanged() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0])})
    spec = {"observation.state": (None, None)}
    result = reconstruct_proprioception(proprio, spec)
    assert result.grippers == {}
    assert result.joint_positions is None


def test_missing_extra_key_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0])})
    spec = {"nonexistent": (StateField(kind="gripper", end_effector="ee"),)}
    with pytest.raises(ValueError, match="no such key"):
        reconstruct_proprioception(proprio, spec)


def test_length_mismatch_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0])})
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}  # len 1 != 2
    with pytest.raises(ValueError, match="entries"):
        reconstruct_proprioception(proprio, spec)


def test_gripper_missing_end_effector_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.array([0.5])})
    spec = {"observation.state": (StateField(kind="gripper"),)}
    with pytest.raises(ValueError, match="'gripper' state field requires end_effector"):
        reconstruct_proprioception(proprio, spec)


def test_joint_missing_joint_index_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.array([0.5])})
    spec = {"observation.state": (StateField(kind="joint"),)}
    with pytest.raises(ValueError, match="'joint' state field requires joint_index"):
        reconstruct_proprioception(proprio, spec)


def test_position_axis_raises_not_implemented() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0])})
    spec = {"observation.state": (StateField(kind="position_axis", end_effector="ee", axis="x"),)}
    with pytest.raises(NotImplementedError, match="position_axis"):
        reconstruct_proprioception(proprio, spec)


def test_orientation_axis_raises_not_implemented() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0])})
    spec = {
        "observation.state": (StateField(kind="orientation_axis", end_effector="ee", axis="x"),)
    }
    with pytest.raises(NotImplementedError, match="orientation_axis"):
        reconstruct_proprioception(proprio, spec)


# --- reconstruct_episode_proprioception ------------------------------------


def _episode_with_state(rows: list[np.ndarray]) -> Episode:
    frames = [
        Frame(
            timestamp=i * 0.1,
            proprioception=_proprio(extra={"observation.state": row}),
            action=np.zeros(1),
            images={
                "cam": CameraFrame(
                    read=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
                    timestamp=i * 0.1,
                    resolution=(2, 2),
                    timestamp_is_measured=False,
                )
            },
            is_first=(i == 0),
            is_last=(i == len(rows) - 1),
        )
        for i, row in enumerate(rows)
    ]
    metadata = EpisodeMetadata(
        episode_id="state-0",
        dataset_name="synthetic",
        robot_embodiment="franka_panda",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    return Episode(metadata=metadata, frames=frames)


def test_reconstruct_episode_proprioception_applies_to_every_frame() -> None:
    episode = _episode_with_state([np.array([0.1]), np.array([0.2]), np.array([0.3])])
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    reconstructed = reconstruct_episode_proprioception(episode, spec)
    grippers = [frame.proprioception.grippers["ee"] for frame in reconstructed]
    assert grippers == pytest.approx([0.1, 0.2, 0.3])


def test_reconstruct_episode_proprioception_preserves_metadata() -> None:
    episode = _episode_with_state([np.array([0.1])])
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    reconstructed = reconstruct_episode_proprioception(episode, spec)
    assert reconstructed.metadata is episode.metadata


def test_reconstruct_episode_proprioception_streams_once() -> None:
    episode = _episode_with_state([np.array([0.1]), np.array([0.2])])
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    reconstructed = reconstruct_episode_proprioception(episode, spec)
    first_pass = list(reconstructed)
    assert len(first_pass) == 2
    assert list(reconstructed) == []  # exhausted, same one-shot rule as Episode
