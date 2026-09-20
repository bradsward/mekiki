"""Tests for `mekiki.state`."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.episode import CameraFrame, Episode, EpisodeMetadata, Frame, Proprioception
from mekiki.rotation import OrientationEncoding
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


# --- ee_poses reconstruction ------------------------------------------------

_EULER_XYZ = OrientationEncoding("euler", "xyz")
_QUAT = OrientationEncoding("quaternion_xyzw")
_ROTVEC = OrientationEncoding("rotvec")


def _position_fields(frame: str = "base_link", ee: str = "ee") -> list[StateField]:
    return [
        StateField(kind="position_axis", end_effector=ee, axis=axis, frame=frame)
        for axis in ("x", "y", "z")
    ]


def _orientation_fields(
    encoding: OrientationEncoding, frame: str = "base_link", ee: str = "ee"
) -> list[StateField]:
    return [
        StateField(
            kind="orientation_axis", end_effector=ee, component=i, frame=frame, encoding=encoding
        )
        for i in range(encoding.n_components)
    ]


def test_pose_reconstructed_from_position_and_euler_orientation() -> None:
    # x, y, z, roll, pitch, yaw -- an Euler 'xyz' (extrinsic) orientation of a
    # single 90deg roll about x
    raw = np.array([0.1, 0.2, 0.3, np.pi / 2, 0.0, 0.0])
    proprio = _proprio(extra={"observation.state": raw})
    spec = {"observation.state": tuple(_position_fields() + _orientation_fields(_EULER_XYZ))}
    pose = reconstruct_proprioception(proprio, spec).ee_poses["ee"]
    half = np.pi / 4
    assert pose.position.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert pose.orientation == pytest.approx([np.sin(half), 0.0, 0.0, np.cos(half)])
    assert pose.frame == "base_link"


def test_pose_reconstructed_from_position_and_quaternion_orientation() -> None:
    raw = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.6, 0.8])
    proprio = _proprio(extra={"observation.state": raw})
    spec = {"observation.state": tuple(_position_fields() + _orientation_fields(_QUAT))}
    pose = reconstruct_proprioception(proprio, spec).ee_poses["ee"]
    assert pose.orientation == pytest.approx([0.0, 0.0, 0.6, 0.8])


def test_pose_reconstructed_from_position_and_rotvec_orientation() -> None:
    raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    proprio = _proprio(extra={"observation.state": raw})
    spec = {"observation.state": tuple(_position_fields() + _orientation_fields(_ROTVEC))}
    pose = reconstruct_proprioception(proprio, spec).ee_poses["ee"]
    half = np.pi / 4
    assert pose.orientation == pytest.approx([0.0, 0.0, np.sin(half), np.cos(half)])


def test_pose_fields_can_come_from_different_extra_keys() -> None:
    proprio = _proprio(
        extra={
            "observation.position": np.array([1.0, 2.0, 3.0]),
            "observation.euler": np.array([0.0, 0.0, np.pi / 2]),
        }
    )
    spec = {
        "observation.position": tuple(_position_fields()),
        "observation.euler": tuple(_orientation_fields(_EULER_XYZ)),
    }
    pose = reconstruct_proprioception(proprio, spec).ee_poses["ee"]
    assert pose.position.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_pose_reconstruction_never_removes_extra() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    proprio = _proprio(extra={"observation.state": raw})
    spec = {"observation.state": tuple(_position_fields() + _orientation_fields(_EULER_XYZ))}
    result = reconstruct_proprioception(proprio, spec)
    assert result.extra["observation.state"].tolist() == pytest.approx(raw.tolist())


def test_bimanual_poses_reconstructed_independently() -> None:
    raw = np.arange(12, dtype=np.float64) * 0.1
    proprio = _proprio(extra={"observation.state": raw})
    fields = (
        _position_fields(ee="left")
        + _orientation_fields(_EULER_XYZ, ee="left")
        + _position_fields(ee="right")
        + _orientation_fields(_EULER_XYZ, ee="right")
    )
    result = reconstruct_proprioception(proprio, {"observation.state": tuple(fields)})
    assert set(result.ee_poses) == {"left", "right"}
    assert result.ee_poses["left"].position.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert result.ee_poses["right"].position.tolist() == pytest.approx([0.6, 0.7, 0.8])


def test_position_without_orientation_raises_instead_of_fabricating_identity() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0, 2.0, 3.0])})
    spec = {"observation.state": tuple(_position_fields())}
    with pytest.raises(ValueError, match="no orientation"):
        reconstruct_proprioception(proprio, spec)


def test_orientation_without_position_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.array([0.0, 0.0, 0.0])})
    spec = {"observation.state": tuple(_orientation_fields(_EULER_XYZ))}
    with pytest.raises(ValueError, match="no position"):
        reconstruct_proprioception(proprio, spec)


def test_incomplete_position_triple_raises() -> None:
    raw = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
    proprio = _proprio(extra={"observation.state": raw})
    fields = _position_fields()[:2] + _orientation_fields(_EULER_XYZ)
    with pytest.raises(ValueError, match=r"missing axes \['z'\]"):
        reconstruct_proprioception(proprio, {"observation.state": tuple(fields)})


def test_incomplete_orientation_components_raise() -> None:
    raw = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    proprio = _proprio(extra={"observation.state": raw})
    fields = _position_fields() + _orientation_fields(_EULER_XYZ)[:2]
    with pytest.raises(ValueError, match=r"needs exactly 0..2"):
        reconstruct_proprioception(proprio, {"observation.state": tuple(fields)})


def test_disagreeing_frames_raise() -> None:
    raw = np.zeros(6)
    proprio = _proprio(extra={"observation.state": raw})
    fields = _position_fields(frame="base_link") + _orientation_fields(_EULER_XYZ, frame="world")
    with pytest.raises(ValueError, match="different frames"):
        reconstruct_proprioception(proprio, {"observation.state": tuple(fields)})


def test_mixed_encodings_raise() -> None:
    raw = np.zeros(6)
    proprio = _proprio(extra={"observation.state": raw})
    mixed = [
        StateField(
            kind="orientation_axis",
            end_effector="ee",
            component=0,
            frame="base_link",
            encoding=_EULER_XYZ,
        ),
        StateField(
            kind="orientation_axis",
            end_effector="ee",
            component=1,
            frame="base_link",
            encoding=_ROTVEC,
        ),
        StateField(
            kind="orientation_axis",
            end_effector="ee",
            component=2,
            frame="base_link",
            encoding=_ROTVEC,
        ),
    ]
    fields = _position_fields() + mixed
    with pytest.raises(ValueError, match="different orientation encodings"):
        reconstruct_proprioception(proprio, {"observation.state": tuple(fields)})


def test_duplicate_position_axis_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.zeros(2)})
    fields = (
        StateField(kind="position_axis", end_effector="ee", axis="x", frame="base_link"),
        StateField(kind="position_axis", end_effector="ee", axis="x", frame="base_link"),
    )
    with pytest.raises(ValueError, match="duplicate position axis"):
        reconstruct_proprioception(proprio, {"observation.state": fields})


def test_duplicate_orientation_component_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.zeros(2)})
    fields = (
        StateField(
            kind="orientation_axis", end_effector="ee", component=0, frame="f", encoding=_ROTVEC
        ),
        StateField(
            kind="orientation_axis", end_effector="ee", component=0, frame="f", encoding=_ROTVEC
        ),
    )
    with pytest.raises(ValueError, match="duplicate orientation component"):
        reconstruct_proprioception(proprio, {"observation.state": fields})


def test_position_axis_missing_required_pieces_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.zeros(1)})
    fields = (StateField(kind="position_axis", end_effector="ee", axis="x"),)  # no frame
    with pytest.raises(ValueError, match="requires end_effector, axis, and frame"):
        reconstruct_proprioception(proprio, {"observation.state": fields})


def test_orientation_axis_missing_required_pieces_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.zeros(1)})
    fields = (StateField(kind="orientation_axis", end_effector="ee", component=0),)  # no frame
    with pytest.raises(ValueError, match="requires end_effector, component, and frame"):
        reconstruct_proprioception(proprio, {"observation.state": fields})


def test_orientation_axis_without_an_encoding_raises() -> None:
    proprio = _proprio(extra={"observation.state": np.zeros(1)})
    fields = (StateField(kind="orientation_axis", end_effector="ee", component=0, frame="f"),)
    with pytest.raises(ValueError, match="requires an encoding"):
        reconstruct_proprioception(proprio, {"observation.state": fields})


def test_materially_non_unit_quaternion_is_rejected_not_smoothed_over() -> None:
    raw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.9])
    proprio = _proprio(extra={"observation.state": raw})
    spec = {"observation.state": tuple(_position_fields() + _orientation_fields(_QUAT))}
    with pytest.raises(ValueError, match="not silently renormalized"):
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


# --- gripper calibration tolerance -------------------------------------------


def test_small_gripper_overshoot_past_one_is_clipped_and_raw_value_survives() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.0181])})  # real max overshoot
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    result = reconstruct_proprioception(proprio, spec)
    assert result.grippers["ee"] == 1.0
    assert result.extra["observation.state"][0] == pytest.approx(1.0181)  # raw untouched


def test_small_gripper_undershoot_below_zero_is_clipped() -> None:
    proprio = _proprio(extra={"observation.state": np.array([-0.01])})
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    assert reconstruct_proprioception(proprio, spec).grippers["ee"] == 0.0


def test_gripper_value_far_outside_the_range_is_rejected_as_a_wrong_column() -> None:
    proprio = _proprio(extra={"observation.state": np.array([37.0])})  # e.g. a mm or degree column
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    with pytest.raises(ValueError, match="wrong column or a non-normalized unit"):
        reconstruct_proprioception(proprio, spec)


def test_gripper_just_past_the_tolerance_is_rejected() -> None:
    proprio = _proprio(extra={"observation.state": np.array([1.06])})
    spec = {"observation.state": (StateField(kind="gripper", end_effector="ee"),)}
    with pytest.raises(ValueError, match="outside"):
        reconstruct_proprioception(proprio, spec)
