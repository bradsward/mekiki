"""Tests for `mekiki.checks.consistency`."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.checks.consistency import (
    ActionTarget,
    ToleranceModel,
    _quaternion_from_rotation_vector,
    _quaternion_multiply,
    _quaternion_to_rotation_matrix,
    check_action_state_consistency,
    predict_next_proprioception,
    validate_action_target_spec,
)
from mekiki.episode import ActionDimSpec, Episode, EpisodeMetadata, Frame, Pose, Proprioception
from tests.conftest import CLEAN_ACTION_SPACE

_POSITION_ACTION_SPACE = (
    ActionDimSpec("x", "delta", "m", "ee"),
    ActionDimSpec("y", "delta", "m", "ee"),
    ActionDimSpec("z", "delta", "m", "ee"),
    ActionDimSpec("gripper", "absolute", "normalized", "ee"),
)


def test_complete_position_and_gripper_spec_is_valid() -> None:
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
        ActionTarget(kind="gripper", end_effector="ee"),
    )
    validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)  # must not raise


def test_all_none_spec_is_valid_nothing_modeled() -> None:
    # e.g. a PushT-shaped dataset: raw, uninterpreted state -- nothing to
    # check yet, but that's not an error
    target_spec = (None, None, None, None)
    validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)  # must not raise


def test_mixed_none_and_modeled_dimensions_is_valid() -> None:
    target_spec = (
        None,
        None,
        None,
        ActionTarget(kind="gripper", end_effector="ee"),
    )
    validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)  # must not raise


def test_bimanual_independent_groups_are_valid() -> None:
    action_space = (
        ActionDimSpec("lx", "delta", "m", "ee"),
        ActionDimSpec("ly", "delta", "m", "ee"),
        ActionDimSpec("lz", "delta", "m", "ee"),
        ActionDimSpec("rx", "delta", "m", "ee"),
        ActionDimSpec("ry", "delta", "m", "ee"),
        ActionDimSpec("rz", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="left", axis="x"),
        ActionTarget(kind="position_axis", end_effector="left", axis="y"),
        ActionTarget(kind="position_axis", end_effector="left", axis="z"),
        ActionTarget(kind="position_axis", end_effector="right", axis="x"),
        ActionTarget(kind="position_axis", end_effector="right", axis="y"),
        ActionTarget(kind="position_axis", end_effector="right", axis="z"),
    )
    validate_action_target_spec(action_space, target_spec)  # must not raise


def test_joint_target_is_valid() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    validate_action_target_spec(action_space, target_spec)  # must not raise


def test_orientation_axis_group_is_valid() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
        ActionDimSpec("rz", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    validate_action_target_spec(action_space, target_spec)  # must not raise


def test_length_mismatch_raises() -> None:
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    with pytest.raises(ValueError, match="entries"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_incomplete_position_triple_raises_naming_missing_axis() -> None:
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        None,  # z missing
        ActionTarget(kind="gripper", end_effector="ee"),
    )
    with pytest.raises(ValueError, match=r"missing axes.*'z'"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_duplicate_axis_raises() -> None:
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),  # dup
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
        ActionTarget(kind="gripper", end_effector="ee"),
    )
    with pytest.raises(ValueError, match="duplicate axis"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_position_axis_missing_end_effector_raises() -> None:
    target_spec = (
        ActionTarget(kind="position_axis", axis="x"),
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="end_effector and axis"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_position_axis_missing_axis_raises() -> None:
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee"),
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="end_effector and axis"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_gripper_missing_end_effector_raises() -> None:
    target_spec = (None, None, None, ActionTarget(kind="gripper"))
    with pytest.raises(ValueError, match="'gripper' target requires end_effector"):
        validate_action_target_spec(_POSITION_ACTION_SPACE, target_spec)


def test_joint_missing_joint_index_raises() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint"),)
    with pytest.raises(ValueError, match="'joint' target requires joint_index"):
        validate_action_target_spec(action_space, target_spec)


def test_orientation_incomplete_triple_raises() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
    )
    with pytest.raises(ValueError, match=r"missing axes.*'z'"):
        validate_action_target_spec(action_space, target_spec)


def test_empty_spec_for_empty_action_space_is_valid() -> None:
    validate_action_target_spec((), ())  # must not raise


def test_inconsistent_mode_within_group_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "absolute", "m", "ee"),  # inconsistent
        ActionDimSpec("z", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    with pytest.raises(ValueError, match="inconsistent modes"):
        validate_action_target_spec(action_space, target_spec)


def test_inconsistent_frame_within_group_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "delta", "m", "base_link"),  # inconsistent
        ActionDimSpec("z", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    with pytest.raises(ValueError, match="inconsistent frames"):
        validate_action_target_spec(action_space, target_spec)


def test_inconsistent_unit_within_group_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "delta", "normalized", "ee"),  # inconsistent
        ActionDimSpec("z", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    with pytest.raises(ValueError, match="inconsistent units"):
        validate_action_target_spec(action_space, target_spec)


# --- quaternion math, hand-verified ---------------------------------------
#
# All (x, y, z, w) throughout, matching mekiki.episode.Pose.orientation.

_IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def test_rotation_matrix_90deg_about_z_rotates_x_axis_to_y_axis() -> None:
    # a 90-degree rotation about z: q = (0, 0, sin(45deg), cos(45deg))
    half = np.pi / 4
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    r = _quaternion_to_rotation_matrix(q)
    rotated = r @ np.array([1.0, 0.0, 0.0])
    assert rotated == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_quaternion_from_zero_rotation_vector_is_identity() -> None:
    q = _quaternion_from_rotation_vector(np.array([0.0, 0.0, 0.0]))
    assert q == pytest.approx(_IDENTITY_Q)


def test_quaternion_from_rotation_vector_90deg_about_z() -> None:
    r = np.array([0.0, 0.0, np.pi / 2])  # axis=z, angle=90deg
    q = _quaternion_from_rotation_vector(r)
    half = np.pi / 4
    assert q == pytest.approx([0.0, 0.0, np.sin(half), np.cos(half)])


def test_quaternion_multiply_by_identity_is_unchanged() -> None:
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)])
    assert _quaternion_multiply(q, _IDENTITY_Q) == pytest.approx(q)


def test_quaternion_multiply_two_90deg_z_rotations_gives_180deg_about_z() -> None:
    # hand-verified: composing two 90deg-about-z rotations gives 180deg about
    # z, i.e. q = (0, 0, 1, 0)
    half = np.pi / 4
    q90 = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    composed = _quaternion_multiply(q90, q90)
    assert composed == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-9)


# --- predict_next_proprioception ------------------------------------------


def _proprio_with_ee(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: np.ndarray = _IDENTITY_Q,
    frame: str = "base_link",
    gripper: float | None = 1.0,
    end_effector: str = "ee",
    joint_positions: np.ndarray | None = None,
) -> Proprioception:
    grippers = {} if gripper is None else {end_effector: gripper}
    return Proprioception(
        joint_positions=joint_positions,
        joint_velocities=None,
        ee_poses={
            end_effector: Pose(
                position=np.array(position, dtype=np.float64),
                orientation=np.asarray(orientation, dtype=np.float64),
                frame=frame,
            )
        },
        grippers=grippers,
        extra={},
    )


def test_position_absolute_same_frame_returns_action_value_directly() -> None:
    action_space = (
        ActionDimSpec("x", "absolute", "m", "base_link"),
        ActionDimSpec("y", "absolute", "m", "base_link"),
        ActionDimSpec("z", "absolute", "m", "base_link"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee(position=(1.0, 2.0, 3.0), frame="base_link")
    action = np.array([0.5, 0.6, 0.7])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.ee_positions["ee"] == pytest.approx([0.5, 0.6, 0.7])


def test_position_absolute_frame_mismatch_raises() -> None:
    action_space = (
        ActionDimSpec("x", "absolute", "m", "world"),
        ActionDimSpec("y", "absolute", "m", "world"),
        ActionDimSpec("z", "absolute", "m", "world"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee(frame="base_link")
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="absolute targets must match"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_position_delta_same_frame_adds_to_current() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "base_link"),
        ActionDimSpec("y", "delta", "m", "base_link"),
        ActionDimSpec("z", "delta", "m", "base_link"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee(position=(1.0, 2.0, 3.0), frame="base_link")
    action = np.array([0.1, -0.2, 0.3])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.ee_positions["ee"] == pytest.approx([1.1, 1.8, 3.3])


def test_position_delta_in_ee_frame_is_rotated_by_current_orientation() -> None:
    # current orientation is 90deg about z; a delta of (1, 0, 0) expressed
    # in the ee frame should rotate to (0, 1, 0) in the base frame before
    # being added -- same hand-verified rotation as the quaternion tests.
    half = np.pi / 4
    q90 = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "delta", "m", "ee"),
        ActionDimSpec("z", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee(position=(0.0, 0.0, 0.0), orientation=q90, frame="base_link")
    action = np.array([1.0, 0.0, 0.0])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.ee_positions["ee"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_position_delta_unsupported_frame_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "wrist_camera"),
        ActionDimSpec("y", "delta", "m", "wrist_camera"),
        ActionDimSpec("z", "delta", "m", "wrist_camera"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee(frame="base_link")
    action = np.array([0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="unsupported frame transform"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_position_wrong_unit_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "normalized", "ee"),
        ActionDimSpec("y", "delta", "normalized", "ee"),
        ActionDimSpec("z", "delta", "normalized", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee()
    action = np.array([0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="meters"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_position_missing_end_effector_raises() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "base_link"),
        ActionDimSpec("y", "delta", "m", "base_link"),
        ActionDimSpec("z", "delta", "m", "base_link"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="left", axis="x"),
        ActionTarget(kind="position_axis", end_effector="left", axis="y"),
        ActionTarget(kind="position_axis", end_effector="left", axis="z"),
    )
    proprio = _proprio_with_ee(end_effector="ee")  # spec references "left", not "ee"
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="no ee_poses"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_orientation_delta_in_ee_frame_composes_quaternion() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
        ActionDimSpec("rz", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    half = np.pi / 4
    q90 = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    proprio = _proprio_with_ee(orientation=q90)
    action = np.array([0.0, 0.0, np.pi / 2])  # another 90deg about z, in ee frame
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    # two composed 90deg-about-z rotations = 180deg about z, same as the
    # quaternion-math test above
    assert result.ee_orientations["ee"] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-9)


def test_orientation_missing_end_effector_raises() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
        ActionDimSpec("rz", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="left", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="left", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="left", axis="z"),
    )
    proprio = _proprio_with_ee(end_effector="ee")  # spec references "left", not "ee"
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="no ee_poses"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_orientation_absolute_mode_raises() -> None:
    action_space = (
        ActionDimSpec("rx", "absolute", "rad", "ee"),
        ActionDimSpec("ry", "absolute", "rad", "ee"),
        ActionDimSpec("rz", "absolute", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee()
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="aren't supported yet"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_orientation_delta_non_ee_frame_raises_with_parked_explanation() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "base_link"),
        ActionDimSpec("ry", "delta", "rad", "base_link"),
        ActionDimSpec("rz", "delta", "rad", "base_link"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee()
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="opposite quaternion composition order"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_orientation_wrong_unit_raises() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "normalized", "ee"),
        ActionDimSpec("ry", "delta", "normalized", "ee"),
        ActionDimSpec("rz", "delta", "normalized", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    proprio = _proprio_with_ee()
    action = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="radians"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_gripper_absolute_normalized_passes_through() -> None:
    action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    proprio = _proprio_with_ee()
    action = np.array([0.42])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.grippers["ee"] == pytest.approx(0.42)


def test_gripper_wrong_unit_raises() -> None:
    action_space = (ActionDimSpec("gripper", "absolute", "m", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    proprio = _proprio_with_ee()
    action = np.array([0.5])
    with pytest.raises(ValueError, match="normalized"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_gripper_delta_mode_raises() -> None:
    action_space = (ActionDimSpec("gripper", "delta", "normalized", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    proprio = _proprio_with_ee()
    action = np.array([0.1])
    with pytest.raises(ValueError, match="only 'absolute' is supported"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_joint_absolute_passes_through() -> None:
    action_space = (ActionDimSpec("j0", "absolute", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    proprio = _proprio_with_ee(joint_positions=np.array([0.0, 0.0]))
    action = np.array([1.23])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.joint_positions[0] == pytest.approx(1.23)


def test_joint_delta_adds_to_current() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    proprio = _proprio_with_ee(joint_positions=np.array([0.5, 0.0]))
    action = np.array([0.1])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.joint_positions[0] == pytest.approx(0.6)


def test_joint_delta_without_joint_positions_raises() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    proprio = _proprio_with_ee(joint_positions=None)
    action = np.array([0.1])
    with pytest.raises(ValueError, match="no joint_positions"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_joint_index_out_of_range_raises() -> None:
    action_space = (ActionDimSpec("j5", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=5),)
    proprio = _proprio_with_ee(joint_positions=np.array([0.0, 0.0]))
    action = np.array([0.1])
    with pytest.raises(ValueError, match="out of range"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_action_shape_mismatch_raises() -> None:
    action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    proprio = _proprio_with_ee()
    action = np.array([0.1, 0.2])  # wrong length
    with pytest.raises(ValueError, match="shape"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_invalid_target_spec_propagates_validation_error() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "delta", "m", "ee"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
    )  # missing z
    proprio = _proprio_with_ee()
    action = np.array([0.1, 0.1])
    with pytest.raises(ValueError, match="missing axes"):
        predict_next_proprioception(proprio, action, action_space, target_spec)


def test_dimensions_mapped_to_none_are_excluded_from_result() -> None:
    action_space = (
        ActionDimSpec("unlabeled", "delta", "normalized", "unknown"),
        ActionDimSpec("gripper", "absolute", "normalized", "ee"),
    )
    target_spec = (None, ActionTarget(kind="gripper", end_effector="ee"))
    proprio = _proprio_with_ee()
    action = np.array([999.0, 0.7])  # the unlabeled dim's value is irrelevant
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.ee_positions == {}
    assert result.ee_orientations == {}
    assert result.joint_positions == {}
    assert result.grippers == {"ee": pytest.approx(0.7)}


def test_everything_together_produces_one_consistent_prediction() -> None:
    action_space = (
        ActionDimSpec("x", "delta", "m", "ee"),
        ActionDimSpec("y", "delta", "m", "ee"),
        ActionDimSpec("z", "delta", "m", "ee"),
        ActionDimSpec("gripper", "absolute", "normalized", "ee"),
        ActionDimSpec("j0", "delta", "rad", "joint"),
    )
    target_spec = (
        ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
        ActionTarget(kind="gripper", end_effector="ee"),
        ActionTarget(kind="joint", joint_index=0),
    )
    proprio = _proprio_with_ee(position=(1.0, 1.0, 1.0), joint_positions=np.array([0.0]))
    action = np.array([0.1, 0.1, 0.1, 0.9, 0.05])
    result = predict_next_proprioception(proprio, action, action_space, target_spec)
    assert result.ee_positions["ee"] == pytest.approx([1.1, 1.1, 1.1])
    assert result.grippers["ee"] == pytest.approx(0.9)
    assert result.joint_positions[0] == pytest.approx(0.05)


# --- ToleranceModel --------------------------------------------------------

_ZERO_TOLERANCE = ToleranceModel(
    position_base_m=0.0,
    position_rate_per_second_m=0.0,
    orientation_base_rad=0.0,
    orientation_rate_per_second_rad=0.0,
    gripper_base=0.0,
    gripper_rate_per_second=0.0,
    joint_base_rad=0.0,
    joint_rate_per_second_rad=0.0,
)


def test_tolerance_model_rejects_negative_field() -> None:
    with pytest.raises(ValueError, match="position_base_m"):
        ToleranceModel(
            position_base_m=-0.01,
            position_rate_per_second_m=0.0,
            orientation_base_rad=0.0,
            orientation_rate_per_second_rad=0.0,
            gripper_base=0.0,
            gripper_rate_per_second=0.0,
            joint_base_rad=0.0,
            joint_rate_per_second_rad=0.0,
        )


def test_tolerance_model_scales_with_dt() -> None:
    model = ToleranceModel(
        position_base_m=0.01,
        position_rate_per_second_m=0.1,
        orientation_base_rad=0.02,
        orientation_rate_per_second_rad=0.2,
        gripper_base=0.03,
        gripper_rate_per_second=0.3,
        joint_base_rad=0.04,
        joint_rate_per_second_rad=0.4,
    )
    assert model.position(dt=0.5) == pytest.approx(0.01 + 0.1 * 0.5)
    assert model.orientation(dt=0.5) == pytest.approx(0.02 + 0.2 * 0.5)
    assert model.gripper(dt=0.5) == pytest.approx(0.03 + 0.3 * 0.5)
    assert model.joint(dt=0.5) == pytest.approx(0.04 + 0.4 * 0.5)


# --- check_action_state_consistency ---------------------------------------


def _frame(
    timestamp: float,
    proprioception: Proprioception,
    action: np.ndarray,
    *,
    is_first: bool = False,
    is_last: bool = False,
) -> Frame:
    return Frame(
        timestamp=timestamp,
        proprioception=proprioception,
        action=action,
        images={},
        is_first=is_first,
        is_last=is_last,
    )


def _episode_from_frames(frames: list[Frame]) -> Episode:
    metadata = EpisodeMetadata(
        episode_id="consistency-0",
        dataset_name="synthetic",
        robot_embodiment="franka_panda",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    return Episode(metadata=metadata, frames=frames)


_POSITION_DELTA_ACTION_SPACE = (
    ActionDimSpec("x", "delta", "m", "base_link"),
    ActionDimSpec("y", "delta", "m", "base_link"),
    ActionDimSpec("z", "delta", "m", "base_link"),
)
_POSITION_TARGET_SPEC = (
    ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
    ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
    ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
)


def test_perfectly_consistent_episode_has_no_violations() -> None:
    # action exactly explains the next frame's position at every step
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(position=(0.0, 0.0, 0.0)),
            np.array([0.1, 0.0, 0.0]),
            is_first=True,
        ),
        _frame(0.1, _proprio_with_ee(position=(0.1, 0.0, 0.0)), np.array([0.1, 0.0, 0.0])),
        _frame(
            0.2, _proprio_with_ee(position=(0.2, 0.0, 0.0)), np.array([0.0, 0.0, 0.0]), is_last=True
        ),
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames),
        _POSITION_DELTA_ACTION_SPACE,
        _POSITION_TARGET_SPEC,
        _ZERO_TOLERANCE,
    )
    result = results["position:ee"]
    assert result.n_steps == 2
    assert result.violation_indices == ()
    assert result.max_residual == pytest.approx(0.0, abs=1e-9)


def test_position_mismatch_flagged_at_known_residual_and_threshold() -> None:
    # action predicts (0.1, 0, 0) but the next frame actually recorded
    # (0.15, 0, 0) -- a known 0.05m residual
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(position=(0.0, 0.0, 0.0)),
            np.array([0.1, 0.0, 0.0]),
            is_first=True,
        ),
        _frame(
            0.1,
            _proprio_with_ee(position=(0.15, 0.0, 0.0)),
            np.array([0.0, 0.0, 0.0]),
            is_last=True,
        ),
    ]
    tolerance = ToleranceModel(
        position_base_m=0.01,
        position_rate_per_second_m=0.0,
        orientation_base_rad=0.0,
        orientation_rate_per_second_rad=0.0,
        gripper_base=0.0,
        gripper_rate_per_second=0.0,
        joint_base_rad=0.0,
        joint_rate_per_second_rad=0.0,
    )
    results = check_action_state_consistency(
        _episode_from_frames(frames), _POSITION_DELTA_ACTION_SPACE, _POSITION_TARGET_SPEC, tolerance
    )
    result = results["position:ee"]
    assert result.violation_indices == (0,)  # the frame carrying the bad action
    assert result.violation_residuals[0] == pytest.approx(0.05)
    assert result.violation_thresholds[0] == pytest.approx(0.01)
    assert result.max_residual == pytest.approx(0.05)


def test_tolerance_growing_with_dt_can_absorb_the_same_residual() -> None:
    # identical 0.05m residual as above, but a much larger dt this time --
    # the rate term should loosen the threshold enough to not flag it
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(position=(0.0, 0.0, 0.0)),
            np.array([0.1, 0.0, 0.0]),
            is_first=True,
        ),
        _frame(
            1.0,
            _proprio_with_ee(position=(0.15, 0.0, 0.0)),
            np.array([0.0, 0.0, 0.0]),
            is_last=True,
        ),
    ]
    tolerance = ToleranceModel(
        position_base_m=0.01,
        position_rate_per_second_m=0.1,  # at dt=1.0s, tolerance = 0.01 + 0.1 = 0.11
        orientation_base_rad=0.0,
        orientation_rate_per_second_rad=0.0,
        gripper_base=0.0,
        gripper_rate_per_second=0.0,
        joint_base_rad=0.0,
        joint_rate_per_second_rad=0.0,
    )
    results = check_action_state_consistency(
        _episode_from_frames(frames), _POSITION_DELTA_ACTION_SPACE, _POSITION_TARGET_SPEC, tolerance
    )
    result = results["position:ee"]
    assert result.violation_indices == ()
    assert result.max_residual == pytest.approx(0.05)


def test_orientation_mismatch_flagged_with_angular_residual() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
        ActionDimSpec("rz", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    q180 = np.array([0.0, 0.0, 1.0, 0.0])  # actual recorded: 180deg about z
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(orientation=_IDENTITY_Q),
            np.array([0.0, 0.0, np.pi / 2]),
            is_first=True,
        ),
        _frame(0.1, _proprio_with_ee(orientation=q180), np.array([0.0, 0.0, 0.0]), is_last=True),
    ]
    # predicted = identity composed with a 90deg-z delta = a 90deg rotation
    # (matches the quaternion-math tests above); actual is q180 -- residual
    # should be the angular distance between a 90deg and a 180deg rotation
    # about the same axis, i.e. exactly 90 degrees = pi/2 radians
    results = check_action_state_consistency(
        _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
    )
    result = results["orientation:ee"]
    assert result.max_residual == pytest.approx(np.pi / 2, abs=1e-9)


def test_gripper_mismatch_flagged_at_known_residual() -> None:
    action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    frames = [
        _frame(0.0, _proprio_with_ee(gripper=1.0), np.array([0.8]), is_first=True),
        _frame(0.1, _proprio_with_ee(gripper=0.7), np.array([0.0]), is_last=True),
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
    )
    result = results["gripper:ee"]
    assert result.max_residual == pytest.approx(0.1)  # |0.8 - 0.7|


def test_joint_mismatch_flagged_at_known_residual() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    frames = [
        _frame(
            0.0, _proprio_with_ee(joint_positions=np.array([0.0])), np.array([0.2]), is_first=True
        ),
        _frame(
            0.1, _proprio_with_ee(joint_positions=np.array([0.25])), np.array([0.0]), is_last=True
        ),
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
    )
    result = results["joint:0"]
    assert result.max_residual == pytest.approx(0.05)  # |0.2 - 0.25|


def test_missing_position_at_next_frame_raises() -> None:
    frames = [
        _frame(0.0, _proprio_with_ee(end_effector="ee"), np.array([0.1, 0.0, 0.0]), is_first=True),
        _frame(
            0.1, _proprio_with_ee(end_effector="other"), np.array([0.0, 0.0, 0.0]), is_last=True
        ),
    ]
    with pytest.raises(ValueError, match="no ee_poses"):
        check_action_state_consistency(
            _episode_from_frames(frames),
            _POSITION_DELTA_ACTION_SPACE,
            _POSITION_TARGET_SPEC,
            _ZERO_TOLERANCE,
        )


def test_all_none_target_spec_returns_empty_dict() -> None:
    action_space = (ActionDimSpec("unlabeled", "delta", "normalized", "unknown"),)
    target_spec = (None,)
    frames = [
        _frame(0.0, _proprio_with_ee(), np.array([0.0]), is_first=True),
        _frame(0.1, _proprio_with_ee(), np.array([0.0]), is_last=True),
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
    )
    assert results == {}


def test_single_frame_episode_returns_empty_dict() -> None:
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(position=(0.0, 0.0, 0.0)),
            np.array([0.1, 0.0, 0.0]),
            is_first=True,
            is_last=True,
        )
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames),
        _POSITION_DELTA_ACTION_SPACE,
        _POSITION_TARGET_SPEC,
        _ZERO_TOLERANCE,
    )
    assert results == {}


def test_multi_step_episode_reports_correct_step_count_and_indices() -> None:
    # 4 frames -> 3 steps; inject one bad action at step index 1 (the
    # second frame carries it)
    frames = [
        _frame(
            0.0,
            _proprio_with_ee(position=(0.0, 0.0, 0.0)),
            np.array([0.1, 0.0, 0.0]),
            is_first=True,
        ),
        _frame(0.1, _proprio_with_ee(position=(0.1, 0.0, 0.0)), np.array([0.1, 0.0, 0.0])),
        _frame(
            0.2, _proprio_with_ee(position=(0.5, 0.0, 0.0)), np.array([0.1, 0.0, 0.0])
        ),  # bad jump
        _frame(
            0.3, _proprio_with_ee(position=(0.6, 0.0, 0.0)), np.array([0.0, 0.0, 0.0]), is_last=True
        ),
    ]
    results = check_action_state_consistency(
        _episode_from_frames(frames),
        _POSITION_DELTA_ACTION_SPACE,
        _POSITION_TARGET_SPEC,
        _ZERO_TOLERANCE,
    )
    result = results["position:ee"]
    assert result.n_steps == 3
    assert result.violation_indices == (1,)
    assert result.violation_residuals[0] == pytest.approx(0.3)  # |0.2 - 0.5|
    assert result.violation_fraction == pytest.approx(1 / 3)


def test_missing_orientation_at_next_frame_raises() -> None:
    action_space = (
        ActionDimSpec("rx", "delta", "rad", "ee"),
        ActionDimSpec("ry", "delta", "rad", "ee"),
        ActionDimSpec("rz", "delta", "rad", "ee"),
    )
    target_spec = (
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
        ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    )
    frames = [
        _frame(0.0, _proprio_with_ee(end_effector="ee"), np.array([0.0, 0.0, 0.0]), is_first=True),
        _frame(
            0.1, _proprio_with_ee(end_effector="other"), np.array([0.0, 0.0, 0.0]), is_last=True
        ),
    ]
    with pytest.raises(ValueError, match="no ee_poses"):
        check_action_state_consistency(
            _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
        )


def test_missing_gripper_at_next_frame_raises() -> None:
    action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
    target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
    frames = [
        _frame(0.0, _proprio_with_ee(gripper=1.0), np.array([0.5]), is_first=True),
        _frame(0.1, _proprio_with_ee(gripper=None), np.array([0.0]), is_last=True),
    ]
    with pytest.raises(ValueError, match="no grippers"):
        check_action_state_consistency(
            _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
        )


def test_missing_joint_at_next_frame_raises() -> None:
    action_space = (ActionDimSpec("j0", "delta", "rad", "joint"),)
    target_spec = (ActionTarget(kind="joint", joint_index=0),)
    frames = [
        _frame(
            0.0, _proprio_with_ee(joint_positions=np.array([0.0])), np.array([0.1]), is_first=True
        ),
        _frame(0.1, _proprio_with_ee(joint_positions=None), np.array([0.0]), is_last=True),
    ]
    with pytest.raises(ValueError, match="no matching"):
        check_action_state_consistency(
            _episode_from_frames(frames), action_space, target_spec, _ZERO_TOLERANCE
        )
