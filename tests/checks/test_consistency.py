"""Tests for `mekiki.checks.consistency`."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.checks.consistency import (
    ActionTarget,
    _quaternion_from_rotation_vector,
    _quaternion_multiply,
    _quaternion_to_rotation_matrix,
    predict_next_proprioception,
    validate_action_target_spec,
)
from mekiki.episode import ActionDimSpec, Pose, Proprioception

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
