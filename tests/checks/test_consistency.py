"""Tests for `mekiki.checks.consistency`."""

from __future__ import annotations

import pytest

from mekiki.checks.consistency import ActionTarget, validate_action_target_spec
from mekiki.episode import ActionDimSpec

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
