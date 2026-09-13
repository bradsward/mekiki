"""Action-state consistency: does the recorded action explain the observed
state transition? See ``docs/consistency.md`` for the full design — this
module implements it, doesn't re-decide it.

Covers the correspondence problem (`ActionTarget`/`ActionTargetSpec`,
`validate_action_target_spec` — deliberately kept separate from
`ActionDimSpec`/`Episode`, see ``docs/consistency.md``) and single-step
forward integration (`predict_next_proprioception`), turning a validated
spec plus one recorded action into a predicted next-step proprioception.
The residual/tolerance check that compares a prediction against what was
actually recorded isn't implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mekiki.episode import ActionDimSpec, ActionSpaceSpec, Proprioception

ActionTargetKind = Literal["position_axis", "orientation_axis", "gripper", "joint"]

_VECTOR_AXES = frozenset({"x", "y", "z"})
_AXIS_ORDER = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class ActionTarget:
    """What one action dimension predicts, physically.

    Attributes:
        kind: Which kind of state quantity this dimension predicts.
        end_effector: For ``"position_axis"``, ``"orientation_axis"``, and
            ``"gripper"`` — the key into `mekiki.episode.Proprioception`'s
            ``ee_poses``/``grippers`` dicts. Required for those kinds,
            unused for ``"joint"``.
        axis: For ``"position_axis"``/``"orientation_axis"`` — which
            component of the 3-vector this dimension is (position: x/y/z
            in the target's frame; orientation: the corresponding
            component of a small-angle rotation vector). All three axes
            for a given ``(kind, end_effector)`` pair must appear together
            in an `ActionTargetSpec` — see `validate_action_target_spec`.
            Required for those two kinds, unused otherwise.
        joint_index: For ``"joint"`` — the index into
            ``Proprioception.joint_positions`` this dimension predicts.
            Required for that kind, unused otherwise.
    """

    kind: ActionTargetKind
    end_effector: str | None = None
    axis: Literal["x", "y", "z"] | None = None
    joint_index: int | None = None


#: One entry per dimension in the episode's `ActionSpaceSpec`, same order,
#: same length. A dimension mekiki can't check yet (e.g. raw, uninterpreted
#: state with nothing in `mekiki.episode.Proprioception` to compare against)
#: maps to ``None`` — never guessed.
ActionTargetSpec = tuple[ActionTarget | None, ...]


@dataclass(frozen=True, slots=True)
class _GroupedTargets:
    """Action dimensions grouped by what they jointly predict.

    ``position``/``orientation`` map end_effector -> {axis -> action index}.
    ``gripper`` maps end_effector -> action index. ``joint`` maps
    joint_index -> action index. Built once by `_group_targets` and reused
    by both `validate_action_target_spec` and `predict_next_proprioception`
    so the grouping logic exists in exactly one place.
    """

    position: dict[str, dict[str, int]]
    orientation: dict[str, dict[str, int]]
    gripper: dict[str, int]
    joint: dict[int, int]


def _group_targets(target_spec: ActionTargetSpec) -> _GroupedTargets:
    position: dict[str, dict[str, int]] = {}
    orientation: dict[str, dict[str, int]] = {}
    gripper: dict[str, int] = {}
    joint: dict[int, int] = {}

    for i, target in enumerate(target_spec):
        if target is None:
            continue

        if target.kind in ("position_axis", "orientation_axis"):
            if target.end_effector is None or target.axis is None:
                raise ValueError(
                    f"a {target.kind!r} target requires both end_effector and axis, "
                    f"got end_effector={target.end_effector!r}, axis={target.axis!r}"
                )
            group = position if target.kind == "position_axis" else orientation
            axes = group.setdefault(target.end_effector, {})
            if target.axis in axes:
                raise ValueError(
                    f"duplicate axis {target.axis!r} for {target.kind} target on "
                    f"end_effector {target.end_effector!r} — each axis may appear "
                    "at most once per (kind, end_effector) group"
                )
            axes[target.axis] = i
        elif target.kind == "gripper":
            if target.end_effector is None:
                raise ValueError("a 'gripper' target requires end_effector, got None")
            gripper[target.end_effector] = i
        elif target.kind == "joint":
            if target.joint_index is None:
                raise ValueError("a 'joint' target requires joint_index, got None")
            joint[target.joint_index] = i

    return _GroupedTargets(position=position, orientation=orientation, gripper=gripper, joint=joint)


def _require_complete_and_consistent_group(
    end_effector: str, kind: str, axes: dict[str, int], action_space: ActionSpaceSpec
) -> None:
    if set(axes) != _VECTOR_AXES:
        missing = sorted(_VECTOR_AXES - set(axes))
        raise ValueError(
            f"{kind} target for end_effector {end_effector!r} is missing axes "
            f"{missing} — x, y, and z must all be present together to integrate "
            "as one vector, per docs/consistency.md (never per-axis independently)"
        )
    specs = [action_space[axes[axis]] for axis in _AXIS_ORDER]
    modes = {s.mode for s in specs}
    frames = {s.frame for s in specs}
    units = {s.unit for s in specs}
    if len(modes) > 1:
        raise ValueError(
            f"{kind} target for end_effector {end_effector!r} has inconsistent modes "
            f"across its x/y/z dimensions ({sorted(modes)}) — a vector must be "
            "integrated with one consistent mode"
        )
    if len(frames) > 1:
        raise ValueError(
            f"{kind} target for end_effector {end_effector!r} has inconsistent frames "
            f"across its x/y/z dimensions ({sorted(frames)}) — a vector must be "
            "integrated in one consistent frame"
        )
    if len(units) > 1:
        raise ValueError(
            f"{kind} target for end_effector {end_effector!r} has inconsistent units "
            f"across its x/y/z dimensions ({sorted(units)})"
        )


def validate_action_target_spec(
    action_space: ActionSpaceSpec, target_spec: ActionTargetSpec
) -> None:
    """Check a caller-supplied `ActionTargetSpec` is complete enough to integrate.

    Per ``docs/consistency.md``: position and orientation deltas are
    integrated as whole 3-vectors, never per-axis — so every
    ``(kind, end_effector)`` group of ``"position_axis"``/``"orientation_axis"``
    targets must supply exactly the three axes ``x``, ``y``, ``z``, each
    exactly once, and all three must share the same
    `mekiki.episode.ActionDimSpec` mode/frame/unit — a vector can't be
    integrated with e.g. one axis delta and another absolute. This is a
    structural check only; it says nothing about whether the resulting
    *values* are correct, only that the spec itself isn't ambiguous.

    Args:
        action_space: The episode's action space (`mekiki.episode.ActionSpaceSpec`).
        target_spec: The caller-supplied correspondence mapping to validate.

    Raises:
        ValueError: ``target_spec`` doesn't have exactly one entry per
            dimension in ``action_space``; a ``"position_axis"`` or
            ``"orientation_axis"`` target is missing its ``end_effector``
            or ``axis``; a ``"gripper"`` target is missing its
            ``end_effector``; a ``"joint"`` target is missing its
            ``joint_index``; a ``(kind, end_effector)`` group has a
            duplicate or missing axis; or a group's three axes disagree on
            mode, frame, or unit.

    Example:
        >>> from mekiki.episode import ActionDimSpec
        >>> action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
        >>> target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
        >>> validate_action_target_spec(action_space, target_spec)  # no error
    """
    if len(target_spec) != len(action_space):
        raise ValueError(
            f"action_target_spec has {len(target_spec)} entries but action_space has "
            f"{len(action_space)} — must be exactly one target per action dimension "
            "(use None for a dimension that can't be checked yet)."
        )

    grouped = _group_targets(target_spec)

    for end_effector, axes in grouped.position.items():
        _require_complete_and_consistent_group(end_effector, "position_axis", axes, action_space)
    for end_effector, axes in grouped.orientation.items():
        _require_complete_and_consistent_group(end_effector, "orientation_axis", axes, action_space)


# --- quaternion math -------------------------------------------------------
#
# Quaternions throughout this module are (x, y, z, w), unit norm, matching
# `mekiki.episode.Pose.orientation`. These are the standard, well-known
# formulas (Hamilton product; the usual unit-quaternion-to-rotation-matrix
# conversion; the exponential map from a rotation vector) — not reinvented,
# and each has a hand-computable test case in tests/checks/test_consistency.py
# (e.g. a 90-degree rotation about z) rather than just "looks plausible."


def _quaternion_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quaternion(
    q: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = _quaternion_to_rotation_matrix(q) @ v
    return result


def _quaternion_from_rotation_vector(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponential map: a rotation vector (axis * angle, radians) to a unit
    quaternion. The zero vector maps to the identity rotation."""
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = r / angle
    half = angle / 2.0
    sin_half = np.sin(half)
    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)],
        dtype=np.float64,
    )


def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product q1 ⊗ q2 — composes q2 as "applied first" in q1's own
    (body) frame, then q1. Both and the result are (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


# --- prediction -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictedProprioception:
    """Predicted next-step values for whichever quantities an
    `ActionTargetSpec` actually modeled.

    Mirrors `mekiki.episode.Proprioception`'s own dict-keyed shape
    deliberately — a quantity that wasn't modeled (mapped to `None` in the
    spec) is simply absent here, the same "nothing to guess" convention as
    `Proprioception.extra` and friends. Comparing a prediction against a
    recorded `Proprioception` is a later step (the residual/tolerance
    check); this type only reports what's predicted.

    Attributes:
        ee_positions: end_effector -> predicted ``(3,)`` position, meters.
        ee_orientations: end_effector -> predicted ``(4,)`` unit quaternion
            ``(x, y, z, w)``.
        grippers: end_effector -> predicted normalized ``[0, 1]`` value.
        joint_positions: joint_index -> predicted value, radians.
    """

    ee_positions: dict[str, NDArray[np.float64]]
    ee_orientations: dict[str, NDArray[np.float64]]
    grippers: dict[str, float]
    joint_positions: dict[int, float]


def _predict_position(
    end_effector: str,
    axes: dict[str, int],
    action: NDArray[np.float64],
    action_space: ActionSpaceSpec,
    proprioception: Proprioception,
) -> NDArray[np.float64]:
    specs = [action_space[axes[axis]] for axis in _AXIS_ORDER]
    if specs[0].unit != "m":
        raise ValueError(
            f"position_axis target for end_effector {end_effector!r} has unit "
            f"{specs[0].unit!r} — position deltas/targets must be in meters ('m')"
        )
    values = np.array([action[axes[axis]] for axis in _AXIS_ORDER], dtype=np.float64)

    current_pose = proprioception.ee_poses.get(end_effector)
    if current_pose is None:
        raise ValueError(
            f"action_target_spec has a position_axis target for end_effector "
            f"{end_effector!r}, but this frame's proprioception has no "
            f"ee_poses[{end_effector!r}]"
        )

    mode = specs[0].mode
    frame = specs[0].frame

    if mode == "absolute":
        if frame != current_pose.frame:
            raise ValueError(
                f"position_axis target for end_effector {end_effector!r} is absolute "
                f"but declared in frame {frame!r}, while the state's own frame is "
                f"{current_pose.frame!r} — absolute targets must match the state's "
                "own frame exactly (docs/consistency.md)"
            )
        return values

    # mode == "delta"
    if frame == current_pose.frame:
        result: NDArray[np.float64] = current_pose.position + values
        return result
    if frame == "ee":
        rotated = _rotate_vector_by_quaternion(current_pose.orientation, values)
        result = current_pose.position + rotated
        return result
    raise ValueError(
        f"position_axis delta target for end_effector {end_effector!r} is declared "
        f"in frame {frame!r}, which is neither the state's own frame "
        f"({current_pose.frame!r}) nor 'ee' — unsupported frame transform, see "
        "docs/consistency.md"
    )


def _predict_orientation(
    end_effector: str,
    axes: dict[str, int],
    action: NDArray[np.float64],
    action_space: ActionSpaceSpec,
    proprioception: Proprioception,
) -> NDArray[np.float64]:
    specs = [action_space[axes[axis]] for axis in _AXIS_ORDER]
    if specs[0].unit != "rad":
        raise ValueError(
            f"orientation_axis target for end_effector {end_effector!r} has unit "
            f"{specs[0].unit!r} — orientation deltas must be in radians ('rad')"
        )
    if specs[0].mode != "delta":
        raise ValueError(
            f"orientation_axis target for end_effector {end_effector!r} has mode "
            f"{specs[0].mode!r} — absolute orientation targets aren't supported yet "
            "(docs/consistency.md v1 scope), only 'delta' is"
        )
    if specs[0].frame != "ee":
        raise ValueError(
            f"orientation_axis delta target for end_effector {end_effector!r} is "
            f"declared in frame {specs[0].frame!r} — only 'ee' (body-frame) "
            "orientation deltas are supported so far. A same-frame (world/base) "
            "orientation delta needs the opposite quaternion composition order "
            "(pre-multiply, not post-multiply), which docs/consistency.md hasn't "
            "specified yet — see the parked item in STATE.md rather than guessing "
            "the composition order here."
        )
    rotation_vector = np.array([action[axes[axis]] for axis in _AXIS_ORDER], dtype=np.float64)
    current_pose = proprioception.ee_poses.get(end_effector)
    if current_pose is None:
        raise ValueError(
            f"action_target_spec has an orientation_axis target for end_effector "
            f"{end_effector!r}, but this frame's proprioception has no "
            f"ee_poses[{end_effector!r}]"
        )
    delta_q = _quaternion_from_rotation_vector(rotation_vector)
    return _quaternion_multiply(current_pose.orientation, delta_q)


def _predict_gripper(action_value: float, spec: ActionDimSpec) -> float:
    if spec.unit != "normalized":
        raise ValueError(
            f"gripper target has unit {spec.unit!r}, expected 'normalized' in "
            "[0, 1] — any other unit would need an explicit scale factor mekiki "
            "doesn't have"
        )
    if spec.mode != "absolute":
        raise ValueError(
            f"gripper target has mode {spec.mode!r} — only 'absolute' is supported "
            "yet; docs/consistency.md doesn't specify a delta-gripper composition"
        )
    return float(action_value)


def _predict_joint(
    joint_index: int,
    action_value: float,
    spec: ActionDimSpec,
    proprioception: Proprioception,
) -> float:
    if spec.mode == "absolute":
        return float(action_value)
    # mode == "delta"
    if proprioception.joint_positions is None:
        raise ValueError(
            f"joint target (index {joint_index}) is a delta, but this frame's "
            "proprioception has no joint_positions to add it to"
        )
    if joint_index >= len(proprioception.joint_positions):
        raise ValueError(
            f"joint target index {joint_index} is out of range for "
            f"joint_positions of length {len(proprioception.joint_positions)}"
        )
    return float(proprioception.joint_positions[joint_index] + action_value)


def predict_next_proprioception(
    proprioception: Proprioception,
    action: NDArray[np.float64],
    action_space: ActionSpaceSpec,
    target_spec: ActionTargetSpec,
) -> PredictedProprioception:
    """Forward-integrate one step: predict proprioception at ``t+1`` from
    the proprioception at ``t`` and the action recorded at ``t``.

    Single-step only — see docs/consistency.md for why multi-step rollout
    is a different, harder problem this function doesn't attempt.

    Args:
        proprioception: Recorded proprioception at ``t``.
        action: The action vector recorded at ``t``, same length and
            dimension order as ``action_space``.
        action_space: The episode's action space.
        target_spec: What each action dimension predicts. Validated here
            via `validate_action_target_spec` — callers don't need to
            validate it themselves first, but doing so once per episode
            rather than once per frame avoids redundant work.

    Returns:
        Predicted values for whichever quantities ``target_spec`` actually
        modeled. Anything mapped to `None` in ``target_spec`` is simply
        absent from the result — there's nothing to compare it against.

    Raises:
        ValueError: ``target_spec`` fails `validate_action_target_spec`;
            ``action``'s shape doesn't match ``action_space``; a
            referenced end-effector or joint index isn't present in
            ``proprioception``; a unit doesn't match what a target kind
            requires; or a frame/mode combination isn't one of the cases
            docs/consistency.md supports (see that doc, and the specific
            per-kind functions in this module, for exactly which).
    """
    validate_action_target_spec(action_space, target_spec)
    if action.shape != (len(action_space),):
        raise ValueError(
            f"action has shape {action.shape} but action_space has {len(action_space)} dimensions"
        )

    grouped = _group_targets(target_spec)

    ee_positions = {
        end_effector: _predict_position(end_effector, axes, action, action_space, proprioception)
        for end_effector, axes in grouped.position.items()
    }
    ee_orientations = {
        end_effector: _predict_orientation(end_effector, axes, action, action_space, proprioception)
        for end_effector, axes in grouped.orientation.items()
    }
    grippers = {
        end_effector: _predict_gripper(float(action[i]), action_space[i])
        for end_effector, i in grouped.gripper.items()
    }
    joint_positions = {
        joint_index: _predict_joint(joint_index, float(action[i]), action_space[i], proprioception)
        for joint_index, i in grouped.joint.items()
    }

    return PredictedProprioception(
        ee_positions=ee_positions,
        ee_orientations=ee_orientations,
        grippers=grippers,
        joint_positions=joint_positions,
    )
