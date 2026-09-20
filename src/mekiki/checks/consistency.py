"""Action-state consistency: does the recorded action explain the observed
state transition? See ``docs/consistency.md`` for the full design — this
module implements it, doesn't re-decide it.

Covers the correspondence problem (`ActionTarget`/`ActionTargetSpec`,
`validate_action_target_spec` — deliberately kept separate from
`ActionDimSpec`/`Episode`, see ``docs/consistency.md``), single-step
forward integration (`predict_next_proprioception`), and the residual +
tolerance check itself (`check_action_state_consistency`, `ToleranceModel`)
that compares a prediction against what was actually recorded, in physical
units, against a caller-supplied — never invented — tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mekiki.episode import ActionDimSpec, ActionSpaceSpec, Episode, Proprioception
from mekiki.rotation import (
    quaternion_angular_distance,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)

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
        rotated = rotate_vector_by_quaternion(current_pose.orientation, values)
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
    rotation_vector = np.array([action[axes[axis]] for axis in _AXIS_ORDER], dtype=np.float64)
    current_pose = proprioception.ee_poses.get(end_effector)
    if current_pose is None:
        raise ValueError(
            f"action_target_spec has an orientation_axis target for end_effector "
            f"{end_effector!r}, but this frame's proprioception has no "
            f"ee_poses[{end_effector!r}]"
        )
    delta_q = quaternion_from_rotation_vector(rotation_vector)
    frame = specs[0].frame
    if frame == "ee":
        # body-frame delta: applied about the end-effector's own current axes
        return quaternion_multiply(current_pose.orientation, delta_q)
    if frame == current_pose.frame:
        # fixed/world-frame delta: applied about the fixed axes
        return quaternion_multiply(delta_q, current_pose.orientation)
    raise ValueError(
        f"orientation_axis delta target for end_effector {end_effector!r} is declared "
        f"in frame {frame!r}, which is neither the state's own frame "
        f"({current_pose.frame!r}) nor 'ee' — unsupported frame transform, see "
        "docs/consistency.md"
    )


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


# --- tolerance ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToleranceModel:
    """Caller-supplied tolerance for action-state consistency residuals.

    Per docs/consistency.md, tolerance is never invented by mekiki — every
    field here comes from the caller (derived from the dataset's own robot
    type and control loop), the same way `nominal_hz` does in
    `mekiki.checks.temporal`. Each quantity's tolerance at a given ``dt``
    (seconds between the two frames being compared) is
    ``base + rate_per_second * dt``: a floor independent of timestep
    (sensor/quantization noise) plus a term that grows with how long the
    integration window is (how much a real, non-guessed controller can
    plausibly drift from a naive open-loop prediction over that time).

    Attributes:
        position_base_m: Position residual floor, meters.
        position_rate_per_second_m: Additional position tolerance per
            second of ``dt``, meters/second.
        orientation_base_rad: Orientation residual floor, radians.
        orientation_rate_per_second_rad: Additional orientation tolerance
            per second of ``dt``, radians/second.
        gripper_base: Gripper residual floor, normalized ``[0, 1]`` fraction.
        gripper_rate_per_second: Additional gripper tolerance per second,
            normalized fraction/second.
        joint_base_rad: Per-joint residual floor, radians.
        joint_rate_per_second_rad: Additional per-joint tolerance per
            second, radians/second.
    """

    position_base_m: float
    position_rate_per_second_m: float
    orientation_base_rad: float
    orientation_rate_per_second_rad: float
    gripper_base: float
    gripper_rate_per_second: float
    joint_base_rad: float
    joint_rate_per_second_rad: float

    def __post_init__(self) -> None:
        for field_name in (
            "position_base_m",
            "position_rate_per_second_m",
            "orientation_base_rad",
            "orientation_rate_per_second_rad",
            "gripper_base",
            "gripper_rate_per_second",
            "joint_base_rad",
            "joint_rate_per_second_rad",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative, got {value}")

    def position(self, dt: float) -> float:
        """Position tolerance at this ``dt`` (seconds), in meters."""
        return self.position_base_m + self.position_rate_per_second_m * dt

    def orientation(self, dt: float) -> float:
        """Orientation tolerance at this ``dt`` (seconds), in radians."""
        return self.orientation_base_rad + self.orientation_rate_per_second_rad * dt

    def gripper(self, dt: float) -> float:
        """Gripper tolerance at this ``dt`` (seconds), normalized fraction."""
        return self.gripper_base + self.gripper_rate_per_second * dt

    def joint(self, dt: float) -> float:
        """Per-joint tolerance at this ``dt`` (seconds), in radians."""
        return self.joint_base_rad + self.joint_rate_per_second_rad * dt


# --- residual / consistency check ---------------------------------------


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Result of checking one quantity's action-state consistency across
    an episode.

    One `ConsistencyResult` covers a single quantity — one end-effector's
    position, one end-effector's orientation, one end-effector's gripper,
    or one joint — never several combined, so a caller can see exactly
    which quantity misbehaved.

    Attributes:
        n_steps: Number of ``(t, t+1)`` step-pairs checked for this quantity.
        max_residual: Largest residual observed across the whole episode,
            in this quantity's own physical unit (meters for position,
            radians for orientation/joint, normalized fraction for gripper).
        violation_indices: Indices of the frame that *carried the action*
            found inconsistent (i.e. ``t``, not ``t+1``) — this is the
            frame an auditor would look at to see what was commanded.
        violation_residuals: The residual at each of ``violation_indices``,
            same order, same unit as ``max_residual``.
        violation_thresholds: The tolerance actually in force at each
            violation (``ToleranceModel`` evaluated at that step's ``dt``),
            same order — since tolerance varies with ``dt``, a bare
            residual number alone wouldn't say how far over the line it was.
    """

    n_steps: int
    max_residual: float
    violation_indices: tuple[int, ...]
    violation_residuals: tuple[float, ...]
    violation_thresholds: tuple[float, ...]

    @property
    def violation_fraction(self) -> float:
        """Violations as a fraction of steps checked for this quantity."""
        return len(self.violation_indices) / self.n_steps if self.n_steps > 0 else 0.0


class _RunningConsistency:
    """Mutable accumulator for one quantity while streaming an episode.

    Not part of the public API — `check_action_state_consistency` builds
    one of these per quantity key and calls `.finish()` once at the end to
    produce the frozen `ConsistencyResult` callers actually see.
    """

    def __init__(self) -> None:
        self.n_steps = 0
        self.max_residual = 0.0
        self.violation_indices: list[int] = []
        self.violation_residuals: list[float] = []
        self.violation_thresholds: list[float] = []

    def update(self, residual: float, threshold: float, frame_index: int) -> None:
        self.n_steps += 1
        self.max_residual = max(self.max_residual, residual)
        if residual > threshold:
            self.violation_indices.append(frame_index)
            self.violation_residuals.append(residual)
            self.violation_thresholds.append(threshold)

    def finish(self) -> ConsistencyResult:
        return ConsistencyResult(
            n_steps=self.n_steps,
            max_residual=self.max_residual,
            violation_indices=tuple(self.violation_indices),
            violation_residuals=tuple(self.violation_residuals),
            violation_thresholds=tuple(self.violation_thresholds),
        )


def check_action_state_consistency(
    episode: Episode,
    action_space: ActionSpaceSpec,
    target_spec: ActionTargetSpec,
    tolerance: ToleranceModel,
) -> dict[str, ConsistencyResult]:
    """Check whether recorded actions explain the observed state transitions.

    Streams the episode once (per docs/episode.md), forward-integrating
    each frame's action via `predict_next_proprioception` and comparing the
    prediction against what the *next* frame actually recorded, in the
    physical units docs/consistency.md specifies. Single-step only — see
    that doc for why multi-step rollout comparison is a different problem.

    Args:
        episode: Episode to check. Its frame timestamps are assumed
            already known monotonic (`mekiki.checks.temporal.check_timestamp_monotonicity`
            is a precondition here, the same way it's a precondition for
            the M2 rate checks) — this function does not re-check that.
        action_space: The episode's action space.
        target_spec: What each action dimension predicts. Validated once,
            up front — not re-validated per frame.
        tolerance: Caller-supplied tolerance model (never invented by
            mekiki) — see `ToleranceModel`.

    Returns:
        A dict keyed by ``"position:<end_effector>"``,
        ``"orientation:<end_effector>"``, ``"gripper:<end_effector>"``, or
        ``"joint:<joint_index>"`` — one `ConsistencyResult` per quantity
        ``target_spec`` actually modeled. Empty if every dimension mapped
        to `None` (nothing to check — not an error).

    Raises:
        ValueError: ``target_spec`` fails `validate_action_target_spec`;
            `predict_next_proprioception` raises for any frame (unsupported
            frame/mode, missing end-effector or joint in a frame's
            proprioception, wrong unit); or a modeled quantity's actual
            value is missing from the *next* frame's proprioception.

    Example:
        >>> from pathlib import Path
        >>> from mekiki.readers.lerobot import read_episodes
        >>> action_space = (ActionDimSpec("gripper", "absolute", "normalized", "ee"),)
        >>> target_spec = (ActionTarget(kind="gripper", end_effector="ee"),)
        >>> tolerance = ToleranceModel(
        ...     position_base_m=0.01, position_rate_per_second_m=0.05,
        ...     orientation_base_rad=0.05, orientation_rate_per_second_rad=0.2,
        ...     gripper_base=0.05, gripper_rate_per_second=0.0,
        ...     joint_base_rad=0.05, joint_rate_per_second_rad=0.2,
        ... )
        >>> dataset_dir = Path("~/data/pusht").expanduser()
        >>> episode = next(read_episodes(dataset_dir, action_space))  # doctest: +SKIP
        >>> results = check_action_state_consistency(
        ...     episode, action_space, target_spec, tolerance
        ... )  # doctest: +SKIP
    """
    validate_action_target_spec(action_space, target_spec)

    running: dict[str, _RunningConsistency] = {}

    def _get(key: str) -> _RunningConsistency:
        if key not in running:
            running[key] = _RunningConsistency()
        return running[key]

    previous_frame = None
    previous_index = -1

    for i, frame in enumerate(episode):
        if previous_frame is not None:
            dt = frame.timestamp - previous_frame.timestamp
            predicted = predict_next_proprioception(
                previous_frame.proprioception, previous_frame.action, action_space, target_spec
            )

            for end_effector, predicted_position in predicted.ee_positions.items():
                actual_pose = frame.proprioception.ee_poses.get(end_effector)
                if actual_pose is None:
                    raise ValueError(
                        f"predicted a position for end_effector {end_effector!r} but "
                        f"frame {i} has no ee_poses[{end_effector!r}] to compare against"
                    )
                residual = float(np.linalg.norm(predicted_position - actual_pose.position))
                _get(f"position:{end_effector}").update(
                    residual, tolerance.position(dt), previous_index
                )

            for end_effector, predicted_orientation in predicted.ee_orientations.items():
                actual_pose = frame.proprioception.ee_poses.get(end_effector)
                if actual_pose is None:
                    raise ValueError(
                        f"predicted an orientation for end_effector {end_effector!r} but "
                        f"frame {i} has no ee_poses[{end_effector!r}] to compare against"
                    )
                residual = quaternion_angular_distance(
                    predicted_orientation, actual_pose.orientation
                )
                _get(f"orientation:{end_effector}").update(
                    residual, tolerance.orientation(dt), previous_index
                )

            for end_effector, predicted_gripper in predicted.grippers.items():
                actual_gripper = frame.proprioception.grippers.get(end_effector)
                if actual_gripper is None:
                    raise ValueError(
                        f"predicted a gripper value for end_effector {end_effector!r} but "
                        f"frame {i} has no grippers[{end_effector!r}] to compare against"
                    )
                residual = abs(predicted_gripper - actual_gripper)
                _get(f"gripper:{end_effector}").update(
                    residual, tolerance.gripper(dt), previous_index
                )

            for joint_index, predicted_joint in predicted.joint_positions.items():
                actual_joints = frame.proprioception.joint_positions
                if actual_joints is None or joint_index >= len(actual_joints):
                    raise ValueError(
                        f"predicted joint {joint_index} but frame {i} has no matching "
                        "joint_positions to compare against"
                    )
                residual = abs(predicted_joint - float(actual_joints[joint_index]))
                _get(f"joint:{joint_index}").update(residual, tolerance.joint(dt), previous_index)

        previous_frame = frame
        previous_index = i

    return {key: acc.finish() for key, acc in running.items()}
