"""Action-state consistency: does the recorded action explain the observed
state transition? See ``docs/consistency.md`` for the full design — this
module implements it, doesn't re-decide it.

This file currently covers the correspondence problem only: `ActionTarget`
says what physical quantity one action dimension predicts, and
`validate_action_target_spec` checks a caller-supplied mapping is complete
enough to integrate. Deliberately kept separate from `ActionDimSpec` and
`Episode` (``mekiki.episode``) — see ``docs/consistency.md``'s "correspondence
problem" section for why. Forward integration itself (turning a validated
spec into a predicted next state) isn't implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mekiki.episode import ActionSpaceSpec

ActionTargetKind = Literal["position_axis", "orientation_axis", "gripper", "joint"]

_VECTOR_AXES = frozenset({"x", "y", "z"})


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


def validate_action_target_spec(
    action_space: ActionSpaceSpec, target_spec: ActionTargetSpec
) -> None:
    """Check a caller-supplied `ActionTargetSpec` is complete enough to integrate.

    Per ``docs/consistency.md``: position and orientation deltas are
    integrated as whole 3-vectors, never per-axis — so every
    ``(kind, end_effector)`` group of ``"position_axis"``/``"orientation_axis"``
    targets must supply exactly the three axes ``x``, ``y``, ``z``, each
    exactly once. This is a structural check only; it says nothing about
    whether the *values* mekiki will compute from this spec are correct,
    only that the spec itself isn't ambiguous or partial.

    Args:
        action_space: The episode's action space (`mekiki.episode.ActionSpaceSpec`).
        target_spec: The caller-supplied correspondence mapping to validate.

    Raises:
        ValueError: ``target_spec`` doesn't have exactly one entry per
            dimension in ``action_space``; a ``"position_axis"`` or
            ``"orientation_axis"`` target is missing its ``end_effector``
            or ``axis``; a ``"gripper"`` target is missing its
            ``end_effector``; a ``"joint"`` target is missing its
            ``joint_index``; or a ``(kind, end_effector)`` group has a
            duplicate or missing axis.

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

    vector_groups: dict[tuple[ActionTargetKind, str], set[str]] = {}

    for target in target_spec:
        if target is None:
            continue

        if target.kind in ("position_axis", "orientation_axis"):
            if target.end_effector is None or target.axis is None:
                raise ValueError(
                    f"a {target.kind!r} target requires both end_effector and axis, "
                    f"got end_effector={target.end_effector!r}, axis={target.axis!r}"
                )
            key = (target.kind, target.end_effector)
            axes = vector_groups.setdefault(key, set())
            if target.axis in axes:
                raise ValueError(
                    f"duplicate axis {target.axis!r} for {target.kind} target on "
                    f"end_effector {target.end_effector!r} — each axis may appear "
                    "at most once per (kind, end_effector) group"
                )
            axes.add(target.axis)
        elif target.kind == "gripper":
            if target.end_effector is None:
                raise ValueError("a 'gripper' target requires end_effector, got None")
        elif target.kind == "joint":
            if target.joint_index is None:
                raise ValueError("a 'joint' target requires joint_index, got None")

    for (kind, end_effector), axes in vector_groups.items():
        if axes != _VECTOR_AXES:
            missing = sorted(_VECTOR_AXES - axes)
            raise ValueError(
                f"{kind} target for end_effector {end_effector!r} is missing axes "
                f"{missing} — x, y, and z must all be present together to integrate "
                "as one vector, per docs/consistency.md (never per-axis independently)"
            )
