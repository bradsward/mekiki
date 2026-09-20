"""Turning a dataset's raw, uninterpreted ``Proprioception.extra`` values
into structured ``grippers``/``joint_positions``/``ee_poses`` fields real
checks can use. See ``docs/consistency.md``'s "Getting real proprioception
in" section for the design — this module implements it, doesn't re-decide it.

Symmetric to ``mekiki.checks.consistency.ActionTarget``/``ActionTargetSpec``,
but for state instead of actions: a caller-supplied, explicit, per-element
mapping from a raw ``extra[]`` array into structured `Proprioception`
fields, never inferred from a column name or shape. Kept out of both
``mekiki.episode`` (the stable M1 core) and ``mekiki.checks.consistency``
(M3-specific) — this is useful to any check that wants real
position/orientation/gripper/joint data, not just the consistency check.

An end-effector pose is reconstructed from three ``position_axis`` values
plus the ``orientation_axis`` components of one caller-declared
`mekiki.rotation.OrientationEncoding`. ``Pose`` requires both, so neither
is accepted alone — see `reconstruct_proprioception`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mekiki.episode import Episode, Frame, Pose, Proprioception
from mekiki.rotation import OrientationEncoding, orientation_to_quaternion

#: How far outside ``[0, 1]`` a raw value declared to be a normalized gripper
#: may sit before it's rejected instead of treated as sensor calibration noise.
#: Chosen from real data, not guessed: in 25 real ``bridge_orig_lerobot``
#: episodes 29% of frames (280 of 955) read above 1.0, by up to 0.018 — a
#: gripper sensor overshooting its calibrated range is ordinary, not a fault.
GRIPPER_RANGE_TOLERANCE = 0.05

StateFieldKind = Literal["position_axis", "orientation_axis", "gripper", "joint"]


@dataclass(frozen=True, slots=True)
class StateField:
    """Where one raw state value belongs, physically.

    Mirrors `mekiki.checks.consistency.ActionTarget`'s shape, but
    describes a *destination* in a reconstructed `Proprioception` for a
    value read out of `Proprioception.extra`, rather than what an action
    dimension predicts.

    Attributes:
        kind: What kind of structured field this value becomes.
        end_effector: For ``"gripper"``/``"position_axis"``/
            ``"orientation_axis"`` — the key to write into
            `Proprioception.grippers` / ``ee_poses``.
        axis: For ``"position_axis"`` — which component (x/y/z) of the
            position this value is.
        joint_index: For ``"joint"`` — which index into
            `Proprioception.joint_positions` this value becomes.
        frame: For ``"position_axis"``/``"orientation_axis"`` — the
            coordinate frame the pose is expressed in (becomes
            `mekiki.episode.Pose.frame`). Every field for one end-effector
            must name the same frame.
        component: For ``"orientation_axis"`` — which raw component of the
            declared encoding this value is (``0`` .. ``n_components - 1``,
            in the encoding's own order: e.g. for Euler angles, the first
            angle of the sequence, *not* necessarily x).
        encoding: For ``"orientation_axis"`` — how the orientation
            components are encoded. Caller-declared, never inferred; every
            field for one end-effector must declare the same encoding.
    """

    kind: StateFieldKind
    end_effector: str | None = None
    axis: Literal["x", "y", "z"] | None = None
    joint_index: int | None = None
    frame: str | None = None
    component: int | None = None
    encoding: OrientationEncoding | None = None


#: Maps a `Proprioception.extra` key to where each of that array's elements
#: belongs. One entry per element, same order, same length as the array.
#: ``None`` for an element that isn't being reconstructed — it simply stays
#: in ``extra``, unchanged, same as every other element (reconstruction
#: never removes anything from ``extra``, see `reconstruct_proprioception`).
StateFieldSpec = dict[str, tuple[StateField | None, ...]]


def _normalized_gripper(value: float, end_effector: str) -> float:
    """Clip small calibration overshoot into ``[0, 1]``; reject anything far outside.

    `Proprioception` requires normalized grippers in ``[0, 1]`` and mekiki
    keeps that contract strict. But a real sensor routinely reads a hair over
    1.0 (see `GRIPPER_RANGE_TOLERANCE`), and rejecting every such frame would
    make real data unusable. Within the tolerance the *structured* copy is
    clipped — the raw, unclipped value is left untouched in ``extra`` — while a
    value well outside means the caller pointed at the wrong column or a
    non-normalized unit, which is worth failing loudly on.
    """
    if not -GRIPPER_RANGE_TOLERANCE <= value <= 1.0 + GRIPPER_RANGE_TOLERANCE:
        raise ValueError(
            f"a raw value of {value} for gripper {end_effector!r} is more than "
            f"{GRIPPER_RANGE_TOLERANCE} outside [0, 1] — that's not calibration noise, "
            "it looks like the wrong column or a non-normalized unit"
        )
    return min(1.0, max(0.0, value))


def _build_pose(
    end_effector: str,
    position: dict[str, float] | None,
    orientation: dict[int, float] | None,
    encodings: set[OrientationEncoding],
    frames: set[str],
) -> Pose:
    if position is None or orientation is None:
        have = "position" if orientation is None else "orientation"
        missing = "orientation" if orientation is None else "position"
        raise ValueError(
            f"end_effector {end_effector!r} has {have} state fields but no {missing} — "
            "a Pose requires both, and mekiki won't fabricate an identity orientation "
            "(or a made-up position) to fill the gap"
        )
    if set(position) != {"x", "y", "z"}:
        missing_axes = sorted({"x", "y", "z"} - set(position))
        raise ValueError(
            f"position for end_effector {end_effector!r} is missing axes {missing_axes} — "
            "x, y, and z must all be present together"
        )
    if len(frames) != 1:
        raise ValueError(
            f"end_effector {end_effector!r} has state fields naming different frames "
            f"({sorted(frames)}) — a pose is expressed in exactly one"
        )
    if len(encodings) != 1:
        raise ValueError(
            f"end_effector {end_effector!r} declares {len(encodings)} different orientation "
            "encodings — a single orientation has exactly one"
        )
    encoding = next(iter(encodings))
    if set(orientation) != set(range(encoding.n_components)):
        raise ValueError(
            f"orientation for end_effector {end_effector!r} has components "
            f"{sorted(orientation)} but a {encoding.kind!r} encoding needs exactly "
            f"0..{encoding.n_components - 1}"
        )
    return Pose(
        position=np.array([position["x"], position["y"], position["z"]], dtype=np.float64),
        orientation=orientation_to_quaternion(
            np.array([orientation[i] for i in range(encoding.n_components)], dtype=np.float64),
            encoding,
        ),
        frame=next(iter(frames)),
    )


def reconstruct_proprioception(
    proprioception: Proprioception, state_field_spec: StateFieldSpec
) -> Proprioception:
    """Move values out of ``extra`` into structured fields, per a caller-supplied spec.

    ``extra`` is left completely unchanged — reconstructed values are
    *added* alongside it, never removed from it, so nothing is ever
    silently discarded even when only some of an array's elements are
    mapped, or a caller only wants to inspect the structured value while
    still keeping the raw one around.

    Args:
        proprioception: Source proprioception, exactly as a reader
            produced it (everything undecided in ``extra``).
        state_field_spec: Caller-supplied mapping — see `StateFieldSpec`.

    Returns:
        A new `Proprioception` with ``grippers``/``joint_positions``/
        ``ee_poses`` updated by whatever `state_field_spec` mapped,
        everything else (including ``extra``) unchanged from the source.

    Raises:
        ValueError: a referenced ``extra`` key doesn't exist on
            ``proprioception``; a spec tuple's length doesn't match that
            array's length; a field is missing something its kind
            requires; or an end-effector's pose is under-specified —
            position without orientation (or the reverse; ``Pose``
            requires both and mekiki won't fabricate an identity
            orientation), an incomplete or duplicated x/y/z position
            triple, orientation components that aren't exactly
            ``0..n-1`` for the declared encoding, mixed encodings, or
            disagreeing frames. An orientation whose values don't fit its
            encoding (wrong length, a materially non-unit quaternion) is
            also rejected — see `mekiki.rotation.orientation_to_quaternion`.

    Example:
        >>> import numpy as np
        >>> from mekiki.episode import Proprioception
        >>> proprioception = Proprioception(
        ...     joint_positions=None, joint_velocities=None, ee_poses={},
        ...     grippers={}, extra={"observation.state": np.array([1.0, 2.0, 0.7])},
        ... )
        >>> gripper_field = StateField(kind="gripper", end_effector="ee")
        >>> spec = {"observation.state": (None, None, gripper_field)}
        >>> result = reconstruct_proprioception(proprioception, spec)
        >>> result.grippers["ee"]
        0.7
        >>> "observation.state" in result.extra  # never removed
        True
    """
    gripper_values: dict[str, float] = {}
    joint_values: dict[int, float] = {}
    position_values: dict[str, dict[str, float]] = {}
    orientation_values: dict[str, dict[int, float]] = {}
    orientation_encodings: dict[str, set[OrientationEncoding]] = {}
    pose_frames: dict[str, set[str]] = {}

    for extra_key, fields in state_field_spec.items():
        if extra_key not in proprioception.extra:
            raise ValueError(
                f"state_field_spec references extra key {extra_key!r}, but this "
                "proprioception's extra has no such key"
            )
        array = proprioception.extra[extra_key]
        if len(fields) != len(array):
            raise ValueError(
                f"state_field_spec for extra key {extra_key!r} has {len(fields)} "
                f"entries but the array has {len(array)} elements"
            )
        for value, field in zip(array, fields, strict=True):
            if field is None:
                continue
            if field.kind == "position_axis":
                if field.end_effector is None or field.axis is None or field.frame is None:
                    raise ValueError(
                        "a 'position_axis' state field requires end_effector, axis, and "
                        f"frame, got end_effector={field.end_effector!r}, "
                        f"axis={field.axis!r}, frame={field.frame!r}"
                    )
                axes = position_values.setdefault(field.end_effector, {})
                if field.axis in axes:
                    raise ValueError(
                        f"duplicate position axis {field.axis!r} for end_effector "
                        f"{field.end_effector!r}"
                    )
                axes[field.axis] = float(value)
                pose_frames.setdefault(field.end_effector, set()).add(field.frame)
                continue
            if field.kind == "orientation_axis":
                if field.end_effector is None or field.component is None or field.frame is None:
                    raise ValueError(
                        "an 'orientation_axis' state field requires end_effector, "
                        f"component, and frame, got end_effector={field.end_effector!r}, "
                        f"component={field.component!r}, frame={field.frame!r}"
                    )
                if field.encoding is None:
                    raise ValueError(
                        "an 'orientation_axis' state field requires an encoding — there is "
                        "no default, how raw orientation numbers are encoded is "
                        "dataset-specific"
                    )
                components = orientation_values.setdefault(field.end_effector, {})
                if field.component in components:
                    raise ValueError(
                        f"duplicate orientation component {field.component!r} for "
                        f"end_effector {field.end_effector!r}"
                    )
                components[field.component] = float(value)
                orientation_encodings.setdefault(field.end_effector, set()).add(field.encoding)
                pose_frames.setdefault(field.end_effector, set()).add(field.frame)
                continue
            if field.kind == "gripper":
                if field.end_effector is None:
                    raise ValueError("a 'gripper' state field requires end_effector, got None")
                gripper_values[field.end_effector] = _normalized_gripper(
                    float(value), field.end_effector
                )
            elif field.kind == "joint":
                if field.joint_index is None:
                    raise ValueError("a 'joint' state field requires joint_index, got None")
                joint_values[field.joint_index] = float(value)

    new_grippers = dict(proprioception.grippers)
    new_grippers.update(gripper_values)

    new_joint_positions: NDArray[np.float64] | None
    if joint_values:
        existing = proprioception.joint_positions
        length = max(max(joint_values) + 1, len(existing) if existing is not None else 0)
        new_joint_positions = np.zeros(length, dtype=np.float64)
        if existing is not None:
            new_joint_positions[: len(existing)] = existing
        for joint_index, joint_value in joint_values.items():
            new_joint_positions[joint_index] = joint_value
    else:
        new_joint_positions = proprioception.joint_positions

    new_ee_poses = dict(proprioception.ee_poses)
    for end_effector in sorted(set(position_values) | set(orientation_values)):
        new_ee_poses[end_effector] = _build_pose(
            end_effector,
            position_values.get(end_effector),
            orientation_values.get(end_effector),
            orientation_encodings.get(end_effector, set()),
            pose_frames[end_effector],
        )

    return Proprioception(
        joint_positions=new_joint_positions,
        joint_velocities=proprioception.joint_velocities,
        ee_poses=new_ee_poses,
        grippers=new_grippers,
        extra=proprioception.extra,
    )


def reconstruct_episode_proprioception(
    episode: Episode, state_field_spec: StateFieldSpec
) -> Episode:
    """Apply `reconstruct_proprioception` to every frame in an episode.

    Streams — per docs/episode.md, an episode's frames may be a one-shot
    generator, so this wraps rather than materializes them.

    Args:
        episode: Episode whose frames' proprioception should be
            reconstructed.
        state_field_spec: Caller-supplied mapping, applied identically to
            every frame — see `StateFieldSpec`.

    Returns:
        A new `Episode` with the same metadata, yielding frames whose
        proprioception has been reconstructed. Iterating it consumes
        ``episode`` in turn — same one-shot-iteration rule as any `Episode`.
    """

    def _frames() -> Iterator[Frame]:
        for frame in episode:
            yield replace(
                frame,
                proprioception=reconstruct_proprioception(frame.proprioception, state_field_spec),
            )

    return Episode(metadata=episode.metadata, frames=_frames())
