"""Turning a dataset's raw, uninterpreted ``Proprioception.extra`` values
into structured ``grippers``/``joint_positions`` fields real checks can use.
See ``docs/consistency.md``'s "Getting real proprioception in" section for
the design — this module implements it, doesn't re-decide it.

Symmetric to ``mekiki.checks.consistency.ActionTarget``/``ActionTargetSpec``,
but for state instead of actions: a caller-supplied, explicit, per-element
mapping from a raw ``extra[]`` array into structured `Proprioception`
fields, never inferred from a column name or shape. Kept out of both
``mekiki.episode`` (the stable M1 core) and ``mekiki.checks.consistency``
(M3-specific) — this is useful to any check that wants real
position/orientation/gripper/joint data, not just the consistency check.

``position_axis``/``orientation_axis`` reconstruction isn't implemented
yet — see `reconstruct_proprioception`'s docstring and docs/consistency.md
for why (it's not only an orientation-encoding problem; ``Pose`` bundles
position and orientation together, so even position reconstruction is
blocked transitively).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mekiki.episode import Episode, Frame, Proprioception

StateFieldKind = Literal["position_axis", "orientation_axis", "gripper", "joint"]


@dataclass(frozen=True, slots=True)
class StateField:
    """Where one raw state value belongs, physically.

    Mirrors `mekiki.checks.consistency.ActionTarget`'s shape, but
    describes a *destination* in a reconstructed `Proprioception` for a
    value read out of `Proprioception.extra`, rather than what an action
    dimension predicts. Only ``"gripper"`` and ``"joint"`` are implemented
    right now — see the module docstring.

    Attributes:
        kind: What kind of structured field this value becomes.
        end_effector: For ``"gripper"`` (and, once implemented,
            ``"position_axis"``/``"orientation_axis"``) — the key to write
            into `Proprioception.grippers` (or ``ee_poses``).
        axis: For ``"position_axis"``/``"orientation_axis"`` — which
            component of the 3-vector this value is. Unused today since
            neither kind is implemented yet.
        joint_index: For ``"joint"`` — which index into
            `Proprioception.joint_positions` this value becomes.
    """

    kind: StateFieldKind
    end_effector: str | None = None
    axis: Literal["x", "y", "z"] | None = None
    joint_index: int | None = None


#: Maps a `Proprioception.extra` key to where each of that array's elements
#: belongs. One entry per element, same order, same length as the array.
#: ``None`` for an element that isn't being reconstructed — it simply stays
#: in ``extra``, unchanged, same as every other element (reconstruction
#: never removes anything from ``extra``, see `reconstruct_proprioception`).
StateFieldSpec = dict[str, tuple[StateField | None, ...]]


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
        A new `Proprioception` with ``grippers``/``joint_positions``
        updated by whatever `state_field_spec` mapped, everything else
        (including ``extra``) unchanged from the source.

    Raises:
        ValueError: a referenced ``extra`` key doesn't exist on
            ``proprioception``; a spec tuple's length doesn't match that
            array's length; or a ``"gripper"``/``"joint"`` field is
            missing its required ``end_effector``/``joint_index``.
        NotImplementedError: any ``"position_axis"`` or
            ``"orientation_axis"`` field — not supported yet. See the
            module docstring and docs/consistency.md for exactly why
            (it's two separate blockers, not one).

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
                raise NotImplementedError(
                    "position_axis state reconstruction isn't supported yet — "
                    "mekiki.episode.Pose bundles position with orientation as one "
                    "required unit, and orientation reconstruction isn't supported "
                    "(see below), so there's no honest way to build a complete Pose "
                    "from position data alone. See docs/consistency.md."
                )
            if field.kind == "orientation_axis":
                raise NotImplementedError(
                    "orientation_axis state reconstruction isn't supported yet — a "
                    "raw orientation encoding (Euler order, axis-angle, ...) is "
                    "dataset-specific and mekiki has no way to know it. See "
                    "docs/consistency.md."
                )
            if field.kind == "gripper":
                if field.end_effector is None:
                    raise ValueError("a 'gripper' state field requires end_effector, got None")
                gripper_values[field.end_effector] = float(value)
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

    return Proprioception(
        joint_positions=new_joint_positions,
        joint_velocities=proprioception.joint_velocities,
        ee_poses=proprioception.ee_poses,
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
