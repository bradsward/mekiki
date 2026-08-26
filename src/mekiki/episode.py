"""Core Episode/Frame data model.

This module implements the conventions fixed in ``docs/episode.md`` — action
space semantics (per-dimension absolute vs. delta), coordinate frame
handling, and streaming/laziness rules. Treat that document as the source of
truth for *why*; this module is just the typed implementation of it. Do not
re-decide those conventions here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ActionMode = Literal["absolute", "delta"]
ActionUnit = Literal["m", "rad", "normalized"]


@dataclass(frozen=True, slots=True)
class ActionDimSpec:
    """Describes one dimension of an episode's action vector.

    Attributes:
        name: Human-readable axis name, e.g. ``"x"``, ``"roll"``, ``"gripper"``.
        mode: Whether this dimension is an absolute target or a delta from
            the current state (docs/episode.md#action-space-delta-vs-absolute).
        unit: Physical unit — meters, radians, or a normalized range
            (``[-1, 1]`` for most axes, ``[0, 1]`` for gripper).
        frame: Name of the coordinate frame this dimension is expressed in,
            as declared by the source dataset (e.g. ``"base_link"``,
            ``"ee"``). mekiki does not convert between frames at read time.
    """

    name: str
    mode: ActionMode
    unit: ActionUnit
    frame: str


ActionSpaceSpec = tuple[ActionDimSpec, ...]


@dataclass(frozen=True, slots=True, eq=False)
class Pose:
    """A rigid-body pose.

    Attributes:
        position: ``(3,)`` array of meters, ``(x, y, z)``.
        orientation: ``(4,)`` unit quaternion, ``(x, y, z, w)``. Never Euler
            angles — see docs/episode.md.
        frame: Name of the coordinate frame this pose is expressed in.
    """

    position: NDArray[np.float64]
    orientation: NDArray[np.float64]
    frame: str

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            raise ValueError(f"Pose.position must have shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Pose.orientation must have shape (4,), got {self.orientation.shape}")


@dataclass(frozen=True, slots=True, eq=False)
class Proprioception:
    """Recorded robot state at one timestep.

    Does not assume a single-arm-with-gripper robot: bimanual setups have
    two end-effectors and two grippers, many datasets record only joint
    positions with no computed end-effector pose, and some tasks aren't
    arm-shaped at all. Every field is optional except ``extra``, an escape
    hatch for state that doesn't fit the structured fields — see
    docs/episode.md.

    Attributes:
        joint_positions: ``(n_joints,)`` array, radians for revolute joints,
            or ``None`` if the source dataset doesn't record joint state.
        joint_velocities: ``(n_joints,)`` array, rad/s, or ``None``.
        ee_poses: End-effector poses keyed by a short name the source
            dataset uses to distinguish them (``"ee"`` for a single arm,
            ``"left"``/``"right"`` for bimanual). Empty if the dataset
            provides no end-effector pose.
        grippers: Normalized gripper state in ``[0.0, 1.0]`` (0 fully
            closed, 1 fully open) — normalized even when the source signal
            is binary — keyed the same way as ``ee_poses`` when there's a
            correspondence. Empty if there's no gripper.
        extra: Anything else the dataset records that doesn't fit the
            fields above (e.g. PushT's raw 2D pusher position), keyed by
            the source dataset's own field name, values as-recorded with no
            assumed unit or frame. A reader populating this must document
            what it put here; mekiki does not infer physical meaning for it.
    """

    joint_positions: NDArray[np.float64] | None
    joint_velocities: NDArray[np.float64] | None
    ee_poses: dict[str, Pose]
    grippers: dict[str, float]
    extra: dict[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        for name, value in self.grippers.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"gripper {name!r} must be in [0.0, 1.0], got {value}")


@dataclass(frozen=True, slots=True, eq=False)
class CameraFrame:
    """One camera's image at one timestep.

    Attributes:
        read: Lazy accessor returning the image as an ``(H, W, 3)`` uint8
            RGB array. Not invoked until pixels are actually needed — see
            docs/episode.md#streaming-and-laziness.
        timestamp: Seconds, relative to episode start. Independent of the
            owning ``Frame.timestamp``; the gap between them is exactly what
            a cross-stream desync check measures.
        resolution: ``(height, width)`` in pixels.
        timestamp_is_measured: ``True`` if the source dataset records this
            camera's timestamp separately from the control timestamp;
            ``False`` if it was assumed equal to the control timestamp
            because the source doesn't distinguish them
            (docs/episode.md#camera-streams).
    """

    read: Callable[[], NDArray[np.uint8]]
    timestamp: float
    resolution: tuple[int, int]
    timestamp_is_measured: bool


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One timestep within an episode.

    Attributes:
        timestamp: Seconds, relative to episode start (``t=0`` at the first
            frame) — never wall-clock. Monotonic non-decreasing is a
            property mekiki *checks* elsewhere; this type does not enforce
            or silently repair it.
        proprioception: Recorded robot state at this timestep.
        action: Commanded action vector for this timestep. Its dimensions
            are interpreted via the owning ``Episode``'s ``action_space``.
        images: Camera frames keyed by camera name (e.g. ``"wrist"``).
        is_first: Whether this is the first frame of the episode.
        is_last: Whether this is the last frame of the episode.
        success: Episode-level outcome label, if the source dataset
            provides one. ``None`` means unlabeled — never coerced to
            ``False``.
        language_instruction: Task description in natural language, if
            present.
    """

    timestamp: float
    proprioception: Proprioception
    action: NDArray[np.float64]
    images: dict[str, CameraFrame]
    is_first: bool
    is_last: bool
    success: bool | None = None
    language_instruction: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Episode-level metadata that doesn't vary frame to frame.

    Attributes:
        episode_id: Identifier unique within the source dataset.
        dataset_name: Name (and ideally version) of the source dataset.
        robot_embodiment: Robot/embodiment identifier, e.g. ``"franka_panda"``.
        action_space: Per-dimension action space spec for every frame in
            this episode (docs/episode.md#action-space-delta-vs-absolute).
        source_format: Format the episode was read from, e.g.
            ``"lerobot"`` or ``"rlds"``.
    """

    episode_id: str
    dataset_name: str
    robot_embodiment: str
    action_space: ActionSpaceSpec
    source_format: str


@dataclass(frozen=True, slots=True, eq=False)
class Episode:
    """One demonstration episode: metadata plus a lazy stream of frames.

    ``frames`` is an ``Iterable[Frame]`` rather than a list — a dataset
    reader yields episodes lazily, and an individual episode's frames may
    themselves be produced lazily from disk
    (docs/episode.md#streaming-and-laziness). Iterate it once; if a
    consumer needs to iterate more than once, it should materialize
    ``list(episode)`` itself and accept the memory cost of doing so.

    Example:
        >>> from mekiki.episode import ActionDimSpec, Episode, EpisodeMetadata
        >>> metadata = EpisodeMetadata(
        ...     episode_id="ep0",
        ...     dataset_name="example",
        ...     robot_embodiment="franka_panda",
        ...     action_space=(ActionDimSpec("gripper", "absolute", "normalized", "ee"),),
        ...     source_format="lerobot",
        ... )
        >>> episode = Episode(metadata=metadata, frames=[])
        >>> list(episode)
        []
    """

    metadata: EpisodeMetadata
    frames: Iterable[Frame]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)
