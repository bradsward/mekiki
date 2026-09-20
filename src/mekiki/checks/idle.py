"""Idle-time analysis: how much of an episode is dead time, and where?

Design in docs/idle.md. "Idle" is declared by the caller as a set of motion
channels with speed thresholds in real units; nothing here infers what
counts as still.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mekiki.episode import Episode, Frame
from mekiki.rotation import quaternion_angular_distance

ChannelKind = Literal["position", "orientation", "gripper", "joint", "extra"]
SegmentKind = Literal["leading", "interior", "trailing", "whole_episode"]


@dataclass(frozen=True, slots=True)
class MotionChannel:
    """One measured quantity and the speed above which it counts as moving.

    Attributes:
        kind: Which quantity. ``position`` (m/s) and ``orientation`` (rad/s)
            and ``gripper`` (normalized fraction per second) need
            ``end_effector``; ``joint`` (rad/s, or m/s if prismatic) needs
            ``joint_index``; ``extra`` (the raw array's own units per
            second) needs ``state_key``.
        threshold: Speed strictly above this is motion. Must be positive and
            finite, in the unit given for ``kind``.
        end_effector: Key into ``Proprioception.ee_poses`` / ``grippers``.
        joint_index: Index into ``Proprioception.joint_positions``.
        state_key: Key into ``Proprioception.extra``.
    """

    kind: ChannelKind
    threshold: float
    end_effector: str | None = None
    joint_index: int | None = None
    state_key: str | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.threshold) and self.threshold > 0.0):
            raise ValueError(
                f"MotionChannel.threshold must be positive and finite: {self.threshold}"
            )
        if self.kind in ("position", "orientation", "gripper") and self.end_effector is None:
            raise ValueError(f"MotionChannel kind {self.kind!r} needs end_effector")
        if self.kind == "joint" and self.joint_index is None:
            raise ValueError("MotionChannel kind 'joint' needs joint_index")
        if self.kind == "extra" and self.state_key is None:
            raise ValueError("MotionChannel kind 'extra' needs state_key")

    @property
    def name(self) -> str:
        """Stable label used as the key in `IdleResult.peak_speed`."""
        if self.kind == "joint":
            return f"joint:{self.joint_index}"
        if self.kind == "extra":
            return f"extra:{self.state_key}"
        return f"{self.kind}:{self.end_effector}"


@dataclass(frozen=True, slots=True)
class IdleSegment:
    """One run of idle time.

    Attributes:
        kind: Where it sits: ``leading`` (from the first frame),
            ``trailing`` (to the last frame), ``interior``, or
            ``whole_episode`` (both ends: nothing moved).
        start_index: First frame index of the run.
        end_index: Last frame index of the run (inclusive).
        start_seconds: Timestamp of ``start_index``.
        duration_seconds: Time from ``start_index`` to ``end_index``.
    """

    kind: SegmentKind
    start_index: int
    end_index: int
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class IdleResult:
    """Idle analysis of one episode.

    Attributes:
        n_frames: Frames read.
        total_seconds: Last timestamp minus first (``0.0`` if under 2 frames).
        segments: Idle runs at least ``min_idle_seconds`` long, in order.
        leading_seconds / trailing_seconds / interior_seconds: Idle time by
            kind. A ``whole_episode`` segment counts as leading.
        min_idle_seconds: The minimum run length that was kept.
        thresholds: Each channel's ``name`` mapped to the speed threshold it
            was held to.
        peak_speed: Each channel's highest observed speed, in the channel's
            unit. Peak at or below its threshold means that channel never
            registered as moving here, so check the threshold is sane.
    """

    n_frames: int
    total_seconds: float
    segments: tuple[IdleSegment, ...]
    leading_seconds: float
    trailing_seconds: float
    interior_seconds: float
    min_idle_seconds: float
    thresholds: dict[str, float]
    peak_speed: dict[str, float]

    @property
    def recoverable_fraction(self) -> float:
        """Leading plus trailing idle over total duration: the part that can
        be trimmed without touching any motion. ``0.0`` for zero duration."""
        if self.total_seconds <= 0.0:
            return 0.0
        return min(1.0, (self.leading_seconds + self.trailing_seconds) / self.total_seconds)

    @property
    def interior_fraction(self) -> float:
        """Interior (hesitation) idle over total duration."""
        if self.total_seconds <= 0.0:
            return 0.0
        return min(1.0, self.interior_seconds / self.total_seconds)


def _speed(channel: MotionChannel, previous: Frame, current: Frame, dt: float) -> float:
    prev, cur = previous.proprioception, current.proprioception
    kind = channel.kind
    if kind in ("position", "orientation"):
        assert channel.end_effector is not None
        try:
            p, c = prev.ee_poses[channel.end_effector], cur.ee_poses[channel.end_effector]
        except KeyError:
            raise ValueError(
                f"end effector {channel.end_effector!r} has no pose in this frame "
                f"(have {sorted(cur.ee_poses)})"
            ) from None
        if p.frame != c.frame:
            raise ValueError(f"pose frame changed between frames: {p.frame!r} -> {c.frame!r}")
        if kind == "position":
            return float(np.linalg.norm(c.position - p.position)) / dt
        return quaternion_angular_distance(p.orientation, c.orientation) / dt
    if kind == "gripper":
        assert channel.end_effector is not None
        try:
            g0, g1 = prev.grippers[channel.end_effector], cur.grippers[channel.end_effector]
        except KeyError:
            raise ValueError(
                f"end effector {channel.end_effector!r} has no gripper state in this frame "
                f"(have {sorted(cur.grippers)})"
            ) from None
        return abs(g1 - g0) / dt
    if kind == "joint":
        assert channel.joint_index is not None
        if prev.joint_positions is None or cur.joint_positions is None:
            raise ValueError("joint channel needs Proprioception.joint_positions")
        if channel.joint_index >= len(cur.joint_positions):
            raise ValueError(
                f"joint_index {channel.joint_index} out of range for "
                f"{len(cur.joint_positions)} joints"
            )
        j = channel.joint_index
        return abs(float(cur.joint_positions[j] - prev.joint_positions[j])) / dt
    assert channel.state_key is not None
    try:
        v0, v1 = prev.extra[channel.state_key], cur.extra[channel.state_key]
    except KeyError:
        raise ValueError(
            f"state_key {channel.state_key!r} isn't in Proprioception.extra "
            f"(have {sorted(cur.extra)})"
        ) from None
    return float(np.linalg.norm(v1 - v0)) / dt


def analyze_idle_time(
    episode: Episode,
    channels: tuple[MotionChannel, ...],
    *,
    min_idle_seconds: float,
) -> IdleResult:
    """Find leading, trailing and interior idle time in one episode.

    An interval between consecutive frames is idle only if every channel's
    speed is at or below its threshold. Runs of idle intervals at least
    ``min_idle_seconds`` long become segments. Streams the episode once.

    Args:
        episode: The episode to analyze.
        channels: Motion channels with declared thresholds. At least one,
            with distinct names.
        min_idle_seconds: Shortest idle run to report; must be positive.

    Returns:
        Segments, per-kind idle seconds, `IdleResult.recoverable_fraction`,
        and the thresholds and peak speeds so the caller can judge whether
        the declared thresholds fit this robot.

    Raises:
        ValueError: No channels, duplicate channel names, non-positive
            ``min_idle_seconds``, a timestamp that doesn't strictly
            increase, or a frame missing what a channel reads.
    """
    if not channels:
        raise ValueError("at least one MotionChannel is required to say what counts as motion")
    names = [c.name for c in channels]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate channels: {sorted(n for n in names if names.count(n) > 1)}")
    if not (math.isfinite(min_idle_seconds) and min_idle_seconds > 0.0):
        raise ValueError(f"min_idle_seconds must be positive and finite: {min_idle_seconds}")

    times: list[float] = []
    idle: list[bool] = []
    peak = dict.fromkeys(names, 0.0)
    previous: Frame | None = None
    for i, frame in enumerate(episode):
        times.append(frame.timestamp)
        if previous is not None:
            dt = frame.timestamp - previous.timestamp
            if dt <= 0.0:
                raise ValueError(
                    f"timestamp doesn't increase at frame {i} (delta {dt} s); "
                    "run the temporal checks first"
                )
            moving = False
            for channel in channels:
                speed = _speed(channel, previous, frame, dt)
                peak[channel.name] = max(peak[channel.name], speed)
                if speed > channel.threshold:
                    moving = True
            idle.append(not moving)
        previous = frame

    n_frames = len(times)
    total = times[-1] - times[0] if n_frames >= 2 else 0.0
    n_intervals = len(idle)

    segments: list[IdleSegment] = []
    start = 0
    while start < n_intervals:
        if not idle[start]:
            start += 1
            continue
        end = start
        while end + 1 < n_intervals and idle[end + 1]:
            end += 1
        # the run of idle intervals start..end spans frames start..end+1
        duration = times[end + 1] - times[start]
        if duration >= min_idle_seconds:
            at_start, at_end = start == 0, end == n_intervals - 1
            kind: SegmentKind = (
                "whole_episode"
                if at_start and at_end
                else "leading"
                if at_start
                else "trailing"
                if at_end
                else "interior"
            )
            segments.append(IdleSegment(kind, start, end + 1, times[start], duration))
        start = end + 1

    def seconds(*kinds: SegmentKind) -> float:
        return sum(s.duration_seconds for s in segments if s.kind in kinds)

    return IdleResult(
        n_frames=n_frames,
        total_seconds=total,
        segments=tuple(segments),
        leading_seconds=seconds("leading", "whole_episode"),
        trailing_seconds=seconds("trailing"),
        interior_seconds=seconds("interior"),
        min_idle_seconds=min_idle_seconds,
        thresholds={c.name: c.threshold for c in channels},
        peak_speed=peak,
    )
