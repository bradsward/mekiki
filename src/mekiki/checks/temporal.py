"""Temporal-integrity checks: is an episode's timestamp sequence sane?

Starts with monotonicity because everything else in this module (jitter,
dropped frames) assumes consecutive timestamps strictly increase — a check
built on ``mean``/``median`` frame-to-frame deltas is meaningless once one
duplicate or out-of-order timestamp is in the mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mekiki.episode import Episode


@dataclass(frozen=True, slots=True)
class TimestampMonotonicityResult:
    """Result of checking one episode's ``Frame.timestamp`` sequence.

    Attributes:
        n_frames: Total frames checked.
        violation_indices: Frame indices ``i`` (0-based, into iteration
            order) where ``frames[i].timestamp - frames[i-1].timestamp``
            was less than or equal to ``threshold_seconds`` — i.e. the
            timestamp did not strictly increase enough from the previous
            frame. Empty if the episode is clean.
        min_delta_seconds: Smallest consecutive-frame delta observed across
            the whole episode, in seconds. Zero or negative whenever
            ``violation_indices`` is non-empty; ``0.0`` if the episode has
            fewer than two frames (no delta to compute).
        threshold_seconds: The minimum delta a consecutive pair had to
            exceed to count as strictly increasing. Defaults to ``0.0``
            (any non-positive delta is a violation); exposed so a caller
            with known clock-quantization noise can loosen it rather than
            get false positives at the resolution limit of their timestamps.
    """

    n_frames: int
    violation_indices: tuple[int, ...]
    min_delta_seconds: float
    threshold_seconds: float

    @property
    def violation_fraction(self) -> float:
        """Violations as a fraction of consecutive-frame pairs checked.

        ``n_frames`` frames have ``n_frames - 1`` consecutive pairs; ``0.0``
        for episodes with fewer than two frames rather than dividing by zero.
        """
        pairs = self.n_frames - 1
        return len(self.violation_indices) / pairs if pairs > 0 else 0.0


def check_timestamp_monotonicity(
    episode: Episode, *, threshold_seconds: float = 0.0
) -> TimestampMonotonicityResult:
    """Check that an episode's frame timestamps strictly increase.

    Streams the episode once (per docs/episode.md — ``Episode`` is iterated
    a single time) rather than materializing every frame; only the previous
    frame's timestamp is held at any point.

    Args:
        episode: The episode to check.
        threshold_seconds: Minimum delta between consecutive timestamps to
            count as strictly increasing. A delta ``<= threshold_seconds``
            is a violation. Defaults to ``0.0``.

    Returns:
        The check result, with a magnitude (``min_delta_seconds``) and the
        threshold it was compared against — never a bare boolean.

    Example:
        >>> from pathlib import Path
        >>> from mekiki.episode import ActionDimSpec
        >>> from mekiki.readers.lerobot import read_episodes
        >>> action_space = (
        ...     ActionDimSpec("x", "absolute", "normalized", "unknown"),
        ...     ActionDimSpec("y", "absolute", "normalized", "unknown"),
        ... )
        >>> dataset_dir = Path("~/data/pusht").expanduser()
        >>> episode = next(read_episodes(dataset_dir, action_space))  # doctest: +SKIP
        >>> check_timestamp_monotonicity(episode).violation_indices  # doctest: +SKIP
        ()
    """
    n_frames = 0
    violation_indices: list[int] = []
    min_delta = math.inf
    previous_timestamp: float | None = None

    for i, frame in enumerate(episode):
        n_frames += 1
        if previous_timestamp is not None:
            delta = frame.timestamp - previous_timestamp
            min_delta = min(min_delta, delta)
            if delta <= threshold_seconds:
                violation_indices.append(i)
        previous_timestamp = frame.timestamp

    return TimestampMonotonicityResult(
        n_frames=n_frames,
        violation_indices=tuple(violation_indices),
        min_delta_seconds=min_delta if math.isfinite(min_delta) else 0.0,
        threshold_seconds=threshold_seconds,
    )
