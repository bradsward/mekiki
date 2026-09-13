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


@dataclass(frozen=True, slots=True)
class ControlFrequencyJitterResult:
    """Result of checking how far consecutive-frame deltas stray from a
    declared nominal control rate.

    Attributes:
        n_frames: Total frames checked.
        nominal_dt_seconds: Expected interval between consecutive frames
            (``1 / nominal_hz``), in seconds.
        max_abs_jitter_seconds: Largest ``|actual_delta - nominal_dt_seconds|``
            observed across all consecutive deltas.
        violation_indices: Frame indices whose delta from the previous
            frame deviated from ``nominal_dt_seconds`` by more than
            ``threshold_seconds``.
        threshold_seconds: Maximum allowed absolute deviation from the
            nominal interval, in seconds, before a delta counts as jitter.
    """

    n_frames: int
    nominal_dt_seconds: float
    max_abs_jitter_seconds: float
    violation_indices: tuple[int, ...]
    threshold_seconds: float

    @property
    def violation_fraction(self) -> float:
        """Violations as a fraction of consecutive-frame pairs checked."""
        pairs = self.n_frames - 1
        return len(self.violation_indices) / pairs if pairs > 0 else 0.0


def check_control_frequency_jitter(
    episode: Episode,
    *,
    nominal_hz: float,
    threshold_fraction: float = 0.2,
) -> ControlFrequencyJitterResult:
    """Check how far consecutive-frame deltas stray from a declared rate.

    Assumes timestamps are already known monotonic — run
    `check_timestamp_monotonicity` first. A non-monotonic delta here would
    just produce a confusing jitter magnitude rather than a meaningful one.

    Args:
        episode: Episode to check.
        nominal_hz: The dataset's own *declared* control frequency (e.g. a
            LeRobotDataset's ``info.json`` ``fps``) — never inferred from
            the data itself. Inferring it (say, from the median delta)
            could never catch a systematic rate error where every frame
            was recorded at the wrong pace: the episode would just look
            self-consistent against its own median. This must come from
            the dataset's own declared metadata, the same way
            `mekiki.readers.lerobot.validate_action_space` requires a
            caller-supplied action space rather than guessing one.
        threshold_fraction: Maximum allowed deviation from the nominal
            interval, as a fraction of it (e.g. ``0.2`` = 20%). A fraction
            rather than a fixed number of seconds because the same
            absolute jitter means very different things at 5 Hz vs. 100 Hz.

    Returns:
        The check result, with the actual magnitude
        (`ControlFrequencyJitterResult.max_abs_jitter_seconds`) against the
        resolved threshold in seconds — never a bare pass/fail.

    Raises:
        ValueError: ``nominal_hz`` is not positive.

    Example:
        >>> from pathlib import Path
        >>> from mekiki.episode import ActionDimSpec
        >>> from mekiki.readers.lerobot import read_episodes, read_info
        >>> action_space = (
        ...     ActionDimSpec("x", "absolute", "normalized", "unknown"),
        ...     ActionDimSpec("y", "absolute", "normalized", "unknown"),
        ... )
        >>> dataset_dir = Path("~/data/pusht").expanduser()
        >>> info = read_info(dataset_dir)  # doctest: +SKIP
        >>> episode = next(read_episodes(dataset_dir, action_space))  # doctest: +SKIP
        >>> result = check_control_frequency_jitter(
        ...     episode, nominal_hz=info.fps
        ... )  # doctest: +SKIP
        >>> result.violation_indices  # doctest: +SKIP
        ()
    """
    if nominal_hz <= 0:
        raise ValueError(f"nominal_hz must be positive, got {nominal_hz}")

    nominal_dt = 1.0 / nominal_hz
    threshold_seconds = threshold_fraction * nominal_dt

    n_frames = 0
    violation_indices: list[int] = []
    max_abs_jitter = 0.0
    previous_timestamp: float | None = None

    for i, frame in enumerate(episode):
        n_frames += 1
        if previous_timestamp is not None:
            delta = frame.timestamp - previous_timestamp
            jitter = abs(delta - nominal_dt)
            max_abs_jitter = max(max_abs_jitter, jitter)
            if jitter > threshold_seconds:
                violation_indices.append(i)
        previous_timestamp = frame.timestamp

    return ControlFrequencyJitterResult(
        n_frames=n_frames,
        nominal_dt_seconds=nominal_dt,
        max_abs_jitter_seconds=max_abs_jitter,
        violation_indices=tuple(violation_indices),
        threshold_seconds=threshold_seconds,
    )


@dataclass(frozen=True, slots=True)
class DroppedFramesResult:
    """Result of checking for likely-dropped frames.

    Distinct from `ControlFrequencyJitterResult`: jitter measures ordinary
    noise around the nominal rate, this measures deltas large enough that
    the more plausible explanation is one or more *whole frames missing*
    in between, not just a late one. The same episode can trip both checks
    for the same gap — they answer different questions and aren't meant to
    be mutually exclusive.

    Attributes:
        n_frames: Total frames checked.
        nominal_dt_seconds: Expected interval between consecutive frames
            (``1 / nominal_hz``), in seconds.
        gap_indices: Frame indices whose delta from the previous frame was
            at least ``threshold_multiple * nominal_dt_seconds`` — large
            enough to imply at least one missing frame.
        estimated_dropped_per_gap: Estimated number of missing frames at
            each of ``gap_indices``, same order, computed as
            ``round(delta / nominal_dt_seconds) - 1`` and clamped to at
            least 1. A rough estimate, not a certainty — it assumes
            whatever was dropped would otherwise have arrived at the
            nominal rate.
        threshold_multiple: Minimum ``delta / nominal_dt_seconds`` ratio to
            count as a gap.
    """

    n_frames: int
    nominal_dt_seconds: float
    gap_indices: tuple[int, ...]
    estimated_dropped_per_gap: tuple[int, ...]
    threshold_multiple: float

    @property
    def total_estimated_dropped(self) -> int:
        """Sum of `estimated_dropped_per_gap` across the whole episode."""
        return sum(self.estimated_dropped_per_gap)

    @property
    def gap_fraction(self) -> float:
        """Gaps as a fraction of consecutive-frame pairs checked."""
        pairs = self.n_frames - 1
        return len(self.gap_indices) / pairs if pairs > 0 else 0.0


def check_dropped_frames(
    episode: Episode,
    *,
    nominal_hz: float,
    threshold_multiple: float = 1.5,
) -> DroppedFramesResult:
    """Check for consecutive-frame gaps implying one or more dropped frames.

    Assumes timestamps are already known monotonic — run
    `check_timestamp_monotonicity` first. A negative or zero delta is
    ignored here (it can never reach a positive multiple of a positive
    ``nominal_dt_seconds``), not because it's fine, but because that's a
    different, already-covered failure mode.

    Args:
        episode: Episode to check.
        nominal_hz: The dataset's own *declared* control frequency — same
            rationale as `check_control_frequency_jitter`: never infer this
            from the data itself, since a systematic rate error would then
            look self-consistent.
        threshold_multiple: A delta must be at least this many times
            ``nominal_dt_seconds`` to count as a gap rather than ordinary
            jitter. Must be greater than ``1.0``. Default ``1.5`` — a delta
            needs to be well past "one frame arrived late" before "a frame
            is missing" is the more plausible read.

    Returns:
        The check result, with each gap's estimated missing-frame count
        against the threshold that flagged it — never a bare pass/fail.

    Raises:
        ValueError: ``nominal_hz`` is not positive, or ``threshold_multiple``
            is not greater than ``1.0``.

    Example:
        >>> from pathlib import Path
        >>> from mekiki.episode import ActionDimSpec
        >>> from mekiki.readers.lerobot import read_episodes, read_info
        >>> action_space = (
        ...     ActionDimSpec("x", "absolute", "normalized", "unknown"),
        ...     ActionDimSpec("y", "absolute", "normalized", "unknown"),
        ... )
        >>> dataset_dir = Path("~/data/pusht").expanduser()
        >>> info = read_info(dataset_dir)  # doctest: +SKIP
        >>> episode = next(read_episodes(dataset_dir, action_space))  # doctest: +SKIP
        >>> result = check_dropped_frames(episode, nominal_hz=info.fps)  # doctest: +SKIP
        >>> result.total_estimated_dropped  # doctest: +SKIP
        0
    """
    if nominal_hz <= 0:
        raise ValueError(f"nominal_hz must be positive, got {nominal_hz}")
    if threshold_multiple <= 1.0:
        raise ValueError(f"threshold_multiple must be greater than 1.0, got {threshold_multiple}")

    nominal_dt = 1.0 / nominal_hz
    gap_threshold = threshold_multiple * nominal_dt

    n_frames = 0
    gap_indices: list[int] = []
    estimated_dropped: list[int] = []
    previous_timestamp: float | None = None

    for i, frame in enumerate(episode):
        n_frames += 1
        if previous_timestamp is not None:
            delta = frame.timestamp - previous_timestamp
            if delta >= gap_threshold:
                gap_indices.append(i)
                estimated_dropped.append(max(1, round(delta / nominal_dt) - 1))
        previous_timestamp = frame.timestamp

    return DroppedFramesResult(
        n_frames=n_frames,
        nominal_dt_seconds=nominal_dt,
        gap_indices=tuple(gap_indices),
        estimated_dropped_per_gap=tuple(estimated_dropped),
        threshold_multiple=threshold_multiple,
    )
