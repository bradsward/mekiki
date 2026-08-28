"""Tests for `mekiki.checks.temporal.check_timestamp_monotonicity`."""

from __future__ import annotations

import pytest

from mekiki.checks.temporal import check_timestamp_monotonicity
from mekiki.episode import Episode, EpisodeMetadata
from tests.conftest import CLEAN_ACTION_SPACE, make_clean_episode, make_clean_frame


def _episode_from_timestamps(timestamps: list[float]) -> Episode:
    """Build an episode from an arbitrary timestamp sequence.

    Everything except the timestamps stays well-formed (via
    `make_clean_frame`) — this exists purely to inject the one defect this
    check cares about, at a precisely known magnitude.
    """
    frames = [
        make_clean_frame(timestamp=t, is_first=(i == 0), is_last=(i == len(timestamps) - 1))
        for i, t in enumerate(timestamps)
    ]
    metadata = EpisodeMetadata(
        episode_id="defect-0",
        dataset_name="synthetic",
        robot_embodiment="franka_panda",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    return Episode(metadata=metadata, frames=frames)


def test_clean_episode_has_no_violations() -> None:
    result = check_timestamp_monotonicity(make_clean_episode(n_frames=5, control_hz=10.0))
    assert result.n_frames == 5
    assert result.violation_indices == ()
    assert result.violation_fraction == 0.0
    assert result.min_delta_seconds == pytest.approx(0.1)


def test_duplicate_timestamp_is_flagged_at_zero_delta() -> None:
    # frame 2 repeats frame 1's timestamp exactly -- delta is 0.0
    episode = _episode_from_timestamps([0.0, 0.1, 0.1, 0.2])
    result = check_timestamp_monotonicity(episode)
    assert result.n_frames == 4
    assert result.violation_indices == (2,)
    assert result.min_delta_seconds == pytest.approx(0.0)


def test_out_of_order_timestamp_is_flagged_with_negative_delta() -> None:
    # frame 2 goes backwards by exactly 0.05s relative to frame 1
    episode = _episode_from_timestamps([0.0, 0.1, 0.05, 0.2])
    result = check_timestamp_monotonicity(episode)
    assert result.violation_indices == (2,)
    assert result.min_delta_seconds == pytest.approx(-0.05)


def test_violation_fraction_matches_hand_count() -> None:
    # 5 frames -> 4 consecutive pairs; exactly one violation (frame 2)
    episode = _episode_from_timestamps([0.0, 0.1, 0.1, 0.2, 0.3])
    result = check_timestamp_monotonicity(episode)
    assert result.n_frames == 5
    assert result.violation_indices == (2,)
    assert result.violation_fraction == pytest.approx(1 / 4)


def test_custom_threshold_flags_gaps_that_default_threshold_would_not() -> None:
    # a 0.02s gap is technically increasing (delta > 0.0) so the default
    # threshold lets it pass, but a caller who knows their clock resolution
    # should be able to demand a stricter minimum gap.
    episode = _episode_from_timestamps([0.0, 0.02, 0.12])
    default_result = check_timestamp_monotonicity(episode)
    assert default_result.violation_indices == ()

    strict_result = check_timestamp_monotonicity(episode, threshold_seconds=0.05)
    assert strict_result.violation_indices == (1,)
    assert strict_result.threshold_seconds == pytest.approx(0.05)


def test_single_frame_episode_reports_no_deltas() -> None:
    episode = _episode_from_timestamps([0.0])
    result = check_timestamp_monotonicity(episode)
    assert result.n_frames == 1
    assert result.violation_indices == ()
    assert result.min_delta_seconds == 0.0
    assert result.violation_fraction == 0.0


def test_empty_episode_reports_zero_frames() -> None:
    metadata = EpisodeMetadata(
        episode_id="empty-0",
        dataset_name="synthetic",
        robot_embodiment="franka_panda",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    result = check_timestamp_monotonicity(Episode(metadata=metadata, frames=[]))
    assert result.n_frames == 0
    assert result.violation_indices == ()
    assert result.violation_fraction == 0.0
