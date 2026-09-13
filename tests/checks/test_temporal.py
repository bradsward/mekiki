"""Tests for `mekiki.checks.temporal`."""

from __future__ import annotations

import pytest

from mekiki.checks.temporal import (
    check_control_frequency_jitter,
    check_dropped_frames,
    check_timestamp_monotonicity,
)
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


# --- check_control_frequency_jitter ------------------------------------


def test_clean_episode_at_nominal_rate_has_no_jitter_violations() -> None:
    episode = make_clean_episode(n_frames=6, control_hz=10.0)
    result = check_control_frequency_jitter(episode, nominal_hz=10.0)
    assert result.n_frames == 6
    assert result.nominal_dt_seconds == pytest.approx(0.1)
    assert result.violation_indices == ()
    assert result.max_abs_jitter_seconds == pytest.approx(0.0, abs=1e-9)


def test_large_single_frame_delay_is_flagged_at_known_magnitude() -> None:
    # nominal 0.1s dt; frame 2 arrives 0.05s late -- well past the default
    # 20% threshold (0.02s)
    episode = _episode_from_timestamps([0.0, 0.1, 0.25, 0.35])
    result = check_control_frequency_jitter(episode, nominal_hz=10.0)
    assert result.violation_indices == (2,)
    assert result.max_abs_jitter_seconds == pytest.approx(0.05)
    assert result.threshold_seconds == pytest.approx(0.02)


def test_small_jitter_within_threshold_fraction_is_not_flagged() -> None:
    # a 0.01s wobble on a 0.1s nominal dt is 10% -- under the default 20%
    episode = _episode_from_timestamps([0.0, 0.1, 0.19, 0.29])
    result = check_control_frequency_jitter(episode, nominal_hz=10.0)
    assert result.violation_indices == ()
    assert result.max_abs_jitter_seconds == pytest.approx(0.01)


def test_custom_threshold_fraction_catches_smaller_jitter() -> None:
    episode = _episode_from_timestamps([0.0, 0.1, 0.19, 0.29])
    strict = check_control_frequency_jitter(episode, nominal_hz=10.0, threshold_fraction=0.05)
    assert strict.violation_indices == (2,)
    assert strict.threshold_seconds == pytest.approx(0.005)


def test_systematic_wrong_nominal_rate_flags_every_frame() -> None:
    # the data was actually recorded at 10Hz (0.1s dt) throughout, but the
    # caller declares 5Hz (0.2s dt) -- a self-referential median-delta
    # estimate would never catch this, since the episode is perfectly
    # consistent with itself. every consecutive pair should be flagged.
    episode = make_clean_episode(n_frames=5, control_hz=10.0)
    result = check_control_frequency_jitter(episode, nominal_hz=5.0)
    assert result.violation_indices == (1, 2, 3, 4)
    assert result.max_abs_jitter_seconds == pytest.approx(0.1)


def test_jitter_rejects_non_positive_nominal_hz() -> None:
    episode = make_clean_episode(n_frames=2)
    with pytest.raises(ValueError, match="nominal_hz"):
        check_control_frequency_jitter(episode, nominal_hz=0.0)
    with pytest.raises(ValueError, match="nominal_hz"):
        check_control_frequency_jitter(episode, nominal_hz=-5.0)


def test_jitter_violation_fraction_matches_hand_count() -> None:
    episode = _episode_from_timestamps([0.0, 0.1, 0.25, 0.35, 0.45])
    result = check_control_frequency_jitter(episode, nominal_hz=10.0)
    assert result.n_frames == 5
    assert result.violation_indices == (2,)
    assert result.violation_fraction == pytest.approx(1 / 4)


def test_jitter_single_frame_episode_reports_no_deltas() -> None:
    episode = _episode_from_timestamps([0.0])
    result = check_control_frequency_jitter(episode, nominal_hz=10.0)
    assert result.n_frames == 1
    assert result.violation_indices == ()
    assert result.max_abs_jitter_seconds == 0.0
    assert result.violation_fraction == 0.0


# --- check_dropped_frames ------------------------------------------------


def test_clean_episode_has_no_gaps() -> None:
    episode = make_clean_episode(n_frames=6, control_hz=10.0)
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.n_frames == 6
    assert result.gap_indices == ()
    assert result.total_estimated_dropped == 0
    assert result.gap_fraction == 0.0


def test_one_dropped_frame_estimated_at_known_magnitude() -> None:
    # nominal 0.1s dt; frame 2 arrives at 2x the interval -- exactly one
    # frame's worth missing
    gapped = _episode_from_timestamps([0.0, 0.1, 0.3, 0.4])
    result = check_dropped_frames(gapped, nominal_hz=10.0)
    assert result.gap_indices == (2,)
    assert result.estimated_dropped_per_gap == (1,)
    assert result.total_estimated_dropped == 1


def test_two_dropped_frames_estimated_at_known_magnitude() -> None:
    # delta is 3x the nominal interval -- two frames' worth missing
    episode = _episode_from_timestamps([0.0, 0.1, 0.4, 0.5])
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.gap_indices == (2,)
    assert result.estimated_dropped_per_gap == (2,)
    assert result.total_estimated_dropped == 2


def test_delta_below_threshold_multiple_is_not_a_gap() -> None:
    # 1.2x nominal dt is plausible ordinary jitter, well under the
    # default 1.5x threshold for "a frame is missing"
    episode = _episode_from_timestamps([0.0, 0.1, 0.22, 0.32])
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.gap_indices == ()
    assert result.total_estimated_dropped == 0


def test_custom_threshold_multiple_catches_smaller_gaps() -> None:
    episode = _episode_from_timestamps([0.0, 0.1, 0.22, 0.32])
    strict = check_dropped_frames(episode, nominal_hz=10.0, threshold_multiple=1.1)
    assert strict.gap_indices == (2,)


def test_multiple_gaps_sum_correctly() -> None:
    # two separate gaps: 2x at index 2, 3x at index 4
    episode = _episode_from_timestamps([0.0, 0.1, 0.3, 0.4, 0.7, 0.8])
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.gap_indices == (2, 4)
    assert result.estimated_dropped_per_gap == (1, 2)
    assert result.total_estimated_dropped == 3
    assert result.gap_fraction == pytest.approx(2 / 5)


def test_negative_delta_is_never_counted_as_a_gap() -> None:
    # an out-of-order timestamp is check_timestamp_monotonicity's job, not
    # this check's -- it must not also get flagged as a "dropped frame"
    episode = _episode_from_timestamps([0.0, 0.1, 0.05, 0.15])
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.gap_indices == ()


def test_dropped_frames_rejects_non_positive_nominal_hz() -> None:
    episode = make_clean_episode(n_frames=2)
    with pytest.raises(ValueError, match="nominal_hz"):
        check_dropped_frames(episode, nominal_hz=0.0)


def test_rejects_threshold_multiple_not_greater_than_one() -> None:
    episode = make_clean_episode(n_frames=2)
    with pytest.raises(ValueError, match="threshold_multiple"):
        check_dropped_frames(episode, nominal_hz=10.0, threshold_multiple=1.0)
    with pytest.raises(ValueError, match="threshold_multiple"):
        check_dropped_frames(episode, nominal_hz=10.0, threshold_multiple=0.5)


def test_dropped_frames_single_frame_episode_reports_no_gaps() -> None:
    episode = _episode_from_timestamps([0.0])
    result = check_dropped_frames(episode, nominal_hz=10.0)
    assert result.n_frames == 1
    assert result.gap_indices == ()
    assert result.total_estimated_dropped == 0
    assert result.gap_fraction == 0.0
