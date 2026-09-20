"""Tests for `mekiki.checks.provenance`.

Every test constructs data with a *known* ground truth (an action that is
derived from states, one that is commanded with noise, one that's off by a
step...) and checks the audit recovers it — a detector that has never been
shown a labeled positive is a guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.checks.provenance import (
    ActionDimensionProvenance,
    audit_action_provenance,
)
from mekiki.episode import Episode, EpisodeMetadata, Frame, Proprioception
from tests.conftest import CLEAN_ACTION_SPACE

KEY = "observation.state"


def _episode(actions: np.ndarray, states: np.ndarray) -> Episode:
    frames = [
        Frame(
            timestamp=i * 0.1,
            proprioception=Proprioception(
                joint_positions=None,
                joint_velocities=None,
                ee_poses={},
                grippers={},
                extra={KEY: states[i]},
            ),
            action=actions[i],
            images={},
            is_first=(i == 0),
            is_last=(i == len(states) - 1),
        )
        for i in range(len(states))
    ]
    metadata = EpisodeMetadata(
        episode_id="p",
        dataset_name="synthetic",
        robot_embodiment="x",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    return Episode(metadata=metadata, frames=frames)


def _walk(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    """A smooth-ish random walk, like real robot state."""
    return np.cumsum(rng.normal(0.0, 0.02, size=(n, dims)), axis=0) + 0.3


def _delta_actions(states: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1] - state[t]; the last frame repeats zero."""
    return np.vstack([np.diff(states, axis=0), np.zeros((1, states.shape[1]))])


def _audit(episodes: list[Episode], **kwargs: object) -> tuple[ActionDimensionProvenance, ...]:
    return audit_action_provenance(iter(episodes), KEY, **kwargs)  # type: ignore[arg-type]


def test_action_derived_from_state_differences_is_detected_as_exact() -> None:
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 60, 3)
        episodes.append(_episode(_delta_actions(states), states))
    for dim in _audit(episodes):
        assert dim.classification == "exact_state_difference"
        assert dim.relation == "delta"
        assert dim.lag == 0
        assert not dim.misaligned
        assert dim.derived_from_state
        assert dim.identity_fraction > 0.99
        assert dim.slope == pytest.approx(1.0, abs=1e-6)


def test_each_action_dimension_finds_its_own_state_dimension() -> None:
    rng = np.random.default_rng(1)
    states = _walk(rng, 200, 4)
    actions = _delta_actions(states)[:, [2, 0]]  # action0 <- state2, action1 <- state0
    results = _audit([_episode(actions, states)])
    assert [r.state_dim for r in results] == [2, 0]


def test_noisy_commanded_delta_tracks_but_is_not_exact() -> None:
    rng = np.random.default_rng(2)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 2)
        noise = rng.normal(0.0, 0.004, size=states.shape)
        episodes.append(_episode(_delta_actions(states) + noise, states))
    for dim in _audit(episodes):
        assert dim.classification == "tracks_state_change"
        assert not dim.derived_from_state
        assert dim.identity_fraction < 0.5


def test_action_labeled_with_the_previous_transition_is_flagged_misaligned() -> None:
    # off-by-one: action[t] is state[t] - state[t-1], the *previous* step.
    rng = np.random.default_rng(3)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 2)
        shifted = np.vstack([np.zeros((1, 2)), np.diff(states, axis=0)])
        episodes.append(_episode(shifted, states))
    for dim in _audit(episodes):
        assert dim.lag == -1
        assert dim.misaligned
        assert dim.classification == "exact_state_difference"  # exact, just shifted
        assert dim.rho_by_lag[-1] > dim.rho_by_lag[0]


def test_a_noisy_action_at_a_shifted_lag_is_reported_but_not_called_misaligned() -> None:
    rng = np.random.default_rng(18)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 1)
        d = np.diff(states, axis=0)
        ahead = np.vstack([d[1:], np.zeros((2, 1))]) + rng.normal(0.0, 0.004, size=(80, 1))
        episodes.append(_episode(ahead, states))
    (dim,) = _audit(episodes)
    assert dim.lag == 1
    assert not dim.derived_from_state
    assert not dim.misaligned


def test_action_a_step_ahead_is_flagged_misaligned_the_other_way() -> None:
    rng = np.random.default_rng(4)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 2)
        d = np.diff(states, axis=0)
        ahead = np.vstack([d[1:], np.zeros((2, 2))])  # action[t] = state[t+2]-state[t+1]
        episodes.append(_episode(ahead, states))
    for dim in _audit(episodes):
        assert dim.lag == 1
        assert dim.misaligned


def test_commanded_target_with_actuator_lag_tracks_state_but_is_not_exact() -> None:
    # gripper-like: state chases the commanded absolute target, s += 0.6*(a - s)
    rng = np.random.default_rng(5)
    episodes = []
    for _ in range(6):
        targets = np.repeat(rng.choice([0.0, 1.0], size=12), 8)
        s = np.zeros(len(targets))
        for t in range(len(targets) - 1):
            s[t + 1] = s[t] + 0.6 * (targets[t] - s[t])
        episodes.append(_episode(targets[:, None], s[:, None]))
    (dim,) = _audit(episodes)
    # a robot chasing a target moves toward it, so the state *change* can
    # track the command as well as the state level; either is a "tracks"
    assert dim.classification in ("tracks_state_level", "tracks_state_change")
    assert not dim.derived_from_state
    assert dim.identity_fraction < 0.5
    assert dim.spearman > 0.9
    # the state lags the command (best lag > 0), which is dynamics, not a bug
    assert dim.lag >= 0
    assert not dim.misaligned


def test_unrelated_noise_has_no_relationship() -> None:
    rng = np.random.default_rng(6)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 3)
        episodes.append(_episode(rng.normal(size=(80, 2)), states))
    for dim in _audit(episodes):
        assert dim.classification == "no_relationship_found"
        assert abs(dim.spearman) < 0.5


def test_constant_action_is_reported_as_constant_not_as_a_relationship() -> None:
    rng = np.random.default_rng(7)
    states = _walk(rng, 80, 2)
    actions = np.zeros((80, 1))
    (dim,) = _audit([_episode(actions, states)])
    assert dim.classification == "constant_action"
    assert dim.state_dim == -1
    assert not dim.misaligned
    assert not dim.derived_from_state


def test_mixed_provenance_is_resolved_per_dimension_like_real_bridge_v2() -> None:
    # dims 0-1 derived from state differences; dim 2 a commanded gripper target
    rng = np.random.default_rng(8)
    episodes = []
    for _ in range(6):
        n = 96
        pose = _walk(rng, n, 2)
        targets = np.repeat(rng.choice([0.0, 1.0], size=n // 8), 8)
        grip = np.zeros(n)
        for t in range(n - 1):
            grip[t + 1] = grip[t] + 0.6 * (targets[t] - grip[t])
        states = np.column_stack([pose, grip])
        actions = np.column_stack([_delta_actions(pose), targets])
        episodes.append(_episode(actions, states))
    a0, a1, a2 = _audit(episodes)
    assert a0.classification == a1.classification == "exact_state_difference"
    assert a2.classification in ("tracks_state_level", "tracks_state_change")
    assert not a2.derived_from_state
    assert a2.state_dim == 2


def test_pairs_never_cross_an_episode_boundary() -> None:
    # two episodes whose states are far apart: if the last step of one were
    # paired with the first of the next, the giant jump would wreck the result
    rng = np.random.default_rng(9)
    a = _walk(rng, 60, 2)
    b = _walk(rng, 60, 2) + 100.0
    episodes = [_episode(_delta_actions(a), a), _episode(_delta_actions(b), b)]
    for dim in _audit(episodes):
        assert dim.classification == "exact_state_difference"


def test_a_few_outliers_do_not_hide_a_derived_dimension() -> None:
    # e.g. an angle wrapping: a handful of steps disagree wildly
    rng = np.random.default_rng(10)
    states = _walk(rng, 300, 1)
    actions = _delta_actions(states)
    actions[[10, 90, 200], 0] += 6.283185
    (dim,) = _audit([_episode(actions, states)])
    assert dim.spearman > 0.95
    assert dim.classification in ("exact_state_difference", "tracks_state_change")
    assert dim.identity_fraction > 0.98


def test_margin_is_small_when_two_state_dimensions_are_identical() -> None:
    rng = np.random.default_rng(11)
    base = _walk(rng, 100, 1)
    states = np.hstack([base, base.copy()])  # two indistinguishable columns
    actions = _delta_actions(base)
    (dim,) = _audit([_episode(actions, states)])
    assert dim.margin == pytest.approx(0.0, abs=1e-9)


def test_max_episodes_caps_how_much_is_read() -> None:
    rng = np.random.default_rng(12)
    consumed = []

    def gen():  # type: ignore[no-untyped-def]
        for i in range(10):
            consumed.append(i)
            states = _walk(rng, 50, 1)
            yield _episode(_delta_actions(states), states)

    audit_action_provenance(gen(), KEY, max_episodes=3)
    assert consumed == [0, 1, 2]


def test_rho_profile_covers_every_tested_lag() -> None:
    rng = np.random.default_rng(13)
    states = _walk(rng, 100, 1)
    (dim,) = _audit([_episode(_delta_actions(states), states)], lags=(-1, 0, 1))
    assert set(dim.rho_by_lag) == {-1, 0, 1}


def test_missing_state_key_raises_with_the_available_keys() -> None:
    rng = np.random.default_rng(14)
    states = _walk(rng, 40, 1)
    with pytest.raises(ValueError, match=r"isn't in this frame's Proprioception\.extra"):
        audit_action_provenance(iter([_episode(_delta_actions(states), states)]), "nope")


def test_no_episodes_raises() -> None:
    with pytest.raises(ValueError, match="no episodes"):
        audit_action_provenance(iter([]), KEY)


def test_too_little_data_raises() -> None:
    rng = np.random.default_rng(15)
    states = _walk(rng, 6, 1)
    with pytest.raises(ValueError, match="too little data"):
        _audit([_episode(_delta_actions(states), states)])


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="max_episodes"):
        audit_action_provenance(iter([]), KEY, max_episodes=0)
    with pytest.raises(ValueError, match="lags"):
        audit_action_provenance(iter([]), KEY, lags=())


def test_inconsistent_width_between_episodes_raises() -> None:
    rng = np.random.default_rng(16)
    s1 = _walk(rng, 40, 2)
    s2 = _walk(rng, 40, 3)
    with pytest.raises(ValueError, match="width changes"):
        _audit([_episode(_delta_actions(s1), s1), _episode(_delta_actions(s2), s2)])


def test_empty_episodes_are_skipped() -> None:
    rng = np.random.default_rng(17)
    states = _walk(rng, 50, 1)
    empty = Episode(
        metadata=EpisodeMetadata("e", "s", "x", CLEAN_ACTION_SPACE, "synthetic"), frames=[]
    )
    results = audit_action_provenance(iter([empty, _episode(_delta_actions(states), states)]), KEY)
    assert results[0].classification == "exact_state_difference"


def test_episodes_too_short_to_pair_are_tolerated_alongside_long_ones() -> None:
    rng = np.random.default_rng(19)
    long_states = _walk(rng, 60, 1)
    tiny = _walk(rng, 2, 1)
    results = _audit(
        [_episode(_delta_actions(tiny), tiny), _episode(_delta_actions(long_states), long_states)]
    )
    assert results[0].classification == "exact_state_difference"


def test_a_state_that_never_moves_gives_no_relationship() -> None:
    rng = np.random.default_rng(20)
    states = np.full((80, 1), 0.5)
    actions = rng.normal(size=(80, 1))
    (dim,) = _audit([_episode(actions, states)])
    assert dim.classification == "no_relationship_found"
    assert dim.spearman == 0.0


def test_a_partly_noisy_relationship_is_weak() -> None:
    rng = np.random.default_rng(21)
    episodes = []
    for _ in range(5):
        states = _walk(rng, 80, 1)
        d = _delta_actions(states)
        episodes.append(_episode(d + rng.normal(0.0, 0.03, size=d.shape), states))
    (dim,) = _audit(episodes)
    assert dim.classification == "weak_relationship"
    assert 0.5 <= abs(dim.spearman) < 0.9


def test_a_binary_command_that_perfectly_tracks_is_not_capped_by_ties() -> None:
    # plain Spearman of a two-valued column against a continuous one tops out
    # near sqrt(3)/2 = 0.866, so a perfectly separable gripper must still read ~1
    rng = np.random.default_rng(22)
    states = np.concatenate([rng.uniform(0.0, 0.4, 200), rng.uniform(0.6, 1.0, 200)])
    actions = np.concatenate([np.zeros(200), np.ones(200)])
    (dim,) = _audit([_episode(actions[:, None], states[:, None])])
    assert dim.spearman > 0.99


def test_multidimensional_frame_values_raise() -> None:
    rng = np.random.default_rng(23)
    states = _walk(rng, 30, 1)
    episode = _episode(_delta_actions(states), states)
    bad = Episode(
        metadata=episode.metadata,
        frames=[
            Frame(
                timestamp=f.timestamp,
                proprioception=Proprioception(
                    joint_positions=None,
                    joint_velocities=None,
                    ee_poses={},
                    grippers={},
                    extra={KEY: np.zeros((2, 2))},
                ),
                action=f.action,
                images={},
                is_first=f.is_first,
                is_last=f.is_last,
            )
            for f in episode
        ],
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        _audit([bad])
