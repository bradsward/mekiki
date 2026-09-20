"""Action provenance: is each action dimension a command, or a relabeled state?

See ``docs/provenance.md`` for the problem and the method — this module
implements it, doesn't re-decide it. For every action dimension it reports,
from the data alone, which recorded state dimension it most resembles, in
what relation (a state *change* or a state *level*), at what lag, and
whether the resemblance is *exact* (the action **is** that state quantity,
to float precision) or merely a correlation.

This is the one place mekiki infers something from data, so its output is a
hypothesis with evidence for a human to confirm — it never feeds an
`ActionTargetSpec` on its own, and the wording of every classification says
"consistent with", never "was".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mekiki.episode import Episode

Relation = Literal["delta", "level"]
Classification = Literal[
    "exact_state_difference",
    "exact_state_level",
    "tracks_state_change",
    "tracks_state_level",
    "weak_relationship",
    "no_relationship_found",
    "constant_action",
]

#: Fraction of steps that must match the state quantity to within
#: ``identity_atol`` for a dimension to count as exactly derived.
EXACT_IDENTITY_FRACTION = 0.99
#: ``|Spearman rho|`` at or above this is a strong tracking relationship.
STRONG_RHO = 0.9
#: ``|Spearman rho|`` below this is no usable relationship.
WEAK_RHO = 0.5

_MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma for a normal
_OUTLIER_MADS = 5.0


@dataclass(frozen=True, slots=True)
class ActionDimensionProvenance:
    """What one action dimension most resembles in the recorded state.

    Attributes:
        action_dim: Index of the action dimension.
        state_dim: Index of the best-matching state dimension, or ``-1`` when
            the action is constant (nothing to match).
        relation: ``"delta"`` (action resembles a state *change*) or
            ``"level"`` (action resembles a state *value*, i.e. a target).
        lag: ``L`` such that the action at ``t`` best matches the state
            quantity around ``t+1+L``. ``0`` is the standard convention; any
            other value is reported as-is; only an *exact* match at a
            non-zero lag counts as misalignment (see `misaligned`).
        classification: See docs/provenance.md. Every value is a statement
            of resemblance, not of intent.
        spearman: Signed Spearman rank correlation at the winning candidate.
        slope: Least-squares slope of action against the state quantity
            (after one round of outlier rejection). ``1.0`` means same units.
        intercept: Matching least-squares intercept.
        identity_fraction: Fraction of steps where ``|action - state
            quantity| <= identity_atol`` — the action *is* it, not just
            correlated with it.
        n_steps: Number of paired steps this candidate used.
        rho_by_lag: Spearman rho at every tested lag for the winning
            (state dim, relation) pair, so a lag conclusion can be checked.
        margin: Winning ``|rho|`` minus the best ``|rho|`` of any *different*
            (state dim, relation) pair — small means the correspondence is
            ambiguous.
    """

    action_dim: int
    state_dim: int
    relation: Relation
    lag: int
    classification: Classification
    spearman: float
    slope: float
    intercept: float
    identity_fraction: float
    n_steps: int
    rho_by_lag: dict[int, float]
    margin: float

    @property
    def misaligned(self) -> bool:
        """True when the action is exactly a state quantity, but not at lag 0.

        Only an exact match can be called misalignment: a commanded action
        that the robot follows through actuator or controller lag also
        peaks at a non-zero lag, and that is dynamics, not a labeling bug.
        An action that *equals* a shifted state difference to float
        precision cannot be explained by dynamics.
        """
        return self.derived_from_state and self.lag != 0

    @property
    def derived_from_state(self) -> bool:
        """True when the dimension is exactly a state difference or level."""
        return self.classification in ("exact_state_difference", "exact_state_level")


def _average_ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks with ties given their average rank (needed: binary
    gripper actions are almost entirely ties)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    is_start = np.concatenate(([True], sorted_x[1:] != sorted_x[:-1]))
    group_of = np.cumsum(is_start) - 1
    starts = np.flatnonzero(is_start)
    ends = np.concatenate((starts[1:], [len(x)]))
    average = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = average[group_of]
    return ranks


def _standardized_ranks(
    m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Column-wise standardized ranks, plus each column's tie ceiling.

    Zero-variance columns become all zeros so they correlate as exactly 0
    instead of NaN.

    The tie ceiling is the highest Spearman rho a column can reach against
    any tie-free variable: the correlation between its tied (average) ranks
    and the same ordering with ties broken arbitrarily. A binary column
    against a continuous one tops out at sqrt(3)/2 (about 0.866) even when
    the relationship is perfectly monotone, so without this a genuinely
    commanded gripper would never read as strongly related.
    """
    out = np.zeros_like(m)
    ceiling = np.ones(m.shape[1], dtype=np.float64)
    for j in range(m.shape[1]):
        ranks = _average_ranks(m[:, j])
        centered = ranks - ranks.mean()
        norm = float(np.linalg.norm(centered))
        if norm == 0.0:
            continue
        out[:, j] = centered / norm
        ordinal = np.argsort(np.argsort(m[:, j], kind="mergesort"), kind="mergesort")
        ordinal_centered = ordinal - ordinal.mean()
        ceiling[j] = float(centered @ ordinal_centered) / (
            norm * float(np.linalg.norm(ordinal_centered))
        )
    return out, ceiling


def _spearman_matrix(a: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tie-adjusted Spearman rho between every column of ``a`` and ``y``.

    Plain rho divided by the most the two columns' ties allow (product of
    their tie ceilings), clipped to [-1, 1]. Identical to Spearman when
    neither column has ties.
    """
    a_ranks, a_ceiling = _standardized_ranks(a)
    y_ranks, y_ceiling = _standardized_ranks(y)
    raw = a_ranks.T @ y_ranks
    adjusted: NDArray[np.float64] = np.clip(raw / np.outer(a_ceiling, y_ceiling), -1.0, 1.0)
    return adjusted


def _pair_episode(
    actions: NDArray[np.float64],
    states: NDArray[np.float64],
    relation: Relation,
    lag: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Action rows paired with the state quantity ``lag`` steps from the
    standard alignment, entirely within one episode (never across episodes)."""
    n = len(actions)
    if relation == "delta":
        lo, hi = max(0, -lag), min(n - 1, n - 2 - lag)
        if hi < lo:
            return actions[:0], states[:0]
        t = np.arange(lo, hi + 1)
        return actions[t], states[t + 1 + lag] - states[t + lag]
    lo, hi = max(0, -1 - lag), min(n - 1, n - 2 - lag)
    if hi < lo:
        return actions[:0], states[:0]
    t = np.arange(lo, hi + 1)
    return actions[t], states[t + 1 + lag]


def _fit_line(a: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Least-squares ``a ~ k*y + b`` with one round of MAD outlier rejection."""

    def ols(yy: NDArray[np.float64], aa: NDArray[np.float64]) -> tuple[float, float]:
        if len(yy) < 2 or float(np.ptp(yy)) == 0.0:
            return 0.0, float(np.mean(aa))
        slope, intercept = np.polyfit(yy, aa, 1)
        return float(slope), float(intercept)

    slope, intercept = ols(y, a)
    residual = a - (slope * y + intercept)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center))) * _MAD_SCALE
    if mad > 0.0:
        keep = np.abs(residual - center) <= _OUTLIER_MADS * mad
        if int(keep.sum()) >= 3:
            slope, intercept = ols(y[keep], a[keep])
    return slope, intercept


def _classify(
    relation: Relation,
    rho: float,
    identity_fraction: float,
) -> Classification:
    if identity_fraction >= EXACT_IDENTITY_FRACTION:
        return "exact_state_difference" if relation == "delta" else "exact_state_level"
    strength = abs(rho)
    if strength >= STRONG_RHO:
        return "tracks_state_change" if relation == "delta" else "tracks_state_level"
    if strength >= WEAK_RHO:
        return "weak_relationship"
    return "no_relationship_found"


def audit_action_provenance(
    episodes: Iterable[Episode],
    state_key: str,
    *,
    max_episodes: int = 200,
    lags: tuple[int, ...] = (-2, -1, 0, 1, 2),
    identity_atol: float = 1e-5,
    min_steps: int = 20,
) -> tuple[ActionDimensionProvenance, ...]:
    """Audit, per action dimension, how it relates to the recorded state.

    Reads at most ``max_episodes`` episodes (the audit is statistical, and
    pairs are held in memory — nothing here materializes a full dataset).

    Args:
        episodes: Episodes to audit. Consumed lazily, once.
        state_key: Which ``Proprioception.extra`` array is the raw state to
            compare against (e.g. ``"observation.state"``). Named by the
            caller — no assumption is made about what its elements mean.
        max_episodes: Cap on episodes read.
        lags: Lags ``L`` to test; ``0`` is the standard alignment.
        identity_atol: An action counts as *equal* to a state quantity at a
            step when they differ by no more than this. Default ``1e-5`` is
            far above float32 rounding (~1e-7 relative) and far below real
            tracking error.
        min_steps: Fewest paired steps a candidate needs to be trusted.

    Returns:
        One `ActionDimensionProvenance` per action dimension, in order.

    Raises:
        ValueError: no episodes; ``state_key`` missing from a frame's
            ``extra``; action or state not one-dimensional or inconsistent
            in width between frames/episodes; empty ``lags``; or fewer than
            ``min_steps`` paired steps at every lag.
    """
    if max_episodes < 1:
        raise ValueError(f"max_episodes must be at least 1, got {max_episodes}")
    if not lags:
        raise ValueError("lags must not be empty")

    per_episode: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for episode in islice(episodes, max_episodes):
        action_rows: list[NDArray[np.float64]] = []
        state_rows: list[NDArray[np.float64]] = []
        for frame in episode:
            if state_key not in frame.proprioception.extra:
                raise ValueError(
                    f"state_key {state_key!r} isn't in this frame's Proprioception.extra "
                    f"(have {sorted(frame.proprioception.extra)}) — name the raw state "
                    "array to compare actions against"
                )
            action_rows.append(np.asarray(frame.action, dtype=np.float64))
            state_rows.append(np.asarray(frame.proprioception.extra[state_key], dtype=np.float64))
        if not action_rows:
            continue
        actions = np.stack(action_rows)
        states = np.stack(state_rows)
        if actions.ndim != 2 or states.ndim != 2:
            raise ValueError("actions and states must be one-dimensional per frame")
        per_episode.append((actions, states))

    if not per_episode:
        raise ValueError("no episodes with frames to audit")
    n_action = per_episode[0][0].shape[1]
    n_state = per_episode[0][1].shape[1]
    for actions, states in per_episode:
        if actions.shape[1] != n_action or states.shape[1] != n_state:
            raise ValueError("action/state width changes between episodes")

    pairs: dict[tuple[Relation, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    rho: dict[tuple[Relation, int], NDArray[np.float64]] = {}
    for relation in ("delta", "level"):
        for lag in lags:
            chunks = [_pair_episode(a, s, relation, lag) for a, s in per_episode]
            a_all = np.concatenate([c[0] for c in chunks])
            y_all = np.concatenate([c[1] for c in chunks])
            if len(a_all) < min_steps:
                continue
            pairs[(relation, lag)] = (a_all, y_all)
            rho[(relation, lag)] = _spearman_matrix(a_all, y_all)

    if not pairs:
        raise ValueError(
            f"fewer than {min_steps} paired steps at every lag — too little data to audit"
        )

    results: list[ActionDimensionProvenance] = []
    for j in range(n_action):
        # constant-action dimensions have nothing to match
        first_pairs = next(iter(pairs.values()))[0][:, j]
        if float(np.ptp(first_pairs)) == 0.0:
            results.append(
                ActionDimensionProvenance(
                    action_dim=j,
                    state_dim=-1,
                    relation="delta",
                    lag=0,
                    classification="constant_action",
                    spearman=0.0,
                    slope=0.0,
                    intercept=float(first_pairs[0]),
                    identity_fraction=0.0,
                    n_steps=len(first_pairs),
                    rho_by_lag={},
                    margin=0.0,
                )
            )
            continue

        best_key: tuple[Relation, int] | None = None
        best_state = -1
        best_abs = -1.0
        best_by_pair: dict[tuple[int, Relation], float] = {}
        for key, matrix in rho.items():
            relation_name, _lag = key
            for i in range(n_state):
                value = abs(float(matrix[j, i]))
                pair_key = (i, relation_name)
                best_by_pair[pair_key] = max(best_by_pair.get(pair_key, 0.0), value)
                if value > best_abs:
                    best_abs, best_key, best_state = value, key, i
        assert best_key is not None
        relation, lag = best_key
        others = [v for (i, r), v in best_by_pair.items() if (i, r) != (best_state, relation)]
        margin = best_abs - (max(others) if others else 0.0)

        a_all, y_all = pairs[best_key]
        a = a_all[:, j]
        y = y_all[:, best_state]
        slope, intercept = _fit_line(a, y)
        identity_fraction = float(np.mean(np.abs(a - y) <= identity_atol))
        signed_rho = float(rho[best_key][j, best_state])
        profile = {
            lag_value: float(rho[(relation, lag_value)][j, best_state])
            for lag_value in lags
            if (relation, lag_value) in rho
        }
        results.append(
            ActionDimensionProvenance(
                action_dim=j,
                state_dim=best_state,
                relation=relation,
                lag=lag,
                classification=_classify(relation, signed_rho, identity_fraction),
                spearman=signed_rho,
                slope=slope,
                intercept=intercept,
                identity_fraction=identity_fraction,
                n_steps=len(a),
                rho_by_lag=profile,
                margin=float(margin),
            )
        )
    return tuple(results)
