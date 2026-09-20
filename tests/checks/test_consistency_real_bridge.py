"""Regression test built from real robot data.

Five consecutive frames copied verbatim from ``IPEC-COMMUNITY/bridge_orig_lerobot``
(LeRobotDataset v2.0, WidowX, episode 3, frames 10-14). The full validation ran
over 25 real episodes (930 steps); these numbers are pinned here so the
properties it established stay under test without needing a download:

* the recorded position action is (to float32 precision) the recorded
  next-state position minus the current one, expressed in the base frame — so
  declaring that frame gives ~zero residual, and declaring the wrong one
  (``ee``) gives a residual orders of magnitude larger: the check can tell a
  correct frame declaration from a wrong one;
* real gripper readings overshoot 1.0 (1.00015 below) and must not crash
  reconstruction;
* orientation actions are Euler-angle *differences*, so reading them as
  rotation vectors leaves a small but real residual (documented in
  docs/consistency.md, not a bug).
"""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.checks.consistency import ActionTarget, ToleranceModel, check_action_state_consistency
from mekiki.episode import ActionDimSpec, Episode, EpisodeMetadata, Frame, Proprioception
from mekiki.rotation import OrientationEncoding
from mekiki.state import StateField, reconstruct_episode_proprioception

# (timestamp, observation.state[8], action[7]) — real values, bridge_orig ep 3
_REAL_FRAMES = [
    (
        2.0,
        [
            0.326736897,
            -0.0648702458,
            0.100998871,
            0.018867841,
            0.0795753822,
            0.150666967,
            0.0,
            1.00015318,
        ],
        [
            -0.00384321809,
            -0.000246286392,
            -0.00413770229,
            0.00266347639,
            0.0138058588,
            0.00741164386,
            1.0,
        ],
    ),
    (
        2.2,
        [
            0.322893679,
            -0.0651165321,
            0.0968611687,
            0.0215313174,
            0.093381241,
            0.158078611,
            0.0,
            0.999488592,
        ],
        [
            -0.00304657221,
            0.000119701028,
            -0.00291308016,
            0.000499185175,
            0.00962290913,
            0.00118398666,
            1.0,
        ],
    ),
    (
        2.4,
        [
            0.319847107,
            -0.0649968311,
            0.0939480886,
            0.0220305026,
            0.10300415,
            0.159262598,
            0.0,
            1.00015318,
        ],
        [
            -0.00310501456,
            0.000159434974,
            -0.0033692494,
            -0.0012686681,
            0.0119792223,
            0.00675864518,
            1.0,
        ],
    ),
    (
        2.6,
        [
            0.316742092,
            -0.0648373961,
            0.0905788392,
            0.0207618345,
            0.114983372,
            0.166021243,
            0.0,
            1.00015318,
        ],
        [
            -0.000984251499,
            -4.19318676e-05,
            -0.00174486637,
            -3.08454037e-05,
            0.00179357082,
            0.00679144263,
            0.0,
        ],
    ),
    (
        2.8,
        [
            0.315757841,
            -0.064879328,
            0.0888339728,
            0.0207309891,
            0.116776943,
            0.172812685,
            0.0,
            0.947663724,
        ],
        [
            0.000341653824,
            -0.000567637384,
            -0.000551052392,
            -0.000169916078,
            -0.00150448084,
            0.00155836344,
            0.0,
        ],
    ),
]

_EULER = OrientationEncoding("euler", "xyz")
_STATE_SPEC = {
    "observation.state": (
        *(
            StateField(kind="position_axis", end_effector="ee", axis=a, frame="base_link")
            for a in ("x", "y", "z")
        ),
        *(
            StateField(
                kind="orientation_axis",
                end_effector="ee",
                component=i,
                frame="base_link",
                encoding=_EULER,
            )
            for i in range(3)
        ),
        None,
        StateField(kind="gripper", end_effector="ee"),
    )
}
_TARGETS = (
    ActionTarget(kind="position_axis", end_effector="ee", axis="x"),
    ActionTarget(kind="position_axis", end_effector="ee", axis="y"),
    ActionTarget(kind="position_axis", end_effector="ee", axis="z"),
    ActionTarget(kind="orientation_axis", end_effector="ee", axis="x"),
    ActionTarget(kind="orientation_axis", end_effector="ee", axis="y"),
    ActionTarget(kind="orientation_axis", end_effector="ee", axis="z"),
    ActionTarget(kind="gripper", end_effector="ee"),
)
_ZERO_TOLERANCE = ToleranceModel(0, 0, 0, 0, 0, 0, 0, 0)


def _action_space(pos_frame: str) -> tuple[ActionDimSpec, ...]:
    return (
        ActionDimSpec("dx", "delta", "m", pos_frame),
        ActionDimSpec("dy", "delta", "m", pos_frame),
        ActionDimSpec("dz", "delta", "m", pos_frame),
        ActionDimSpec("drx", "delta", "rad", "ee"),
        ActionDimSpec("dry", "delta", "rad", "ee"),
        ActionDimSpec("drz", "delta", "rad", "ee"),
        ActionDimSpec("gripper", "absolute", "normalized", "ee"),
    )


def _episode(pos_frame: str) -> Episode:
    frames = [
        Frame(
            timestamp=t,
            proprioception=Proprioception(
                joint_positions=None,
                joint_velocities=None,
                ee_poses={},
                grippers={},
                extra={"observation.state": np.array(state, dtype=np.float64)},
            ),
            action=np.array(action, dtype=np.float64),
            images={},
            is_first=(i == 0),
            is_last=(i == len(_REAL_FRAMES) - 1),
        )
        for i, (t, state, action) in enumerate(_REAL_FRAMES)
    ]
    metadata = EpisodeMetadata(
        episode_id="3",
        dataset_name="bridge_orig_lerobot",
        robot_embodiment="widowx",
        action_space=_action_space(pos_frame),
        source_format="lerobot",
    )
    return Episode(metadata=metadata, frames=frames)


def _residuals(pos_frame: str) -> dict[str, float]:
    reconstructed = reconstruct_episode_proprioception(_episode(pos_frame), _STATE_SPEC)
    results = check_action_state_consistency(
        reconstructed, _action_space(pos_frame), _TARGETS, _ZERO_TOLERANCE
    )
    return {key: result.max_residual for key, result in results.items()}


def test_correct_frame_gives_essentially_zero_position_residual() -> None:
    assert _residuals("base_link")["position:ee"] < 1e-6


def test_wrong_frame_declaration_is_visibly_worse() -> None:
    right = _residuals("base_link")["position:ee"]
    wrong = _residuals("ee")["position:ee"]
    assert wrong > 1e-4  # a real ~mm-scale error, not float noise
    assert wrong > 1000 * right


def test_real_gripper_overshoot_past_one_does_not_crash_reconstruction() -> None:
    # the first frame's raw gripper is 1.00015 -- outside [0, 1], ordinary
    # sensor calibration behavior, must be accepted (and clipped) not raised on
    reconstructed = reconstruct_episode_proprioception(_episode("base_link"), _STATE_SPEC)
    grippers = [frame.proprioception.grippers["ee"] for frame in reconstructed]
    assert all(0.0 <= g <= 1.0 for g in grippers)
    assert grippers[0] == pytest.approx(1.0)


def test_euler_difference_actions_read_as_rotation_vectors_leave_a_small_residual() -> None:
    # not zero (Euler differences aren't rotation vectors away from the zero
    # configuration) but small and bounded -- well under a null "no motion"
    # prediction would be for steps this size
    residual = _residuals("base_link")["orientation:ee"]
    assert 1e-4 < residual < 0.05
