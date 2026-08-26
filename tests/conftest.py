"""Shared test fixtures for mekiki.

``make_clean_episode`` builds a minimal, valid episode with no defects —
the "clean control" every detector's test suite is expected to run its
checks against alongside a fixture with the defect deliberately injected.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from mekiki.episode import (
    ActionDimSpec,
    CameraFrame,
    Episode,
    EpisodeMetadata,
    Frame,
    Pose,
    Proprioception,
)

#: A minimal action space: three translational deltas plus a gripper target,
#: all expressed in the end-effector frame.
CLEAN_ACTION_SPACE: tuple[ActionDimSpec, ...] = (
    ActionDimSpec("x", "delta", "m", "ee"),
    ActionDimSpec("y", "delta", "m", "ee"),
    ActionDimSpec("z", "delta", "m", "ee"),
    ActionDimSpec("gripper", "absolute", "normalized", "ee"),
)


def _identity_quaternion() -> NDArray[np.float64]:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _blank_image(resolution: tuple[int, int]) -> NDArray[np.uint8]:
    height, width = resolution
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_clean_frame(
    *,
    timestamp: float,
    is_first: bool = False,
    is_last: bool = False,
    control_hz: float = 10.0,
) -> Frame:
    """Build one well-formed frame with a small, deterministic action.

    Args:
        timestamp: Frame timestamp in seconds, relative to episode start.
        is_first: Whether this is the episode's first frame.
        is_last: Whether this is the episode's last frame.
        control_hz: Control frequency used only to size the per-step delta
            action so it stays physically plausible (small, constant-speed
            motion) — not stored anywhere.

    Returns:
        A ``Frame`` with a single ``"exterior"`` camera, arbitrary but valid
        proprioception, and an action matching ``CLEAN_ACTION_SPACE``.
    """
    step = 0.01 / control_hz  # 1 cm/s nominal end-effector speed
    action = np.array([step, 0.0, 0.0, 1.0], dtype=np.float64)
    resolution = (64, 64)
    return Frame(
        timestamp=timestamp,
        proprioception=Proprioception(
            joint_positions=np.zeros(7, dtype=np.float64),
            joint_velocities=np.zeros(7, dtype=np.float64),
            ee_poses={
                "ee": Pose(
                    position=np.array([timestamp * step, 0.0, 0.2], dtype=np.float64),
                    orientation=_identity_quaternion(),
                    frame="base_link",
                )
            },
            grippers={"ee": 1.0},
            extra={},
        ),
        action=action,
        images={
            "exterior": CameraFrame(
                read=lambda: _blank_image(resolution),
                timestamp=timestamp,
                resolution=resolution,
                timestamp_is_measured=False,
            )
        },
        is_first=is_first,
        is_last=is_last,
        success=True,
        language_instruction="pick up the cube",
    )


def make_clean_episode(*, n_frames: int = 5, control_hz: float = 10.0) -> Episode:
    """Build a minimal, defect-free episode for use as a test control.

    Args:
        n_frames: Number of frames in the episode.
        control_hz: Nominal control frequency (frames evenly spaced at
            ``1 / control_hz`` seconds apart).

    Returns:
        An ``Episode`` with ``n_frames`` well-formed frames and
        ``CLEAN_ACTION_SPACE`` metadata.
    """
    frames = [
        make_clean_frame(
            timestamp=i / control_hz,
            is_first=(i == 0),
            is_last=(i == n_frames - 1),
            control_hz=control_hz,
        )
        for i in range(n_frames)
    ]
    metadata = EpisodeMetadata(
        episode_id="clean-0",
        dataset_name="synthetic",
        robot_embodiment="franka_panda",
        action_space=CLEAN_ACTION_SPACE,
        source_format="synthetic",
    )
    return Episode(metadata=metadata, frames=frames)


@pytest.fixture
def clean_episode() -> Iterator[Episode]:
    """A minimal, defect-free episode — the control for detector tests."""
    yield make_clean_episode()
