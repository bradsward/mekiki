"""Tests for `mekiki.rotation` — hand-verified against known rotations."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.rotation import (
    quaternion_from_rotation_vector,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
)

# --- quaternion math, hand-verified ---------------------------------------
#
# All (x, y, z, w) throughout, matching mekiki.episode.Pose.orientation.

_IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def test_rotation_matrix_90deg_about_z_rotates_x_axis_to_y_axis() -> None:
    # a 90-degree rotation about z: q = (0, 0, sin(45deg), cos(45deg))
    half = np.pi / 4
    q = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    r = quaternion_to_rotation_matrix(q)
    rotated = r @ np.array([1.0, 0.0, 0.0])
    assert rotated == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def testquaternion_from_zero_rotation_vector_is_identity() -> None:
    q = quaternion_from_rotation_vector(np.array([0.0, 0.0, 0.0]))
    assert q == pytest.approx(_IDENTITY_Q)


def testquaternion_from_rotation_vector_90deg_about_z() -> None:
    r = np.array([0.0, 0.0, np.pi / 2])  # axis=z, angle=90deg
    q = quaternion_from_rotation_vector(r)
    half = np.pi / 4
    assert q == pytest.approx([0.0, 0.0, np.sin(half), np.cos(half)])


def testquaternion_multiply_by_identity_is_unchanged() -> None:
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)])
    assert quaternion_multiply(q, _IDENTITY_Q) == pytest.approx(q)


def testquaternion_multiply_two_90deg_z_rotations_gives_180deg_about_z() -> None:
    # hand-verified: composing two 90deg-about-z rotations gives 180deg about
    # z, i.e. q = (0, 0, 1, 0)
    half = np.pi / 4
    q90 = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    composed = quaternion_multiply(q90, q90)
    assert composed == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-9)
