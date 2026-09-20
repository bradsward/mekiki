"""Tests for `mekiki.rotation` — hand-verified against known rotations."""

from __future__ import annotations

import numpy as np
import pytest

from mekiki.rotation import (
    OrientationEncoding,
    orientation_to_quaternion,
    quaternion_from_euler,
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


# --- Euler angles ---------------------------------------------------------

# Golden values generated with scipy.spatial.transform.Rotation.from_euler
# (scipy 1.18.1, 2026-09-20). scipy is not a mekiki dependency — it was used
# once as an independent oracle, and a randomized comparison over every valid
# sequence in both modes (4,800 cases) agreed to 2e-16 before these six were
# pinned here as regression values.
_SCIPY_EULER_GOLDEN = [
    (
        "xyz",
        (0.3, -0.5, 1.1),
        (0.2513019482416863, -0.1328683898180115, 0.5322705776530124, 0.7974216914293402),
    ),
    (
        "XYZ",
        (0.3, -0.5, 1.1),
        (-0.0044236978962037, -0.2842307321522804, 0.4692322109021089, 0.8360708427214887),
    ),
    (
        "zyx",
        (-0.7, 0.2, 0.9),
        (0.3757287827012926, 0.2328482438677356, -0.2664274062243827, 0.8565197104504160),
    ),
    (
        "ZYX",
        (-0.7, 0.2, 0.9),
        (0.4373781811288254, -0.0639589672753088, -0.3480102268106739, 0.8267396562477815),
    ),
    (
        "xzx",
        (1.0, 0.4, -0.6),
        (0.1947091711543252, 0.1425166545207693, 0.1384142557064306, 0.9605304970014426),
    ),
    (
        "YXY",
        (0.25, -1.2, 0.8),
        (-0.5434261440465140, 0.4136689434131764, -0.1533269341257453, 0.7141826674554368),
    ),
]


@pytest.mark.parametrize(("sequence", "angles", "expected"), _SCIPY_EULER_GOLDEN)
def test_euler_matches_scipy_golden_values(
    sequence: str, angles: tuple[float, float, float], expected: tuple[float, ...]
) -> None:
    q = quaternion_from_euler(np.array(angles), sequence)
    # q and -q are the same rotation
    assert min(np.abs(q - expected).max(), np.abs(q + expected).max()) < 1e-12


def test_euler_single_90deg_about_x_is_hand_computable() -> None:
    q = quaternion_from_euler(np.array([np.pi / 2, 0.0, 0.0]), "xyz")
    half = np.pi / 4
    assert q == pytest.approx([np.sin(half), 0.0, 0.0, np.cos(half)])


def test_extrinsic_and_intrinsic_of_the_same_angles_are_different_rotations() -> None:
    # hand-worked: extrinsic xyz with (90, 90, 0) maps e_x to -e_z; intrinsic
    # XYZ with the same angles maps e_x to +e_y. The whole reason there's no
    # default sequence case.
    angles = np.array([np.pi / 2, np.pi / 2, 0.0])
    e_x = np.array([1.0, 0.0, 0.0])
    extrinsic = quaternion_to_rotation_matrix(quaternion_from_euler(angles, "xyz")) @ e_x
    intrinsic = quaternion_to_rotation_matrix(quaternion_from_euler(angles, "XYZ")) @ e_x
    assert extrinsic == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    assert intrinsic == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("sequence", ["xy", "xyzx", "xya", "xYz", "xxz", "xzz", ""])
def test_invalid_euler_sequence_raises(sequence: str) -> None:
    with pytest.raises(ValueError, match="euler sequence"):
        quaternion_from_euler(np.zeros(3), sequence)


def test_euler_wrong_number_of_angles_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        quaternion_from_euler(np.zeros(2), "xyz")


# --- OrientationEncoding / orientation_to_quaternion ------------------------


def test_encoding_n_components() -> None:
    assert OrientationEncoding("quaternion_xyzw").n_components == 4
    assert OrientationEncoding("rotvec").n_components == 3
    assert OrientationEncoding("euler", "xyz").n_components == 3


def test_euler_encoding_requires_a_sequence() -> None:
    with pytest.raises(ValueError, match="requires euler_sequence"):
        OrientationEncoding("euler")


def test_euler_encoding_rejects_an_invalid_sequence() -> None:
    with pytest.raises(ValueError, match="euler sequence"):
        OrientationEncoding("euler", "xxy")


def test_non_euler_encoding_rejects_a_sequence() -> None:
    with pytest.raises(ValueError, match="only meaningful"):
        OrientationEncoding("rotvec", "xyz")


def test_orientation_to_quaternion_euler() -> None:
    q = orientation_to_quaternion(
        np.array([np.pi / 2, 0.0, 0.0]), OrientationEncoding("euler", "xyz")
    )
    half = np.pi / 4
    assert q == pytest.approx([np.sin(half), 0.0, 0.0, np.cos(half)])


def test_orientation_to_quaternion_rotvec() -> None:
    q = orientation_to_quaternion(np.array([0.0, 0.0, np.pi / 2]), OrientationEncoding("rotvec"))
    half = np.pi / 4
    assert q == pytest.approx([0.0, 0.0, np.sin(half), np.cos(half)])


def test_orientation_to_quaternion_passes_a_unit_quaternion_through() -> None:
    q_in = np.array([0.0, 0.0, 0.6, 0.8])
    q = orientation_to_quaternion(q_in, OrientationEncoding("quaternion_xyzw"))
    assert q == pytest.approx(q_in)


def test_orientation_to_quaternion_renormalizes_float32_scale_drift() -> None:
    q_in = np.array([0.0, 0.0, 0.6, 0.8]) * 1.0001  # within tolerance
    q = orientation_to_quaternion(q_in, OrientationEncoding("quaternion_xyzw"))
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_orientation_to_quaternion_rejects_a_materially_non_unit_quaternion() -> None:
    with pytest.raises(ValueError, match="not silently renormalized"):
        orientation_to_quaternion(
            np.array([0.0, 0.0, 0.6, 0.9]), OrientationEncoding("quaternion_xyzw")
        )


def test_orientation_to_quaternion_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="needs 3 values"):
        orientation_to_quaternion(np.zeros(4), OrientationEncoding("rotvec"))


def test_quaternion_angular_distance_ignores_double_cover() -> None:
    from mekiki.rotation import quaternion_angular_distance

    q = np.array([0.0, 0.0, 0.6, 0.8])
    assert quaternion_angular_distance(q, -q) == pytest.approx(0.0, abs=1e-9)
    assert quaternion_angular_distance(q, q) == pytest.approx(0.0, abs=1e-9)
