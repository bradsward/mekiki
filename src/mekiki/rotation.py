"""Quaternion and rotation math shared by ``mekiki.checks`` and ``mekiki.state``.

Quaternions throughout are ``(x, y, z, w)``, unit norm, matching
`mekiki.episode.Pose.orientation`. These are the standard, well-known
formulas (Hamilton product; the usual unit-quaternion-to-rotation-matrix
conversion; the exponential map from a rotation vector; quaternion angular
distance) — not reinvented, and each has a hand-computable test case in
``tests/test_rotation.py`` (e.g. a 90-degree rotation about z) rather than
just "looks plausible."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quaternion_to_rotation_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix for a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_vector_by_quaternion(
    q: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rotate a 3-vector ``v`` by unit quaternion ``q`` (active rotation)."""
    result: NDArray[np.float64] = quaternion_to_rotation_matrix(q) @ v
    return result


def quaternion_from_rotation_vector(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponential map: a rotation vector (axis * angle, radians) to a unit
    quaternion. The zero vector maps to the identity rotation."""
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = r / angle
    half = angle / 2.0
    sin_half = np.sin(half)
    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)],
        dtype=np.float64,
    )


def quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product ``q1 ⊗ q2`` — as a rotation, ``q2`` applied first in
    ``q1``'s own (body) frame, then ``q1``. Both and the result are
    ``(x, y, z, w)``."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_angular_distance(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
    """Angular distance in radians between two unit quaternions.

    ``θ = 2 · arccos(|dot(q1, q2)|)``. The absolute value is not optional —
    ``q`` and ``-q`` represent the same rotation, and skipping it turns
    every correct prediction into a reported ~180° error (docs/consistency.md).
    ``dot`` is clipped to ``[0, 1]`` before ``arccos`` purely to absorb
    floating-point overshoot past 1.0 for two very-nearly-equal quaternions.
    """
    dot = float(np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))
