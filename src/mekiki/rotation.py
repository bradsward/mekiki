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

from dataclasses import dataclass
from typing import Literal

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


_AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

#: A quaternion whose norm is further than this from 1 is rejected rather
#: than silently renormalized — see `orientation_to_quaternion`.
QUATERNION_NORM_TOLERANCE = 1e-3


def _single_axis_quaternion(axis: str, angle: float) -> NDArray[np.float64]:
    half = angle / 2.0
    vector = _AXIS_VECTORS[axis.lower()] * np.sin(half)
    return np.array([vector[0], vector[1], vector[2], np.cos(half)], dtype=np.float64)


def _validate_euler_sequence(sequence: str) -> None:
    if len(sequence) != 3 or any(c not in "xyzXYZ" for c in sequence):
        raise ValueError(f"euler sequence must be three axis letters from x/y/z, got {sequence!r}")
    if not (sequence.islower() or sequence.isupper()):
        raise ValueError(
            f"euler sequence {sequence!r} mixes cases — use all lowercase for extrinsic "
            "(fixed-axis) rotations or all uppercase for intrinsic (body-axis) ones"
        )
    lowered = sequence.lower()
    if lowered[0] == lowered[1] or lowered[1] == lowered[2]:
        raise ValueError(
            f"euler sequence {sequence!r} repeats an axis on consecutive rotations, "
            "which isn't a valid three-angle parameterization"
        )


def quaternion_from_euler(angles: NDArray[np.float64], sequence: str) -> NDArray[np.float64]:
    """Unit quaternion for three Euler angles, in scipy's notation.

    Args:
        angles: Three angles in radians, in the order named by ``sequence``.
        sequence: Three axis letters. **Lowercase is extrinsic** — rotations
            about the fixed axes, the first letter's rotation applied first
            (so ``"xyz"`` is roll about fixed x, then pitch about fixed y,
            then yaw about fixed z). **Uppercase is intrinsic** — each
            rotation about the axes of the body as already rotated. Mixed
            case, a wrong length, and consecutive repeated axes are
            rejected. There is no default: the same three angles under
            ``"xyz"`` and ``"XYZ"`` are different rotations.

    Returns:
        The unit quaternion ``(x, y, z, w)``. Verified against scipy's
        ``Rotation.from_euler`` for every valid sequence in both modes.

    Raises:
        ValueError: ``sequence`` is invalid, or ``angles`` doesn't have
            exactly three values.
    """
    _validate_euler_sequence(sequence)
    if angles.shape != (3,):
        raise ValueError(f"euler angles must have shape (3,), got {angles.shape}")
    q1, q2, q3 = (
        _single_axis_quaternion(axis, float(angle))
        for axis, angle in zip(sequence, angles, strict=True)
    )
    if sequence.islower():  # extrinsic
        return quaternion_multiply(q3, quaternion_multiply(q2, q1))
    return quaternion_multiply(q1, quaternion_multiply(q2, q3))  # intrinsic


@dataclass(frozen=True, slots=True)
class OrientationEncoding:
    """How a run of raw numbers encodes an orientation — always declared by
    the caller, never inferred (docs/consistency.md, "Orientation encodings").

    Attributes:
        kind: ``"quaternion_xyzw"`` (4 values, already a quaternion),
            ``"rotvec"`` (3 values, axis times angle, in radians), or
            ``"euler"`` (3 angles in radians, needs ``euler_sequence``).
        euler_sequence: Required for, and only allowed with, ``"euler"`` —
            scipy-style sequence, lowercase extrinsic / uppercase intrinsic
            (see `quaternion_from_euler`).
    """

    kind: Literal["quaternion_xyzw", "rotvec", "euler"]
    euler_sequence: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "euler":
            if self.euler_sequence is None:
                raise ValueError(
                    "an 'euler' orientation encoding requires euler_sequence — there is "
                    "deliberately no default, the same angles under 'xyz' (extrinsic) "
                    "and 'XYZ' (intrinsic) are different rotations"
                )
            _validate_euler_sequence(self.euler_sequence)
        elif self.euler_sequence is not None:
            raise ValueError(
                f"euler_sequence is only meaningful for an 'euler' encoding, not {self.kind!r}"
            )

    @property
    def n_components(self) -> int:
        """How many raw values this encoding consumes."""
        return 4 if self.kind == "quaternion_xyzw" else 3


def orientation_to_quaternion(
    values: NDArray[np.float64], encoding: OrientationEncoding
) -> NDArray[np.float64]:
    """Convert raw orientation numbers to a unit quaternion ``(x, y, z, w)``.

    Args:
        values: Exactly ``encoding.n_components`` raw values.
        encoding: How ``values`` are encoded — caller-declared.

    Returns:
        A unit quaternion.

    Raises:
        ValueError: ``values`` has the wrong length; or a
            ``"quaternion_xyzw"`` input's norm is more than
            `QUATERNION_NORM_TOLERANCE` from 1. That's rejected instead of
            being renormalized because a materially non-unit quaternion in a
            recorded dataset is a data problem worth surfacing, not
            smoothing over; within tolerance it is renormalized to absorb
            float32 storage error.
    """
    if values.shape != (encoding.n_components,):
        raise ValueError(
            f"a {encoding.kind!r} orientation needs {encoding.n_components} values, "
            f"got shape {values.shape}"
        )
    if encoding.kind == "quaternion_xyzw":
        norm = float(np.linalg.norm(values))
        if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
            raise ValueError(
                f"quaternion has norm {norm:.6f}, more than {QUATERNION_NORM_TOLERANCE} "
                "from 1 — not silently renormalized, this looks like a data problem"
            )
        result: NDArray[np.float64] = values / norm
        return result
    if encoding.kind == "rotvec":
        return quaternion_from_rotation_vector(values)
    assert encoding.euler_sequence is not None  # guaranteed by __post_init__
    return quaternion_from_euler(values, encoding.euler_sequence)
