"""
2D Vector utilities for crowd simulation.
Uses NumPy arrays for efficient computation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# Type alias for 2D vectors
Vec2 = NDArray[np.float64]


def vec2(x: float, y: float) -> Vec2:
    """Create a 2D vector."""
    return np.array([x, y], dtype=np.float64)


def zero() -> Vec2:
    """Return zero vector."""
    return np.array([0.0, 0.0], dtype=np.float64)


def magnitude(v: Vec2) -> float:
    """Compute vector magnitude."""
    return float(np.linalg.norm(v))


def magnitude_squared(v: Vec2) -> float:
    """Compute squared magnitude (avoids sqrt)."""
    return float(np.dot(v, v))


def normalize(v: Vec2) -> Vec2:
    """Normalize vector to unit length. Returns zero if magnitude is zero."""
    mag = magnitude(v)
    if mag < 1e-10:
        return zero()
    return v / mag


def clamp_magnitude(v: Vec2, max_mag: float) -> Vec2:
    """Clamp vector magnitude to maximum value."""
    mag = magnitude(v)
    if mag > max_mag and mag > 1e-10:
        return (v / mag) * max_mag
    return v.copy()


def distance(a: Vec2, b: Vec2) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def distance_squared(a: Vec2, b: Vec2) -> float:
    """Squared distance (avoids sqrt)."""
    diff = a - b
    return float(np.dot(diff, diff))


def dot(a: Vec2, b: Vec2) -> float:
    """Dot product."""
    return float(np.dot(a, b))


def cross_2d(a: Vec2, b: Vec2) -> float:
    """2D cross product (returns scalar)."""
    return float(a[0] * b[1] - a[1] * b[0])


def rotate(v: Vec2, angle: float) -> Vec2:
    """Rotate vector by angle (radians)."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return vec2(
        v[0] * cos_a - v[1] * sin_a,
        v[0] * sin_a + v[1] * cos_a
    )


def angle_of(v: Vec2) -> float:
    """Get angle of vector from positive x-axis (radians)."""
    return float(np.arctan2(v[1], v[0]))


def from_angle(angle: float, magnitude: float = 1.0) -> Vec2:
    """Create vector from angle and magnitude."""
    return vec2(np.cos(angle) * magnitude, np.sin(angle) * magnitude)


def lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    """Linear interpolation between vectors."""
    return a + (b - a) * t


def random_in_circle(radius: float) -> Vec2:
    """Random point within a circle of given radius."""
    angle = np.random.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(np.random.uniform(0, 1))  # sqrt for uniform distribution
    return vec2(r * np.cos(angle), r * np.sin(angle))


def random_unit() -> Vec2:
    """Random unit vector."""
    angle = np.random.uniform(0, 2 * np.pi)
    return vec2(np.cos(angle), np.sin(angle))


# Batch operations for performance

def batch_distances(positions: NDArray, point: Vec2) -> NDArray:
    """Compute distances from all positions to a single point."""
    diff = positions - point
    return np.sqrt(np.sum(diff * diff, axis=1))


def batch_magnitudes(vectors: NDArray) -> NDArray:
    """Compute magnitudes of multiple vectors."""
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def circular_mean(angles: NDArray) -> float:
    """Compute circular mean of angles (radians)."""
    if len(angles) == 0:
        return 0.0
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return float(np.arctan2(sin_sum, cos_sum))


def circular_variance(angles: NDArray) -> float:
    """
    Compute circular variance of angles.
    Returns value in [0, 1]: 0 = all same direction, 1 = uniform distribution.
    """
    if len(angles) == 0:
        return 0.0
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    return float(1 - r)
