"""Guard schemas and guard application helpers.

Supported guard specs are intentionally simple and JSON serializable:

- {"type": "quadrant"}: x >= 0 and y >= 0.
- {"type": "halfspace", "normal": [a, b], "threshold": c}.
- {"type": "halfspaces", "halfspaces": [{"normal": [a, b], "threshold": c}, ...]}.

Legacy specs with {"constraints": ["x >= 0", "y >= 0"]} are accepted for the
quadrant guard used by the current experiments.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .geometry import clip_polygon_with_halfspace

Halfspace = tuple[np.ndarray, float]


def quadrant_guard_spec() -> dict:
    return {"type": "quadrant", "constraints": ["x >= 0", "y >= 0"]}


def quadrant_halfspaces() -> list[Halfspace]:
    return [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)]


def guard_halfspaces(guard: dict | None) -> list[Halfspace]:
    """Convert a JSON guard spec to halfspaces normal^T x >= threshold."""
    if guard is None:
        return quadrant_halfspaces()

    guard_type = guard.get("type", "quadrant")
    if guard_type == "quadrant":
        return quadrant_halfspaces()
    if guard_type == "halfspace":
        return [(np.asarray(guard["normal"], dtype=float).reshape(2), float(guard["threshold"]))]
    if guard_type == "halfspaces":
        return [
            (np.asarray(item["normal"], dtype=float).reshape(2), float(item["threshold"]))
            for item in guard.get("halfspaces", [])
        ]

    constraints = guard.get("constraints")
    if constraints == ["x >= 0", "y >= 0"] or constraints == ["y >= 0", "x >= 0"]:
        return quadrant_halfspaces()
    raise ValueError(f"Unsupported guard spec: {guard}")


def apply_guard_to_polygon(poly: np.ndarray, guard: dict | None = None) -> np.ndarray:
    out = np.asarray(poly, dtype=float).reshape(-1, 2)
    for normal, threshold in guard_halfspaces(guard):
        out = clip_polygon_with_halfspace(out, normal, threshold)
    return out


def points_satisfying_guard(points: np.ndarray, guard: dict | None = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(pts), dtype=bool)
    for normal, threshold in guard_halfspaces(guard):
        mask &= (pts @ normal) >= threshold
    return mask


def guarded_points(points: np.ndarray, guard: dict | None = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return pts[points_satisfying_guard(pts, guard)]


def apply_quadrant_guard(poly: np.ndarray) -> np.ndarray:
    return apply_guard_to_polygon(poly, quadrant_guard_spec())


def guarded_output(points: np.ndarray, A: np.ndarray, b: np.ndarray, guard: dict | None = None) -> np.ndarray:
    out = np.asarray(points, dtype=float).reshape(-1, 2) @ np.asarray(A, dtype=float).T + np.asarray(b, dtype=float)
    return guarded_points(out, guard)
