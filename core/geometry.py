"""Reusable 2D convex geometry utilities for the POPL precision-loss experiments."""

from __future__ import annotations

import numpy as np


def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return monotone-chain convex hull vertices in counter-clockwise order."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return pts.copy()

    pts = np.unique(pts, axis=0)
    if len(pts) <= 1:
        return pts.copy()

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def convex_hull_indices(points: np.ndarray) -> np.ndarray:
    """Return indices of monotone-chain hull vertices in the original point array."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return np.arange(len(pts), dtype=int)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]

    lower: list[int] = []
    for local_idx, point in enumerate(sorted_pts):
        while len(lower) >= 2 and cross(sorted_pts[lower[-2]], sorted_pts[lower[-1]], point) <= 0:
            lower.pop()
        lower.append(local_idx)

    upper: list[int] = []
    for local_idx in range(len(sorted_pts) - 1, -1, -1):
        point = sorted_pts[local_idx]
        while len(upper) >= 2 and cross(sorted_pts[upper[-2]], sorted_pts[upper[-1]], point) <= 0:
            upper.pop()
        upper.append(local_idx)

    hull_local = lower[:-1] + upper[:-1]
    return order[np.asarray(hull_local, dtype=int)].astype(int)


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def box_polygon(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)


def affine_polygon(poly: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) == 0:
        return poly.copy()
    return convex_hull(poly @ np.asarray(A, dtype=float).T + np.asarray(b, dtype=float))


def clip_polygon_with_halfspace(
    poly: np.ndarray,
    normal: np.ndarray,
    threshold: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Clip a convex polygon by normal^T x >= threshold."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) == 0:
        return np.empty((0, 2), dtype=float)

    normal = np.asarray(normal, dtype=float).reshape(2)

    def is_inside(p: np.ndarray) -> bool:
        return float(np.dot(normal, p)) >= threshold - eps

    def intersection(s: np.ndarray, e: np.ndarray) -> np.ndarray:
        direction = e - s
        denom = float(np.dot(normal, direction))
        if abs(denom) < eps:
            return s.copy()
        t = (threshold - float(np.dot(normal, s))) / denom
        t = min(1.0, max(0.0, t))
        return s + t * direction

    out: list[np.ndarray] = []
    n = len(poly)
    for i in range(n):
        s = poly[i]
        e = poly[(i + 1) % n]
        s_in = is_inside(s)
        e_in = is_inside(e)
        if s_in and e_in:
            out.append(e)
        elif s_in and not e_in:
            out.append(intersection(s, e))
        elif not s_in and e_in:
            out.append(intersection(s, e))
            out.append(e)

    if not out:
        return np.empty((0, 2), dtype=float)
    return convex_hull(np.asarray(out, dtype=float))
