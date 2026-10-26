"""Posterior halfspace generation and lower-bound correction utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .geometry import clip_polygon_with_halfspace, convex_hull, polygon_area
from .guards import guard_halfspaces

EPS = 1e-12


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= EPS:
        return None
    return float(numerator / denominator)


def finite_float_or_none(value: float) -> float | None:
    if math.isfinite(value):
        return float(value)
    return None


def polygon_satisfies_halfspace(poly: np.ndarray, normal: np.ndarray, threshold: float, eps: float = 1e-10) -> bool:
    if len(poly) == 0:
        return True
    vals = np.asarray(poly, dtype=float) @ np.asarray(normal, dtype=float)
    return bool(np.all(vals >= threshold - eps))


def points_satisfy_halfspace(points: np.ndarray, normal: np.ndarray, threshold: float, eps: float = 1e-10) -> bool:
    if len(points) == 0:
        return True
    vals = np.asarray(points, dtype=float) @ np.asarray(normal, dtype=float)
    return bool(np.all(vals >= threshold - eps))


def projection_range(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(points, dtype=float) @ np.asarray(normal, dtype=float)
    return float(np.min(vals)), float(np.max(vals))


def generate_constraints(
    P: np.ndarray,
    T: np.ndarray,
    count: int,
    seed: int,
    mode: str,
    beta_a: float,
    beta_b: float,
    threshold_noise_scale: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    constraints: list[dict] = []
    if len(P) == 0:
        return constraints

    for i in range(count):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        p_min, p_max = projection_range(P, normal)

        if mode == "projection_uniform" or len(T) == 0:
            threshold = float(rng.uniform(p_min, p_max))
            generation_mode = "projection_uniform"
        elif mode == "boundary_gap":
            t_min, _ = projection_range(T, normal)
            gap = max(0.0, t_min - p_min)
            if gap <= EPS:
                threshold = float(rng.uniform(p_min, p_max))
                generation_mode = "fallback_projection_uniform"
            else:
                r = float(rng.beta(beta_a, beta_b))
                noise = float(rng.normal(0.0, threshold_noise_scale * gap))
                threshold = float(p_min + r * gap + noise)
                threshold = min(p_max, max(p_min, threshold))
                generation_mode = "boundary_gap"
        else:
            raise ValueError(f"Unknown constraint mode: {mode}")

        constraints.append(
            {
                "constraint_id": i,
                "normal": normal.tolist(),
                "threshold": threshold,
                "theta": theta,
                "generation_mode": generation_mode,
            }
        )
    return constraints


def evaluate_constraint_trace(
    P: np.ndarray,
    true_survivors: np.ndarray,
    old_lower_bound: float,
    true_loss_area: float,
    constraints: Iterable[dict],
) -> tuple[list[dict], np.ndarray]:
    P_area = polygon_area(P)
    current_polygon = np.asarray(P, dtype=float).reshape(-1, 2).copy()
    useful_count = 0
    triggered_count = 0
    trace: list[dict] = []

    for item in constraints:
        normal = np.asarray(item["normal"], dtype=float).reshape(2)
        threshold = float(item["threshold"])
        abstract_satisfies = polygon_satisfies_halfspace(P, normal, threshold)
        triggered = not abstract_satisfies
        useful = False
        useful_area_increment = 0.0

        if triggered:
            triggered_count += 1
            useful = points_satisfy_halfspace(true_survivors, normal, threshold)

        if useful:
            useful_count += 1
            before_area = polygon_area(current_polygon)
            current_polygon = clip_polygon_with_halfspace(current_polygon, normal, threshold)
            after_area = polygon_area(current_polygon)
            useful_area_increment = max(0.0, before_area - after_area)

        posterior_lower_bound = max(0.0, P_area - polygon_area(current_polygon))
        corrected_lower_bound = max(float(old_lower_bound), posterior_lower_bound)
        trace.append(
            {
                "constraint_id": int(item["constraint_id"]),
                "normal": item["normal"],
                "threshold": threshold,
                "generation_mode": item.get("generation_mode"),
                "triggered": bool(triggered),
                "useful": bool(useful),
                "triggered_count": int(triggered_count),
                "useful_count": int(useful_count),
                "useful_area_increment": float(useful_area_increment),
                "posterior_lower_bound": float(posterior_lower_bound),
                "old_lower_bound": float(old_lower_bound),
                "corrected_lower_bound": float(corrected_lower_bound),
                "true_loss_area": float(true_loss_area),
                "corrected_to_true_loss_ratio": finite_float_or_none(
                    ratio_or_none(corrected_lower_bound, true_loss_area) if true_loss_area > EPS else 0.0
                ),
            }
        )
    return trace, current_polygon



def pullback_halfspace(
    normal: np.ndarray,
    threshold: float,
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pull normal^T(Ax+b) >= threshold back to the affine input."""
    normal = np.asarray(normal, dtype=float).reshape(2)
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2)
    return A.T @ normal, float(threshold - normal @ b)


def evaluate_intermediate_pullback_trace(
    intermediate_polygon: np.ndarray,
    intermediate_true_points: np.ndarray,
    old_intermediate_lower_bound: float,
    second_layer: dict,
    final_trace: Iterable[dict],
) -> tuple[list[dict], np.ndarray]:
    """Refine the first guarded-layer result using final useful constraints.

    A useful final constraint H is pulled back as G_2 -> H. For multiple
    constraints the allowed region is not G_2 or (H_1 and ... and H_k).
    Because this region is generally non-convex, its convex hull is used so the
    removed area remains a sound lower bound relative to the intermediate
    concrete-point convex hull.
    """
    P = np.asarray(intermediate_polygon, dtype=float).reshape(-1, 2)
    true_points = np.asarray(intermediate_true_points, dtype=float).reshape(-1, 2)
    A = np.asarray(second_layer["A"], dtype=float).reshape(2, 2)
    b = np.asarray(second_layer["b"], dtype=float).reshape(2)
    P_area = polygon_area(P)
    true_hull_area = polygon_area(convex_hull(true_points))
    true_loss_area = max(0.0, P_area - true_hull_area)

    pass_polygon = P.copy()
    fail_polygons: list[np.ndarray] = []
    for guard_normal, guard_threshold in guard_halfspaces(second_layer["guard"]):
        pass_normal, pass_threshold = pullback_halfspace(
            guard_normal, guard_threshold, A, b
        )
        pass_polygon = clip_polygon_with_halfspace(
            pass_polygon, pass_normal, pass_threshold
        )
        fail_polygons.append(
            clip_polygon_with_halfspace(P, -pass_normal, -pass_threshold)
        )

    trace: list[dict] = []
    useful_count = 0
    allowed_hull = P.copy()
    for row in final_trace:
        if bool(row["useful"]):
            useful_count += 1
            pulled_normal, pulled_threshold = pullback_halfspace(
                np.asarray(row["normal"], dtype=float),
                float(row["threshold"]),
                A,
                b,
            )
            pass_polygon = clip_polygon_with_halfspace(
                pass_polygon, pulled_normal, pulled_threshold
            )

        if bool(row["useful"]):
            pieces = [piece for piece in fail_polygons + [pass_polygon] if len(piece)]
            allowed_hull = (
                convex_hull(np.vstack(pieces))
                if pieces
                else np.empty((0, 2), dtype=float)
            )
        posterior_lower_bound = max(0.0, P_area - polygon_area(allowed_hull))
        corrected_lower_bound = max(
            float(old_intermediate_lower_bound), posterior_lower_bound
        )
        trace.append(
            {
                "constraint_id": int(row["constraint_id"]),
                "triggered": bool(row.get("triggered", False)),
                "useful": bool(row["useful"]),
                "final_useful": bool(row["useful"]),
                "triggered_count": int(row.get("triggered_count", 0)),
                "useful_count": int(useful_count),
                "allowed_convex_area": float(polygon_area(allowed_hull)),
                "posterior_lower_bound": float(posterior_lower_bound),
                "old_lower_bound": float(old_intermediate_lower_bound),
                "corrected_lower_bound": float(corrected_lower_bound),
                "true_loss_area": float(true_loss_area),
                "recovery_ratio": finite_float_or_none(
                    ratio_or_none(corrected_lower_bound, true_loss_area)
                    if true_loss_area > EPS
                    else 0.0
                ),
            }
        )
    return trace, allowed_hull
