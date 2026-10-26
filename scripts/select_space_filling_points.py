"""
select_space_filling_points.py

Select a small subset of the original triangle point cloud for point-wise
program evaluation.

Selection policy:
  1. Keep all support points of the original convex hull.
  2. Compute the polygon centroid of the convex hull and triangulate the hull
     by connecting the centroid to each hull edge.
  3. Allocate the remaining budget to triangles by area, then perform
     barycentric stratified random sampling inside each triangle.
  4. Fill any remaining budget with a randomized farthest-point rule, choosing
     from the top-k farthest candidates at each step.

Optional grid-cover policy:
  1. Keep all support points of the original convex hull.
  2. Partition the plane by either square grid cells or equilateral triangular
     grid cells.
  3. For each occupied cell without a support point, keep the original point
     nearest to the cell center.
  4. If a target h is supplied, use the finest cell size that gives cell
     diameter <= h. If only max_keep is supplied, binary-search the finest cell
     size whose representatives fit the budget.

Default:
  max_keep = 800
  seed = 283

Output:
  <out_dir>/selected_point_indices.npy
  <out_dir>/selected_points.npy
  <out_dir>/selected_points_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from core.geometry import convex_hull_indices, cross, polygon_area


def normalize_points(points: np.ndarray, mins: np.ndarray, spans: np.ndarray) -> np.ndarray:
    return (points - mins) / spans


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Return the area centroid of a non-degenerate polygon."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return np.mean(poly, axis=0)

    x = poly[:, 0]
    y = poly[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross_terms = x * y_next - x_next * y
    signed_area = 0.5 * float(np.sum(cross_terms))
    if abs(signed_area) < 1e-12:
        return np.mean(poly, axis=0)

    cx = float(np.sum((x + x_next) * cross_terms) / (6.0 * signed_area))
    cy = float(np.sum((y + y_next) * cross_terms) / (6.0 * signed_area))
    return np.array([cx, cy], dtype=float)


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs(cross(a, b, c))


def barycentric_coordinates(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Return barycentric coordinates relative to triangle (a, b, c)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    v0 = b - a
    v1 = c - a
    v2 = pts - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return np.full((len(pts), 3), np.nan, dtype=float)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.column_stack([u, v, w])


def allocate_integer_quotas(weights: np.ndarray, total: int) -> np.ndarray:
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=int)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        quotas = np.zeros(len(weights), dtype=int)
        quotas[: min(total, len(weights))] = 1
        return quotas

    raw = total * weights / weight_sum
    quotas = np.floor(raw).astype(int)
    remainder = total - int(np.sum(quotas))
    if remainder > 0:
        order = np.argsort(raw - quotas)[::-1]
        quotas[order[:remainder]] += 1
    return quotas


def centroid_triangulated_stratified_candidates(
    points: np.ndarray,
    hull_indices: np.ndarray,
    candidate_indices: np.ndarray,
    selected_mask: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    grid_bins: int | None,
    stratum_scale: float,
) -> tuple[list[int], dict]:
    """
    Stratify the convex hull using centroid-based triangulation.

    Each hull edge and the polygon centroid form one triangle. The budget is
    allocated by triangle area, and candidates in each triangle are grouped by
    discretized barycentric coordinates. At most one point is sampled from each
    occupied stratum before moving to fallback farthest-point completion.
    """
    if budget <= 0 or len(candidate_indices) == 0:
        return [], {
            "triangulation_method": "polygon_centroid_fan",
            "triangle_count": 0,
            "polygon_centroid": None,
            "stratum_scale": stratum_scale,
            "grid_bins_override": grid_bins,
        }

    hull_vertices = points[hull_indices]
    centroid = polygon_centroid(hull_vertices)
    n_triangles = len(hull_vertices)
    triangles = [
        (centroid, hull_vertices[i], hull_vertices[(i + 1) % n_triangles])
        for i in range(n_triangles)
    ]
    triangle_areas = np.asarray(
        [triangle_area(a, b, c) for a, b, c in triangles], dtype=float
    )
    quotas = allocate_integer_quotas(triangle_areas, budget)

    chosen: list[int] = []
    per_triangle_selected: list[int] = []
    per_triangle_candidates: list[int] = []
    per_triangle_bins: list[int] = []
    unassigned: set[int] = set(int(idx) for idx in candidate_indices)
    candidate_points = points[candidate_indices]

    for tri_id, (a, b, c) in enumerate(triangles):
        quota = int(quotas[tri_id])
        if quota <= 0 or not unassigned:
            per_triangle_selected.append(0)
            per_triangle_candidates.append(0)
            per_triangle_bins.append(0)
            continue

        active_indices = np.fromiter(unassigned, dtype=int)
        active_points = points[active_indices]
        bary = barycentric_coordinates(active_points, a, b, c)
        inside_mask = np.all(bary >= -1e-12, axis=1) & np.all(bary <= 1.0 + 1e-12, axis=1)
        tri_candidate_indices = active_indices[inside_mask]
        tri_bary = bary[inside_mask]
        per_triangle_candidates.append(int(len(tri_candidate_indices)))

        if len(tri_candidate_indices) == 0:
            per_triangle_selected.append(0)
            per_triangle_bins.append(0)
            continue

        if grid_bins is None:
            bins = max(1, int(np.ceil(stratum_scale * np.sqrt(quota + 1))))
        else:
            bins = max(1, int(grid_bins))
        per_triangle_bins.append(int(bins))

        clipped = np.clip(tri_bary, 0.0, 1.0)
        ij = np.floor(clipped[:, :2] * bins).astype(int)
        ij = np.clip(ij, 0, bins - 1)
        # Barycentric lambda_0/lambda_1 bins are enough; lambda_2 is determined.
        strata: dict[tuple[int, int], list[int]] = {}
        for idx, cell in zip(tri_candidate_indices, ij):
            strata.setdefault((int(cell[0]), int(cell[1])), []).append(int(idx))

        keys = list(strata)
        rng.shuffle(keys)
        tri_chosen: list[int] = []
        for key in keys:
            if len(tri_chosen) >= quota or len(chosen) >= budget:
                break
            available = [idx for idx in strata[key] if not selected_mask[idx]]
            if not available:
                continue
            picked = int(rng.choice(available))
            selected_mask[picked] = True
            tri_chosen.append(picked)
            chosen.append(picked)

        for idx in tri_candidate_indices:
            unassigned.discard(int(idx))
        per_triangle_selected.append(int(len(tri_chosen)))

        if len(chosen) >= budget:
            break

    meta = {
        "triangulation_method": "polygon_centroid_fan",
        "triangle_count": int(n_triangles),
        "polygon_centroid": centroid.tolist(),
        "polygon_area": polygon_area(hull_vertices),
        "triangle_area_min": float(np.min(triangle_areas)) if len(triangle_areas) else 0.0,
        "triangle_area_max": float(np.max(triangle_areas)) if len(triangle_areas) else 0.0,
        "triangle_area_sum": float(np.sum(triangle_areas)),
        "triangle_quotas": quotas.astype(int).tolist(),
        "triangle_selected_counts": per_triangle_selected,
        "triangle_candidate_counts": per_triangle_candidates,
        "triangle_bins": per_triangle_bins,
        "stratum_scale": float(stratum_scale),
        "grid_bins_override": None if grid_bins is None else int(grid_bins),
    }
    return chosen, meta

def randomized_farthest_fill(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    selected_indices: list[int],
    selected_mask: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    top_k: int,
) -> list[int]:
    """
    Fill remaining slots by repeatedly choosing from the top-k farthest points
    from the current selected set. Distances are computed in normalized input
    coordinates so x/y scale differences do not dominate.
    """
    if budget <= 0:
        return []

    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-12)
    norm_points = normalize_points(points, mins, spans)
    norm_candidates = norm_points[candidate_indices]
    active = ~selected_mask[candidate_indices]

    if not np.any(active):
        return []

    selected_norm = norm_points[np.asarray(selected_indices, dtype=int)]
    min_d2 = np.full(len(candidate_indices), np.inf, dtype=float)
    for p in selected_norm:
        diff = norm_candidates - p
        min_d2 = np.minimum(min_d2, np.einsum("ij,ij->i", diff, diff))
    min_d2[~active] = -np.inf

    chosen: list[int] = []
    for _ in range(budget):
        active_positions = np.flatnonzero(active)
        if len(active_positions) == 0:
            break

        k = min(top_k, len(active_positions))
        top_positions = active_positions[np.argpartition(min_d2[active_positions], -k)[-k:]]
        picked_pos = int(rng.choice(top_positions))
        picked_idx = int(candidate_indices[picked_pos])

        selected_mask[picked_idx] = True
        active[picked_pos] = False
        min_d2[picked_pos] = -np.inf
        chosen.append(picked_idx)

        picked_norm = norm_candidates[picked_pos]
        diff = norm_candidates - picked_norm
        new_d2 = np.einsum("ij,ij->i", diff, diff)
        min_d2 = np.minimum(min_d2, new_d2)
        min_d2[~active] = -np.inf

    return chosen


def grid_cell_size_from_cover_h(grid_kind: str, cover_h: float) -> float:
    if cover_h <= 0.0:
        raise ValueError("cover_h must be positive.")
    if grid_kind == "square":
        return cover_h / math.sqrt(2.0)
    if grid_kind == "triangle":
        return cover_h
    raise ValueError(f"Unknown grid_kind: {grid_kind!r}.")


def grid_cell_diameter_upper_bound(grid_kind: str, cell_size: float) -> float:
    if grid_kind == "square":
        return math.sqrt(2.0) * cell_size
    if grid_kind == "triangle":
        return cell_size
    raise ValueError(f"Unknown grid_kind: {grid_kind!r}.")


def square_grid_cells(
    points: np.ndarray,
    cell_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.floor((points - origin) / cell_size).astype(np.int64)
    centers = origin + (coords.astype(float) + 0.5) * cell_size
    return coords, centers


def triangle_grid_cells(
    points: np.ndarray,
    cell_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points to an equilateral triangular tiling.

    The lattice basis is e1=(s,0), e2=(s/2,sqrt(3)s/2). Each fundamental
    parallelogram is split into two equilateral triangles. The triangle side
    length is s, so the cell diameter is also s.
    """
    shifted = points - origin
    v = shifted[:, 1] / ((math.sqrt(3.0) / 2.0) * cell_size)
    u = shifted[:, 0] / cell_size - 0.5 * v
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    fu = u - i
    fv = v - j
    upper = (fu + fv) > 1.0
    tri = upper.astype(np.int64)
    cell_ids = np.column_stack([i, j, tri])

    center_u = i.astype(float) + np.where(upper, 2.0 / 3.0, 1.0 / 3.0)
    center_v = j.astype(float) + np.where(upper, 2.0 / 3.0, 1.0 / 3.0)
    centers = np.column_stack(
        [
            origin[0] + cell_size * (center_u + 0.5 * center_v),
            origin[1] + (math.sqrt(3.0) / 2.0) * cell_size * center_v,
        ]
    )
    return cell_ids, centers


def grid_cells(
    points: np.ndarray,
    grid_kind: str,
    cell_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_size <= 0.0:
        raise ValueError("cell_size must be positive.")
    if grid_kind == "square":
        return square_grid_cells(points, cell_size, origin)
    if grid_kind == "triangle":
        return triangle_grid_cells(points, cell_size, origin)
    raise ValueError(f"Unknown grid_kind: {grid_kind!r}.")


def grid_cover_for_cell_size(
    points: np.ndarray,
    hull_indices: np.ndarray,
    grid_kind: str,
    cell_size: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, dict]:
    support_mask = np.zeros(len(points), dtype=bool)
    support_mask[hull_indices] = True
    cell_ids, centers = grid_cells(points, grid_kind, cell_size, origin)

    support_cells: set[tuple[int, ...]] = set()
    best_by_cell: dict[tuple[int, ...], tuple[int, float]] = {}

    for idx, (cell_id, center) in enumerate(zip(cell_ids, centers)):
        key = tuple(int(v) for v in cell_id)
        if support_mask[idx]:
            support_cells.add(key)
            continue

        diff = points[idx] - center
        dist2 = float(np.dot(diff, diff))
        previous = best_by_cell.get(key)
        if previous is None or dist2 < previous[1]:
            best_by_cell[key] = (int(idx), dist2)

    representatives = [idx for key, (idx, _) in best_by_cell.items() if key not in support_cells]
    representatives.sort()
    selected = np.asarray([int(idx) for idx in hull_indices] + representatives, dtype=int)

    meta = {
        "grid_kind": grid_kind,
        "grid_cell_size": float(cell_size),
        "grid_cell_diameter_upper_bound": float(
            grid_cell_diameter_upper_bound(grid_kind, cell_size)
        ),
        "occupied_cell_count": int(len({tuple(int(v) for v in row) for row in cell_ids})),
        "support_occupied_cell_count": int(len(support_cells)),
        "representative_cell_count": int(len(representatives)),
    }
    return selected, meta


def compute_cover_radius(
    points: np.ndarray,
    selected_indices: np.ndarray,
    chunk_size: int = 8192,
) -> float:
    if len(points) == 0:
        return 0.0
    if len(selected_indices) == 0:
        return float("inf")

    selected_points = points[np.asarray(selected_indices, dtype=int)]
    max_min_d2 = 0.0
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        diff = chunk[:, None, :] - selected_points[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        chunk_min = np.min(d2, axis=1)
        max_min_d2 = max(max_min_d2, float(np.max(chunk_min)))
    return float(math.sqrt(max_min_d2))


def budgeted_grid_cover(
    points: np.ndarray,
    hull_indices: np.ndarray,
    grid_kind: str,
    max_keep: int,
    origin: np.ndarray,
    iterations: int = 50,
) -> tuple[np.ndarray, dict]:
    spans = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-12)
    low = 0.0
    high = float(max(np.linalg.norm(spans), np.max(spans), 1.0))

    selected, meta = grid_cover_for_cell_size(points, hull_indices, grid_kind, high, origin)
    while len(selected) > max_keep:
        low = high
        high *= 2.0
        selected, meta = grid_cover_for_cell_size(points, hull_indices, grid_kind, high, origin)

    best_selected = selected
    best_meta = meta
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        trial_selected, trial_meta = grid_cover_for_cell_size(
            points, hull_indices, grid_kind, mid, origin
        )
        if len(trial_selected) <= max_keep:
            high = mid
            best_selected = trial_selected
            best_meta = trial_meta
        else:
            low = mid

    best_meta = dict(best_meta)
    best_meta["budget_search_iterations"] = int(iterations)
    best_meta["budget_search_low_cell_size"] = float(low)
    best_meta["budget_search_high_cell_size"] = float(high)
    return best_selected, best_meta


def select_grid_cover_points(
    points: np.ndarray,
    max_keep: int | None,
    seed: int,
    cover_h: float | None,
    grid_kind: str,
) -> tuple[np.ndarray, dict]:
    if grid_kind not in {"square", "triangle"}:
        raise ValueError("grid_kind must be either 'square' or 'triangle'.")
    if max_keep is None and cover_h is None:
        raise ValueError("grid_cover requires cover_h, max_keep, or both.")
    if max_keep is not None and max_keep <= 0:
        raise ValueError("max_keep must be positive when it is used as a budget.")
    if cover_h is not None and cover_h <= 0.0:
        raise ValueError("cover_h must be positive.")

    hull_indices = convex_hull_indices(points)
    support_count = len(hull_indices)
    origin = points.min(axis=0) - 1e-12
    target_meta: dict = {}
    required_count_for_target_h: int | None = None
    selection_mode = "budget"
    selected_policy = "support_plus_budgeted_grid_cover"

    if max_keep is not None and support_count >= max_keep:
        selected_array = np.asarray([int(idx) for idx in hull_indices], dtype=int)
        achieved_h = compute_cover_radius(points, selected_array)
        meta = {
            "selection_method": "support_points_plus_grid_cover",
            "selected_policy": "support_only_due_to_budget",
            "budget_policy": "keep_all_support_points_even_if_over_budget",
            "selection_mode": "support_only_due_to_budget",
            "seed": int(seed),
            "max_keep": int(max_keep),
            "target_h": None if cover_h is None else float(cover_h),
            "achieved_h": float(achieved_h),
            "is_h_cover": None if cover_h is None else bool(achieved_h <= cover_h * (1.0 + 1e-12)),
            "selected_count": int(len(selected_array)),
            "support_count": int(support_count),
            "support_count_exceeds_budget": bool(support_count > max_keep),
            "extra_sampled_count": 0,
            "stratified_count": 0,
            "farthest_count": 0,
            "grid_kind": grid_kind,
            "grid_cover_meta": {},
            "num_input_points": int(len(points)),
        }
        return selected_array, meta

    if cover_h is not None:
        target_cell_size = grid_cell_size_from_cover_h(grid_kind, cover_h)
        target_selected, target_meta = grid_cover_for_cell_size(
            points, hull_indices, grid_kind, target_cell_size, origin
        )
        required_count_for_target_h = int(len(target_selected))
        if max_keep is None or len(target_selected) <= max_keep:
            selected_array = target_selected
            grid_meta = target_meta
            selection_mode = "target_h" if max_keep is None else "target_h_within_budget"
            selected_policy = "support_plus_grid_h_cover"
        else:
            selected_array, grid_meta = budgeted_grid_cover(
                points, hull_indices, grid_kind, max_keep, origin
            )
            selection_mode = "target_h_over_budget_budget_fallback"
            selected_policy = "support_plus_budgeted_grid_cover"
    else:
        selected_array, grid_meta = budgeted_grid_cover(
            points, hull_indices, grid_kind, max_keep, origin
        )

    achieved_h = compute_cover_radius(points, selected_array)
    is_h_cover = None if cover_h is None else bool(achieved_h <= cover_h * (1.0 + 1e-12))
    support_set = set(int(idx) for idx in hull_indices)
    selected_support_count = sum(1 for idx in selected_array if int(idx) in support_set)

    grid_meta = dict(grid_meta)
    grid_meta.update(
        {
            "target_grid_meta": target_meta,
            "required_count_for_target_h": required_count_for_target_h,
            "representative_rule": "support_if_present_else_nearest_to_cell_center",
            "origin": origin.astype(float).tolist(),
        }
    )

    meta = {
        "selection_method": "support_points_plus_grid_cover",
        "selected_policy": selected_policy,
        "budget_policy": (
            "no_budget_target_h_only" if max_keep is None else "sample_extra_points_up_to_max_keep"
        ),
        "selection_mode": selection_mode,
        "seed": int(seed),
        "max_keep": None if max_keep is None else int(max_keep),
        "target_h": None if cover_h is None else float(cover_h),
        "achieved_h": float(achieved_h),
        "is_h_cover": is_h_cover,
        "selected_count": int(len(selected_array)),
        "support_count": int(support_count),
        "selected_support_count": int(selected_support_count),
        "support_count_exceeds_budget": False,
        "extra_sampled_count": int(len(selected_array) - support_count),
        "stratified_count": 0,
        "farthest_count": 0,
        "grid_kind": grid_kind,
        "grid_cell_size": float(grid_meta["grid_cell_size"]),
        "grid_cell_diameter_upper_bound": float(grid_meta["grid_cell_diameter_upper_bound"]),
        "grid_cover_meta": grid_meta,
        "num_input_points": int(len(points)),
    }
    return selected_array, meta


def select_points(
    points: np.ndarray,
    max_keep: int,
    seed: int,
    grid_bins: int | None,
    top_k: int,
    stratum_scale: float,
    selection_method: str = "centroid_triangulated",
    cover_h: float | None = None,
    grid_kind: str = "square",
) -> tuple[np.ndarray, dict]:
    if selection_method == "grid_cover":
        max_keep_limit = None if max_keep is None or max_keep <= 0 else int(max_keep)
        return select_grid_cover_points(
            points=points,
            max_keep=max_keep_limit,
            seed=seed,
            cover_h=cover_h,
            grid_kind=grid_kind,
        )
    if selection_method not in {
        "centroid_triangulated",
        "centroid",
        "space_filling",
    }:
        raise ValueError(
            "selection_method must be 'centroid_triangulated' or 'grid_cover'."
        )

    if max_keep <= 0:
        raise ValueError("max_keep must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if stratum_scale <= 0.0:
        raise ValueError("stratum_scale must be positive.")

    rng = np.random.default_rng(seed)
    hull_indices = convex_hull_indices(points)
    support_count = len(hull_indices)

    if support_count >= max_keep:
        selected_array = np.asarray([int(idx) for idx in hull_indices], dtype=int)
        meta = {
            "selection_method": "support_only_due_to_budget",
            "selected_policy": "support_only_due_to_budget",
            "budget_policy": "keep_all_support_points_even_if_over_budget",
            "seed": seed,
            "max_keep": max_keep,
            "selected_count": int(len(selected_array)),
            "support_count": int(support_count),
            "support_count_exceeds_budget": bool(support_count > max_keep),
            "extra_sampled_count": 0,
            "stratified_count": 0,
            "farthest_count": 0,
            "grid_bins": None if grid_bins is None else int(grid_bins),
            "stratum_scale": float(stratum_scale),
            "top_k": int(top_k),
            "num_input_points": int(len(points)),
            "stratified_meta": {},
        }
        return selected_array, meta

    selected_mask = np.zeros(len(points), dtype=bool)
    selected_mask[hull_indices] = True
    selected_indices = [int(idx) for idx in hull_indices]
    candidate_indices = np.flatnonzero(~selected_mask)

    remaining = max_keep - len(selected_indices)
    stratified, stratified_meta = centroid_triangulated_stratified_candidates(
        points=points,
        hull_indices=hull_indices,
        candidate_indices=candidate_indices,
        selected_mask=selected_mask,
        budget=remaining,
        rng=rng,
        grid_bins=grid_bins,
        stratum_scale=stratum_scale,
    )
    selected_indices.extend(stratified)

    remaining = max_keep - len(selected_indices)
    farthest = randomized_farthest_fill(
        points=points,
        candidate_indices=candidate_indices,
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        budget=remaining,
        rng=rng,
        top_k=top_k,
    )
    selected_indices.extend(farthest)

    selected_array = np.asarray(selected_indices, dtype=int)
    meta = {
        "selection_method": "support_points_plus_centroid_triangulated_stratified_randomized_maximin",
        "selected_policy": "support_plus_sampling",
        "budget_policy": "sample_extra_points_up_to_max_keep",
        "seed": seed,
        "max_keep": max_keep,
        "selected_count": int(len(selected_array)),
        "support_count": int(support_count),
        "support_count_exceeds_budget": False,
        "extra_sampled_count": int(len(selected_array) - support_count),
        "stratified_count": int(len(stratified)),
        "farthest_count": int(len(farthest)),
        "grid_bins": None if grid_bins is None else int(grid_bins),
        "stratum_scale": float(stratum_scale),
        "top_k": int(top_k),
        "num_input_points": int(len(points)),
        "stratified_meta": stratified_meta,
    }
    return selected_array, meta

def main() -> None:
    parser = argparse.ArgumentParser(description="Select a small space-filling point subset.")
    parser.add_argument(
        "--points-file",
        type=str,
        default="triangle_lower_bound_outputs/triangle_points.npy",
    )
    parser.add_argument("--out-dir", type=str, default="point_subset_area_outputs")
    parser.add_argument("--max-keep", type=int, default=800)
    parser.add_argument("--seed", type=int, default=283)
    parser.add_argument(
        "--grid-bins",
        type=int,
        default=None,
        help="Optional fixed barycentric stratum bins per centroid triangle.",
    )
    parser.add_argument("--stratum-scale", type=float, default=1.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--selection-method",
        choices=["centroid_triangulated", "grid_cover"],
        default="centroid_triangulated",
        help="Point selection policy. The default preserves the original behavior.",
    )
    parser.add_argument(
        "--cover-h",
        type=float,
        default=None,
        help="Target discrete h-cover radius for --selection-method grid_cover.",
    )
    parser.add_argument(
        "--grid-kind",
        choices=["square", "triangle"],
        default="square",
        help="Grid type for --selection-method grid_cover.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    points_path = root / args.points_file
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(points_path)
    selected_indices, meta = select_points(
        points=points,
        max_keep=args.max_keep,
        seed=args.seed,
        grid_bins=args.grid_bins,
        top_k=args.top_k,
        stratum_scale=args.stratum_scale,
        selection_method=args.selection_method,
        cover_h=args.cover_h,
        grid_kind=args.grid_kind,
    )

    np.save(out_dir / "selected_point_indices.npy", selected_indices)
    np.save(out_dir / "selected_points.npy", points[selected_indices])
    (out_dir / "selected_points_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Saved selected indices: {(out_dir / 'selected_point_indices.npy').resolve()}")
    print(f"Saved selected points:  {(out_dir / 'selected_points.npy').resolve()}")
    print(f"Saved selection meta:   {(out_dir / 'selected_points_meta.json').resolve()}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
