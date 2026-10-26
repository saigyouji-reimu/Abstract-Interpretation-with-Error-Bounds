"""
select_concrete_points.py

Select concrete representative points directly from a finite 2D point cloud by
constructing a discrete h-cover, following the representative coverage guidance:

  1. Preserve the current abstraction:
       Conv(R) = Conv(I)
     by retaining all convex-hull support points.

  2. Cover the concrete input set:
       for every x in I, there exists r in R with ||x - r||_2 <= h.

The algorithm is a greedy farthest-point h-cover over the discrete point set I.
It does not modify or run the existing experiments; it only writes selected
indices, selected points, and metadata.

Typical usage:

  python select_concrete_points.py \
    --points-file triangle_lower_bound_outputs/triangle_points.npy \
    --out-dir concrete_h_cover_outputs \
    --h 0.25

Budgeted usage:

  python select_concrete_points.py \
    --points-file triangle_lower_bound_outputs/triangle_points.npy \
    --out-dir concrete_h_cover_outputs \
    --max-keep 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from core.geometry import convex_hull_indices, polygon_area


def pairwise_min_dist2_to_selected(points: np.ndarray, selected_points: np.ndarray) -> np.ndarray:
    """
    Initial nearest-selected squared distances for all points.

    This is O(|I| |R0|), but R0 is the hull support set and is small in the
    intended experiments.
    """
    min_dist2 = np.full(len(points), np.inf, dtype=float)
    for selected in selected_points:
        diff = points - selected
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
    return min_dist2


def update_min_dist2(points: np.ndarray, min_dist2: np.ndarray, new_point: np.ndarray) -> None:
    diff = points - new_point
    dist2 = np.einsum("ij,ij->i", diff, diff)
    np.minimum(min_dist2, dist2, out=min_dist2)


def greedy_h_cover_select(
    points: np.ndarray,
    h: float | None,
    max_keep: int | None,
) -> tuple[np.ndarray, dict]:
    """
    Greedy discrete h-cover selection.

    Start with hull support points. Repeatedly add the currently farthest
    uncovered concrete point until either:

      - the achieved cover radius is <= h, if h is provided;
      - max_keep is reached, if max_keep is provided;
      - all points are selected.

    If both h and max_keep are provided, h is the target and max_keep is a
    stopping budget. The metadata records whether the requested h-cover was
    actually achieved.
    """
    if h is not None and h < 0.0:
        raise ValueError("h must be non-negative.")
    if max_keep is not None and max_keep <= 0:
        raise ValueError("max_keep must be positive when provided.")

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("points must be nonempty.")

    hull_indices = convex_hull_indices(pts)
    selected_mask = np.zeros(len(pts), dtype=bool)
    selected_mask[hull_indices] = True
    selected_indices = [int(idx) for idx in hull_indices]

    min_dist2 = pairwise_min_dist2_to_selected(pts, pts[hull_indices])
    min_dist2[selected_mask] = 0.0

    target_h2 = None if h is None else float(h * h)
    termination_reason = ""

    while True:
        farthest_idx = int(np.argmax(min_dist2))
        achieved_h = float(np.sqrt(max(0.0, min_dist2[farthest_idx])))

        if target_h2 is not None and min_dist2[farthest_idx] <= target_h2 + 1e-15:
            termination_reason = "target_h_achieved"
            break

        if max_keep is not None and len(selected_indices) >= max_keep:
            termination_reason = "max_keep_reached"
            break

        if selected_mask[farthest_idx]:
            termination_reason = "all_points_selected"
            break

        selected_mask[farthest_idx] = True
        selected_indices.append(farthest_idx)
        update_min_dist2(pts, min_dist2, pts[farthest_idx])
        min_dist2[selected_mask] = 0.0

        if len(selected_indices) == len(pts):
            termination_reason = "all_points_selected"
            break

    selected_array = np.asarray(selected_indices, dtype=int)
    farthest_idx = int(np.argmax(min_dist2))
    achieved_h = float(np.sqrt(max(0.0, min_dist2[farthest_idx])))
    is_h_cover = bool(h is not None and achieved_h <= float(h) + 1e-12)

    support_count = int(len(hull_indices))
    if h is not None and is_h_cover:
        selected_policy = "support_plus_greedy_h_cover"
    elif h is not None:
        selected_policy = "support_plus_budgeted_greedy_cover"
    else:
        selected_policy = "support_plus_budgeted_greedy_cover"

    meta = {
        "selection_method": "support_points_plus_greedy_discrete_h_cover",
        "selected_policy": selected_policy,
        "target_h": None if h is None else float(h),
        "achieved_h": achieved_h,
        "is_h_cover": is_h_cover,
        "termination_reason": termination_reason,
        "max_keep": None if max_keep is None else int(max_keep),
        "selected_count": int(len(selected_array)),
        "support_count": support_count,
        "extra_sampled_count": int(len(selected_array) - support_count),
        "num_input_points": int(len(pts)),
        "input_hull_area": polygon_area(pts[hull_indices]),
        "cover_radius_point_index": farthest_idx,
        "conv_preserved_by_support_points": True,
        "distance_norm": "euclidean_l2",
    }
    return selected_array, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Select concrete h-cover representatives.")
    parser.add_argument(
        "--points-file",
        type=str,
        default="triangle_lower_bound_outputs/triangle_points.npy",
    )
    parser.add_argument("--out-dir", type=str, default="concrete_h_cover_outputs")
    parser.add_argument("--h", type=float, default=None, help="Target h-cover radius.")
    parser.add_argument(
        "--max-keep",
        type=int,
        default=800,
        help=(
            "Optional selection budget. Use --max-keep 0 to disable the budget "
            "and select until h is achieved."
        ),
    )
    args = parser.parse_args()

    if args.max_keep == 0:
        max_keep: int | None = None
    else:
        max_keep = args.max_keep
    if args.h is None and max_keep is None:
        raise ValueError("At least one of --h or --max-keep must be provided.")

    root = Path(__file__).resolve().parent
    points_path = root / args.points_file
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(points_path)
    selected_indices, meta = greedy_h_cover_select(points, h=args.h, max_keep=max_keep)

    np.save(out_dir / "selected_point_indices.npy", selected_indices)
    np.save(out_dir / "selected_points.npy", points[selected_indices])
    (out_dir / "selected_points_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=True), encoding="utf-8"
    )

    print(f"Saved selected indices: {(out_dir / 'selected_point_indices.npy').resolve()}")
    print(f"Saved selected points:  {(out_dir / 'selected_points.npy').resolve()}")
    print(f"Saved selection meta:   {(out_dir / 'selected_points_meta.json').resolve()}")
    print(json.dumps(meta, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
