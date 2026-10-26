"""Evaluate output hull area retained by a selected point subset.

This is a point-cloud experiment, not abstract interpretation propagation. The
program schema is shared with the lower-bound experiments and supports arbitrary
numbers of layers, with a JSON guard executed after every affine layer.
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

from core.geometry import box_polygon, convex_hull_indices, polygon_area
from core.lower_bound import hull_area_after_program
from core.programs import composed_affine_from_program, execute_program_polygon


EPS = 1e-12


def loss_to_abstraction_ratio(loss: float, abstraction_loss: float) -> float:
    if abstraction_loss > EPS:
        return float(loss / abstraction_loss)
    if abs(loss) <= EPS:
        return 0.0
    return float("nan")


def compute_one(
    program: dict,
    points: np.ndarray,
    selected_points: np.ndarray,
    support_only_points: np.ndarray,
) -> dict:
    _, _, abs_det_A, num_affine_layers = composed_affine_from_program(program)

    true_area, true_survivors, true_hull_vertices = hull_area_after_program(
        points, program
    )
    selected_area, selected_survivors, selected_hull_vertices = (
        hull_area_after_program(selected_points, program)
    )
    support_only_area, support_only_survivors, support_only_hull_vertices = (
        hull_area_after_program(support_only_points, program)
    )

    box_abstract_polygon = execute_program_polygon(box_polygon(points), program)
    box_abstract_area = polygon_area(box_abstract_polygon)

    area_gap = max(0.0, float(true_area - selected_area))
    abs_area_gap = float(abs(area_gap))
    support_only_area_gap = max(0.0, float(true_area - support_only_area))
    support_only_abs_area_gap = float(abs(support_only_area_gap))
    box_abstraction_loss_area = max(0.0, float(box_abstract_area - true_area))
    sampling_loss_to_box_loss_ratio = loss_to_abstraction_ratio(
        area_gap, box_abstraction_loss_area
    )
    support_only_loss_to_box_loss_ratio = loss_to_abstraction_ratio(
        support_only_area_gap, box_abstraction_loss_area
    )
    true_area_is_zero = bool(true_area <= EPS)
    if true_area_is_zero and selected_area <= EPS:
        area_retention_ratio = 1.0
        support_only_area_retention_ratio = 1.0
        relative_area_gap = 0.0
        support_only_relative_area_gap = 0.0
    elif true_area_is_zero:
        area_retention_ratio = float("nan")
        support_only_area_retention_ratio = float("nan")
        relative_area_gap = float("nan")
        support_only_relative_area_gap = float("nan")
    else:
        area_retention_ratio = float(selected_area / true_area)
        support_only_area_retention_ratio = float(support_only_area / true_area)
        relative_area_gap = float(area_gap / true_area)
        support_only_relative_area_gap = float(support_only_area_gap / true_area)

    return {
        "program_id": int(program["program_id"]),
        "abs_det_A": float(abs_det_A),
        "num_affine_layers": int(num_affine_layers),
        "execution_order": program.get("execution_order"),
        "true_area": float(true_area),
        "selected_area": float(selected_area),
        "area_retention_ratio": area_retention_ratio,
        "area_gap": area_gap,
        "support_only_area": float(support_only_area),
        "support_only_area_retention_ratio": support_only_area_retention_ratio,
        "support_only_area_gap": support_only_area_gap,
        "box_abstract_area": float(box_abstract_area),
        "box_abstraction_loss_area": box_abstraction_loss_area,
        "sampling_loss_to_box_loss_ratio": sampling_loss_to_box_loss_ratio,
        "support_only_loss_to_box_loss_ratio": support_only_loss_to_box_loss_ratio,
        "abs_area_gap": abs_area_gap,
        "support_only_abs_area_gap": support_only_abs_area_gap,
        "relative_area_gap": relative_area_gap,
        "support_only_relative_area_gap": support_only_relative_area_gap,
        "true_area_is_zero": true_area_is_zero,
        "selected_area_leq_true_area": bool(selected_area <= true_area + 1e-9),
        "num_true_survivors": int(true_survivors),
        "num_selected_survivors": int(selected_survivors),
        "num_support_only_survivors": int(support_only_survivors),
        "true_survival_ratio": float(true_survivors / len(points)),
        "selected_survival_ratio": float(selected_survivors / len(selected_points)),
        "support_only_survival_ratio": float(
            support_only_survivors / len(support_only_points)
        ),
        "true_output_hull_vertices": int(true_hull_vertices),
        "selected_output_hull_vertices": int(selected_hull_vertices),
        "support_only_output_hull_vertices": int(support_only_hull_vertices),
    }


def write_stats_tsv(rows: list[dict], path: Path) -> None:
    columns = [
        "program_id",
        "abs_det_A",
        "num_affine_layers",
        "execution_order",
        "true_area",
        "selected_area",
        "area_retention_ratio",
        "area_gap",
        "support_only_area",
        "support_only_area_retention_ratio",
        "support_only_area_gap",
        "box_abstract_area",
        "box_abstraction_loss_area",
        "sampling_loss_to_box_loss_ratio",
        "support_only_loss_to_box_loss_ratio",
        "relative_area_gap",
        "support_only_relative_area_gap",
        "num_true_survivors",
        "num_selected_survivors",
        "num_support_only_survivors",
        "true_survival_ratio",
        "selected_survival_ratio",
        "support_only_survival_ratio",
        "true_output_hull_vertices",
        "selected_output_hull_vertices",
        "support_only_output_hull_vertices",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in columns) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate point-subset output area retention.")
    parser.add_argument(
        "--points-file",
        type=str,
        default="triangle_lower_bound_outputs/triangle_points.npy",
    )
    parser.add_argument(
        "--programs-file",
        type=str,
        default="triangle_lower_bound_outputs/affine_programs.json",
    )
    parser.add_argument(
        "--selected-indices-file",
        type=str,
        default="point_subset_area_outputs/selected_point_indices.npy",
    )
    parser.add_argument("--out-dir", type=str, default="point_subset_area_outputs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    points = np.load(root / args.points_file)
    selected_indices = np.load(root / args.selected_indices_file)
    selected_points = points[selected_indices]
    support_only_indices = convex_hull_indices(points)
    support_only_points = points[support_only_indices]
    programs = json.loads((root / args.programs_file).read_text(encoding="utf-8"))

    rows = [
        compute_one(program, points, selected_points, support_only_points)
        for program in programs
    ]

    results_path = out_dir / "point_subset_area_results.json"
    stats_path = out_dir / "point_subset_area_stats.tsv"
    results_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    write_stats_tsv(rows, stats_path)

    print(f"Loaded points: {points.shape}")
    print(f"Loaded selected points: {selected_points.shape}")
    print(f"Loaded support-only points: {support_only_points.shape}")
    print(f"Loaded programs: {len(programs)}")
    print(f"Saved per-program results: {results_path.resolve()}")
    print(f"Saved stats TSV:           {stats_path.resolve()}")


if __name__ == "__main__":
    main()
