"""Small semantic and end-to-end checks for the branching experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.branching import (
    BRANCH_EXECUTION_ORDER,
    BRANCH_PROGRAM_SCHEMA_VERSION,
    BRANCH_PROGRAM_TYPE,
    execute_branch_program_points,
    execute_branch_program_polygon,
    split_polygon_by_guard,
)
from core.geometry import box_polygon, convex_hull, polygon_area
from core.guards import quadrant_guard_spec
from scripts.evaluate_branch_point_subset_area import compute_branch_one


def main() -> None:
    guard = quadrant_guard_spec()
    program = {
        "program_id": 0,
        "program_type": BRANCH_PROGRAM_TYPE,
        "schema_version": BRANCH_PROGRAM_SCHEMA_VERSION,
        "execution_order": BRANCH_EXECUTION_ORDER,
        "num_affine_nodes": 3,
        "max_affine_depth": 2,
        "affine1": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        "guard1": guard,
        "affine2": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        "guard2": guard,
        "affine3": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.25, -0.25]},
    }
    points = np.array(
        [
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    output, trace = execute_branch_program_points(points, program, return_trace=True)
    assert trace["then_input_count"] == 2
    assert trace["else_input_count"] == 3
    assert trace["merged_output_count"] == len(points)

    square = box_polygon(points)
    then_polygon, else_pieces = split_polygon_by_guard(square, guard)
    split_area = polygon_area(then_polygon) + sum(
        polygon_area(piece) for piece in else_pieces
    )
    assert abs(split_area - polygon_area(square)) <= 1e-9

    abstract = execute_branch_program_polygon(square, program)
    concrete_hull = convex_hull(output)
    assert polygon_area(abstract) + 1e-9 >= polygon_area(concrete_hull)

    selected = points[[0, 1, 2, 3]]
    row = compute_branch_one(program, points, selected, selected)
    assert row["selected_area"] <= row["true_area"] + 1e-9
    assert row["box_abstract_area"] + 1e-9 >= row["true_area"]
    assert 0.0 <= row["area_retention_ratio"] <= 1.0 + 1e-9
    print("smoke_branching: ok")


if __name__ == "__main__":
    main()
