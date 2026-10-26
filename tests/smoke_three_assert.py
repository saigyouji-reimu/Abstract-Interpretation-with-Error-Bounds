"""Smoke checks for the current three-assertion paper experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.branch_feedback import generate_program_constraint_pair
from core.geometry import convex_hull, polygon_area
from core.three_assert import (
    ASSERT_SITES,
    execute_three_assert_abstract,
    execute_three_assert_points,
    read_three_assert_programs,
    refine_polygon,
    three_assert_program_collection_fingerprint,
)
from core.three_assert_feedback import evaluate_three_assert_feedback_trace


EXPECTED_FINGERPRINT = (
    "ae198d0e35ea37f500a74b131ef501d581313d404f4bb7622f1bcb6c520d3237"
)


def main() -> None:
    programs = read_three_assert_programs(
        ROOT / "data/programs/three_assert_affine_programs.json"
    )
    assert len(programs) == 100
    assert three_assert_program_collection_fingerprint(programs) == EXPECTED_FINGERPRINT

    rng = np.random.default_rng(283)
    points = rng.uniform(-4.0, 4.0, size=(500, 2))
    input_hull = convex_hull(points)
    concrete = execute_three_assert_points(points, programs[0])
    abstract = execute_three_assert_abstract(input_hull, programs[0])
    for site in ASSERT_SITES:
        concrete_area = polygon_area(convex_hull(concrete[site]))
        abstract_area = polygon_area(abstract[site])
        assert abstract_area + 1e-8 >= concrete_area

    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    hexagon = np.column_stack((np.cos(angles), np.sin(angles)))
    boxed = refine_polygon(hexagon, edge_threshold=5)
    assert len(boxed) == 4
    assert polygon_area(boxed) + 1e-10 >= polygon_area(hexagon)

    assert1, assert2, _ = generate_program_constraint_pair(
        program_id=0,
        constraints_per_branch=12,
        seed=283,
        projection_radius=12.0,
        central_radius=4.0,
        central_fraction=0.5,
    )
    trace, final = evaluate_three_assert_feedback_trace(
        points=points,
        program=programs[0],
        assert1_constraints=assert1,
        assert2_constraints=assert2,
    )
    assert len(trace) == 12
    assert final["old_guard1_lower_bound"] <= final["guard1_true_loss"] + 1e-8
    assert (
        final["combined_corrected_lower_bound"]
        <= final["guard1_true_loss"] + 1e-8
    )
    assert (
        final["assert3_combined_corrected_lower_bound"]
        <= final["assert3_true_loss"] + 1e-8
    )
    print("smoke_three_assert: ok")


if __name__ == "__main__":
    main()
