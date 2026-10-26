"""Lightweight smoke tests for reusable experiment components.

Run from this directory or the repository root:

    python tests/smoke_core.py

The test intentionally uses tiny synthetic data and does not touch large
experiment outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.geometry import box_polygon, convex_hull, polygon_area
from core.guards import apply_guard_to_polygon, guarded_points, quadrant_guard_spec
from core.branching import (
    BRANCH_EXECUTION_ORDER,
    BRANCH_PROGRAM_SCHEMA_VERSION,
    BRANCH_PROGRAM_TYPE,
    execute_branch_program_points,
    execute_branch_program_polygon,
)
from core.lower_bound import compute_lower_bound_components
from core.posterior import (
    evaluate_constraint_trace,
    evaluate_intermediate_pullback_trace,
    generate_constraints,
)
from core.programs import (
    composed_affine_from_program,
    execute_program_points,
    generate_programs,
    program_layers,
)
from scripts.select_concrete_points import greedy_h_cover_select
from scripts.select_space_filling_points import select_points


def make_points() -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, 7)
    ys = np.linspace(-1.0, 1.0, 7)
    return np.asarray([(x, y) for x in xs for y in ys if abs(x) + abs(y) <= 1.25], dtype=float)


def main() -> None:
    points = make_points()
    assert len(points) > 10

    hull = convex_hull(points)
    assert polygon_area(hull) > 0.0
    assert polygon_area(box_polygon(points)) >= polygon_area(hull)

    clipped = apply_guard_to_polygon(hull, quadrant_guard_spec())
    assert polygon_area(clipped) >= 0.0
    if len(clipped):
        assert np.all(clipped[:, 0] >= -1e-10)
        assert np.all(clipped[:, 1] >= -1e-10)

    programs = generate_programs(
        m=2,
        seed=283,
        a_low=-1.5,
        a_high=1.5,
        b_low=-0.5,
        b_high=0.5,
        min_abs_det=0.05,
        num_affine_layers=3,
    )
    A, b, det_product, k = composed_affine_from_program(programs[0])
    assert A.shape == (2, 2)
    assert b.shape == (2,)
    assert det_product > 0.0
    assert k == 3
    assert all("guard" in layer for layer in programs[0]["affine_layers"])

    legacy_program = {
        "affine_layers": [
            {key: value for key, value in layer.items() if key != "guard"}
            for layer in programs[0]["affine_layers"]
        ],
        "guard": quadrant_guard_spec(),
    }
    try:
        program_layers(legacy_program)
    except ValueError:
        pass
    else:
        raise AssertionError("final-guard-only legacy programs must be rejected")

    guard = quadrant_guard_spec()
    guarded_program = {
        "program_id": 0,
        "program_type": "guarded_affine_sequence",
        "execution_order": "affine_then_guard_per_layer",
        "guard_after_each_affine": True,
        "num_affine_layers": 2,
        "guard": guard,
        "affine_layers": [
            {
                "layer_id": 0,
                "A": [[1.0, 0.0], [0.0, 1.0]],
                "b": [0.0, 0.0],
                "abs_det_A": 1.0,
                "guard": guard,
            },
            {
                "layer_id": 1,
                "A": [[-1.0, 0.0], [0.0, 1.0]],
                "b": [1.0, 0.0],
                "abs_det_A": 1.0,
                "guard": guard,
            },
        ],
    }
    probe = np.array([[-0.5, 0.5], [0.5, 0.5]], dtype=float)
    probe_output, probe_trace = execute_program_points(
        probe, guarded_program, return_trace=True
    )
    assert len(probe_output) == 1
    assert probe_trace[0]["after_guard_count"] == 1
    assert probe_trace[1]["after_guard_count"] == 1

    branch_program = {
        "program_id": 0,
        "program_type": BRANCH_PROGRAM_TYPE,
        "schema_version": BRANCH_PROGRAM_SCHEMA_VERSION,
        "execution_order": BRANCH_EXECUTION_ORDER,
        "num_affine_nodes": 3,
        "max_affine_depth": 2,
        "affine1": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0]},
        "guard1": guard,
        "affine2": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [0.5, 0.0]},
        "guard2": guard,
        "affine3": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [-0.5, 0.0]},
    }
    branch_output, branch_trace = execute_branch_program_points(
        points, branch_program, return_trace=True
    )
    assert branch_trace["then_input_count"] + branch_trace["else_input_count"] == len(points)
    assert (
        branch_trace["then_output_count"] + branch_trace["else_output_count"]
        == len(branch_output)
    )
    branch_abstract = execute_branch_program_polygon(
        box_polygon(points), branch_program
    )
    branch_true_hull = convex_hull(branch_output)
    assert polygon_area(branch_abstract) + 1e-9 >= polygon_area(branch_true_hull)

    stats = compute_lower_bound_components(points, guarded_program)
    for key in [
        "old_lower_bound",
        "true_loss_area",
        "box_abstract_polygon",
        "true_hull",
        "true_survivors",
        "layer_trace",
    ]:
        assert key in stats
    assert stats["num_affine_layers"] == 2
    assert len(stats["layer_trace"]) == 2
    assert stats["old_lower_bound"] >= -1e-10
    assert stats["true_loss_area"] >= -1e-10
    assert stats["old_lower_bound"] <= stats["true_loss_area"] + 1e-9

    constraints = generate_constraints(
        P=stats["box_abstract_polygon"],
        T=stats["true_hull"],
        count=8,
        seed=284,
        mode="boundary_gap",
        beta_a=2.0,
        beta_b=1.0,
        threshold_noise_scale=0.1,
    )
    trace, final_polygon = evaluate_constraint_trace(
        P=stats["box_abstract_polygon"],
        true_survivors=stats["true_survivors"],
        old_lower_bound=float(stats["old_lower_bound"]),
        true_loss_area=float(stats["true_loss_area"]),
        constraints=constraints,
    )
    assert len(trace) == 8
    assert polygon_area(final_polygon) <= polygon_area(stats["box_abstract_polygon"]) + 1e-10

    layers = program_layers(guarded_program)
    first_points = guarded_points(
        points @ layers[0]["A"].T + layers[0]["b"], layers[0]["guard"]
    )
    intermediate_trace, _ = evaluate_intermediate_pullback_trace(
        intermediate_polygon=stats["layer_trace"][0]["box_after_guard"],
        intermediate_true_points=first_points,
        old_intermediate_lower_bound=float(
            stats["layer_trace"][0]["lower_bound_after_guard"]
        ),
        second_layer=layers[1],
        final_trace=trace,
    )
    assert len(intermediate_trace) == len(trace)
    assert intermediate_trace[-1]["useful_count"] == trace[-1]["useful_count"]
    intermediate_true_loss = max(
        0.0,
        polygon_area(stats["layer_trace"][0]["box_after_guard"])
        - polygon_area(convex_hull(first_points)),
    )
    assert (
        intermediate_trace[-1]["corrected_lower_bound"]
        <= intermediate_true_loss + 1e-9
    )

    selected, meta = select_points(
        points=points,
        max_keep=12,
        seed=283,
        grid_bins=None,
        top_k=3,
        stratum_scale=1.25,
        selection_method="centroid_triangulated",
    )
    assert len(selected) <= max(12, int(meta["support_count"]))
    assert meta["selected_count"] == len(selected)

    grid_selected, grid_meta = select_points(
        points=points,
        max_keep=12,
        seed=283,
        grid_bins=None,
        top_k=3,
        stratum_scale=1.25,
        selection_method="grid_cover",
        grid_kind="square",
    )
    assert len(grid_selected) >= int(grid_meta["support_count"])

    cover_selected, cover_meta = greedy_h_cover_select(points, h=None, max_keep=12)
    assert len(cover_selected) == int(cover_meta["selected_count"])
    assert cover_meta["achieved_h"] >= 0.0

    print("smoke_core: ok")


if __name__ == "__main__":
    main()
