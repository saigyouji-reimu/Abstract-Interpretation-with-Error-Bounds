"""Lower-bound and concrete-output metrics for guarded affine sequences."""

from __future__ import annotations

import numpy as np

from .geometry import affine_polygon, box_polygon, convex_hull, polygon_area
from .guards import apply_guard_to_polygon, guarded_points
from .programs import (
    composed_affine_from_program,
    execute_program_points,
    program_layers,
)


def hull_area_after_program(
    points: np.ndarray,
    program: dict,
) -> tuple[float, int, int]:
    survivors = execute_program_points(points, program)
    hull = convex_hull(survivors)
    return polygon_area(hull), int(len(survivors)), int(len(hull))


def compute_lower_bound_components(
    points: np.ndarray,
    program: dict,
    input_hull: np.ndarray | None = None,
    input_box: np.ndarray | None = None,
) -> dict:
    """Compute lower-bound components under affine -> guard per-layer semantics."""
    layers = program_layers(program)
    composed_A, composed_b, total_abs_det, num_affine_layers = (
        composed_affine_from_program(program)
    )

    input_hull = convex_hull(points) if input_hull is None else input_hull
    input_box = box_polygon(points) if input_box is None else input_box
    input_hull_area = polygon_area(input_hull)
    input_box_area = polygon_area(input_box)
    initial_error = max(0.0, input_box_area - input_hull_area)

    method_polygon = np.asarray(input_hull, dtype=float).reshape(-1, 2)
    box_abstract_polygon = np.asarray(input_box, dtype=float).reshape(-1, 2)
    concrete_points = np.asarray(points, dtype=float).reshape(-1, 2)
    lower_bound = float(initial_error)
    layer_trace: list[dict] = []

    for layer in layers:
        method_after_affine = affine_polygon(
            method_polygon, layer["A"], layer["b"]
        )
        box_after_affine = affine_polygon(
            box_abstract_polygon, layer["A"], layer["b"]
        )
        lower_bound_after_affine = float(lower_bound * layer["abs_det_A"])

        method_after_guard = apply_guard_to_polygon(
            method_after_affine, layer["guard"]
        )
        box_after_guard = apply_guard_to_polygon(
            box_after_affine, layer["guard"]
        )
        guard_removed_from_box = max(
            0.0,
            polygon_area(box_after_affine) - polygon_area(box_after_guard),
        )
        lower_bound_after_guard = max(
            0.0, lower_bound_after_affine - guard_removed_from_box
        )

        concrete_after_affine = concrete_points @ layer["A"].T + layer["b"]
        concrete_after_guard = guarded_points(
            concrete_after_affine, layer["guard"]
        )

        layer_trace.append(
            {
                "layer_id": int(layer["layer_id"]),
                "abs_det_A": float(layer["abs_det_A"]),
                "guard": layer["guard"],
                "method_input_polygon": method_polygon,
                "box_input_polygon": box_abstract_polygon,
                "method_after_affine": method_after_affine,
                "box_after_affine": box_after_affine,
                "method_after_guard": method_after_guard,
                "box_after_guard": box_after_guard,
                "method_after_affine_area": polygon_area(method_after_affine),
                "box_after_affine_area": polygon_area(box_after_affine),
                "method_after_guard_area": polygon_area(method_after_guard),
                "box_after_guard_area": polygon_area(box_after_guard),
                "guard_removed_from_box": float(guard_removed_from_box),
                "lower_bound_input": float(lower_bound),
                "lower_bound_after_affine": float(lower_bound_after_affine),
                "lower_bound_after_guard": float(lower_bound_after_guard),
                "concrete_input_count": int(len(concrete_points)),
                "concrete_after_guard_count": int(len(concrete_after_guard)),
            }
        )

        method_polygon = method_after_guard
        box_abstract_polygon = box_after_guard
        concrete_points = concrete_after_guard
        lower_bound = lower_bound_after_guard

    final_layer = layer_trace[-1]
    true_hull = convex_hull(concrete_points)
    method1_area = polygon_area(method_polygon)
    box_abstract_area = polygon_area(box_abstract_polygon)
    pointwise_true_area = polygon_area(true_hull)
    true_loss_area = max(0.0, box_abstract_area - pointwise_true_area)

    return {
        "A": composed_A,
        "b": composed_b,
        "guard": layers[-1]["guard"],
        "program_type": program.get("program_type"),
        "execution_order": "affine_then_guard_per_layer",
        "num_affine_layers": int(num_affine_layers),
        "input_hull": input_hull,
        "input_box": input_box,
        "input_hull_area": float(input_hull_area),
        "input_box_area": float(input_box_area),
        "initial_error": float(initial_error),
        "total_abs_det": float(total_abs_det),
        "affine1_scale": float(total_abs_det),
        "after_affine_value": float(final_layer["lower_bound_after_affine"]),
        "after_affine1_value": float(final_layer["lower_bound_after_affine"]),
        "hull_after_affine": final_layer["method_after_affine"],
        "box_after_affine": final_layer["box_after_affine"],
        "method1_polygon": method_polygon,
        "box_abstract_polygon": box_abstract_polygon,
        "hull_after_affine_area": float(final_layer["method_after_affine_area"]),
        "box_after_affine_area": float(final_layer["box_after_affine_area"]),
        "method1_area": float(method1_area),
        "method1_output_hull_area": float(method1_area),
        "box_abstract_area": float(box_abstract_area),
        "method2_output_hull_area": float(box_abstract_area),
        "coarse_removed_area_by_guard": float(
            final_layer["guard_removed_from_box"]
        ),
        "guard_removed_area_by_layer": [
            float(item["guard_removed_from_box"]) for item in layer_trace
        ],
        "old_lower_bound": float(lower_bound),
        "after_guard_lower_bound": float(lower_bound),
        "after_guard_true_error": max(0.0, box_abstract_area - method1_area),
        "output_abs_area_gap": abs(box_abstract_area - method1_area),
        "true_survivors": concrete_points,
        "true_survivor_count": int(len(concrete_points)),
        "true_hull": true_hull,
        "true_hull_vertices": int(len(true_hull)),
        "pointwise_true_area": float(pointwise_true_area),
        "pointwise_true_survivors": int(len(concrete_points)),
        "pointwise_true_hull_vertices": int(len(true_hull)),
        "true_loss_area": float(true_loss_area),
        "abstraction_loss_area": float(true_loss_area),
        "layer_trace": layer_trace,
    }
