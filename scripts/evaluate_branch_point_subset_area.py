"""Evaluate area retention for affine programs with one guarded branch."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.branching import (
    BRANCH_EXECUTION_ORDER,
    branch_components,
    execute_branch_program_points,
    execute_branch_program_polygon,
)
from core.geometry import box_polygon, convex_hull, polygon_area
from scripts.evaluate_point_subset_area import EPS, loss_to_abstraction_ratio


def _point_result(points: np.ndarray, program: dict) -> dict:
    output, trace = execute_branch_program_points(points, program, return_trace=True)
    hull = convex_hull(output)
    return {
        "area": polygon_area(hull),
        "output_count": int(len(output)),
        "hull_vertices": int(len(hull)),
        **trace,
    }


def _retention(area: float, true_area: float) -> tuple[float, float, float]:
    gap = max(0.0, float(true_area - area))
    if true_area <= EPS and area <= EPS:
        return 1.0, gap, 0.0
    if true_area <= EPS:
        return float("nan"), gap, float("nan")
    return float(area / true_area), gap, float(gap / true_area)


def compute_branch_reference(program: dict, points: np.ndarray) -> dict:
    """Compute full-cloud and initial-box results shared by all selections."""
    true = _point_result(points, program)
    abstract_polygon, abstract_trace = execute_branch_program_polygon(
        box_polygon(points), program, return_trace=True
    )
    return {
        "true": true,
        "box_abstract_area": float(polygon_area(abstract_polygon)),
        "abstract_then_area": float(abstract_trace["then_output_area"]),
        "abstract_else_area": float(abstract_trace["else_output_area"]),
    }


def compute_branch_one(
    program: dict,
    points: np.ndarray,
    selected_points: np.ndarray,
    support_only_points: np.ndarray,
    *,
    reference: dict | None = None,
) -> dict:
    components = branch_components(program)
    reference = compute_branch_reference(program, points) if reference is None else reference
    true = reference["true"]
    selected = _point_result(selected_points, program)
    support = _point_result(support_only_points, program)
    box_abstract_area = float(reference["box_abstract_area"])

    retention, area_gap, relative_gap = _retention(selected["area"], true["area"])
    support_retention, support_gap, support_relative_gap = _retention(
        support["area"], true["area"]
    )
    abstraction_loss = max(0.0, float(box_abstract_area - true["area"]))

    return {
        "program_id": int(program["program_id"]),
        "program_type": program["program_type"],
        "schema_version": int(program["schema_version"]),
        "execution_order": BRANCH_EXECUTION_ORDER,
        "num_affine_nodes": 3,
        "max_affine_depth": 2,
        "affine1_abs_det": components["affine1"]["abs_det_A"],
        "affine2_abs_det": components["affine2"]["abs_det_A"],
        "affine3_abs_det": components["affine3"]["abs_det_A"],
        "then_path_abs_det": (
            components["affine1"]["abs_det_A"]
            * components["affine2"]["abs_det_A"]
        ),
        "else_path_abs_det": (
            components["affine1"]["abs_det_A"]
            * components["affine3"]["abs_det_A"]
        ),
        "true_area": float(true["area"]),
        "selected_area": float(selected["area"]),
        "area_retention_ratio": retention,
        "area_gap": area_gap,
        "relative_area_gap": relative_gap,
        "support_only_area": float(support["area"]),
        "support_only_area_retention_ratio": support_retention,
        "support_only_area_gap": support_gap,
        "support_only_relative_area_gap": support_relative_gap,
        "box_abstract_area": float(box_abstract_area),
        "box_abstraction_loss_area": abstraction_loss,
        "sampling_loss_to_box_loss_ratio": loss_to_abstraction_ratio(
            area_gap, abstraction_loss
        ),
        "support_only_loss_to_box_loss_ratio": loss_to_abstraction_ratio(
            support_gap, abstraction_loss
        ),
        "true_area_is_zero": bool(true["area"] <= EPS),
        "selected_area_leq_true_area": bool(selected["area"] <= true["area"] + 1e-9),
        "num_true_survivors": int(true["output_count"]),
        "num_selected_survivors": int(selected["output_count"]),
        "num_support_only_survivors": int(support["output_count"]),
        "true_survival_ratio": float(true["output_count"] / len(points)),
        "selected_survival_ratio": float(
            selected["output_count"] / len(selected_points)
        ),
        "support_only_survival_ratio": float(
            support["output_count"] / len(support_only_points)
        ),
        "true_output_hull_vertices": int(true["hull_vertices"]),
        "selected_output_hull_vertices": int(selected["hull_vertices"]),
        "support_only_output_hull_vertices": int(support["hull_vertices"]),
        "true_then_input_count": int(true["then_input_count"]),
        "true_else_input_count": int(true["else_input_count"]),
        "true_then_output_count": int(true["then_output_count"]),
        "true_else_output_count": int(true["else_output_count"]),
        "selected_then_output_count": int(selected["then_output_count"]),
        "selected_else_output_count": int(selected["else_output_count"]),
        "abstract_then_area": float(reference["abstract_then_area"]),
        "abstract_else_area": float(reference["abstract_else_area"]),
    }
