"""Evaluate upper-bound area metrics at all three assertion sites."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.geometry import convex_hull, polygon_area
from core.three_assert import (
    ASSERT_SITES,
    THREE_ASSERT_EXECUTION_ORDER,
    execute_three_assert_abstract,
    execute_three_assert_points,
    polygon_edges,
    site_hull_area,
    three_assert_components,
)
from evaluate_point_subset_area import EPS


def _retention(area: float, true_area: float) -> tuple[float, float, float]:
    gap = max(0.0, float(true_area - area))
    if true_area <= EPS and area <= EPS:
        return 1.0, gap, 0.0
    if true_area <= EPS:
        return float("nan"), gap, float("nan")
    return float(area / true_area), gap, float(gap / true_area)


def _point_site_summaries(points: np.ndarray, program: dict) -> dict[str, dict]:
    output = execute_three_assert_points(points, program, return_trace=True)
    site_points = output["sites"]
    trace = output["trace"]
    summaries: dict[str, dict] = {}
    for site in ASSERT_SITES:
        area, count, hull_vertices = site_hull_area(site_points[site])
        summaries[site] = {
            "area": float(area),
            "count": int(count),
            "hull_vertices": int(hull_vertices),
        }
    summaries["_trace"] = trace
    return summaries


def compute_three_assert_reference(program: dict, points: np.ndarray) -> dict:
    input_hull = convex_hull(points)
    abstract = execute_three_assert_abstract(input_hull, program, return_trace=True)
    true = _point_site_summaries(points, program)
    site_refs = {}
    for site in ASSERT_SITES:
        abstract_poly = abstract["sites"][site]
        abstract_area = polygon_area(abstract_poly)
        true_area = true[site]["area"]
        site_refs[site] = {
            "abstract_area": float(abstract_area),
            "abstract_edges": polygon_edges(abstract_poly),
            "true_area": float(true_area),
            "true_loss": max(0.0, float(abstract_area - true_area)),
            "true_count": int(true[site]["count"]),
            "true_hull_vertices": int(true[site]["hull_vertices"]),
        }
    return {
        "input_hull_area": float(polygon_area(input_hull)),
        "input_hull_vertices": int(len(input_hull)),
        "site_refs": site_refs,
        "true_trace": true["_trace"],
        "abstract_trace": abstract["trace"],
    }


def compute_three_assert_one(
    program: dict,
    selected_points: np.ndarray,
    generator_points: np.ndarray,
    *,
    reference: dict,
) -> list[dict]:
    components = three_assert_components(program)
    selected = _point_site_summaries(selected_points, program)
    generator = _point_site_summaries(generator_points, program)
    rows: list[dict] = []
    for site in ASSERT_SITES:
        site_ref = reference["site_refs"][site]
        abstract_area = float(site_ref["abstract_area"])
        true_area = float(site_ref["true_area"])
        selected_area = float(selected[site]["area"])
        generator_area = float(generator[site]["area"])
        retention, area_gap, relative_gap = _retention(selected_area, true_area)
        generator_retention, generator_gap, generator_relative_gap = _retention(
            generator_area, true_area
        )
        true_loss = max(0.0, float(abstract_area - true_area))
        selected_upper_loss = max(0.0, float(abstract_area - selected_area))
        generator_upper_loss = max(0.0, float(abstract_area - generator_area))
        rows.append(
            {
                "program_id": int(program["program_id"]),
                "assert_site": site,
                "program_type": program["program_type"],
                "schema_version": int(program["schema_version"]),
                "execution_order": THREE_ASSERT_EXECUTION_ORDER,
                "edge_threshold": int(components["edge_threshold"]),
                "affine1_abs_det": float(components["affine1"]["abs_det_A"]),
                "affine2_abs_det": float(components["affine2"]["abs_det_A"]),
                "affine3_abs_det": float(components["affine3"]["abs_det_A"]),
                "affine4_abs_det": float(components["affine4"]["abs_det_A"]),
                "abstract_area": abstract_area,
                "abstract_edges": int(site_ref["abstract_edges"]),
                "true_area": true_area,
                "true_loss": true_loss,
                "selected_area": selected_area,
                "area_retention_ratio": retention,
                "area_gap": area_gap,
                "relative_area_gap": relative_gap,
                "generator_area": generator_area,
                "generator_retention_ratio": generator_retention,
                "generator_area_gap": generator_gap,
                "generator_relative_area_gap": generator_relative_gap,
                "selected_upper_loss": selected_upper_loss,
                "generator_upper_loss": generator_upper_loss,
                "selected_upper_loss_reduction_vs_generator": float(
                    generator_upper_loss - selected_upper_loss
                ),
                "selected_area_gain_vs_generator": float(selected_area - generator_area),
                "true_area_is_zero": bool(true_area <= EPS),
                "selected_area_leq_true_area": bool(selected_area <= true_area + 1e-9),
                "generator_area_leq_true_area": bool(generator_area <= true_area + 1e-9),
                "num_true_survivors": int(site_ref["true_count"]),
                "num_selected_survivors": int(selected[site]["count"]),
                "num_generator_survivors": int(generator[site]["count"]),
                "true_output_hull_vertices": int(site_ref["true_hull_vertices"]),
                "selected_output_hull_vertices": int(selected[site]["hull_vertices"]),
                "generator_output_hull_vertices": int(generator[site]["hull_vertices"]),
            }
        )
    return rows
