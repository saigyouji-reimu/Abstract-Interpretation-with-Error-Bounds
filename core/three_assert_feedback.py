"""Posterior lower-bound learning for the three-assertion branch program."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .branch_feedback import _inverse_affine_polygon
from .branching import split_polygon_by_guard
from .geometry import affine_polygon, convex_hull, polygon_area
from .guards import points_satisfying_guard
from .posterior import EPS, finite_float_or_none, points_satisfy_halfspace, polygon_satisfies_halfspace, ratio_or_none
from .three_assert import (
    execute_three_assert_abstract,
    execute_three_assert_points,
    refine_polygon,
    stack_hull,
    three_assert_components,
    transfer_affine,
    transfer_join,
)
from .geometry import clip_polygon_with_halfspace

THREE_ASSERT_FEEDBACK_SCHEMA_VERSION = 1


def _not_guard_hull(poly: np.ndarray, guard: dict) -> np.ndarray:
    _, pieces = split_polygon_by_guard(poly, guard)
    return stack_hull(pieces)


def _clip_to_not_guard(poly: np.ndarray, guard: dict) -> np.ndarray:
    return _not_guard_hull(poly, guard)


def _corrected_metrics(
    *,
    original_area: float,
    allowed_after_affine2: np.ndarray,
    affine2: dict,
    old_lower_bound: float,
    true_loss_area: float,
) -> tuple[float, float, float | None]:
    allowed_input = _inverse_affine_polygon(
        allowed_after_affine2, affine2["A"], affine2["b"]
    )
    posterior = max(0.0, float(original_area - polygon_area(allowed_input)))
    corrected = max(float(old_lower_bound), posterior)
    recovery = finite_float_or_none(
        ratio_or_none(corrected, true_loss_area) if true_loss_area > EPS else 0.0
    )
    return float(posterior), float(corrected), recovery


def _assert3_metrics(
    *,
    original_assert3: np.ndarray,
    posterior_assert1: np.ndarray,
    posterior_assert2: np.ndarray,
    outer_else: np.ndarray,
    edge_threshold: int,
    old_lower_bound: float,
    true_loss_area: float,
) -> tuple[float, float, float | None, float]:
    posterior_poly = transfer_join(
        [posterior_assert1, posterior_assert2, outer_else], edge_threshold
    )
    posterior = max(0.0, float(polygon_area(original_assert3) - polygon_area(posterior_poly)))
    corrected = max(float(old_lower_bound), posterior)
    recovery = finite_float_or_none(
        ratio_or_none(corrected, true_loss_area) if true_loss_area > EPS else 0.0
    )
    return float(posterior), float(corrected), recovery, float(polygon_area(posterior_poly))


def _initial_guard1_lower(points: np.ndarray, program: dict, input_hull: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    c = three_assert_components(program)
    threshold = c["edge_threshold"]
    raw_after_affine1 = affine_polygon(input_hull, c["affine1"]["A"], c["affine1"]["b"])
    refined_after_affine1 = refine_polygon(raw_after_affine1, threshold)
    introduced = max(0.0, polygon_area(refined_after_affine1) - polygon_area(raw_after_affine1))
    then_raw, _ = split_polygon_by_guard(refined_after_affine1, c["guard1"])
    then_g1 = refine_polygon(then_raw, threshold)
    removed = max(0.0, polygon_area(refined_after_affine1) - polygon_area(then_g1))
    old_lb = max(0.0, introduced - removed)

    after_affine1_points = points @ c["affine1"]["A"].T + c["affine1"]["b"]
    after_g1_points = after_affine1_points[
        points_satisfying_guard(after_affine1_points, c["guard1"])
    ]
    true_loss = max(0.0, polygon_area(then_g1) - polygon_area(convex_hull(after_g1_points)))
    return float(old_lb), then_g1, after_g1_points, float(true_loss)


def evaluate_three_assert_feedback_trace(
    *,
    points: np.ndarray,
    program: dict,
    assert1_constraints: Iterable[dict],
    assert2_constraints: Iterable[dict],
) -> tuple[list[dict], dict]:
    c = three_assert_components(program)
    threshold = c["edge_threshold"]
    source = np.asarray(points, dtype=float).reshape(-1, 2)
    input_hull = convex_hull(source)
    old_guard1_lb, P1, C1, guard1_true_loss = _initial_guard1_lower(
        source, program, input_hull
    )

    abstract = execute_three_assert_abstract(input_hull, program, return_trace=True)
    abstract_trace = abstract["trace"]
    point_exec = execute_three_assert_points(source, program, return_trace=True)
    point_sites = point_exec["sites"]

    affine2 = c["affine2"]
    affine3 = c["affine3"]
    if abs(np.linalg.det(affine2["A"])) <= EPS or abs(np.linalg.det(affine3["A"])) <= EPS:
        raise ValueError("Feedback pullback requires invertible affine2 and affine3.")

    original_assert1 = np.asarray(abstract_trace["assert1"], dtype=float).reshape(-1, 2)
    original_before_affine3 = np.asarray(abstract_trace["before_affine3"], dtype=float).reshape(-1, 2)
    original_assert2 = np.asarray(abstract_trace["assert2"], dtype=float).reshape(-1, 2)
    original_outer_else = np.asarray(abstract_trace["outer_else"], dtype=float).reshape(-1, 2)
    original_assert3 = np.asarray(abstract_trace["assert3"], dtype=float).reshape(-1, 2)

    concrete_assert1_hull = convex_hull(point_sites["assert1"])
    concrete_assert2_hull = convex_hull(point_sites["assert2"])
    concrete_assert3_hull = convex_hull(point_sites["assert3"])
    assert3_true_loss = max(0.0, polygon_area(original_assert3) - polygon_area(concrete_assert3_hull))
    old_assert3_lb = 0.0

    assert1_current = original_assert1.copy()
    assert2_current = original_assert2.copy()
    assert1_useful_count = assert2_useful_count = 0
    assert1_triggered_count = assert2_triggered_count = 0
    trace: list[dict] = []
    a1_constraints = list(assert1_constraints)
    a2_constraints = list(assert2_constraints)
    if len(a1_constraints) != len(a2_constraints):
        raise ValueError("assert1/assert2 constraint lists must have the same length.")

    for checked_count, (a1_item, a2_item) in enumerate(zip(a1_constraints, a2_constraints), start=1):
        n1 = np.asarray(a1_item["normal"], dtype=float).reshape(2)
        b1 = float(a1_item["threshold"])
        n2 = np.asarray(a2_item["normal"], dtype=float).reshape(2)
        b2 = float(a2_item["threshold"])

        assert1_triggered = not polygon_satisfies_halfspace(original_assert1, n1, b1)
        assert2_triggered = not polygon_satisfies_halfspace(original_assert2, n2, b2)
        assert1_useful = bool(assert1_triggered and points_satisfy_halfspace(concrete_assert1_hull, n1, b1))
        assert2_useful = bool(assert2_triggered and points_satisfy_halfspace(concrete_assert2_hull, n2, b2))
        assert1_triggered_count += int(assert1_triggered)
        assert2_triggered_count += int(assert2_triggered)
        assert1_useful_count += int(assert1_useful)
        assert2_useful_count += int(assert2_useful)

        if assert1_useful:
            assert1_current = clip_polygon_with_halfspace(assert1_current, n1, b1)
        if assert2_useful:
            assert2_current = clip_polygon_with_halfspace(assert2_current, n2, b2)

        assert2_preimage = _inverse_affine_polygon(assert2_current, affine3["A"], affine3["b"])
        assert2_preimage = _clip_to_not_guard(assert2_preimage, c["guard2"])

        assert1_only_after_affine2 = stack_hull([assert1_current, original_before_affine3])
        assert2_only_after_affine2 = stack_hull([original_assert1, assert2_preimage])
        combined_after_affine2 = stack_hull([assert1_current, assert2_preimage])

        a1_post, a1_corr, a1_rec = _corrected_metrics(
            original_area=polygon_area(P1),
            allowed_after_affine2=assert1_only_after_affine2,
            affine2=affine2,
            old_lower_bound=old_guard1_lb,
            true_loss_area=guard1_true_loss,
        )
        a2_post, a2_corr, a2_rec = _corrected_metrics(
            original_area=polygon_area(P1),
            allowed_after_affine2=assert2_only_after_affine2,
            affine2=affine2,
            old_lower_bound=old_guard1_lb,
            true_loss_area=guard1_true_loss,
        )
        combined_post, combined_corr, combined_rec = _corrected_metrics(
            original_area=polygon_area(P1),
            allowed_after_affine2=combined_after_affine2,
            affine2=affine2,
            old_lower_bound=old_guard1_lb,
            true_loss_area=guard1_true_loss,
        )

        a1_assert3_post, a1_assert3_corr, a1_assert3_rec, a1_assert3_area = _assert3_metrics(
            original_assert3=original_assert3,
            posterior_assert1=assert1_current,
            posterior_assert2=original_assert2,
            outer_else=original_outer_else,
            edge_threshold=threshold,
            old_lower_bound=old_assert3_lb,
            true_loss_area=assert3_true_loss,
        )
        a2_assert3_post, a2_assert3_corr, a2_assert3_rec, a2_assert3_area = _assert3_metrics(
            original_assert3=original_assert3,
            posterior_assert1=original_assert1,
            posterior_assert2=assert2_current,
            outer_else=original_outer_else,
            edge_threshold=threshold,
            old_lower_bound=old_assert3_lb,
            true_loss_area=assert3_true_loss,
        )
        combined_assert3_post, combined_assert3_corr, combined_assert3_rec, combined_assert3_area = _assert3_metrics(
            original_assert3=original_assert3,
            posterior_assert1=assert1_current,
            posterior_assert2=assert2_current,
            outer_else=original_outer_else,
            edge_threshold=threshold,
            old_lower_bound=old_assert3_lb,
            true_loss_area=assert3_true_loss,
        )

        trace.append(
            {
                "checked_count_per_site": int(checked_count),
                "total_checks": int(2 * checked_count),
                "assert1_constraint_id": int(a1_item["constraint_id"]),
                "assert2_constraint_id": int(a2_item["constraint_id"]),
                "assert1_triggered": bool(assert1_triggered),
                "assert2_triggered": bool(assert2_triggered),
                "assert1_useful": bool(assert1_useful),
                "assert2_useful": bool(assert2_useful),
                "assert1_triggered_count": int(assert1_triggered_count),
                "assert2_triggered_count": int(assert2_triggered_count),
                "assert1_useful_count": int(assert1_useful_count),
                "assert2_useful_count": int(assert2_useful_count),
                "total_useful_count": int(assert1_useful_count + assert2_useful_count),
                "assert1_posterior_lower_bound": a1_post,
                "assert2_posterior_lower_bound": a2_post,
                "combined_posterior_lower_bound": combined_post,
                "assert1_corrected_lower_bound": a1_corr,
                "assert2_corrected_lower_bound": a2_corr,
                "combined_corrected_lower_bound": combined_corr,
                "assert1_recovery_ratio": a1_rec,
                "assert2_recovery_ratio": a2_rec,
                "combined_recovery_ratio": combined_rec,
                "assert3_assert1_posterior_lower_bound": a1_assert3_post,
                "assert3_assert2_posterior_lower_bound": a2_assert3_post,
                "assert3_combined_posterior_lower_bound": combined_assert3_post,
                "assert3_assert1_corrected_lower_bound": a1_assert3_corr,
                "assert3_assert2_corrected_lower_bound": a2_assert3_corr,
                "assert3_combined_corrected_lower_bound": combined_assert3_corr,
                "assert3_assert1_recovery_ratio": a1_assert3_rec,
                "assert3_assert2_recovery_ratio": a2_assert3_rec,
                "assert3_combined_recovery_ratio": combined_assert3_rec,
                "assert3_assert1_posterior_area": a1_assert3_area,
                "assert3_assert2_posterior_area": a2_assert3_area,
                "assert3_combined_posterior_area": combined_assert3_area,
                "old_guard1_lower_bound": float(old_guard1_lb),
                "guard1_true_loss": float(guard1_true_loss),
                "old_assert3_lower_bound": float(old_assert3_lb),
                "assert3_true_loss": float(assert3_true_loss),
            }
        )

    final = trace[-1] if trace else None
    final_summary = {
        "schema_version": THREE_ASSERT_FEEDBACK_SCHEMA_VERSION,
        "constraint_count_per_site": int(len(trace)),
        "total_checks": int(2 * len(trace)),
        "old_guard1_lower_bound": float(old_guard1_lb),
        "guard1_abstract_area": float(polygon_area(P1)),
        "guard1_true_hull_area": float(polygon_area(convex_hull(C1))),
        "guard1_true_loss": float(guard1_true_loss),
        "old_guard1_recovery_ratio": finite_float_or_none(ratio_or_none(old_guard1_lb, guard1_true_loss)) if guard1_true_loss > EPS else 0.0,
        "old_assert3_lower_bound": float(old_assert3_lb),
        "assert3_abstract_area": float(polygon_area(original_assert3)),
        "assert3_true_hull_area": float(polygon_area(concrete_assert3_hull)),
        "assert3_true_loss": float(assert3_true_loss),
        "old_assert3_recovery_ratio": 0.0 if assert3_true_loss > EPS else 0.0,
        "assert1_concrete_count": int(len(point_sites["assert1"])),
        "assert2_concrete_count": int(len(point_sites["assert2"])),
        "assert3_concrete_count": int(len(point_sites["assert3"])),
    }
    if final is not None:
        for key in [
            "assert1_triggered_count",
            "assert2_triggered_count",
            "assert1_useful_count",
            "assert2_useful_count",
            "total_useful_count",
            "assert1_corrected_lower_bound",
            "assert2_corrected_lower_bound",
            "combined_corrected_lower_bound",
            "assert1_recovery_ratio",
            "assert2_recovery_ratio",
            "combined_recovery_ratio",
            "assert3_assert1_corrected_lower_bound",
            "assert3_assert2_corrected_lower_bound",
            "assert3_combined_corrected_lower_bound",
            "assert3_assert1_recovery_ratio",
            "assert3_assert2_recovery_ratio",
            "assert3_combined_recovery_ratio",
            "assert3_combined_posterior_area",
        ]:
            final_summary[key] = final[key]
    return trace, final_summary
