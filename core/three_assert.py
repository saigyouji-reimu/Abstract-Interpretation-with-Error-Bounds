"""Three-assertion branching programs for the new POPL precision-loss experiments.

Program shape::

    affine1;
    if G1 then
        affine2;
        if G2 then assert1 else affine3; assert2
    else
        affine4;
    assert3

The third assertion is placed after the outer branch merge.  Abstract execution
uses Poly with a small complexity budget: after every affine/guard/join transfer,
a polygon with more than ``edge_threshold`` edges is replaced by its smallest
axis-aligned enclosing box.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .branching import split_polygon_by_guard
from .geometry import affine_polygon, box_polygon, convex_hull, polygon_area
from .guards import apply_guard_to_polygon, guarded_points, points_satisfying_guard

THREE_ASSERT_PROGRAM_SCHEMA_VERSION = 1
THREE_ASSERT_PROGRAM_TYPE = "three_assert_guarded_affine_branch"
THREE_ASSERT_EXECUTION_ORDER = (
    "affine1_then_if_x_ge_0_then_affine2_if_y_ge_0_assert1_else_affine3_assert2_"
    "else_affine4_merge_assert3"
)
DEFAULT_EDGE_THRESHOLD = 5

ASSERT_SITES = ["assert1", "assert2", "assert3"]


def x_nonnegative_guard() -> dict:
    return {"type": "halfspace", "normal": [1.0, 0.0], "threshold": 0.0, "name": "x >= 0"}


def y_nonnegative_guard() -> dict:
    return {"type": "halfspace", "normal": [0.0, 1.0], "threshold": 0.0, "name": "y >= 0"}


def _affine(payload: dict, name: str) -> dict:
    A = np.asarray(payload["A"], dtype=float).reshape(2, 2)
    return {
        "name": name,
        "A": A,
        "b": np.asarray(payload["b"], dtype=float).reshape(2),
        "abs_det_A": float(payload.get("abs_det_A", abs(np.linalg.det(A)))),
    }


def affine_payload(layer: dict, name: str) -> dict:
    return {
        "name": name,
        "A": deepcopy(layer["A"]),
        "b": deepcopy(layer["b"]),
        "abs_det_A": float(layer["abs_det_A"]),
    }


def three_assert_components(program: dict) -> dict:
    if program.get("program_type") != THREE_ASSERT_PROGRAM_TYPE:
        raise ValueError(f"Unexpected program type: {program.get('program_type')}")
    if program.get("schema_version") != THREE_ASSERT_PROGRAM_SCHEMA_VERSION:
        raise ValueError(f"Unexpected schema: {program.get('schema_version')}")
    if program.get("execution_order") != THREE_ASSERT_EXECUTION_ORDER:
        raise ValueError(f"Unexpected execution order: {program.get('execution_order')}")
    return {
        "affine1": _affine(program["affine1"], "affine1"),
        "guard1": program["guard1"],
        "affine2": _affine(program["affine2"], "affine2"),
        "guard2": program["guard2"],
        "affine3": _affine(program["affine3"], "affine3"),
        "affine4": _affine(program["affine4"], "affine4"),
        "edge_threshold": int(program.get("edge_threshold", DEFAULT_EDGE_THRESHOLD)),
    }


def validate_three_assert_program_collection(programs: list[dict]) -> None:
    if not programs:
        raise ValueError("Program collection must be nonempty.")
    seen: set[int] = set()
    for program in programs:
        program_id = int(program["program_id"])
        if program_id in seen:
            raise ValueError(f"Duplicate program_id: {program_id}")
        seen.add(program_id)
        three_assert_components(program)
        if int(program.get("num_affine_nodes", -1)) != 4:
            raise ValueError(f"Program {program_id} must contain four affine nodes.")


def read_three_assert_programs(path: Path) -> list[dict]:
    programs = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_three_assert_program_collection(programs)
    return programs


def three_assert_program_collection_fingerprint(programs: list[dict]) -> str:
    validate_three_assert_program_collection(programs)
    canonical = json.dumps(programs, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def merge_point_sets(*sets: np.ndarray) -> np.ndarray:
    nonempty = [np.asarray(points, dtype=float).reshape(-1, 2) for points in sets if len(points)]
    return np.vstack(nonempty) if nonempty else np.empty((0, 2), dtype=float)


def stack_hull(polygons: Iterable[np.ndarray]) -> np.ndarray:
    pieces = [np.asarray(poly, dtype=float).reshape(-1, 2) for poly in polygons if len(poly)]
    if not pieces:
        return np.empty((0, 2), dtype=float)
    return convex_hull(np.vstack(pieces))


def refine_polygon(poly: np.ndarray, edge_threshold: int = DEFAULT_EDGE_THRESHOLD) -> np.ndarray:
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) > int(edge_threshold):
        return box_polygon(poly)
    return poly.copy()


def transfer_affine(poly: np.ndarray, affine: dict, edge_threshold: int) -> np.ndarray:
    return refine_polygon(affine_polygon(poly, affine["A"], affine["b"]), edge_threshold)


def transfer_guard(poly: np.ndarray, guard: dict, edge_threshold: int) -> np.ndarray:
    return refine_polygon(apply_guard_to_polygon(poly, guard), edge_threshold)


def transfer_join(polygons: Iterable[np.ndarray], edge_threshold: int) -> np.ndarray:
    return refine_polygon(stack_hull(polygons), edge_threshold)


def execute_three_assert_points(points: np.ndarray, program: dict, *, return_trace: bool = False) -> dict:
    c = three_assert_components(program)
    source = np.asarray(points, dtype=float).reshape(-1, 2)
    after_affine1 = source @ c["affine1"]["A"].T + c["affine1"]["b"]
    g1_mask = points_satisfying_guard(after_affine1, c["guard1"])
    after_g1 = after_affine1[g1_mask]
    after_not_g1 = after_affine1[~g1_mask]

    after_affine2 = after_g1 @ c["affine2"]["A"].T + c["affine2"]["b"]
    g2_mask = points_satisfying_guard(after_affine2, c["guard2"])
    assert1 = after_affine2[g2_mask]
    before_affine3 = after_affine2[~g2_mask]
    assert2 = before_affine3 @ c["affine3"]["A"].T + c["affine3"]["b"]
    outer_else = after_not_g1 @ c["affine4"]["A"].T + c["affine4"]["b"]
    assert3 = merge_point_sets(assert1, assert2, outer_else)

    results = {"assert1": assert1, "assert2": assert2, "assert3": assert3}
    if not return_trace:
        return results
    counts = {
        "input_count": int(len(source)),
        "after_g1_count": int(len(after_g1)),
        "after_not_g1_count": int(len(after_not_g1)),
        "assert1_count": int(len(assert1)),
        "assert2_count": int(len(assert2)),
        "assert3_count": int(len(assert3)),
    }
    return {"sites": results, "trace": counts}


def execute_three_assert_abstract(input_hull: np.ndarray, program: dict, *, return_trace: bool = False) -> dict:
    c = three_assert_components(program)
    threshold = c["edge_threshold"]
    trace: dict[str, np.ndarray | float | int] = {}

    a0 = np.asarray(input_hull, dtype=float).reshape(-1, 2)
    after_affine1_raw = affine_polygon(a0, c["affine1"]["A"], c["affine1"]["b"])
    after_affine1 = refine_polygon(after_affine1_raw, threshold)
    then_g1_raw, else_g1_pieces_raw = split_polygon_by_guard(after_affine1, c["guard1"])
    then_g1 = refine_polygon(then_g1_raw, threshold)
    else_g1_pieces = [refine_polygon(piece, threshold) for piece in else_g1_pieces_raw]
    else_g1 = transfer_join(else_g1_pieces, threshold)

    after_affine2 = transfer_affine(then_g1, c["affine2"], threshold)
    assert1_raw, before_affine3_pieces_raw = split_polygon_by_guard(after_affine2, c["guard2"])
    assert1 = refine_polygon(assert1_raw, threshold)
    before_affine3_pieces = [refine_polygon(piece, threshold) for piece in before_affine3_pieces_raw]
    before_affine3 = transfer_join(before_affine3_pieces, threshold)
    assert2 = transfer_affine(before_affine3, c["affine3"], threshold)
    outer_else = transfer_affine(else_g1, c["affine4"], threshold)
    assert3 = transfer_join([assert1, assert2, outer_else], threshold)

    sites = {"assert1": assert1, "assert2": assert2, "assert3": assert3}
    if not return_trace:
        return sites
    trace.update(
        {
            "input_hull": a0,
            "after_affine1_raw": after_affine1_raw,
            "after_affine1": after_affine1,
            "then_g1": then_g1,
            "else_g1": else_g1,
            "after_affine2": after_affine2,
            "assert1": assert1,
            "before_affine3": before_affine3,
            "assert2": assert2,
            "outer_else": outer_else,
            "assert3": assert3,
        }
    )
    return {"sites": sites, "trace": trace}


def site_hull_area(points: np.ndarray) -> tuple[float, int, int]:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    hull = convex_hull(pts)
    return polygon_area(hull), int(len(pts)), int(len(hull))


def polygon_edges(poly: np.ndarray) -> int:
    return int(len(np.asarray(poly, dtype=float).reshape(-1, 2)))
