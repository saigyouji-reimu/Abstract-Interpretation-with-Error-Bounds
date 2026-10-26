"""Branching affine-program semantics for the upper-bound experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import affine_polygon, convex_hull, polygon_area
from .guards import (
    apply_guard_to_polygon,
    guard_halfspaces,
    guarded_points,
    points_satisfying_guard,
)

BRANCH_PROGRAM_SCHEMA_VERSION = 1
BRANCH_EXECUTION_ORDER = (
    "affine1_then_if_guard1_then_affine2_guard2_else_affine3_then_merge"
)
BRANCH_PROGRAM_TYPE = "guarded_affine_branch"


def _affine(payload: dict, name: str) -> dict:
    A = np.asarray(payload["A"], dtype=float).reshape(2, 2)
    return {
        "name": name,
        "A": A,
        "b": np.asarray(payload["b"], dtype=float).reshape(2),
        "abs_det_A": float(payload.get("abs_det_A", abs(np.linalg.det(A)))),
    }


def branch_components(program: dict) -> dict:
    """Return normalized components of a branching affine program."""
    if program.get("program_type") != BRANCH_PROGRAM_TYPE:
        raise ValueError(f"Unexpected branch program type: {program.get('program_type')}")
    if program.get("schema_version") != BRANCH_PROGRAM_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected branch schema version: {program.get('schema_version')}"
        )
    if program.get("execution_order") != BRANCH_EXECUTION_ORDER:
        raise ValueError(
            f"Unexpected branch execution order: {program.get('execution_order')}"
        )
    return {
        "affine1": _affine(program["affine1"], "affine1"),
        "guard1": program["guard1"],
        "affine2": _affine(program["affine2"], "affine2"),
        "guard2": program["guard2"],
        "affine3": _affine(program["affine3"], "affine3"),
    }


def validate_branch_program_collection(programs: list[dict]) -> None:
    if not programs:
        raise ValueError("Branch program collection must be nonempty.")
    seen: set[int] = set()
    for program in programs:
        program_id = int(program["program_id"])
        if program_id in seen:
            raise ValueError(f"Duplicate branch program_id: {program_id}")
        seen.add(program_id)
        branch_components(program)
        if int(program.get("num_affine_nodes", -1)) != 3:
            raise ValueError(f"Branch program {program_id} must contain three affine nodes.")
        if int(program.get("max_affine_depth", -1)) != 2:
            raise ValueError(f"Branch program {program_id} must have affine depth two.")


def read_branch_programs(path: Path) -> list[dict]:
    programs = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_branch_program_collection(programs)
    return programs


def branch_program_collection_fingerprint(programs: list[dict]) -> str:
    validate_branch_program_collection(programs)
    canonical = json.dumps(
        programs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def split_polygon_by_guard(
    polygon: np.ndarray, guard: dict
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Split a convex polygon into the guard region and complement pieces.

    If the guard is H1 and ... and Hk, the closed complement is partitioned as
    not H1, H1 and not H2, ..., H1 and ... and H(k-1) and not Hk. Boundaries
    may occur in both sides but do not affect area or the convex join.
    """
    from .geometry import clip_polygon_with_halfspace

    prefix = np.asarray(polygon, dtype=float).reshape(-1, 2)
    else_pieces: list[np.ndarray] = []
    for normal, threshold in guard_halfspaces(guard):
        failed = clip_polygon_with_halfspace(prefix, -normal, -threshold)
        if len(failed):
            else_pieces.append(failed)
        prefix = clip_polygon_with_halfspace(prefix, normal, threshold)
        if len(prefix) == 0:
            break
    return prefix, else_pieces


def _merge_point_sets(*sets: np.ndarray) -> np.ndarray:
    nonempty = [np.asarray(points, dtype=float).reshape(-1, 2) for points in sets if len(points)]
    return np.vstack(nonempty) if nonempty else np.empty((0, 2), dtype=float)


def execute_branch_program_points(
    points: np.ndarray,
    program: dict,
    *,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Execute affine1; if guard1 then affine2; guard2 else affine3."""
    components = branch_components(program)
    source = np.asarray(points, dtype=float).reshape(-1, 2)
    after_affine1 = (
        source @ components["affine1"]["A"].T + components["affine1"]["b"]
    )
    then_mask = points_satisfying_guard(after_affine1, components["guard1"])
    then_input = after_affine1[then_mask]
    else_input = after_affine1[~then_mask]

    then_after_affine2 = (
        then_input @ components["affine2"]["A"].T + components["affine2"]["b"]
    )
    then_output = guarded_points(then_after_affine2, components["guard2"])
    else_output = (
        else_input @ components["affine3"]["A"].T + components["affine3"]["b"]
    )
    merged = _merge_point_sets(then_output, else_output)

    if not return_trace:
        return merged
    return merged, {
        "input_count": int(len(source)),
        "then_input_count": int(len(then_input)),
        "else_input_count": int(len(else_input)),
        "then_output_count": int(len(then_output)),
        "else_output_count": int(len(else_output)),
        "merged_output_count": int(len(merged)),
    }


def execute_branch_program_polygon(
    polygon: np.ndarray,
    program: dict,
    *,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Execute the branch in the Poly domain and join with a convex hull."""
    components = branch_components(program)
    after_affine1 = affine_polygon(
        polygon, components["affine1"]["A"], components["affine1"]["b"]
    )
    then_input, else_pieces = split_polygon_by_guard(
        after_affine1, components["guard1"]
    )
    then_after_affine2 = affine_polygon(
        then_input, components["affine2"]["A"], components["affine2"]["b"]
    )
    then_output = apply_guard_to_polygon(
        then_after_affine2, components["guard2"]
    )

    else_vertices = _merge_point_sets(*else_pieces)
    else_input = convex_hull(else_vertices)
    else_output = affine_polygon(
        else_input, components["affine3"]["A"], components["affine3"]["b"]
    )
    merged = convex_hull(_merge_point_sets(then_output, else_output))

    if not return_trace:
        return merged
    return merged, {
        "after_affine1": after_affine1,
        "then_input": then_input,
        "else_input": else_input,
        "then_output": then_output,
        "else_output": else_output,
        "merged_output": merged,
        "then_output_area": polygon_area(then_output),
        "else_output_area": polygon_area(else_output),
        "merged_output_area": polygon_area(merged),
    }
