"""Fixed-bank posterior learning from both sides of a guarded branch."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .branching import split_polygon_by_guard
from .geometry import (
    affine_polygon,
    clip_polygon_with_halfspace,
    convex_hull,
    polygon_area,
)
from .guards import apply_guard_to_polygon, points_satisfying_guard
from .posterior import (
    EPS,
    finite_float_or_none,
    points_satisfy_halfspace,
    polygon_satisfies_halfspace,
    ratio_or_none,
)

CONSTRAINT_BANK_SCHEMA_VERSION = 1
BRANCH_FEEDBACK_SCHEMA_VERSION = 1


def _stratified_values(
    rng: np.random.Generator,
    count: int,
    low: float,
    high: float,
) -> np.ndarray:
    values = low + (high - low) * (
        np.arange(count, dtype=float) + rng.random(count)
    ) / count
    rng.shuffle(values)
    return values


def generate_fixed_constraint_bank(
    *,
    count: int = 10_000,
    seed: int = 283,
    num_blocks: int = 10,
    projection_radius: float = 120.0,
    central_radius: float = 30.0,
    central_fraction: float = 0.4,
) -> dict:
    """Generate a fixed, pair-independent bank of diverse 2D halfspaces.

    Every block independently covers all directions. Thresholds combine broad
    and center-dense stratification so each 1000-constraint block remains useful
    across differently scaled affine outputs.
    """
    if count <= 0 or num_blocks <= 0 or count % num_blocks:
        raise ValueError("count must be positive and divisible by num_blocks.")
    if not 0.0 <= central_fraction <= 1.0:
        raise ValueError("central_fraction must lie in [0, 1].")
    if not 0.0 < central_radius <= projection_radius:
        raise ValueError("Require 0 < central_radius <= projection_radius.")

    rng = np.random.default_rng(seed)
    block_size = count // num_blocks
    constraints: list[dict] = []
    for block_id in range(num_blocks):
        thetas = _stratified_values(rng, block_size, 0.0, 2.0 * np.pi)
        central_count = int(round(block_size * central_fraction))
        broad_count = block_size - central_count
        thresholds = np.concatenate(
            [
                _stratified_values(
                    rng, broad_count, -projection_radius, projection_radius
                ),
                _stratified_values(
                    rng, central_count, -central_radius, central_radius
                ),
            ]
        )
        rng.shuffle(thresholds)
        for local_id, (theta, threshold) in enumerate(zip(thetas, thresholds)):
            constraint_id = block_id * block_size + local_id
            constraints.append(
                {
                    "constraint_id": int(constraint_id),
                    "block_id": int(block_id),
                    "block_local_id": int(local_id),
                    "normal": [float(math.cos(theta)), float(math.sin(theta))],
                    "threshold": float(threshold),
                    "theta": float(theta),
                    "generation_mode": "fixed_stratified_global",
                }
            )

    return {
        "schema_version": CONSTRAINT_BANK_SCHEMA_VERSION,
        "seed": int(seed),
        "count": int(count),
        "num_blocks": int(num_blocks),
        "block_size": int(block_size),
        "projection_radius": float(projection_radius),
        "central_radius": float(central_radius),
        "central_fraction": float(central_fraction),
        "constraints": constraints,
    }



def _generate_stratified_constraints(
    *,
    count: int,
    rng: np.random.Generator,
    projection_radius: float,
    central_radius: float,
    central_fraction: float,
    constraint_id_offset: int,
    generation_mode: str,
    program_id: int | None = None,
    branch: str | None = None,
) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if not 0.0 <= central_fraction <= 1.0:
        raise ValueError("central_fraction must lie in [0, 1].")
    if not 0.0 < central_radius <= projection_radius:
        raise ValueError("Require 0 < central_radius <= projection_radius.")

    thetas = _stratified_values(rng, count, 0.0, 2.0 * np.pi)
    central_count = int(round(count * central_fraction))
    broad_count = count - central_count
    thresholds = np.concatenate(
        [
            _stratified_values(rng, broad_count, -projection_radius, projection_radius),
            _stratified_values(rng, central_count, -central_radius, central_radius),
        ]
    )
    rng.shuffle(thresholds)
    constraints: list[dict] = []
    for local_id, (theta, threshold) in enumerate(zip(thetas, thresholds)):
        item = {
            "constraint_id": int(constraint_id_offset + local_id),
            "block_id": int(program_id if program_id is not None else 0),
            "block_local_id": int(local_id),
            "normal": [float(math.cos(theta)), float(math.sin(theta))],
            "threshold": float(threshold),
            "theta": float(theta),
            "generation_mode": generation_mode,
        }
        if program_id is not None:
            item["program_id"] = int(program_id)
        if branch is not None:
            item["branch"] = str(branch)
        constraints.append(item)
    return constraints


def generate_program_constraint_pair(
    *,
    program_id: int,
    constraints_per_branch: int = 1000,
    seed: int = 283,
    projection_radius: float = 120.0,
    central_radius: float = 30.0,
    central_fraction: float = 0.4,
) -> tuple[list[dict], list[dict], str]:
    """Generate deterministic program-specific then/else feedback constraints."""
    program_seed = int(seed) + 1_000_003 * int(program_id)
    rng = np.random.default_rng(program_seed)
    then_constraints = _generate_stratified_constraints(
        count=constraints_per_branch,
        rng=rng,
        projection_radius=projection_radius,
        central_radius=central_radius,
        central_fraction=central_fraction,
        constraint_id_offset=int(program_id) * 2 * int(constraints_per_branch),
        generation_mode="program_stratified_then",
        program_id=program_id,
        branch="then",
    )
    else_constraints = _generate_stratified_constraints(
        count=constraints_per_branch,
        rng=rng,
        projection_radius=projection_radius,
        central_radius=central_radius,
        central_fraction=central_fraction,
        constraint_id_offset=(int(program_id) * 2 + 1) * int(constraints_per_branch),
        generation_mode="program_stratified_else",
        program_id=program_id,
        branch="else",
    )
    config = {
        "mode": "per_program",
        "program_id": int(program_id),
        "program_seed": int(program_seed),
        "constraints_per_branch": int(constraints_per_branch),
        "total_unique_constraints": int(2 * constraints_per_branch),
        "projection_radius": float(projection_radius),
        "central_radius": float(central_radius),
        "central_fraction": float(central_fraction),
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return then_constraints, else_constraints, fingerprint


def validate_constraint_bank(bank: dict) -> None:
    if bank.get("schema_version") != CONSTRAINT_BANK_SCHEMA_VERSION:
        raise ValueError("Unsupported fixed constraint-bank schema.")
    constraints = bank.get("constraints", [])
    if int(bank.get("count", -1)) != len(constraints):
        raise ValueError("Constraint-bank count does not match its payload.")
    num_blocks = int(bank["num_blocks"])
    block_size = int(bank["block_size"])
    if len(constraints) != num_blocks * block_size:
        raise ValueError("Constraint-bank block dimensions are inconsistent.")
    ids = [int(item["constraint_id"]) for item in constraints]
    if ids != list(range(len(constraints))):
        raise ValueError("Constraint IDs must be contiguous and ordered.")
    for item in constraints:
        np.asarray(item["normal"], dtype=float).reshape(2)
        float(item["threshold"])


def read_constraint_bank(path: Path) -> dict:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_constraint_bank(bank)
    return bank


def constraint_bank_fingerprint(bank: dict) -> str:
    validate_constraint_bank(bank)
    payload = {key: value for key, value in bank.items() if key != "fingerprint"}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def constraints_for_program(
    bank: dict,
    program_id: int,
    constraints_per_branch: int = 1000,
) -> tuple[list[dict], int]:
    """Select one fixed block by program ID; no pair-specific RNG is used."""
    validate_constraint_bank(bank)
    block_size = int(bank["block_size"])
    if constraints_per_branch <= 0 or constraints_per_branch > block_size:
        raise ValueError(
            f"constraints_per_branch must lie in [1, {block_size}]."
        )
    block_id = int(program_id) % int(bank["num_blocks"])
    start = block_id * block_size
    selected = bank["constraints"][start : start + constraints_per_branch]
    return selected, block_id


def _stack_hull(polygons: Iterable[np.ndarray]) -> np.ndarray:
    pieces = [
        np.asarray(polygon, dtype=float).reshape(-1, 2)
        for polygon in polygons
        if len(polygon)
    ]
    if not pieces:
        return np.empty((0, 2), dtype=float)
    return convex_hull(np.vstack(pieces))


def _inverse_affine_polygon(
    polygon: np.ndarray, A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(polygon) == 0:
        return polygon.copy()
    inverse = np.linalg.inv(np.asarray(A, dtype=float).reshape(2, 2))
    return convex_hull(
        (polygon - np.asarray(b, dtype=float).reshape(2)) @ inverse.T
    )


def _corrected_metrics(
    *,
    original_area: float,
    allowed_output_hull: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    old_lower_bound: float,
    true_loss_area: float,
) -> tuple[float, float, float | None]:
    allowed_input = _inverse_affine_polygon(allowed_output_hull, A, b)
    posterior = max(0.0, original_area - polygon_area(allowed_input))
    corrected = max(float(old_lower_bound), posterior)
    recovery = finite_float_or_none(
        ratio_or_none(corrected, true_loss_area)
        if true_loss_area > EPS
        else 0.0
    )
    return float(posterior), float(corrected), recovery


def evaluate_two_branch_feedback_trace(
    *,
    intermediate_polygon: np.ndarray,
    intermediate_true_points: np.ndarray,
    old_intermediate_lower_bound: float,
    second_layer: dict,
    constraints: Iterable[dict],
    else_constraints: Iterable[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Learn at guard1 from false-positive feedback on both guard2 branches.

    Then feedback represents ``G2 => H`` and else feedback represents
    ``not G2 => H``. The implementation keeps the two output-space branch
    regions separate, clips only the branch justified by feedback, joins them,
    and applies the affine inverse. This is the geometric realization of the
    two implication weakest preconditions.
    """
    P1 = np.asarray(intermediate_polygon, dtype=float).reshape(-1, 2)
    C1 = np.asarray(intermediate_true_points, dtype=float).reshape(-1, 2)
    A = np.asarray(second_layer["A"], dtype=float).reshape(2, 2)
    b = np.asarray(second_layer["b"], dtype=float).reshape(2)
    if abs(np.linalg.det(A)) <= EPS:
        raise ValueError("Two-branch geometric pullback requires invertible affine2.")

    P1_area = polygon_area(P1)
    true_loss_area = max(0.0, P1_area - polygon_area(convex_hull(C1)))
    abstract_after_affine = affine_polygon(P1, A, b)
    then_original, else_original_pieces = split_polygon_by_guard(
        abstract_after_affine, second_layer["guard"]
    )
    else_abstract = _stack_hull(else_original_pieces)

    concrete_after_affine = C1 @ A.T + b
    then_mask = points_satisfying_guard(
        concrete_after_affine, second_layer["guard"]
    )
    concrete_then = concrete_after_affine[then_mask]
    concrete_else = concrete_after_affine[~then_mask]
    concrete_then_hull = convex_hull(concrete_then)
    concrete_else_hull = convex_hull(concrete_else)

    then_current = then_original.copy()
    else_current = [piece.copy() for piece in else_original_pieces]
    then_useful_count = else_useful_count = 0
    then_triggered_count = else_triggered_count = 0
    trace: list[dict] = []
    then_constraints = list(constraints)
    paired_else_constraints = (
        then_constraints if else_constraints is None else list(else_constraints)
    )
    if len(then_constraints) != len(paired_else_constraints):
        raise ValueError("then/else constraint lists must have the same length.")

    for checked_count, (then_item, else_item) in enumerate(
        zip(then_constraints, paired_else_constraints), start=1
    ):
        then_normal = np.asarray(then_item["normal"], dtype=float).reshape(2)
        then_threshold = float(then_item["threshold"])
        else_normal = np.asarray(else_item["normal"], dtype=float).reshape(2)
        else_threshold = float(else_item["threshold"])

        then_triggered = not polygon_satisfies_halfspace(
            then_original, then_normal, then_threshold
        )
        else_triggered = not polygon_satisfies_halfspace(
            else_abstract, else_normal, else_threshold
        )
        then_useful = bool(
            then_triggered
            and points_satisfy_halfspace(
                concrete_then_hull, then_normal, then_threshold
            )
        )
        else_useful = bool(
            else_triggered
            and points_satisfy_halfspace(
                concrete_else_hull, else_normal, else_threshold
            )
        )
        then_triggered_count += int(then_triggered)
        else_triggered_count += int(else_triggered)
        then_useful_count += int(then_useful)
        else_useful_count += int(else_useful)

        if then_useful:
            then_current = clip_polygon_with_halfspace(
                then_current, then_normal, then_threshold
            )
        if else_useful:
            else_current = [
                clip_polygon_with_halfspace(piece, else_normal, else_threshold)
                for piece in else_current
            ]

        then_only_hull = _stack_hull([then_current, *else_original_pieces])
        else_only_hull = _stack_hull([then_original, *else_current])
        combined_hull = _stack_hull([then_current, *else_current])
        then_posterior, then_corrected, then_recovery = _corrected_metrics(
            original_area=P1_area,
            allowed_output_hull=then_only_hull,
            A=A,
            b=b,
            old_lower_bound=old_intermediate_lower_bound,
            true_loss_area=true_loss_area,
        )
        else_posterior, else_corrected, else_recovery = _corrected_metrics(
            original_area=P1_area,
            allowed_output_hull=else_only_hull,
            A=A,
            b=b,
            old_lower_bound=old_intermediate_lower_bound,
            true_loss_area=true_loss_area,
        )
        combined_posterior, combined_corrected, combined_recovery = (
            _corrected_metrics(
                original_area=P1_area,
                allowed_output_hull=combined_hull,
                A=A,
                b=b,
                old_lower_bound=old_intermediate_lower_bound,
                true_loss_area=true_loss_area,
            )
        )
        trace.append(
            {
                "constraint_id": int(then_item["constraint_id"]),
                "then_constraint_id": int(then_item["constraint_id"]),
                "else_constraint_id": int(else_item["constraint_id"]),
                "block_id": int(then_item["block_id"]),
                "then_block_id": int(then_item["block_id"]),
                "else_block_id": int(else_item["block_id"]),
                "checked_count_per_branch": int(checked_count),
                "total_branch_checks": int(2 * checked_count),
                "then_triggered": bool(then_triggered),
                "else_triggered": bool(else_triggered),
                "then_useful": bool(then_useful),
                "else_useful": bool(else_useful),
                "then_triggered_count": int(then_triggered_count),
                "else_triggered_count": int(else_triggered_count),
                "then_useful_count": int(then_useful_count),
                "else_useful_count": int(else_useful_count),
                "total_useful_count": int(
                    then_useful_count + else_useful_count
                ),
                "then_posterior_lower_bound": then_posterior,
                "else_posterior_lower_bound": else_posterior,
                "combined_posterior_lower_bound": combined_posterior,
                "then_corrected_lower_bound": then_corrected,
                "else_corrected_lower_bound": else_corrected,
                "combined_corrected_lower_bound": combined_corrected,
                "then_recovery_ratio": then_recovery,
                "else_recovery_ratio": else_recovery,
                "combined_recovery_ratio": combined_recovery,
                "combined_gain_over_best_branch": float(
                    combined_corrected - max(then_corrected, else_corrected)
                ),
                "old_intermediate_lower_bound": float(
                    old_intermediate_lower_bound
                ),
                "true_loss_area": float(true_loss_area),
            }
        )

    final = trace[-1] if trace else None
    final_summary = {
        "constraint_count_per_branch": int(len(trace)),
        "total_branch_checks": int(2 * len(trace)),
        "then_concrete_count": int(len(concrete_then)),
        "else_concrete_count": int(len(concrete_else)),
        "then_branch_empty": bool(len(concrete_then) == 0),
        "else_branch_empty": bool(len(concrete_else) == 0),
        "intermediate_abstract_area": float(P1_area),
        "intermediate_true_hull_area": float(polygon_area(convex_hull(C1))),
        "true_loss_area": float(true_loss_area),
        "old_lower_bound": float(old_intermediate_lower_bound),
    }
    if final is not None:
        final_summary.update(
            {
                key: final[key]
                for key in [
                    "then_triggered_count",
                    "else_triggered_count",
                    "then_useful_count",
                    "else_useful_count",
                    "total_useful_count",
                    "then_posterior_lower_bound",
                    "else_posterior_lower_bound",
                    "combined_posterior_lower_bound",
                    "then_corrected_lower_bound",
                    "else_corrected_lower_bound",
                    "combined_corrected_lower_bound",
                    "then_recovery_ratio",
                    "else_recovery_ratio",
                    "combined_recovery_ratio",
                    "combined_gain_over_best_branch",
                ]
            }
        )
    return trace, final_summary
