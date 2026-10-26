"""Program generation and execution for guarded affine sequences."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import affine_polygon
from .guards import apply_guard_to_polygon, guarded_points, quadrant_guard_spec

PROGRAM_SCHEMA_VERSION = 2
EXECUTION_ORDER = "affine_then_guard_per_layer"


def read_programs(path: Path) -> list[dict]:
    programs = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_program_collection(programs)
    return programs


def validate_program_collection(programs: list[dict]) -> None:
    if not programs:
        raise ValueError("Program collection must be nonempty.")
    seen_ids: set[int] = set()
    for program in programs:
        program_id = int(program["program_id"])
        if program_id in seen_ids:
            raise ValueError(f"Duplicate program_id: {program_id}")
        seen_ids.add(program_id)
        if program.get("schema_version") != PROGRAM_SCHEMA_VERSION:
            raise ValueError(
                f"Program {program_id} has schema_version={program.get('schema_version')}; "
                f"expected {PROGRAM_SCHEMA_VERSION}."
            )
        if program.get("execution_order") != EXECUTION_ORDER:
            raise ValueError(
                f"Program {program_id} has execution_order={program.get('execution_order')}; "
                f"expected {EXECUTION_ORDER}."
            )
        if not program.get("guard_after_each_affine"):
            raise ValueError(f"Program {program_id} does not guard every affine layer.")
        layers = program_layers(program)
        if int(program.get("num_affine_layers", -1)) != len(layers):
            raise ValueError(f"Program {program_id} has inconsistent layer count.")


def program_collection_fingerprint(programs: list[dict]) -> str:
    validate_program_collection(programs)
    canonical = json.dumps(
        programs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sample_affine_layer(
    rng: np.random.Generator,
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
    min_abs_det: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    attempts = 0
    while True:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to generate a matrix with |det(A)| >= {min_abs_det}. "
                "Try reducing --min-abs-det."
            )
        A = rng.uniform(a_low, a_high, size=(2, 2))
        det_abs = float(abs(np.linalg.det(A)))
        if det_abs < min_abs_det:
            continue
        b = rng.uniform(b_low, b_high, size=(2,))
        return A, b, det_abs


def compose_affine_layers(layers: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Compose affine maps while deliberately ignoring intermediate guards.

    The result is useful for determinant diagnostics, but it is not semantically
    equivalent to executing a guarded affine sequence.
    """
    composed_A = np.eye(2, dtype=float)
    composed_b = np.zeros(2, dtype=float)
    det_product = 1.0
    for layer in layers:
        A = np.asarray(layer["A"], dtype=float).reshape(2, 2)
        b = np.asarray(layer["b"], dtype=float).reshape(2)
        composed_b = A @ composed_b + b
        composed_A = A @ composed_A
        det_product *= float(abs(np.linalg.det(A)))
    return composed_A, composed_b, float(det_product)


def program_layers(program: dict) -> list[dict]:
    """Return validated affine+guard layers in execution order."""
    layers = program.get("affine_layers")
    if not layers:
        raise ValueError("Program must contain a nonempty affine_layers list.")

    normalized: list[dict] = []
    shared_guard = program.get("guard")
    for index, layer in enumerate(layers):
        guard = layer.get("guard")
        if guard is None and program.get("guard_after_each_affine"):
            guard = shared_guard
        if guard is None:
            raise ValueError(
                "Every affine layer must be followed by a guard. "
                f"Layer {index} has no guard; this looks like the archived "
                "final-guard-only program schema."
            )
        A = np.asarray(layer["A"], dtype=float).reshape(2, 2)
        normalized.append(
            {
                "layer_id": int(layer.get("layer_id", index)),
                "A": A,
                "b": np.asarray(layer["b"], dtype=float).reshape(2),
                "abs_det_A": float(layer.get("abs_det_A", abs(np.linalg.det(A)))),
                "guard": guard,
            }
        )
    return normalized


def composed_affine_from_program(
    program: dict,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return the affine-only composition for diagnostics, not execution."""
    layers = program_layers(program)
    A, b, det_product = compose_affine_layers(layers)
    return A, b, det_product, len(layers)


def execute_program_points(
    points: np.ndarray,
    program: dict,
    *,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Execute affine -> guard for every layer on a finite point cloud."""
    current = np.asarray(points, dtype=float).reshape(-1, 2)
    trace: list[dict] = []
    for layer in program_layers(program):
        before_count = int(len(current))
        after_affine = current @ layer["A"].T + layer["b"]
        current = guarded_points(after_affine, layer["guard"])
        if return_trace:
            trace.append(
                {
                    "layer_id": layer["layer_id"],
                    "input_count": before_count,
                    "after_affine_count": int(len(after_affine)),
                    "after_guard_count": int(len(current)),
                }
            )
    if return_trace:
        return current, trace
    return current


def execute_program_polygon(
    polygon: np.ndarray,
    program: dict,
    *,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Execute affine -> guard for every layer on a convex polygon."""
    current = np.asarray(polygon, dtype=float).reshape(-1, 2)
    trace: list[dict] = []
    for layer in program_layers(program):
        after_affine = affine_polygon(current, layer["A"], layer["b"])
        after_guard = apply_guard_to_polygon(after_affine, layer["guard"])
        if return_trace:
            trace.append(
                {
                    "layer_id": layer["layer_id"],
                    "after_affine": after_affine,
                    "after_guard": after_guard,
                }
            )
        current = after_guard
    if return_trace:
        return current, trace
    return current


def generate_programs(
    m: int,
    seed: int,
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
    min_abs_det: float,
    num_affine_layers: int,
    guard: dict | None = None,
) -> list[dict]:
    if m <= 0:
        raise ValueError("m must be positive.")
    if num_affine_layers <= 0:
        raise ValueError("num_affine_layers must be positive.")
    if a_low >= a_high:
        raise ValueError("a_low must be less than a_high.")
    if b_low >= b_high:
        raise ValueError("b_low must be less than b_high.")

    rng = np.random.default_rng(seed)
    programs: list[dict] = []
    layer_max_attempts = max(10_000, 200 * m * num_affine_layers)
    guard_spec = deepcopy(guard or quadrant_guard_spec())

    for pid in range(m):
        layers: list[dict] = []
        for layer_id in range(num_affine_layers):
            A, b, det_abs = sample_affine_layer(
                rng=rng,
                a_low=a_low,
                a_high=a_high,
                b_low=b_low,
                b_high=b_high,
                min_abs_det=min_abs_det,
                max_attempts=layer_max_attempts,
            )
            layers.append(
                {
                    "layer_id": layer_id,
                    "A": A.tolist(),
                    "b": b.tolist(),
                    "abs_det_A": det_abs,
                    "guard": deepcopy(guard_spec),
                }
            )

        composed_A, composed_b, det_product = compose_affine_layers(layers)
        programs.append(
            {
                "program_id": pid,
                "program_type": "guarded_affine_sequence",
                "schema_version": PROGRAM_SCHEMA_VERSION,
                "execution_order": EXECUTION_ORDER,
                "guard_after_each_affine": True,
                "num_affine_layers": num_affine_layers,
                "affine_layers": layers,
                "A": composed_A.tolist(),
                "b": composed_b.tolist(),
                "abs_det_A": det_product,
                "affine_layer_abs_dets": [
                    float(layer["abs_det_A"]) for layer in layers
                ],
                "guard": deepcopy(guard_spec),
                "composed_affine_is_diagnostic_only": True,
            }
        )
    return programs
