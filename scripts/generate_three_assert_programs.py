"""Generate the three-assertion branching programs for the new experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.programs import generate_programs, program_collection_fingerprint, sample_affine_layer  # noqa: E402
from core.three_assert import (  # noqa: E402
    DEFAULT_EDGE_THRESHOLD,
    THREE_ASSERT_EXECUTION_ORDER,
    THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
    THREE_ASSERT_PROGRAM_TYPE,
    affine_payload,
    validate_three_assert_program_collection,
    x_nonnegative_guard,
    y_nonnegative_guard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three-assertion branching programs.")
    parser.add_argument("--source-programs", default="triangle_lower_bound_outputs/affine_programs.json")
    parser.add_argument("--out-file", default="triangle_lower_bound_outputs/three_assert_affine_programs.json")
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--seed", type=int, default=283)
    parser.add_argument("--a-low", type=float, default=-1.5)
    parser.add_argument("--a-high", type=float, default=1.5)
    parser.add_argument("--b-low", type=float, default=-2.0)
    parser.add_argument("--b-high", type=float, default=2.0)
    parser.add_argument("--min-abs-det", type=float, default=0.05)
    parser.add_argument("--edge-threshold", type=int, default=DEFAULT_EDGE_THRESHOLD)
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    source_path = root / args.source_programs
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if any(int(program.get("num_affine_layers", -1)) != 2 for program in source):
        raise ValueError("Every source program must have exactly two affine layers.")

    reproduced = generate_programs(
        m=len(source),
        seed=args.seed,
        a_low=args.a_low,
        a_high=args.a_high,
        b_low=args.b_low,
        b_high=args.b_high,
        min_abs_det=args.min_abs_det,
        num_affine_layers=2,
    )
    if reproduced != source:
        raise RuntimeError(
            "The supplied RNG parameters do not reproduce the source programs exactly."
        )

    rng = np.random.default_rng(args.seed)
    max_attempts = max(10_000, 200 * len(source) * 4)
    for _ in range(2 * len(source)):
        sample_affine_layer(
            rng,
            args.a_low,
            args.a_high,
            args.b_low,
            args.b_high,
            args.min_abs_det,
            max_attempts,
        )

    source_fingerprint = program_collection_fingerprint(source)
    programs: list[dict] = []
    for source_program in source:
        A3, b3, det3 = sample_affine_layer(
            rng,
            args.a_low,
            args.a_high,
            args.b_low,
            args.b_high,
            args.min_abs_det,
            max_attempts,
        )
        A4, b4, det4 = sample_affine_layer(
            rng,
            args.a_low,
            args.a_high,
            args.b_low,
            args.b_high,
            args.min_abs_det,
            max_attempts,
        )
        layer1, layer2 = source_program["affine_layers"]
        programs.append(
            {
                "program_id": int(source_program["program_id"]),
                "program_type": THREE_ASSERT_PROGRAM_TYPE,
                "schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
                "execution_order": THREE_ASSERT_EXECUTION_ORDER,
                "num_affine_nodes": 4,
                "edge_threshold": int(args.edge_threshold),
                "affine1": affine_payload(layer1, "affine1"),
                "guard1": x_nonnegative_guard(),
                "affine2": affine_payload(layer2, "affine2"),
                "guard2": y_nonnegative_guard(),
                "affine3": {
                    "name": "affine3",
                    "A": A3.tolist(),
                    "b": b3.tolist(),
                    "abs_det_A": float(det3),
                },
                "affine4": {
                    "name": "affine4",
                    "A": A4.tolist(),
                    "b": b4.tolist(),
                    "abs_det_A": float(det4),
                },
                "source_program_id": int(source_program["program_id"]),
                "source_program_fingerprint": source_fingerprint,
                "affine3_affine4_rng_policy": "continue_source_rng_stream",
            }
        )

    validate_three_assert_program_collection(programs)
    out_path = root / args.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(programs, indent=2), encoding="utf-8")

    metadata_path = (
        root / args.metadata_file
        if args.metadata_file
        else out_path.with_name("three_assert_affine_programs_meta.json")
    )
    metadata = {
        "source_programs": str(source_path.relative_to(ARTIFACT_ROOT)),
        "source_program_fingerprint": source_fingerprint,
        "seed": int(args.seed),
        "a_low": float(args.a_low),
        "a_high": float(args.a_high),
        "b_low": float(args.b_low),
        "b_high": float(args.b_high),
        "min_abs_det": float(args.min_abs_det),
        "num_programs": int(len(programs)),
        "program_type": THREE_ASSERT_PROGRAM_TYPE,
        "schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "guard1": "x >= 0",
        "guard2": "y >= 0",
        "edge_threshold": int(args.edge_threshold),
        "affine3_affine4_rng_policy": "continue_source_rng_stream_after_source_affine1_affine2",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved three-assert programs: {out_path}")
    print(f"Saved generation metadata: {metadata_path}")


if __name__ == "__main__":
    main()
