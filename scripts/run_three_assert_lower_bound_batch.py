"""Run lower-bound feedback for the three-assertion branch program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.branch_feedback import generate_program_constraint_pair  # noqa: E402
from core.three_assert import (  # noqa: E402
    THREE_ASSERT_EXECUTION_ORDER,
    THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
    read_three_assert_programs,
    three_assert_program_collection_fingerprint,
)
from core.three_assert_feedback import THREE_ASSERT_FEEDBACK_SCHEMA_VERSION, evaluate_three_assert_feedback_trace  # noqa: E402
from run_point_subset_area_batch import parse_distributions  # noqa: E402
from core.posterior import ratio_or_none  # noqa: E402


def _metrics_from_row(row: dict | None, *, old_lower_bound: float, true_loss_area: float, final_total_useful: int) -> dict:
    if row is None:
        values = {
            "checked_count_per_site": 0,
            "total_checks": 0,
            "assert1_triggered_count": 0,
            "assert2_triggered_count": 0,
            "assert1_useful_count": 0,
            "assert2_useful_count": 0,
            "total_useful_count": 0,
            "assert1_corrected_lower_bound": old_lower_bound,
            "assert2_corrected_lower_bound": old_lower_bound,
            "combined_corrected_lower_bound": old_lower_bound,
            "assert1_recovery_ratio": ratio_or_none(old_lower_bound, true_loss_area),
            "assert2_recovery_ratio": ratio_or_none(old_lower_bound, true_loss_area),
            "combined_recovery_ratio": ratio_or_none(old_lower_bound, true_loss_area),
            "assert3_assert1_corrected_lower_bound": 0.0,
            "assert3_assert2_corrected_lower_bound": 0.0,
            "assert3_combined_corrected_lower_bound": 0.0,
            "assert3_assert1_recovery_ratio": 0.0,
            "assert3_assert2_recovery_ratio": 0.0,
            "assert3_combined_recovery_ratio": 0.0,
        }
    else:
        keys = [
            "checked_count_per_site",
            "total_checks",
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
        ]
        values = {key: row.get(key) for key in keys}
    values["useful_fraction_observed"] = ratio_or_none(int(values["total_useful_count"]), final_total_useful)
    values["old_lower_bound"] = float(old_lower_bound)
    values["true_loss_area"] = float(true_loss_area)
    return values


def build_three_checkpoints(
    trace: list[dict],
    checked_checkpoints: list[int],
    useful_checkpoints: list[int],
    useful_fraction_checkpoints: list[float],
    *,
    old_lower_bound: float,
    true_loss_area: float,
) -> dict:
    final_useful = int(trace[-1]["total_useful_count"]) if trace else 0
    output = {"checked_count": [], "useful_count": [], "useful_fraction": []}
    for checkpoint in checked_checkpoints:
        row = trace[min(checkpoint, len(trace)) - 1] if checkpoint > 0 and trace else None
        item = _metrics_from_row(row, old_lower_bound=old_lower_bound, true_loss_area=true_loss_area, final_total_useful=final_useful)
        item.update({"axis": "checked_count", "checkpoint": int(checkpoint), "reached_checkpoint": bool(len(trace) >= checkpoint)})
        output["checked_count"].append(item)
    for checkpoint in useful_checkpoints:
        row = next((item for item in trace if int(item["total_useful_count"]) >= checkpoint), trace[-1] if trace else None)
        item = _metrics_from_row(row if checkpoint > 0 else None, old_lower_bound=old_lower_bound, true_loss_area=true_loss_area, final_total_useful=final_useful)
        item.update({"axis": "useful_count", "checkpoint": int(checkpoint), "reached_checkpoint": bool(final_useful >= checkpoint)})
        output["useful_count"].append(item)
    for checkpoint in useful_fraction_checkpoints:
        target = int(np.ceil(float(checkpoint) * final_useful))
        row = next((item for item in trace if int(item["total_useful_count"]) >= target), trace[-1] if trace else None)
        item = _metrics_from_row(row if target > 0 else None, old_lower_bound=old_lower_bound, true_loss_area=true_loss_area, final_total_useful=final_useful)
        item.update({"axis": "useful_fraction", "checkpoint": float(checkpoint), "target_useful_count": int(target), "reached_checkpoint": True})
        output["useful_fraction"].append(item)
    return output


def run_pair(
    *,
    sample_dir: Path,
    distribution: str,
    sample_id: int,
    program: dict,
    program_fingerprint: str,
    out_root: Path,
    constraints_per_site: int,
    constraint_seed: int,
    projection_radius: float,
    central_radius: float,
    central_fraction: float,
    checked_checkpoints: list[int],
    useful_checkpoints: list[int],
    useful_fraction_checkpoints: list[float],
    force: bool,
    save_trace: bool,
) -> None:
    points_path = sample_dir / "triangle_points.npy"
    meta_path = sample_dir / "triangle_meta.json"
    if not points_path.exists() or not meta_path.exists():
        print(f"[missing points] {sample_dir}")
        return
    pair_dir = out_root / distribution / f"sample_{sample_id:03d}" / f"program_{int(program['program_id']):03d}"
    summary_path = pair_dir / "pair_summary.json"
    trace_path = pair_dir / "feedback_trace.json"
    if summary_path.exists() and (not save_trace or trace_path.exists()) and not force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "experiment": "three_assert_lower_bound_feedback",
            "program_fingerprint": program_fingerprint,
            "program_id": int(program["program_id"]),
            "constraints_per_site": int(constraints_per_site),
            "constraint_seed": int(constraint_seed),
        }
        mismatches = {k: (existing.get(k), v) for k, v in expected.items() if existing.get(k) != v}
        if mismatches:
            raise RuntimeError(f"Existing result mismatch at {pair_dir}: {mismatches}")
        print(f"[skip] {distribution} sample_{sample_id:03d} program_{int(program['program_id']):03d}")
        return

    points = np.load(points_path)
    point_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert1_constraints, assert2_constraints, constraint_fingerprint = generate_program_constraint_pair(
        program_id=int(program["program_id"]),
        constraints_per_branch=constraints_per_site,
        seed=constraint_seed,
        projection_radius=projection_radius,
        central_radius=central_radius,
        central_fraction=central_fraction,
    )
    trace, final = evaluate_three_assert_feedback_trace(
        points=points,
        program=program,
        assert1_constraints=assert1_constraints,
        assert2_constraints=assert2_constraints,
    )
    checkpoints = build_three_checkpoints(
        trace,
        checked_checkpoints,
        useful_checkpoints,
        useful_fraction_checkpoints,
        old_lower_bound=float(final["old_guard1_lower_bound"]),
        true_loss_area=float(final["guard1_true_loss"]),
    )

    pair_dir.mkdir(parents=True, exist_ok=True)
    if save_trace:
        trace_path.write_text(
            json.dumps(trace, indent=2, allow_nan=True), encoding="utf-8"
        )
    payload = {
        "experiment": "three_assert_lower_bound_feedback",
        "schema_version": THREE_ASSERT_FEEDBACK_SCHEMA_VERSION,
        "program_execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "program_schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": program_fingerprint,
        "program_id": int(program["program_id"]),
        "distribution": distribution,
        "sample_id": int(sample_id),
        "point_seed": point_meta.get("seed"),
        "num_points": point_meta.get("n"),
        "constraints_per_site": int(constraints_per_site),
        "constraint_seed": int(constraint_seed),
        "constraint_fingerprint": constraint_fingerprint,
        "constraint_generation": "program_stratified_assert1_assert2",
        "assert1_constraint_count": len(assert1_constraints),
        "assert2_constraint_count": len(assert2_constraints),
        "trace_saved": bool(save_trace),
        "final": final,
        "checkpoints": checkpoints,
    }
    summary_path.write_text(
        json.dumps(payload, allow_nan=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"[done] {distribution} sample_{sample_id:03d} program_{int(program['program_id']):03d} "
        f"combined_guard1={final.get('combined_recovery_ratio')} "
        f"combined_assert3={final.get('assert3_combined_recovery_ratio')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three-assert lower-bound feedback experiments.")
    parser.add_argument("--points-root", default="point_distribution_outputs")
    parser.add_argument("--programs-file", default="triangle_lower_bound_outputs/three_assert_affine_programs.json")
    parser.add_argument("--out-dir", default="three_assert_lower_bound_outputs/batch")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=100)
    parser.add_argument("--program-start", type=int, default=0)
    parser.add_argument("--program-end", type=int, default=None)
    parser.add_argument("--constraints-per-site", type=int, default=2000)
    parser.add_argument("--constraint-seed", type=int, default=283)
    parser.add_argument("--projection-radius", type=float, default=120.0)
    parser.add_argument("--central-radius", type=float, default=30.0)
    parser.add_argument("--central-fraction", type=float, default=0.4)
    parser.add_argument("--checked-checkpoints", type=int, nargs="*", default=[0, 100, 250, 500, 1000, 1500, 2000])
    parser.add_argument("--useful-checkpoints", type=int, nargs="*", default=[0, 1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--useful-fraction-checkpoints", type=float, nargs="*", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save the full per-constraint trace. Disabled by default to keep output compact.",
    )
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    programs = read_three_assert_programs(root / args.programs_file)
    fingerprint = three_assert_program_collection_fingerprint(programs)
    distributions = parse_distributions(args.distributions)
    program_end = len(programs) if args.program_end is None else min(args.program_end, len(programs))
    selected_programs = programs[args.program_start:program_end]
    if not selected_programs:
        raise ValueError("The selected program range is empty.")
    out_root = root / args.out_dir

    for distribution in distributions:
        for sample_id in range(args.sample_start, args.sample_end):
            sample_dir = root / args.points_root / distribution / f"sample_{sample_id:03d}"
            for program in selected_programs:
                run_pair(
                    sample_dir=sample_dir,
                    distribution=distribution,
                    sample_id=sample_id,
                    program=program,
                    program_fingerprint=fingerprint,
                    out_root=out_root,
                    constraints_per_site=args.constraints_per_site,
                    constraint_seed=args.constraint_seed,
                    projection_radius=args.projection_radius,
                    central_radius=args.central_radius,
                    central_fraction=args.central_fraction,
                    checked_checkpoints=args.checked_checkpoints,
                    useful_checkpoints=args.useful_checkpoints,
                    useful_fraction_checkpoints=args.useful_fraction_checkpoints,
                    force=args.force,
                    save_trace=args.save_trace,
                )


if __name__ == "__main__":
    main()
