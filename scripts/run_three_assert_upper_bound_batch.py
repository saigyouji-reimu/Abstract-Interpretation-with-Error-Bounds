"""Run three-assertion upper-bound experiments using existing selections."""

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

from core.geometry import convex_hull_indices  # noqa: E402
from core.three_assert import (  # noqa: E402
    ASSERT_SITES,
    THREE_ASSERT_EXECUTION_ORDER,
    THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
    read_three_assert_programs,
    three_assert_program_collection_fingerprint,
)
from evaluate_three_assert_upper_bound import (  # noqa: E402
    compute_three_assert_one,
    compute_three_assert_reference,
)
from run_point_subset_area_batch import (  # noqa: E402
    parse_distributions,
    parse_selection_algorithms,
    summarize_values,
)

OUTPUT_DIR_NAME = "three_assert_upper_bound_outputs"
SOURCE_OUTPUT_DIR_NAME = "upper_bound_outputs"

SUMMARY_METRICS = [
    "abstract_area",
    "true_area",
    "true_loss",
    "selected_area",
    "area_retention_ratio",
    "area_gap",
    "relative_area_gap",
    "generator_area",
    "generator_retention_ratio",
    "generator_area_gap",
    "selected_upper_loss",
    "generator_upper_loss",
    "selected_upper_loss_reduction_vs_generator",
    "selected_area_gain_vs_generator",
    "num_true_survivors",
    "num_selected_survivors",
    "num_generator_survivors",
]


def build_summary(
    rows: list[dict],
    point_meta: dict,
    selection_meta: dict,
    algorithm_name: str,
    programs_file: Path,
    fingerprint: str,
    source_selection_dir: Path,
) -> dict:
    metrics = {}
    for site in ASSERT_SITES:
        site_rows = [row for row in rows if row["assert_site"] == site]
        metrics[site] = {
            metric: summarize_values(
                np.asarray([float(row[metric]) for row in site_rows], dtype=float)
            )
            for metric in SUMMARY_METRICS
        }
    metrics["all_sites"] = {
        metric: summarize_values(
            np.asarray([float(row[metric]) for row in rows], dtype=float)
        )
        for metric in SUMMARY_METRICS
    }
    return {
        "experiment": "three_assert_upper_bound",
        "distribution": point_meta.get("distribution"),
        "sample_id": point_meta.get("sample_id"),
        "point_seed": point_meta.get("seed"),
        "selection_algorithm": algorithm_name,
        "selection_method": selection_meta.get("selection_method"),
        "grid_kind": selection_meta.get("grid_kind"),
        "selection_mode": selection_meta.get("selection_mode"),
        "target_h": selection_meta.get("target_h"),
        "achieved_h": selection_meta.get("achieved_h"),
        "is_h_cover": selection_meta.get("is_h_cover"),
        "num_programs": len({int(row["program_id"]) for row in rows}),
        "program_ids": sorted({int(row["program_id"]) for row in rows}),
        "num_rows": len(rows),
        "assert_sites": list(ASSERT_SITES),
        "program_execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "program_schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": fingerprint,
        "programs_file": str(programs_file.relative_to(ARTIFACT_ROOT)),
        "selection_reused": True,
        "source_selection_dir": str(source_selection_dir.relative_to(ARTIFACT_ROOT)),
        "num_points": point_meta.get("n"),
        "support_count": selection_meta.get("support_count"),
        "selected_count": selection_meta.get("selected_count"),
        "selected_policy": selection_meta.get("selected_policy"),
        "support_count_exceeds_budget": selection_meta.get("support_count_exceeds_budget"),
        "max_keep": selection_meta.get("max_keep"),
        "extra_sampled_count": selection_meta.get("extra_sampled_count"),
        "stratified_count": selection_meta.get("stratified_count"),
        "farthest_count": selection_meta.get("farthest_count"),
        "grid_cell_size": selection_meta.get("grid_cell_size"),
        "grid_cell_diameter_upper_bound": selection_meta.get("grid_cell_diameter_upper_bound"),
        "metrics": metrics,
    }


def _existing_matches(
    summary_path: Path,
    fingerprint: str,
    algorithm_name: str,
    program_ids: list[int],
) -> bool:
    if not summary_path.exists():
        return False
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "three_assert_upper_bound",
        "program_execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "program_schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": fingerprint,
        "selection_algorithm": algorithm_name,
        "selection_reused": True,
        "program_ids": program_ids,
    }
    mismatches = {
        key: (existing.get(key), value)
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing result {summary_path.parent} mismatches: {mismatches}")
    return True


def run_sample(
    sample_dir: Path,
    programs: list[dict],
    programs_file: Path,
    fingerprint: str,
    algorithm_names: list[str],
    force: bool,
) -> None:
    points_path = sample_dir / "triangle_points.npy"
    point_meta_path = sample_dir / "triangle_meta.json"
    if not points_path.exists() or not point_meta_path.exists():
        print(f"[missing points] {sample_dir}")
        return

    pending: list[tuple[str, Path, Path, dict]] = []
    for algorithm_name in algorithm_names:
        source_dir = sample_dir / SOURCE_OUTPUT_DIR_NAME / algorithm_name
        indices_path = source_dir / "selected_point_indices.npy"
        selection_meta_path = source_dir / "selected_points_meta.json"
        if not indices_path.exists() or not selection_meta_path.exists():
            raise FileNotFoundError(f"Missing reused selection for {sample_dir} {algorithm_name}")
        output_dir = sample_dir / OUTPUT_DIR_NAME / algorithm_name
        summary_path = output_dir / "sample_summary.json"
        results_path = output_dir / "three_assert_upper_results.json"
        if summary_path.exists() and results_path.exists() and not force:
            if _existing_matches(
                summary_path, fingerprint, algorithm_name,
                [int(program["program_id"]) for program in programs],
            ):
                print(f"[skip] {sample_dir} {algorithm_name}")
                continue
        selection_meta = json.loads(selection_meta_path.read_text(encoding="utf-8"))
        pending.append((algorithm_name, source_dir, output_dir, selection_meta))

    if not pending:
        return

    points = np.load(points_path)
    point_meta = json.loads(point_meta_path.read_text(encoding="utf-8"))
    support_indices = convex_hull_indices(points)
    generator_points = points[support_indices]
    references = [compute_three_assert_reference(program, points) for program in programs]

    for algorithm_name, source_dir, output_dir, selection_meta in pending:
        selected_indices = np.load(source_dir / "selected_point_indices.npy")
        selected_points = points[selected_indices]
        rows: list[dict] = []
        for program, reference in zip(programs, references):
            rows.extend(
                compute_three_assert_one(
                    program,
                    selected_points,
                    generator_points,
                    reference=reference,
                )
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "three_assert_upper_results.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
        )
        summary = build_summary(
            rows,
            point_meta,
            selection_meta,
            algorithm_name,
            programs_file,
            fingerprint,
            source_dir,
        )
        (output_dir / "sample_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(
            f"[done] {point_meta.get('distribution')} sample_{int(point_meta.get('sample_id') or 0):03d} "
            f"{algorithm_name}: rows={len(rows)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three-assertion upper-bound experiments.")
    parser.add_argument("--points-root", default="point_distribution_outputs")
    parser.add_argument("--programs-file", default="triangle_lower_bound_outputs/three_assert_affine_programs.json")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--selection-algorithms", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=100)
    parser.add_argument("--program-start", type=int, default=0)
    parser.add_argument("--program-end", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    programs_file = root / args.programs_file
    programs = read_three_assert_programs(programs_file)
    fingerprint = three_assert_program_collection_fingerprint(programs)
    program_end = len(programs) if args.program_end is None else min(args.program_end, len(programs))
    programs = programs[args.program_start:program_end]
    if not programs:
        raise ValueError("The selected program range is empty.")
    distributions = parse_distributions(args.distributions)
    algorithm_names = [algorithm.name for algorithm in parse_selection_algorithms(args.selection_algorithms)]

    for distribution in distributions:
        for sample_id in range(args.sample_start, args.sample_end):
            run_sample(
                root / args.points_root / distribution / f"sample_{sample_id:03d}",
                programs,
                programs_file,
                fingerprint,
                algorithm_names,
                args.force,
            )


if __name__ == "__main__":
    main()
