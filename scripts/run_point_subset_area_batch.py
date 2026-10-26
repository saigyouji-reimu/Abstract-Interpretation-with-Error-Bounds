"""
run_point_subset_area_batch.py

Batch runner for the point-subset area-retention upper-bound experiment.

It traverses generated point-cloud samples, runs each requested point-selection
algorithm for each sample, evaluates all reused affine+guard programs, and
writes one independent result directory per (sample, algorithm) so interrupted
runs can resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.programs import (  # noqa: E402
    EXECUTION_ORDER,
    PROGRAM_SCHEMA_VERSION,
    program_collection_fingerprint,
    read_programs,
)
from evaluate_point_subset_area import compute_one  # noqa: E402
from select_space_filling_points import select_points  # noqa: E402

DISTRIBUTIONS = [
    "triangle_uniform",
    "truncated_gaussian_center",
    "gaussian_mixture",
    "edge_biased_dirichlet",
    "center_biased_dirichlet",
    "vertex_biased_dirichlet",
]

SUMMARY_METRICS = [
    "true_area",
    "selected_area",
    "area_retention_ratio",
    "area_gap",
    "support_only_area",
    "support_only_area_retention_ratio",
    "support_only_area_gap",
    "box_abstract_area",
    "box_abstraction_loss_area",
    "sampling_loss_to_box_loss_ratio",
    "support_only_loss_to_box_loss_ratio",
    "relative_area_gap",
    "support_only_relative_area_gap",
    "num_true_survivors",
    "num_selected_survivors",
    "num_support_only_survivors",
]


@dataclass(frozen=True)
class SelectionAlgorithm:
    name: str
    selection_method: str
    grid_kind: str | None = None


SELECTION_ALGORITHMS = [
    SelectionAlgorithm(
        name="centroid_triangulated",
        selection_method="centroid_triangulated",
    ),
    SelectionAlgorithm(
        name="grid_cover_square",
        selection_method="grid_cover",
        grid_kind="square",
    ),
    SelectionAlgorithm(
        name="grid_cover_triangle",
        selection_method="grid_cover",
        grid_kind="triangle",
    ),
]
SELECTION_ALGORITHM_BY_NAME = {algorithm.name: algorithm for algorithm in SELECTION_ALGORITHMS}


def parse_distributions(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0] == "all":
        return list(DISTRIBUTIONS)
    for dist in raw:
        if dist not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution {dist}; choices are {DISTRIBUTIONS}")
    return raw


def parse_selection_algorithms(raw: list[str]) -> list[SelectionAlgorithm]:
    if len(raw) == 1 and raw[0] == "all":
        return list(SELECTION_ALGORITHMS)
    algorithms = []
    for name in raw:
        if name not in SELECTION_ALGORITHM_BY_NAME:
            choices = ["all"] + list(SELECTION_ALGORITHM_BY_NAME)
            raise ValueError(f"Unknown selection algorithm {name}; choices are {choices}")
        algorithms.append(SELECTION_ALGORITHM_BY_NAME[name])
    return algorithms


def summarize_values(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {k: float("nan") for k in ["min", "p10", "p25", "mean", "median", "p75", "p90", "max", "std"]} | {"count": int(len(values)), "finite_count": 0}
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def build_sample_summary(
    rows: list[dict],
    point_meta: dict,
    selection_meta: dict,
    algorithm: SelectionAlgorithm,
    programs_file: Path,
    program_fingerprint: str,
    selection_config: dict,
) -> dict:
    summary = {
        "distribution": point_meta.get("distribution"),
        "sample_id": point_meta.get("sample_id"),
        "point_seed": point_meta.get("seed"),
        "selection_algorithm": algorithm.name,
        "selection_method_arg": algorithm.selection_method,
        "grid_kind_arg": algorithm.grid_kind,
        "selection_method": selection_meta.get("selection_method"),
        "grid_kind": selection_meta.get("grid_kind"),
        "selection_mode": selection_meta.get("selection_mode"),
        "target_h": selection_meta.get("target_h"),
        "achieved_h": selection_meta.get("achieved_h"),
        "is_h_cover": selection_meta.get("is_h_cover"),
        "grid_cell_size": selection_meta.get("grid_cell_size"),
        "grid_cell_diameter_upper_bound": selection_meta.get("grid_cell_diameter_upper_bound"),
        "num_programs": len(rows),
        "program_execution_order": (rows[0].get("execution_order") if rows else None),
        "program_schema_version": PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": program_fingerprint,
        "programs_file": str(programs_file),
        "program_ids": [int(row["program_id"]) for row in rows],
        "selection_config": selection_config,
        "num_points": point_meta.get("n"),
        "support_count": selection_meta.get("support_count"),
        "selected_count": selection_meta.get("selected_count"),
        "selected_policy": selection_meta.get("selected_policy"),
        "support_count_exceeds_budget": selection_meta.get("support_count_exceeds_budget"),
        "max_keep": selection_meta.get("max_keep"),
        "extra_sampled_count": selection_meta.get("extra_sampled_count"),
        "stratified_count": selection_meta.get("stratified_count"),
        "farthest_count": selection_meta.get("farthest_count"),
        "metrics": {},
    }
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary["metrics"][metric] = summarize_values(values)
    return summary


def run_one_sample_algorithm(
    sample_dir: Path,
    programs: list[dict],
    programs_file: Path,
    program_fingerprint: str,
    algorithm: SelectionAlgorithm,
    max_keep: int,
    base_seed: int,
    grid_bins: int | None,
    stratum_scale: float,
    top_k: int,
    cover_h: float | None,
    force: bool,
) -> None:
    output_dir = sample_dir / "upper_bound_outputs" / algorithm.name
    results_path = output_dir / "point_subset_area_results.json"
    summary_path = output_dir / "sample_summary.json"
    selection_config = {
        "max_keep": int(max_keep),
        "base_seed": int(base_seed),
        "grid_bins": None if grid_bins is None else int(grid_bins),
        "stratum_scale": float(stratum_scale),
        "top_k": int(top_k),
        "cover_h": None if cover_h is None else float(cover_h),
        "selection_algorithm": algorithm.name,
    }
    if results_path.exists() and summary_path.exists() and not force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "program_execution_order": EXECUTION_ORDER,
            "program_schema_version": PROGRAM_SCHEMA_VERSION,
            "program_fingerprint": program_fingerprint,
            "selection_algorithm": algorithm.name,
            "selection_config": selection_config,
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing result at {output_dir} belongs to a different experiment: "
                f"{mismatches}. Use --force only if replacement is intended."
            )
        print(f"[skip] {sample_dir} {algorithm.name} already has matching results")
        return

    points_path = sample_dir / "triangle_points.npy"
    meta_path = sample_dir / "triangle_meta.json"
    if not points_path.exists() or not meta_path.exists():
        print(f"[missing] {sample_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.load(points_path)
    point_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_id = int(point_meta.get("sample_id") or 0)
    selection_seed = base_seed + sample_id

    selected_indices, selection_meta = select_points(
        points=points,
        max_keep=max_keep,
        seed=selection_seed,
        grid_bins=grid_bins,
        top_k=top_k,
        stratum_scale=stratum_scale,
        selection_method=algorithm.selection_method,
        cover_h=cover_h if algorithm.selection_method == "grid_cover" else None,
        grid_kind=algorithm.grid_kind or "square",
    )
    selected_points = points[selected_indices]
    # compute_one needs an explicit support-only point set. Reuse the leading
    # support_count entries because select_points always stores support points first.
    support_only_points = points[selected_indices[: int(selection_meta["support_count"])]]
    rows = [compute_one(program, points, selected_points, support_only_points) for program in programs]

    np.save(output_dir / "selected_point_indices.npy", selected_indices)
    np.save(output_dir / "selected_points.npy", selected_points)
    (output_dir / "selected_points_meta.json").write_text(
        json.dumps(selection_meta, indent=2, allow_nan=True), encoding="utf-8"
    )
    results_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    sample_summary = build_sample_summary(
        rows,
        point_meta,
        selection_meta,
        algorithm,
        programs_file,
        program_fingerprint,
        selection_config,
    )
    summary_path.write_text(json.dumps(sample_summary, indent=2, allow_nan=True), encoding="utf-8")
    print(
        f"[done] {point_meta.get('distribution')} sample_{sample_id:03d} {algorithm.name}: "
        f"selected={selection_meta['selected_count']} support={selection_meta['support_count']} "
        f"policy={selection_meta.get('selected_policy')}"
    )


def run_one_sample(
    sample_dir: Path,
    programs: list[dict],
    programs_file: Path,
    program_fingerprint: str,
    algorithms: list[SelectionAlgorithm],
    max_keep: int,
    base_seed: int,
    grid_bins: int | None,
    stratum_scale: float,
    top_k: int,
    cover_h: float | None,
    force: bool,
) -> None:
    for algorithm in algorithms:
        run_one_sample_algorithm(
            sample_dir=sample_dir,
            programs=programs,
            programs_file=programs_file,
            program_fingerprint=program_fingerprint,
            algorithm=algorithm,
            max_keep=max_keep,
            base_seed=base_seed,
            grid_bins=grid_bins,
            stratum_scale=stratum_scale,
            top_k=top_k,
            cover_h=cover_h,
            force=force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch upper-bound point-subset experiments.")
    parser.add_argument("--points-root", type=str, default="point_distribution_outputs")
    parser.add_argument("--programs-file", type=str, default="triangle_lower_bound_outputs/affine_programs.json")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--selection-algorithms", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=100)
    parser.add_argument("--max-keep", type=int, default=800)
    parser.add_argument("--base-seed", type=int, default=283)
    parser.add_argument("--cover-h", type=float, default=None)
    parser.add_argument("--grid-bins", type=int, default=None)
    parser.add_argument("--stratum-scale", type=float, default=1.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    points_root = root / args.points_root
    programs_file = root / args.programs_file
    programs = read_programs(programs_file)
    program_fingerprint = program_collection_fingerprint(programs)
    distributions = parse_distributions(args.distributions)
    algorithms = parse_selection_algorithms(args.selection_algorithms)

    for distribution in distributions:
        for sample_id in range(args.sample_start, args.sample_end):
            sample_dir = points_root / distribution / f"sample_{sample_id:03d}"
            run_one_sample(
                sample_dir=sample_dir,
                programs=programs,
                programs_file=programs_file,
                program_fingerprint=program_fingerprint,
                algorithms=algorithms,
                max_keep=args.max_keep,
                base_seed=args.base_seed,
                grid_bins=args.grid_bins,
                stratum_scale=args.stratum_scale,
                top_k=args.top_k,
                cover_h=args.cover_h,
                force=args.force,
            )


if __name__ == "__main__":
    main()
