"""Generate deterministic input clouds and representative selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_triangle_points import (  # noqa: E402
    build_meta,
    sample_distribution,
    write_sample,
)
from run_point_subset_area_batch import (  # noqa: E402
    parse_distributions,
    parse_selection_algorithms,
)
from select_space_filling_points import select_points  # noqa: E402


def points_fingerprint(points: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(points, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def prepare_sample(
    *,
    points_root: Path,
    distribution: str,
    sample_id: int,
    n: int,
    base_seed: int,
    max_keep: int,
    algorithms: list,
    grid_bins: int | None,
    stratum_scale: float,
    top_k: int,
    force: bool,
) -> None:
    sample_dir = points_root / distribution / f"sample_{sample_id:03d}"
    points_path = sample_dir / "triangle_points.npy"
    meta_path = sample_dir / "triangle_meta.json"
    seed = base_seed + sample_id

    if points_path.exists() and meta_path.exists() and not force:
        point_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"n": n, "seed": seed, "distribution": distribution}
        mismatches = {
            key: (point_meta.get(key), value)
            for key, value in expected.items()
            if point_meta.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing point cloud at {sample_dir} has incompatible metadata: "
                f"{mismatches}. Use --force to replace it."
            )
        points = np.load(points_path)
    else:
        points, params = sample_distribution(n, seed, distribution)
        point_meta = build_meta(n, seed, distribution, params, sample_id=sample_id)
        point_meta["points_fingerprint"] = points_fingerprint(points)
        write_sample(sample_dir, points, point_meta)

    fingerprint = point_meta.get("points_fingerprint") or points_fingerprint(points)
    for algorithm in algorithms:
        selection_seed = base_seed + sample_id
        selection_config = {
            "algorithm": algorithm.name,
            "max_keep": max_keep,
            "seed": selection_seed,
            "grid_bins": grid_bins,
            "stratum_scale": stratum_scale,
            "top_k": top_k,
            "points_fingerprint": fingerprint,
        }
        output_dir = sample_dir / "upper_bound_outputs" / algorithm.name
        indices_path = output_dir / "selected_point_indices.npy"
        selection_meta_path = output_dir / "selected_points_meta.json"
        if indices_path.exists() and selection_meta_path.exists() and not force:
            existing = json.loads(selection_meta_path.read_text(encoding="utf-8"))
            if existing.get("artifact_selection_config") != selection_config:
                raise RuntimeError(
                    f"Existing selection at {output_dir} has incompatible metadata. "
                    "Use --force to replace it."
                )
            print(f"[skip] {distribution} sample_{sample_id:03d} {algorithm.name}")
            continue

        indices, selection_meta = select_points(
            points=points,
            max_keep=max_keep,
            seed=selection_seed,
            grid_bins=grid_bins,
            top_k=top_k,
            stratum_scale=stratum_scale,
            selection_method=algorithm.selection_method,
            grid_kind=algorithm.grid_kind or "square",
        )
        selection_meta["artifact_selection_config"] = selection_config
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(indices_path, indices)
        selection_meta_path.write_text(
            json.dumps(selection_meta, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print(
            f"[done] {distribution} sample_{sample_id:03d} {algorithm.name}: "
            f"selected={len(indices)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate point clouds and representative selections."
    )
    parser.add_argument("--points-root", default="work/fast/points")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--selection-algorithms", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--base-seed", type=int, default=283)
    parser.add_argument("--max-keep", type=int, default=800)
    parser.add_argument("--grid-bins", type=int, default=None)
    parser.add_argument("--stratum-scale", type=float, default=1.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    points_root = ARTIFACT_ROOT / args.points_root
    distributions = parse_distributions(args.distributions)
    algorithms = parse_selection_algorithms(args.selection_algorithms)
    if args.sample_start < 0 or args.sample_start >= args.sample_end:
        raise ValueError("Invalid sample range.")

    for distribution in distributions:
        for sample_id in range(args.sample_start, args.sample_end):
            prepare_sample(
                points_root=points_root,
                distribution=distribution,
                sample_id=sample_id,
                n=args.n,
                base_seed=args.base_seed,
                max_keep=args.max_keep,
                algorithms=algorithms,
                grid_bins=args.grid_bins,
                stratum_scale=args.stratum_scale,
                top_k=args.top_k,
                force=args.force,
            )


if __name__ == "__main__":
    main()
