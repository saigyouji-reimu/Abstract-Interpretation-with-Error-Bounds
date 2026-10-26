"""
generate_triangle_points.py

Generate point clouds inside the fixed equilateral triangle centered at the
origin. The script supports the original single-sample output as well as a
batch mode for multiple representative distributions.

Triangle:
  top vertex:        (0, 8)
  bottom-left:       (-4*sqrt(3), -4)
  bottom-right:      ( 4*sqrt(3), -4)

Single-sample output:
  <out_dir>/triangle_points.npy
  <out_dir>/triangle_meta.json

Batch output:
  <batch_out_dir>/<distribution>/sample_<id>/triangle_points.npy
  <batch_out_dir>/<distribution>/sample_<id>/triangle_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from core.geometry import polygon_area

DISTRIBUTIONS = [
    "triangle_uniform",
    "truncated_gaussian_center",
    "gaussian_mixture",
    "edge_biased_dirichlet",
    "center_biased_dirichlet",
    "vertex_biased_dirichlet",
]


def triangle_vertices() -> np.ndarray:
    """Return the vertices of the equilateral triangle."""
    return np.array(
        [
            [0.0, 8.0],
            [-4.0 * np.sqrt(3.0), -4.0],
            [4.0 * np.sqrt(3.0), -4.0],
        ],
        dtype=float,
    )


def from_barycentric(weights: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return weights @ vertices


def sample_uniform_triangle_rng(n: int, rng: np.random.Generator) -> np.ndarray:
    vertices = triangle_vertices()
    weights = rng.dirichlet(np.ones(3), size=n)
    return from_barycentric(weights, vertices)


def sample_uniform_triangle(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    rng = np.random.default_rng(seed)
    return sample_uniform_triangle_rng(n, rng)


def points_in_triangle(points: np.ndarray, vertices: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a, b, c = vertices
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= -eps) & (v >= -eps) & (w >= -eps)


def sample_truncated_gaussian(
    n: int,
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    vertices = triangle_vertices()
    accepted: list[np.ndarray] = []
    remaining = n
    while remaining > 0:
        batch_size = max(remaining * 3, 4096)
        batch = rng.multivariate_normal(mean, cov, size=batch_size)
        inside = batch[points_in_triangle(batch, vertices)]
        if len(inside) > 0:
            accepted.append(inside[:remaining])
            remaining -= min(remaining, len(inside))
    return np.vstack(accepted)[:n]


def sample_gaussian_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    vertices = triangle_vertices()
    centers = np.array(
        [
            0.62 * vertices[0] + 0.19 * vertices[1] + 0.19 * vertices[2],
            0.18 * vertices[0] + 0.64 * vertices[1] + 0.18 * vertices[2],
            0.18 * vertices[0] + 0.18 * vertices[1] + 0.64 * vertices[2],
        ],
        dtype=float,
    )
    covs = [
        np.array([[1.4, 0.0], [0.0, 1.1]], dtype=float),
        np.array([[1.0, 0.35], [0.35, 0.8]], dtype=float),
        np.array([[1.0, -0.35], [-0.35, 0.8]], dtype=float),
    ]
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    accepted: list[np.ndarray] = []
    remaining = n
    while remaining > 0:
        batch_size = max(remaining * 3, 4096)
        comps = rng.choice(3, size=batch_size, p=weights)
        batch = np.empty((batch_size, 2), dtype=float)
        for comp in range(3):
            mask = comps == comp
            if np.any(mask):
                batch[mask] = rng.multivariate_normal(centers[comp], covs[comp], size=int(np.sum(mask)))
        inside = batch[points_in_triangle(batch, vertices)]
        if len(inside) > 0:
            accepted.append(inside[:remaining])
            remaining -= min(remaining, len(inside))
    return np.vstack(accepted)[:n]


def sample_edge_biased_dirichlet(n: int, rng: np.random.Generator) -> np.ndarray:
    vertices = triangle_vertices()
    weights = np.empty((n, 3), dtype=float)
    edges = rng.integers(0, 3, size=n)
    t = rng.random(n)
    # Small positive distance from the chosen edge; this biases samples toward
    # edges without concentrating almost entirely at vertices.
    eps = rng.beta(0.6, 8.0, size=n) * 0.35
    for i, edge in enumerate(edges):
        w = np.zeros(3, dtype=float)
        if edge == 0:
            w[0] = eps[i]
            w[1] = (1.0 - eps[i]) * t[i]
            w[2] = (1.0 - eps[i]) * (1.0 - t[i])
        elif edge == 1:
            w[1] = eps[i]
            w[0] = (1.0 - eps[i]) * t[i]
            w[2] = (1.0 - eps[i]) * (1.0 - t[i])
        else:
            w[2] = eps[i]
            w[0] = (1.0 - eps[i]) * t[i]
            w[1] = (1.0 - eps[i]) * (1.0 - t[i])
        weights[i] = w
    return from_barycentric(weights, vertices)


def sample_center_biased_dirichlet(n: int, rng: np.random.Generator) -> np.ndarray:
    vertices = triangle_vertices()
    weights = rng.dirichlet(np.array([6.0, 6.0, 6.0]), size=n)
    return from_barycentric(weights, vertices)


def sample_vertex_biased_dirichlet(n: int, rng: np.random.Generator) -> np.ndarray:
    vertices = triangle_vertices()
    weights = np.empty((n, 3), dtype=float)
    comps = rng.integers(0, 3, size=n)
    for comp in range(3):
        mask = comps == comp
        if not np.any(mask):
            continue
        alpha = np.ones(3, dtype=float) * 0.8
        alpha[comp] = 9.0
        weights[mask] = rng.dirichlet(alpha, size=int(np.sum(mask)))
    return from_barycentric(weights, vertices)


def sample_distribution(n: int, seed: int, distribution: str) -> tuple[np.ndarray, dict]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution}")

    rng = np.random.default_rng(seed)
    if distribution == "triangle_uniform":
        points = sample_uniform_triangle_rng(n, rng)
        params = {"barycentric_distribution": "Dirichlet(1, 1, 1)"}
    elif distribution == "truncated_gaussian_center":
        points = sample_truncated_gaussian(
            n=n,
            rng=rng,
            mean=np.array([0.0, 0.0], dtype=float),
            cov=np.array([[5.0, 0.0], [0.0, 5.0]], dtype=float),
        )
        params = {"mean": [0.0, 0.0], "cov": [[5.0, 0.0], [0.0, 5.0]], "truncated_to_triangle": True}
    elif distribution == "gaussian_mixture":
        points = sample_gaussian_mixture(n, rng)
        params = {"components": 3, "truncated_to_triangle": True}
    elif distribution == "edge_biased_dirichlet":
        points = sample_edge_biased_dirichlet(n, rng)
        params = {"description": "choose an edge uniformly, sample along it, then add a small Beta(0.6, 8.0)-scaled barycentric distance from the edge"}
    elif distribution == "center_biased_dirichlet":
        points = sample_center_biased_dirichlet(n, rng)
        params = {"barycentric_distribution": "Dirichlet(6, 6, 6)"}
    elif distribution == "vertex_biased_dirichlet":
        points = sample_vertex_biased_dirichlet(n, rng)
        params = {"description": "uniform mixture over vertices; chosen vertex has Dirichlet alpha 9.0 and others alpha 0.8"}
    else:
        raise AssertionError("unreachable")
    return points, params


def write_sample(out_dir: Path, points: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "triangle_points.npy", points)
    (out_dir / "triangle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_meta(n: int, seed: int, distribution: str, params: dict, sample_id: int | None) -> dict:
    vertices = triangle_vertices()
    return {
        "n": n,
        "seed": seed,
        "base_seed": None if sample_id is None else seed - sample_id,
        "sample_id": sample_id,
        "distribution": distribution,
        "distribution_params": params,
        "triangle_vertices": vertices.tolist(),
        "triangle_area_exact": polygon_area(vertices),
        "description": "Points sampled inside the fixed centered equilateral triangle.",
    }


def parse_distributions(raw: list[str]) -> list[str]:
    if len(raw) == 1 and raw[0] == "all":
        return list(DISTRIBUTIONS)
    out: list[str] = []
    for dist in raw:
        if dist not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution {dist}; choices are {DISTRIBUTIONS}")
        out.append(dist)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate triangle point clouds from several distributions.")
    parser.add_argument("--n", type=int, default=100_000, help="Number of initial points per sample.")
    parser.add_argument("--seed", type=int, default=283, help="Random seed for single-sample mode.")
    parser.add_argument("--distribution", type=str, default="triangle_uniform", choices=DISTRIBUTIONS)
    parser.add_argument("--out-dir", type=str, default="triangle_lower_bound_outputs")
    parser.add_argument("--batch", action="store_true", help="Generate multiple samples per distribution.")
    parser.add_argument("--batch-out-dir", type=str, default="point_distribution_outputs")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=283)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    args = parser.parse_args()

    if not args.batch:
        points, params = sample_distribution(args.n, args.seed, args.distribution)
        meta = build_meta(args.n, args.seed, args.distribution, params, sample_id=None)
        write_sample(Path(args.out_dir), points, meta)
        print(f"Saved points: {(Path(args.out_dir) / 'triangle_points.npy').resolve()}")
        print(f"Saved meta:   {(Path(args.out_dir) / 'triangle_meta.json').resolve()}")
        print(f"Distribution: {args.distribution}")
        return

    distributions = parse_distributions(args.distributions)
    sample_end = args.num_samples if args.sample_end is None else args.sample_end
    if args.sample_start < 0 or sample_end > args.num_samples or args.sample_start >= sample_end:
        raise ValueError("Invalid sample range.")

    root = Path(args.batch_out_dir)
    written = 0
    for distribution in distributions:
        for sample_id in range(args.sample_start, sample_end):
            seed = args.base_seed + sample_id
            points, params = sample_distribution(args.n, seed, distribution)
            meta = build_meta(args.n, seed, distribution, params, sample_id=sample_id)
            out_dir = root / distribution / f"sample_{sample_id:03d}"
            write_sample(out_dir, points, meta)
            written += 1
            print(f"[{distribution} sample_{sample_id:03d}] seed={seed} saved to {out_dir}")
    print(f"Generated {written} point-cloud samples under {root.resolve()}")


if __name__ == "__main__":
    main()
