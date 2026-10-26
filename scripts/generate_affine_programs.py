"""Generate random guarded affine sequences.

The number of affine layers is parameterized by --num-affine-layers. Every
affine layer is immediately followed by the fixed quadrant guard x >= 0,
y >= 0. The schema and evaluators support any positive k.

Output:
  <out_dir>/affine_programs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from core.programs import generate_programs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random guarded affine sequences.")
    parser.add_argument("--m", type=int, default=100, help="Number of programs.")
    parser.add_argument("--seed", type=int, default=283, help="Random seed.")
    parser.add_argument("--num-affine-layers", type=int, default=2, help="Program complexity parameter k.")
    parser.add_argument("--a-low", type=float, default=-1.5)
    parser.add_argument("--a-high", type=float, default=1.5)
    parser.add_argument("--b-low", type=float, default=-2.0)
    parser.add_argument("--b-high", type=float, default=2.0)
    parser.add_argument("--min-abs-det", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="triangle_lower_bound_outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    programs = generate_programs(
        m=args.m,
        seed=args.seed,
        a_low=args.a_low,
        a_high=args.a_high,
        b_low=args.b_low,
        b_high=args.b_high,
        min_abs_det=args.min_abs_det,
        num_affine_layers=args.num_affine_layers,
    )

    out_path = out_dir / "affine_programs.json"
    out_path.write_text(json.dumps(programs, indent=2), encoding="utf-8")
    print(f"Saved programs: {out_path.resolve()}")
    print(f"Number of programs: {len(programs)}")
    print(f"Affine layers per program: {args.num_affine_layers}")


if __name__ == "__main__":
    main()
