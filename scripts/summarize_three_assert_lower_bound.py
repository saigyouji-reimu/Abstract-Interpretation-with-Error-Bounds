"""Aggregate three-assertion lower-bound feedback results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.three_assert import THREE_ASSERT_EXECUTION_ORDER, THREE_ASSERT_PROGRAM_SCHEMA_VERSION  # noqa: E402
from run_point_subset_area_batch import parse_distributions, summarize_values  # noqa: E402

FINAL_METRICS = [
    "old_guard1_recovery_ratio",
    "assert1_recovery_ratio",
    "assert2_recovery_ratio",
    "combined_recovery_ratio",
    "assert3_assert1_recovery_ratio",
    "assert3_assert2_recovery_ratio",
    "assert3_combined_recovery_ratio",
    "assert1_useful_count",
    "assert2_useful_count",
    "total_useful_count",
    "guard1_true_loss",
    "assert3_true_loss",
]


def _stats(series: pd.Series) -> dict:
    return summarize_values(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))


def _md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rendered = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def collect(batch_dir: Path, distributions: list[str], sample_start: int, sample_end: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    final_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    fingerprints: set[str] = set()
    constraints_per_site: set[int] = set()
    for distribution in distributions:
        for sample_id in range(sample_start, sample_end):
            sample_dir = batch_dir / distribution / f"sample_{sample_id:03d}"
            if not sample_dir.exists():
                continue
            for summary_path in sorted(sample_dir.glob("program_*/pair_summary.json")):
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if payload.get("program_execution_order") != THREE_ASSERT_EXECUTION_ORDER:
                    raise ValueError(f"Unexpected execution order in {summary_path}")
                if payload.get("program_schema_version") != THREE_ASSERT_PROGRAM_SCHEMA_VERSION:
                    raise ValueError(f"Unexpected schema in {summary_path}")
                fingerprints.add(str(payload["program_fingerprint"]))
                constraints_per_site.add(int(payload["constraints_per_site"]))
                final = payload["final"]
                row = {
                    "distribution": distribution,
                    "sample_id": sample_id,
                    "program_id": int(payload["program_id"]),
                    "constraints_per_site": int(payload["constraints_per_site"]),
                    **final,
                }
                final_rows.append(row)
                checkpoints = payload.get("checkpoints", {})
                for axis, rows in checkpoints.items():
                    for cp in rows:
                        checkpoint_rows.append({"distribution": distribution, "sample_id": sample_id, "program_id": int(payload["program_id"]), "axis": axis, **cp})
    if not final_rows:
        raise RuntimeError("No three-assert lower-bound results found.")
    if len(fingerprints) != 1:
        raise ValueError(f"Expected one program fingerprint, found {fingerprints}")
    final_df = pd.DataFrame(final_rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    summary = {
        "experiment": "three_assert_lower_bound_feedback",
        "program_execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "program_schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": next(iter(fingerprints)),
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "constraints_per_site_values": sorted(constraints_per_site),
        "num_pairs": int(len(final_df)),
    }
    return final_df, checkpoint_df, summary


def aggregate_final(final_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for distribution, group in final_df.groupby("distribution", sort=False):
        row = {"distribution": distribution, "num_pairs": int(len(group))}
        for metric in FINAL_METRICS:
            if metric in group:
                stats = _stats(group[metric])
                for stat in ["mean", "median", "p25", "p75", "min", "max"]:
                    row[f"{metric}_{stat}"] = stats[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_checkpoints(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    if checkpoint_df.empty:
        return pd.DataFrame()
    rows = []
    metrics = [
        "combined_recovery_ratio",
        "assert3_combined_recovery_ratio",
        "assert1_useful_count",
        "assert2_useful_count",
        "total_useful_count",
    ]
    for keys, group in checkpoint_df.groupby(["axis", "checkpoint"], sort=False):
        axis, checkpoint = keys
        row = {"axis": axis, "checkpoint": checkpoint, "num_rows": int(len(group))}
        for metric in metrics:
            if metric in group:
                stats = _stats(group[metric])
                for stat in ["mean", "median", "p25", "p75"]:
                    row[f"{metric}_{stat}"] = stats[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def draw_checked_curve(checkpoint_summary: pd.DataFrame, ax) -> None:
    data = checkpoint_summary[checkpoint_summary["axis"] == "checked_count"].sort_values("checkpoint")
    for metric, label, color in [
        ("combined_recovery_ratio", "guard1 combined", "tab:blue"),
        ("assert3_combined_recovery_ratio", "assert3 combined", "tab:green"),
    ]:
        mean = data[f"{metric}_mean"].to_numpy(dtype=float)
        p25 = data[f"{metric}_p25"].to_numpy(dtype=float)
        p75 = data[f"{metric}_p75"].to_numpy(dtype=float)
        x = data["checkpoint"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", label=label, color=color)
        ax.fill_between(x, p25, p75, color=color, alpha=0.15)
    ax.set_xlabel("constraints checked per site")
    ax.set_ylabel("recovery ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_checked_curve(checkpoint_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_checked_curve(checkpoint_summary, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_useful_curve(checkpoint_summary: pd.DataFrame, ax) -> None:
    data = checkpoint_summary[checkpoint_summary["axis"] == "checked_count"].sort_values("checkpoint")
    ax.plot(data["checkpoint"], data["assert1_useful_count_mean"], marker="o", label="assert1 useful")
    ax.plot(data["checkpoint"], data["assert2_useful_count_mean"], marker="o", label="assert2 useful")
    ax.set_xlabel("constraints checked per site")
    ax.set_ylabel("mean useful constraints")
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_useful_curve(checkpoint_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_useful_curve(checkpoint_summary, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_box(final_df: pd.DataFrame, path: Path) -> None:
    data = [
        final_df["old_guard1_recovery_ratio"].dropna(),
        final_df["assert1_recovery_ratio"].dropna(),
        final_df["assert2_recovery_ratio"].dropna(),
        final_df["combined_recovery_ratio"].dropna(),
        final_df["assert3_combined_recovery_ratio"].dropna(),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.boxplot(data, tick_labels=["old g1", "a1->g1", "a2->g1", "combined g1", "combined a3"])
    ax.set_ylabel("recovery ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_notebook(path: Path, summary: dict, aggregate: pd.DataFrame, figures_dir: Path, out_dir: Path) -> None:
    compact = aggregate[[
        "distribution",
        "num_pairs",
        "combined_recovery_ratio_mean",
        "combined_recovery_ratio_median",
        "assert3_combined_recovery_ratio_mean",
        "assert3_combined_recovery_ratio_median",
        "total_useful_count_mean",
    ]].copy()
    nb = {
        "cells": [
            _md_cell(
                "# Three-Assertion Lower-Bound Feedback Experiment\n\n"
                "This notebook summarizes posterior feedback from `assert1` and `assert2`, "
                "pulled back to the program point after `affine1; G1`, and then propagated "
                "forward to the outer-merge `assert3`.\n\n"
                f"Program fingerprint: `{summary['program_fingerprint']}`. Pairs: **{summary['num_pairs']}**. "
                f"Constraints per site: `{summary['constraints_per_site_values']}`."
            ),
            _md_cell("## Distribution Summary\n\n" + _markdown_table(compact)),
            _md_cell(
                "## Figures\n\n"
                f"![checked recovery](../{figures_dir.name}/checked_recovery.png)\n\n"
                f"![useful counts](../{figures_dir.name}/useful_counts.png)\n\n"
                f"![final recovery](../{figures_dir.name}/final_recovery_boxplot.png)"
            ),
            _code_cell(
                "from pathlib import Path\nimport pandas as pd\n"
                "ROOT = Path.cwd()\n"
                f"summary_dir = ROOT.parent / '{out_dir.name}' if ROOT.name == 'notebooks' else ROOT / '{out_dir.name}'\n"
                "final_df = pd.read_csv(summary_dir / 'three_assert_lower_bound_pair_final_metrics.csv.gz')\n"
                "checkpoint_df = pd.read_csv(summary_dir / 'three_assert_lower_bound_checkpoint_summary.csv')\n"
                "final_df\n"
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize three-assert lower-bound feedback results.")
    parser.add_argument("--batch-dir", default="three_assert_lower_bound_outputs/batch")
    parser.add_argument("--out-dir", default="three_assert_lower_bound_summary")
    parser.add_argument("--figures-dir", default="three_assert_lower_bound_figures")
    parser.add_argument("--notebook", default="notebooks/three_assert_lower_bound_experiment.ipynb")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=100)
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    distributions = parse_distributions(args.distributions)
    final_df, checkpoint_df, summary = collect(root / args.batch_dir, distributions, args.sample_start, args.sample_end)
    aggregate = aggregate_final(final_df)
    checkpoint_summary = aggregate_checkpoints(checkpoint_df)

    out_dir = root / args.out_dir
    figures_dir = root / args.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(
        out_dir / "three_assert_lower_bound_pair_final_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    checkpoint_df.to_csv(
        out_dir / "three_assert_lower_bound_checkpoint_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    aggregate.to_csv(out_dir / "three_assert_lower_bound_distribution_summary.csv", index=False)
    checkpoint_summary.to_csv(out_dir / "three_assert_lower_bound_checkpoint_summary.csv", index=False)
    (out_dir / "three_assert_lower_bound_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    plot_checked_curve(checkpoint_summary, figures_dir / "checked_recovery.png")
    plot_useful_curve(checkpoint_summary, figures_dir / "useful_counts.png")
    plot_final_box(final_df, figures_dir / "final_recovery_boxplot.png")
    write_notebook(root / args.notebook, summary, aggregate, figures_dir, out_dir)
    print(f"Saved summary: {out_dir}")
    print(f"Saved figures: {figures_dir}")
    print(f"Saved notebook: {root / args.notebook}")


if __name__ == "__main__":
    main()
