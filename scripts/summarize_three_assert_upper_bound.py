"""Aggregate and report three-assertion upper-bound experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.three_assert import ASSERT_SITES, THREE_ASSERT_EXECUTION_ORDER, THREE_ASSERT_PROGRAM_SCHEMA_VERSION  # noqa: E402
from run_point_subset_area_batch import parse_distributions, parse_selection_algorithms, summarize_values  # noqa: E402
from run_three_assert_upper_bound_batch import OUTPUT_DIR_NAME, SUMMARY_METRICS  # noqa: E402

ALGORITHM_LABELS = {
    "centroid_triangulated": "Centroid",
    "grid_cover_square": "Square grid",
    "grid_cover_triangle": "Triangular grid",
}

DERIVED_UPPER_METRICS = [
    "selected_upper_loss_slack_vs_true",
    "generator_upper_loss_slack_vs_true",
    "selected_to_generator_slack_ratio",
    "slack_reduction_ratio_vs_generator",
]

LIGHT_WARM_CMAP = LinearSegmentedColormap.from_list(
    "light_warm_readable",
    ["#fffdf7", "#fff4c2", "#ffe0b5", "#ffd0d0"],
)


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


def _heatmap(
    frame: pd.DataFrame,
    value: str,
    title: str,
    path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.3f}",
    value_scale: float = 1.0,
    colorbar_label: str | None = None,
) -> None:
    table = frame.pivot_table(index="assert_site", columns="algorithm", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    values = table.to_numpy(dtype=float) * value_scale
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(
        range(len(table.columns)),
        [ALGORITHM_LABELS.get(c, c) for c in table.columns],
        rotation=12,
        ha="right",
        fontsize=16,
    )
    ax.set_yticks(range(len(table.index)), table.index, fontsize=17)
    ax.tick_params(axis="both", length=0)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            ax.text(
                j,
                i,
                fmt.format(values[i, j]),
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color="#24170f",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "#fffaf2", "edgecolor": "none", "alpha": 0.86},
            )
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    cbar.ax.tick_params(labelsize=14)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _loss_arrow_plot(program_df: pd.DataFrame, path: Path) -> None:
    rows = []
    for site, group in program_df.groupby("assert_site", sort=False):
        rows.append(
            {
                "assert_site": site,
                "selected_slack": float(group["selected_upper_loss_slack_vs_true"].mean()),
                "generator_slack": float(group["generator_upper_loss_slack_vs_true"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    y_positions = np.array([1.15, 0.58, 0.0])[: len(table)]
    fig, ax = plt.subplots(figsize=(14.6, 3.25))
    colors = {
        "true": "#1f5f99",
        "selected": "#d97925",
        "generator": "#8b4ba7",
    }

    # Piecewise display coordinate: [0,10] is expanded, [10,30] is compressed
    # into a short zigzag interval, and [30,32] is shown compactly.
    break_start = 10.0
    break_end = 30.0
    break_width = 0.95
    right_scale = 0.72

    def xpos(x: float) -> float:
        if x <= break_start:
            return x
        if x < break_end:
            return break_start + (x - break_start) / (break_end - break_start) * break_width
        return break_start + break_width + (x - break_end) * right_scale

    def draw_compressed_arrow(x_from: float, x_to: float, y: float, *, color: str, lw: float, linestyle: str = "-") -> None:
        # Direct connection when both endpoints lie in the same displayed region.
        if (x_from <= break_start and x_to <= break_start) or (x_from >= break_end and x_to >= break_end):
            ax.annotate(
                "",
                xy=(xpos(x_to), y),
                xytext=(xpos(x_from), y),
                arrowprops={"arrowstyle": "-|>", "lw": lw, "color": color, "linestyle": linestyle, "shrinkA": 7, "shrinkB": 8},
                zorder=2,
            )
            return

        # Otherwise draw a continuous path that crosses the compressed interval.
        descending = x_from >= x_to
        points: list[tuple[float, float]] = []
        if descending:
            points.append((xpos(x_from), y))
            if x_from > break_end:
                points.append((xpos(break_end), y))
            zxs = np.linspace(xpos(break_end), xpos(break_start), 7)
            amp = 0.035
            for k, zx in enumerate(zxs):
                points.append((float(zx), y + (amp if k % 2 else -amp)))
            points.append((xpos(break_start), y))
            points.append((xpos(x_to), y))
        else:
            points.append((xpos(x_from), y))
            if x_from < break_start:
                points.append((xpos(break_start), y))
            zxs = np.linspace(xpos(break_start), xpos(break_end), 7)
            amp = 0.035
            for k, zx in enumerate(zxs):
                points.append((float(zx), y + (amp if k % 2 else -amp)))
            points.append((xpos(break_end), y))
            points.append((xpos(x_to), y))

        cleaned: list[tuple[float, float]] = []
        for pt in points:
            if not cleaned or abs(cleaned[-1][0] - pt[0]) > 1e-9 or abs(cleaned[-1][1] - pt[1]) > 1e-9:
                cleaned.append(pt)
        if len(cleaned) < 2:
            return
        xs, ys = zip(*cleaned[:-1])
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, lw=lw, linestyle=linestyle, zorder=1.5)
        ax.annotate(
            "",
            xy=cleaned[-1],
            xytext=cleaned[-2],
            arrowprops={"arrowstyle": "-|>", "lw": lw, "color": color, "linestyle": linestyle, "shrinkA": 7, "shrinkB": 8},
            zorder=2,
        )

    for y, (_, row) in zip(y_positions, table.iterrows()):
        selected = float(row["selected_slack"])
        generator = float(row["generator_slack"])
        draw_compressed_arrow(generator, selected, y, color="#9a6a2f", lw=3.2)
        draw_compressed_arrow(selected, 0.0, y, color="#c9a36a", lw=2.3, linestyle="--")
        ax.scatter([xpos(0.0)], [y], s=135, color=colors["true"], edgecolor="white", linewidth=1.4, zorder=4)
        ax.scatter([xpos(selected)], [y], s=135, color=colors["selected"], edgecolor="white", linewidth=1.4, zorder=4)
        ax.scatter([xpos(generator)], [y], s=135, color=colors["generator"], edgecolor="white", linewidth=1.4, zorder=4)

    ax.set_yticks(y_positions, table["assert_site"], fontsize=17)
    ax.set_ylim(-0.35, 1.5)
    ax.set_xlim(-0.25, xpos(32.0) + 0.42)
    tick_values = list(range(0, 11, 1)) + [30, 32]
    ax.set_xticks([xpos(float(t)) for t in tick_values], [str(t) for t in tick_values], fontsize=14)
    ax.grid(axis="x", alpha=0.22)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("excess over true loss (area)", fontsize=16)
    ax.set_title("Slack relative to true loss", fontsize=20, fontweight="bold", pad=12)

    # Put the axis break directly on the x-axis, marking the compressed interval.
    axis_y = ax.get_ylim()[0]
    zxs = np.linspace(xpos(break_start), xpos(break_end), 9)
    amp = 0.055
    zys = [axis_y + (amp if k % 2 else 0.0) for k in range(len(zxs))]
    ax.plot(zxs, zys, color="#4a3b2a", lw=1.6, clip_on=False, zorder=5)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["true"], markeredgecolor="white", markersize=11, label="True loss"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["selected"], markeredgecolor="white", markersize=11, label="Representative slack"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["generator"], markeredgecolor="white", markersize=11, label="Generator slack"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=12.5, frameon=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.82, bottom=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_site_metric_table(program_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, group in program_df.groupby("assert_site", sort=False):
        true_loss = float(group["true_loss"].mean())
        selected_upper = float(group["selected_upper_loss"].mean())
        generator_upper = float(group["generator_upper_loss"].mean())
        selected_slack = float(group["selected_upper_loss_slack_vs_true"].mean())
        generator_slack = float(group["generator_upper_loss_slack_vs_true"].mean())
        rows.append(
            {
                "assert_site": site,
                "true_loss": true_loss,
                "selected_upper_bound": selected_upper,
                "generator_upper_bound": generator_upper,
                "selected_area_retention": float(group["area_retention_ratio"].mean()),
                "generator_area_retention": float(group["generator_retention_ratio"].mean()),
                "selected_slack": selected_slack,
                "generator_slack": generator_slack,
                "slack_reduction_vs_generator": 1.0 - selected_slack / generator_slack if generator_slack else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _site_boxplot(program_df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    algorithms = list(dict.fromkeys(program_df["algorithm"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, site in zip(axes, ASSERT_SITES):
        group = program_df[program_df["assert_site"] == site]
        ax.boxplot(
            [group[group["algorithm"] == alg][metric].dropna() for alg in algorithms],
            tick_labels=[ALGORITHM_LABELS.get(a, a) for a in algorithms],
        )
        ax.set_title(site)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(True, alpha=0.2)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def collect(points_root: Path, distributions: list[str], algorithms: list[str], sample_start: int, sample_end: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sample_rows: list[dict] = []
    program_rows: list[dict] = []
    fingerprints: set[str] = set()
    for distribution in distributions:
        for sample_id in range(sample_start, sample_end):
            for algorithm in algorithms:
                output_dir = points_root / distribution / f"sample_{sample_id:03d}" / OUTPUT_DIR_NAME / algorithm
                summary_path = output_dir / "sample_summary.json"
                results_path = output_dir / "three_assert_upper_results.json"
                if not summary_path.exists() or not results_path.exists():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("program_execution_order") != THREE_ASSERT_EXECUTION_ORDER:
                    raise ValueError(f"Unexpected execution order in {summary_path}")
                if summary.get("program_schema_version") != THREE_ASSERT_PROGRAM_SCHEMA_VERSION:
                    raise ValueError(f"Unexpected schema in {summary_path}")
                fingerprints.add(str(summary["program_fingerprint"]))
                rows = json.loads(results_path.read_text(encoding="utf-8"))
                sample_row = {
                    "distribution": distribution,
                    "sample_id": sample_id,
                    "algorithm": algorithm,
                    "selected_count": summary.get("selected_count"),
                    "support_count": summary.get("support_count"),
                    "achieved_h": summary.get("achieved_h"),
                }
                for site in ASSERT_SITES:
                    for metric in SUMMARY_METRICS:
                        stats = summary["metrics"][site][metric]
                        sample_row[f"{site}_{metric}_mean"] = stats["mean"]
                        sample_row[f"{site}_{metric}_median"] = stats["median"]
                sample_rows.append(sample_row)
                for row in rows:
                    program_rows.append({"distribution": distribution, "sample_id": sample_id, "algorithm": algorithm, **row})
    if not program_rows:
        raise RuntimeError("No three-assert upper-bound results found.")
    if len(fingerprints) != 1:
        raise ValueError(f"Expected one program fingerprint, found {fingerprints}")
    sample_df = pd.DataFrame(sample_rows)
    program_df = pd.DataFrame(program_rows)
    program_df["selected_upper_loss_slack_vs_true"] = program_df["selected_upper_loss"] - program_df["true_loss"]
    program_df["generator_upper_loss_slack_vs_true"] = program_df["generator_upper_loss"] - program_df["true_loss"]
    denominator = program_df["generator_upper_loss_slack_vs_true"].replace(0.0, np.nan)
    program_df["selected_to_generator_slack_ratio"] = program_df["selected_upper_loss_slack_vs_true"] / denominator
    program_df["slack_reduction_ratio_vs_generator"] = 1.0 - program_df["selected_to_generator_slack_ratio"]
    aggregate_rows: list[dict] = []
    for (distribution, algorithm, site), group in program_df.groupby(["distribution", "algorithm", "assert_site"], sort=False):
        row = {"distribution": distribution, "algorithm": algorithm, "assert_site": site, "num_rows": int(len(group))}
        for metric in [*SUMMARY_METRICS, *DERIVED_UPPER_METRICS]:
            stats = summarize_values(group[metric].to_numpy(dtype=float))
            for stat in ["mean", "median", "p10", "p25", "p75", "p90", "min", "max", "std"]:
                row[f"{metric}_{stat}"] = stats[stat]
        generator_slack_mean = row["generator_upper_loss_slack_vs_true_mean"]
        row["selected_to_generator_mean_slack_ratio"] = (
            row["selected_upper_loss_slack_vs_true_mean"] / generator_slack_mean
            if generator_slack_mean not in (0.0, np.nan)
            else np.nan
        )
        row["slack_reduction_mean_slack_ratio_vs_generator"] = (
            1.0 - row["selected_to_generator_mean_slack_ratio"]
            if np.isfinite(row["selected_to_generator_mean_slack_ratio"])
            else np.nan
        )
        aggregate_rows.append(row)
    unique_inputs = sample_df.drop_duplicates(["distribution", "sample_id"])
    payload = {
        "experiment": "three_assert_upper_bound",
        "program_execution_order": THREE_ASSERT_EXECUTION_ORDER,
        "program_schema_version": THREE_ASSERT_PROGRAM_SCHEMA_VERSION,
        "program_fingerprint": next(iter(fingerprints)),
        "sample_start": int(sample_start),
        "sample_end": int(sample_end),
        "num_sample_algorithm_rows": int(len(sample_df)),
        "num_program_site_rows": int(len(program_df)),
        "expected_sample_algorithm_rows": int(len(distributions) * (sample_end - sample_start) * len(algorithms)),
        "complete": bool(len(sample_df) == len(distributions) * (sample_end - sample_start) * len(algorithms)),
        "mean_generator_support_count_per_input": float(unique_inputs["support_count"].mean()),
        "median_generator_support_count_per_input": float(unique_inputs["support_count"].median()),
        "mean_selected_count": float(unique_inputs["selected_count"].mean()),
    }
    return sample_df, pd.DataFrame(aggregate_rows), {"summary": payload, "program_df": program_df}


def write_notebook(path: Path, summary: dict, table: pd.DataFrame, figures_dir: Path, summary_dir: Path) -> None:
    compact = table.copy()
    for col in ["selected_area_retention", "generator_area_retention", "slack_reduction_vs_generator"]:
        compact[col] = compact[col] * 100.0
    nb = {
        "cells": [
            _md_cell(
                "# Three-Assertion Upper-Bound Experiment\n\n"
                "This notebook summarizes the new program shape `affine1; if G1 then affine2; "
                "(if G2 then assert1 else affine3; assert2) else affine4; assert3`, "
                "with `G1: x >= 0`, `G2: y >= 0`, and Poly results boxed whenever a "
                "post-transfer polygon has more than five edges.\n\n"
                f"Program fingerprint: `{summary['program_fingerprint']}`. Completed "
                f"sample/algorithm groups: **{summary['num_sample_algorithm_rows']} / "
                f"{summary['expected_sample_algorithm_rows']}**. "
                f"The generator baseline uses **{summary['mean_generator_support_count_per_input']:.1f}** support points per input set on average, "
                f"while the representative policies use **{summary['mean_selected_count']:.0f}** selected points. "
                "The main upper-bound metrics are area retention, selected slack, and the fraction of generator slack removed by selected representatives. "
                "Selected slack is numerically equal to area gap, but is interpreted as upper-bound looseness. "
                "The aggregate table below averages over the three representative-selection policies."
            ),
            _md_cell("## Aggregate Metrics\n\n" + _markdown_table(compact)),
            _md_cell(
                "## Figures\n\n"
                "The table includes the selected slack over the true loss, the generator slack over the true loss, "
                "and their aggregate ratio. A smaller ratio means the representative set estimates the true loss "
                "much more tightly than the generator baseline.\n\n"
                f"![loss arrows](../{figures_dir.name}/loss_arrow_by_site.png)"

                ""
            ),
            _code_cell(
                "from pathlib import Path\nimport pandas as pd\n"
                "ROOT = Path.cwd()\n"
                f"summary_dir = ROOT.parent / '{summary_dir.name}' if ROOT.name == 'notebooks' else ROOT / '{summary_dir.name}'\n"
                "aggregate_df = pd.read_csv(summary_dir / 'three_assert_upper_bound_program_level_summary.csv')\n"
                "sample_df = pd.read_csv(summary_dir / 'three_assert_upper_bound_sample_level_metrics.csv')\n"
                "aggregate_df\n"
            ),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize three-assert upper-bound results.")
    parser.add_argument("--points-root", default="point_distribution_outputs")
    parser.add_argument("--out-dir", default="three_assert_upper_bound_summary")
    parser.add_argument("--figures-dir", default="three_assert_upper_bound_figures")
    parser.add_argument("--notebook", default="notebooks/three_assert_upper_bound_experiment.ipynb")
    parser.add_argument("--distributions", nargs="+", default=["all"])
    parser.add_argument("--selection-algorithms", nargs="+", default=["all"])
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=100)
    args = parser.parse_args()

    root = ARTIFACT_ROOT
    distributions = parse_distributions(args.distributions)
    algorithms = [a.name for a in parse_selection_algorithms(args.selection_algorithms)]
    sample_df, aggregate_df, extra = collect(root / args.points_root, distributions, algorithms, args.sample_start, args.sample_end)
    summary = extra["summary"]
    program_df = extra["program_df"]
    site_metric_df = make_site_metric_table(program_df)

    out_dir = root / args.out_dir
    figures_dir = root / args.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out_dir / "three_assert_upper_bound_sample_level_metrics.csv", index=False)
    aggregate_df.to_csv(out_dir / "three_assert_upper_bound_program_level_summary.csv", index=False)
    program_df.to_csv(
        out_dir / "three_assert_upper_bound_program_site_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    site_metric_df.to_csv(out_dir / "three_assert_upper_bound_site_metric_table.csv", index=False)
    (out_dir / "three_assert_upper_bound_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    _loss_arrow_plot(program_df, figures_dir / "loss_arrow_by_site.png")
    write_notebook(root / args.notebook, summary, site_metric_df, figures_dir, out_dir)
    print(f"Saved summary: {out_dir}")
    print(f"Saved figures: {figures_dir}")
    print(f"Saved notebook: {root / args.notebook}")


if __name__ == "__main__":
    main()
