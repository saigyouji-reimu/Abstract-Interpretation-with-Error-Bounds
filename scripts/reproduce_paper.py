"""Recompute paper Tables 1/2 and Figures 9/11 from reference or current rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from summarize_three_assert_upper_bound import _loss_arrow_plot, make_site_metric_table
from summarize_three_assert_lower_bound import (
    aggregate_checkpoints, draw_checked_curve, draw_useful_curve,
)

ROW_FILES = {
    "upper": "upper/three_assert_upper_bound_program_site_rows.csv.gz",
    "lower": "lower/three_assert_lower_bound_pair_final_metrics.csv.gz",
    "checkpoints": "lower/three_assert_lower_bound_checkpoint_rows.csv.gz",
}
LOWER_COLUMNS = {
    "p_t": {
        "true_loss": "guard1_true_loss",
        "initial_lower_bound": "old_guard1_lower_bound",
        "initial_recovery": "old_guard1_recovery_ratio",
        "posterior_lower_bound": "combined_corrected_lower_bound",
        "posterior_recovery": "combined_recovery_ratio",
        "q1_posterior_recovery": "assert1_recovery_ratio",
        "q2_posterior_recovery": "assert2_recovery_ratio",
    },
    "q_3": {
        "true_loss": "assert3_true_loss",
        "initial_lower_bound": "old_assert3_lower_bound",
        "initial_recovery": "old_assert3_recovery_ratio",
        "posterior_lower_bound": "assert3_combined_corrected_lower_bound",
        "posterior_recovery": "assert3_combined_recovery_ratio",
        "q1_posterior_recovery": "assert3_assert1_recovery_ratio",
        "q2_posterior_recovery": "assert3_assert2_recovery_ratio",
    },
}


def make_lower_table(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for location, columns in LOWER_COLUMNS.items():
        entry = {"location": location}
        for metric, column in columns.items():
            entry[f"{metric}_mean"] = rows[column].mean()
            entry[f"{metric}_median"] = rows[column].median()
        result.append(entry)
    return pd.DataFrame(result)


def table_tex(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             "\\caption{" + caption + "}", "\\label{" + label + "}",
             "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def write_tables(upper: pd.DataFrame, lower: pd.DataFrame, output: Path) -> None:
    upper.to_csv(output / "table1.csv", index=False)
    lower.to_csv(output / "table2.csv", index=False)
    upper_rows = []
    for row in upper.to_dict("records"):
        upper_rows.append([
            r"$\mathtt{assert}_{" + row["assert_site"][-1] + "}$",
            *[f"{row[name]:.2f}" for name in
              ("true_loss", "selected_upper_bound", "generator_upper_bound")],
            *[f"{100 * row[name]:.1f}" for name in
              ("selected_area_retention", "generator_area_retention")],
            *[f"{row[name]:.2f}" for name in ("selected_slack", "generator_slack")],
            f"{100 * row['slack_reduction_vs_generator']:.1f}",
        ])
    (output / "table1.tex").write_text(table_tex(
        ["Site", "True loss", "Rep. upper", "Gen. upper", r"Rep. ret. (\%)",
         r"Gen. ret. (\%)", "Rep. slack", "Gen. slack", r"Slack red. (\%)"],
        upper_rows,
        "Upper-bound results averaged over the three representative-selection policies. "
        "Ret. denotes area retention; slack is upper bound minus true loss.",
        "tab:artifact-upper"), encoding="utf-8")
    lower_rows = []
    for row in lower.to_dict("records"):
        cells = ["$" + row["location"] + "$"]
        for metric in LOWER_COLUMNS[row["location"]]:
            digits = 3 if "recovery" in metric else 2
            cells.append(f"{row[metric + '_mean']:.{digits}f} / "
                         f"{row[metric + '_median']:.{digits}f}")
        lower_rows.append(cells)
    (output / "table2.tex").write_text(table_tex(
        ["Location", "True loss", "Initial LB", "Initial rec.", "Post. LB",
         "Post. rec.", "$q_1$-post. rec.", "$q_2$-post. rec."], lower_rows,
        "Lower-bound results (mean / median). LB denotes lower bound and rec. denotes "
        "recovery. Post. uses both feedback sites; the last two columns use one site each.",
        "tab:artifact-lower"), encoding="utf-8")


def validate_rows(upper: pd.DataFrame, lower: pd.DataFrame, checkpoints: pd.DataFrame,
                  config: dict) -> None:
    pair_keys = ["distribution", "sample_id", "program_id"]
    expected = pd.MultiIndex.from_product([
        config["distributions"], range(config["sample_start"], config["sample_end"]),
        range(config["program_start"], config["program_end"]),
    ], names=pair_keys)

    def check_index(frame: pd.DataFrame, keys: list[str], wanted: pd.MultiIndex) -> None:
        actual = pd.MultiIndex.from_frame(frame[keys])
        if actual.has_duplicates or len(actual) != len(wanted) or len(wanted.difference(actual)):
            raise ValueError(f"Incomplete, duplicate, or unexpected rows for {keys}.")

    check_index(lower, pair_keys, expected)
    for site in ("assert1", "assert2", "assert3"):
        for algorithm in ("centroid_triangulated", "grid_cover_square", "grid_cover_triangle"):
            selected = upper[(upper["assert_site"] == site) & (upper["algorithm"] == algorithm)]
            check_index(selected, pair_keys, expected)
    if len(upper) != len(expected) * 9:
        raise ValueError("Unexpected upper-bound sites or selection algorithms.")
    checked = checkpoints[checkpoints["axis"] == "checked_count"]
    if checked.empty or set(checked["checkpoint"]) != {0, 100, 250, 500, 1000, 1500, 2000}:
        raise ValueError("Missing or unexpected checked-count checkpoints.")
    for _, group in checked.groupby("checkpoint"):
        check_index(group, pair_keys, expected)
    if set(lower["constraints_per_site"]) != {config["constraints_per_site"]}:
        raise ValueError("Feedback budget does not match the selected configuration.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(source: str, mode: str, output: Path | None = None) -> Path:
    effective_mode = "full" if source == "reference" else mode
    base = ROOT / "reference_results" if source == "reference" else ROOT / "results" / mode
    if output is None:
        output = ROOT / "results" / ("reference_report" if source == "reference" else mode) / "paper"
    paths = {key: base / name for key, name in ROW_FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run the report stage first for current results.")
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    config = json.loads((ROOT / "configs" / f"{effective_mode}.json").read_text())
    validate_rows(frames["upper"], frames["lower"], frames["checkpoints"], config)
    upper_table = make_site_metric_table(frames["upper"]).set_index("assert_site").loc[
        ["assert1", "assert2", "assert3"]].reset_index()
    lower_table = make_lower_table(frames["lower"])
    checkpoint_summary = aggregate_checkpoints(frames["checkpoints"])
    output.mkdir(parents=True, exist_ok=True)
    write_tables(upper_table, lower_table, output)
    checked = checkpoint_summary[checkpoint_summary["axis"] == "checked_count"].sort_values("checkpoint")
    checked = checked.copy()
    raw_checked = frames["checkpoints"][frames["checkpoints"]["axis"] == "checked_count"]
    for metric in ("combined_recovery_ratio", "assert3_combined_recovery_ratio"):
        counts = raw_checked.groupby("checkpoint")[metric].apply(lambda values: np.isfinite(values).sum())
        checked[metric + "_valid_count"] = checked["checkpoint"].map(counts)
    checked.to_csv(output / "figure11_data.csv", index=False)
    for suffix in ("png", "pdf"):
        _loss_arrow_plot(frames["upper"], output / f"figure9.{suffix}")
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8), layout="constrained")
        draw_checked_curve(checkpoint_summary, axes[0])
        draw_useful_curve(checkpoint_summary, axes[1])
        axes[0].set_title("(a) Recovery vs. checked constraints")
        axes[1].set_title("(b) Useful constraints")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"figure11.{suffix}", dpi=220)
        plt.close(fig)
    manifest = {
        "source": source, "mode": effective_mode, "config": config,
        "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                         "rows": len(frames[key])} for key, path in paths.items()},
        "aggregation": {
            "table1": "Means over input sets, programs and three policies; reduction = 1 - mean selected slack / mean generator slack.",
            "table2": "Mean and median of per-pair metrics; recovery ratios are not ratios of mean areas.",
            "figure9": "Mean upper-bound slack relative to true loss; the x-axis compresses the interval 10 to 30.",
            "figure11": "Checks per feedback site; mean recovery with p25-p75 band, and mean useful count per site.",
            "zero_loss_convention": "Historical final rows store initial recovery as zero when true loss is zero. "
                "The p_t checkpoint at zero checks instead leaves 0/0 undefined (600 reference pairs); "
                "Figure 11 excludes those undefined entries, as in the original paper. "
                "No historical values are rewritten; figure11_data.csv includes finite recovery counts.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Tables 1/2 and Figures 9/11 ({source}, {effective_mode}): {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["reference", "current"], default="reference")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    generate(args.source, args.mode, args.out_dir)


if __name__ == "__main__":
    main()
