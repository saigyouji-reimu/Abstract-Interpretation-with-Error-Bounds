"""Regenerate paper-facing tables, figures, and notebooks from reference rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent
for path in (ARTIFACT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_three_assert_lower_bound import (  # noqa: E402
    aggregate_checkpoints,
    aggregate_final,
    plot_checked_curve,
    plot_final_box,
    plot_useful_curve,
    write_notebook as write_lower_notebook,
)
from summarize_three_assert_upper_bound import (  # noqa: E402
    _loss_arrow_plot,
    make_site_metric_table,
    write_notebook as write_upper_notebook,
)
from reproduce_paper import generate as generate_paper_outputs  # noqa: E402


def main() -> None:
    reference = ARTIFACT_ROOT / "reference_results"
    output = ARTIFACT_ROOT / "results" / "reference_report"
    upper_out = output / "upper"
    lower_out = output / "lower"
    upper_figures = output / "figures" / "upper"
    lower_figures = output / "figures" / "lower"
    notebooks = output / "notebooks"
    for path in (
        upper_out,
        lower_out,
        upper_figures,
        lower_figures,
        notebooks,
    ):
        path.mkdir(parents=True, exist_ok=True)

    upper_rows = pd.read_csv(
        reference
        / "upper"
        / "three_assert_upper_bound_program_site_rows.csv.gz"
    )
    upper_summary = json.loads(
        (
            reference
            / "upper"
            / "three_assert_upper_bound_summary.json"
        ).read_text(encoding="utf-8")
    )
    upper_table = make_site_metric_table(upper_rows)
    upper_table.to_csv(
        upper_out / "three_assert_upper_bound_site_metric_table.csv",
        index=False,
    )
    _loss_arrow_plot(upper_rows, upper_figures / "loss_arrow_by_site.png")
    write_upper_notebook(
        notebooks / "upper_bound_experiment.ipynb",
        upper_summary,
        upper_table,
        upper_figures,
        upper_out,
    )

    lower_rows = pd.read_csv(
        reference
        / "lower"
        / "three_assert_lower_bound_pair_final_metrics.csv.gz"
    )
    checkpoint_rows = pd.read_csv(
        reference
        / "lower"
        / "three_assert_lower_bound_checkpoint_rows.csv.gz"
    )
    lower_summary = json.loads(
        (
            reference
            / "lower"
            / "three_assert_lower_bound_summary.json"
        ).read_text(encoding="utf-8")
    )
    lower_aggregate = aggregate_final(lower_rows)
    checkpoint_summary = aggregate_checkpoints(checkpoint_rows)
    lower_aggregate.to_csv(
        lower_out / "three_assert_lower_bound_distribution_summary.csv",
        index=False,
    )
    checkpoint_summary.to_csv(
        lower_out / "three_assert_lower_bound_checkpoint_summary.csv",
        index=False,
    )
    plot_checked_curve(checkpoint_summary, lower_figures / "checked_recovery.png")
    plot_useful_curve(checkpoint_summary, lower_figures / "useful_counts.png")
    plot_final_box(lower_rows, lower_figures / "final_recovery_boxplot.png")
    write_lower_notebook(
        notebooks / "lower_bound_experiment.ipynb",
        lower_summary,
        lower_aggregate,
        lower_figures,
        lower_out,
    )
    generate_paper_outputs("reference", "full")
    print(f"Reference report regenerated under {output}")


if __name__ == "__main__":
    main()
