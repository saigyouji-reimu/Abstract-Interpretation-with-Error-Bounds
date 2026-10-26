#!/usr/bin/env python3
"""Single entry point for evaluating the artifact."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd

from scripts.progress import worker_specs

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
PROGRAMS_FILE = ROOT / "data/programs/three_assert_affine_programs.json"
EXPECTED_PROGRAM_FINGERPRINT = (
    "ae198d0e35ea37f500a74b131ef501d581313d404f4bb7622f1bcb6c520d3237"
)


def load_config(mode: str) -> dict:
    path = ROOT / "configs" / f"{mode}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("mode") != mode:
        raise ValueError(f"Configuration mode mismatch in {path}")
    return config


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def mode_paths(mode: str) -> dict[str, Path]:
    work = ROOT / "work" / mode
    results = ROOT / "results" / mode
    return {
        "work": work,
        "points": work / "points",
        "lower_pairs": work / "lower_pairs",
        "results": results,
        "upper_summary": results / "upper",
        "lower_summary": results / "lower",
        "upper_figures": results / "figures" / "upper",
        "lower_figures": results / "figures" / "lower",
        "notebooks": results / "notebooks",
        "logs": results / "logs",
    }


def worker_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def run_logged(command: list[str], log_path: Path) -> None:
    from scripts.progress import read_json

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        run = read_json(log_path.parent / "progress.json")
        if run.get("status") == "running" and run.get("run_id"):
            log.write(f"# artifact-run-id: {run['run_id']}\n")
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=worker_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_parallel(
    commands: Iterable[tuple[list[str], Path]],
    jobs: int,
    stage: str,
) -> None:
    jobs = max(1, jobs)
    failures: list[tuple[Path, BaseException]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_logged, command, log_path): log_path
            for command, log_path in commands
        }
        for future in as_completed(futures):
            log_path = futures[future]
            try:
                future.result()
                print(f"[{stage}] completed {log_path.name}")
            except BaseException as exc:
                failures.append((log_path, exc))
                print(f"[{stage}] FAILED {log_path}: {exc}", file=sys.stderr)
    if failures:
        details = ", ".join(str(path) for path, _ in failures)
        raise RuntimeError(f"{stage} failed; inspect logs: {details}")


def common_range_args(config: dict, distribution: str, start: int, end: int) -> list[str]:
    return [
        "--distributions",
        distribution,
        "--sample-start",
        str(start),
        "--sample-end",
        str(end),
    ]


def prepare(mode: str, config: dict, jobs: int, force: bool) -> None:
    paths = mode_paths(mode)
    commands = []
    for distribution, start, end in worker_specs(config):
        command = [
            sys.executable,
            str(SCRIPTS / "prepare_inputs.py"),
            "--points-root",
            relative(paths["points"]),
            *common_range_args(config, distribution, start, end),
            "--n",
            str(config["num_points"]),
            "--base-seed",
            str(config["base_seed"]),
            "--max-keep",
            str(config["max_keep"]),
        ]
        if force:
            command.append("--force")
        log_path = paths["logs"] / f"prepare_{distribution}_{start:03d}_{end:03d}.log"
        commands.append((command, log_path))
    run_parallel(commands, jobs, "prepare")


def run_upper(mode: str, config: dict, jobs: int, force: bool) -> None:
    paths = mode_paths(mode)
    commands = []
    for distribution, start, end in worker_specs(config):
        command = [
            sys.executable,
            str(SCRIPTS / "run_three_assert_upper_bound_batch.py"),
            "--points-root",
            relative(paths["points"]),
            "--programs-file",
            relative(PROGRAMS_FILE),
            *common_range_args(config, distribution, start, end),
            "--selection-algorithms",
            "all",
            "--program-start",
            str(config["program_start"]),
            "--program-end",
            str(config["program_end"]),
        ]
        if force:
            command.append("--force")
        log_path = paths["logs"] / f"upper_{distribution}_{start:03d}_{end:03d}.log"
        commands.append((command, log_path))
    run_parallel(commands, jobs, "upper")


def run_lower(
    mode: str,
    config: dict,
    jobs: int,
    force: bool,
    save_trace: bool,
) -> None:
    paths = mode_paths(mode)
    commands = []
    for distribution, start, end in worker_specs(config):
        command = [
            sys.executable,
            str(SCRIPTS / "run_three_assert_lower_bound_batch.py"),
            "--points-root",
            relative(paths["points"]),
            "--programs-file",
            relative(PROGRAMS_FILE),
            "--out-dir",
            relative(paths["lower_pairs"]),
            *common_range_args(config, distribution, start, end),
            "--program-start",
            str(config["program_start"]),
            "--program-end",
            str(config["program_end"]),
            "--constraints-per-site",
            str(config["constraints_per_site"]),
            "--constraint-seed",
            str(config["constraint_seed"]),
        ]
        if force:
            command.append("--force")
        if save_trace:
            command.append("--save-trace")
        log_path = paths["logs"] / f"lower_{distribution}_{start:03d}_{end:03d}.log"
        commands.append((command, log_path))
    run_parallel(commands, jobs, "lower")


def report_current(mode: str, config: dict, progress=None) -> None:
    paths = mode_paths(mode)
    paths["notebooks"].mkdir(parents=True, exist_ok=True)
    upper_command = [
        sys.executable,
        str(SCRIPTS / "summarize_three_assert_upper_bound.py"),
        "--points-root",
        relative(paths["points"]),
        "--out-dir",
        relative(paths["upper_summary"]),
        "--figures-dir",
        relative(paths["upper_figures"]),
        "--notebook",
        relative(paths["notebooks"] / "upper_bound_experiment.ipynb"),
        "--distributions",
        *config["distributions"],
        "--selection-algorithms",
        "all",
        "--sample-start",
        str(config["sample_start"]),
        "--sample-end",
        str(config["sample_end"]),
    ]
    lower_command = [
        sys.executable,
        str(SCRIPTS / "summarize_three_assert_lower_bound.py"),
        "--batch-dir",
        relative(paths["lower_pairs"]),
        "--out-dir",
        relative(paths["lower_summary"]),
        "--figures-dir",
        relative(paths["lower_figures"]),
        "--notebook",
        relative(paths["notebooks"] / "lower_bound_experiment.ipynb"),
        "--distributions",
        *config["distributions"],
        "--sample-start",
        str(config["sample_start"]),
        "--sample-end",
        str(config["sample_end"]),
    ]
    run_logged(upper_command, paths["logs"] / "report_upper.log")
    if progress:
        progress(1)
    run_logged(lower_command, paths["logs"] / "report_lower.log")
    if progress:
        progress(2)
    run_logged(
        [sys.executable, str(SCRIPTS / "reproduce_paper.py"),
         "--source", "current", "--mode", mode],
        paths["logs"] / "report_paper.log",
    )
    if progress:
        progress(3)
    print(f"[report] saved under {paths['results']}")


def report_reference() -> None:
    command = [sys.executable, str(SCRIPTS / "report_reference.py")]
    log_path = ROOT / "results" / "reference_report" / "report.log"
    run_logged(command, log_path)
    print(f"[report] regenerated reference report under {log_path.parent}")


def check_programs() -> None:
    from core.three_assert import (
        read_three_assert_programs,
        three_assert_program_collection_fingerprint,
    )

    programs = read_three_assert_programs(PROGRAMS_FILE)
    fingerprint = three_assert_program_collection_fingerprint(programs)
    if fingerprint != EXPECTED_PROGRAM_FINGERPRINT:
        raise AssertionError(
            f"Program fingerprint mismatch: {fingerprint} != "
            f"{EXPECTED_PROGRAM_FINGERPRINT}"
        )
    if len(programs) != 100:
        raise AssertionError(f"Expected 100 programs, found {len(programs)}")


def load_claim_frames(
    mode: str,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "reference":
        upper_path = (
            ROOT
            / "reference_results/upper/three_assert_upper_bound_site_metric_table.csv"
        )
        lower_path = (
            ROOT
            / "reference_results/lower/three_assert_lower_bound_pair_final_metrics.csv.gz"
        )
    else:
        paths = mode_paths(mode)
        upper_path = (
            paths["upper_summary"]
            / "three_assert_upper_bound_site_metric_table.csv"
        )
        lower_path = (
            paths["lower_summary"]
            / "three_assert_lower_bound_pair_final_metrics.csv.gz"
        )
    if not upper_path.exists() or not lower_path.exists():
        raise FileNotFoundError(
            f"Missing claim inputs: upper={upper_path.exists()}, "
            f"lower={lower_path.exists()}"
        )
    return pd.read_csv(upper_path), pd.read_csv(lower_path)


def _record(
    rows: list[dict],
    claim: str,
    passed: bool,
    observed: object,
    expected: object,
) -> None:
    rows.append(
        {
            "claim": claim,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def check_claims(mode: str, source: str) -> dict:
    upper, lower = load_claim_frames(mode, source)
    rows: list[dict] = []

    upper_by_site = upper.set_index("assert_site")
    for site in ("assert1", "assert2", "assert3"):
        selected_retention = float(
            upper_by_site.loc[site, "selected_area_retention"]
        )
        generator_retention = float(
            upper_by_site.loc[site, "generator_area_retention"]
        )
        selected_slack = float(upper_by_site.loc[site, "selected_slack"])
        generator_slack = float(upper_by_site.loc[site, "generator_slack"])
        _record(
            rows,
            f"RQ1 retention improvement at {site}",
            selected_retention > generator_retention,
            selected_retention,
            f"> {generator_retention}",
        )
        _record(
            rows,
            f"RQ1 slack improvement at {site}",
            selected_slack < generator_slack,
            selected_slack,
            f"< {generator_slack}",
        )

    initial_mean = float(lower["old_guard1_recovery_ratio"].mean())
    initial_median = float(lower["old_guard1_recovery_ratio"].median())
    posterior_mean = float(lower["combined_recovery_ratio"].mean())
    posterior_median = float(lower["combined_recovery_ratio"].median())
    q3_initial_mean = float(lower["old_assert3_recovery_ratio"].mean())
    q3_posterior_mean = float(lower["assert3_combined_recovery_ratio"].mean())
    useful_mean = float(lower["total_useful_count"].mean())
    useful_median = float(lower["total_useful_count"].median())

    _record(rows, "RQ2 informative initial lower bound at p_t", initial_mean > 0.0, initial_mean, "> 0")
    _record(rows, "RQ3 posterior improves p_t", posterior_mean > initial_mean, posterior_mean, f"> {initial_mean}")
    _record(rows, "RQ3 posterior improves q3", q3_posterior_mean > q3_initial_mean, q3_posterior_mean, f"> {q3_initial_mean}")
    _record(rows, "RQ3 useful feedback exists", useful_mean > 0.0, useful_mean, "> 0")

    full_claims = source == "reference" or mode == "full"
    if full_claims:
        expected_upper = {
            "assert1": (0.932, 0.962),
            "assert2": (0.913, 0.966),
            "assert3": (0.980, 0.918),
        }
        for site, (retention, reduction) in expected_upper.items():
            observed_retention = float(
                upper_by_site.loc[site, "selected_area_retention"]
            )
            observed_reduction = float(
                upper_by_site.loc[site, "slack_reduction_vs_generator"]
            )
            _record(
                rows,
                f"RQ1 paper retention at {site}",
                round(observed_retention * 100, 1) == round(retention * 100, 1),
                observed_retention,
                retention,
            )
            _record(
                rows,
                f"RQ1 paper slack reduction at {site}",
                round(observed_reduction * 100, 1) == round(reduction * 100, 1),
                observed_reduction,
                reduction,
            )
        expected_lower = {
            "RQ2 initial p_t mean": (initial_mean, 0.463, 3),
            "RQ2 initial p_t median": (initial_median, 0.477, 3),
            "RQ3 posterior p_t mean": (posterior_mean, 0.644, 3),
            "RQ3 posterior p_t median": (posterior_median, 0.711, 3),
            "RQ3 posterior q3 mean": (q3_posterior_mean, 0.228, 3),
            "RQ3 useful mean": (useful_mean, 242.3, 1),
            "RQ3 useful median": (useful_median, 68.0, 1),
        }
        for claim, (observed, expected, digits) in expected_lower.items():
            _record(
                rows,
                claim,
                round(observed, digits) == round(expected, digits),
                observed,
                expected,
            )

    passed = all(row["status"] == "PASS" for row in rows)
    payload = {
        "mode": mode,
        "source": source,
        "passed": passed,
        "checks": rows,
    }
    output_dir = mode_paths(mode)["results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"claim_check_{source}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['status']:4s} {row['claim']}: "
            f"observed={row['observed']} expected={row['expected']}"
        )
    print(f"Claim check: {'PASS' if passed else 'FAIL'} ({output_path})")
    if not passed:
        raise AssertionError("One or more artifact claims failed.")
    return payload


def portability_check() -> None:
    forbidden = "/home/" + "yfshi"
    suffixes = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml", ".sh"}
    violations = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in {"work", "results", "__pycache__", ".venv", ".git", "dist"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if forbidden in text:
            violations.append(str(path.relative_to(ROOT)))
    if violations:
        raise AssertionError(f"Absolute development paths found in: {violations}")


def sanity() -> None:
    check_programs()
    tests = [
        "smoke_core.py",
        "smoke_branching.py",
        "smoke_branch_feedback.py",
        "smoke_three_assert.py",
    ]
    for test in tests:
        subprocess.run(
            [sys.executable, str(ROOT / "tests" / test)],
            cwd=ROOT,
            env=worker_environment(),
            check=True,
        )
    portability_check()
    check_claims("full", "reference")
    print("Sanity checks: PASS")


def evaluate(args: argparse.Namespace) -> None:
    from scripts.background import LOCK_ENV, now, write_state
    from scripts.progress import begin_run, read_json

    config = load_config(args.mode)
    jobs = args.jobs or min(12, os.cpu_count() or 1)
    stages = (
        ["prepare", "upper", "lower", "report", "check"]
        if args.stage == "all"
        else [args.stage]
    )
    logs = mode_paths(args.mode)["logs"]
    background_started_at = (read_json(logs / "background.json").get("started_at")
                             if LOCK_ENV in os.environ else None)
    run = begin_run(logs, config, stages, jobs, args.force, background_started_at)
    for stage in stages:
        entry = run["stages"][stage]
        entry.update(status="running", started_at=now())
        write_state(logs / "progress.json", run)
        print(f"[{stage}] started", flush=True)
        try:
            if stage == "prepare":
                prepare(args.mode, config, jobs, args.force)
            elif stage == "upper":
                run_upper(args.mode, config, jobs, args.force)
            elif stage == "lower":
                run_lower(args.mode, config, jobs, args.force, args.save_trace)
            elif stage == "report":
                def report_progress(completed: int) -> None:
                    entry["completed_units"] = completed
                    write_state(logs / "progress.json", run)

                report_current(args.mode, config, progress=report_progress)
            elif stage == "check":
                check_claims(args.mode, "current")
            else:
                raise AssertionError(stage)
        except BaseException as exc:
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            entry.update(status=state, finished_at=now(), error=str(exc))
            run.update(status=state, finished_at=entry["finished_at"])
            write_state(logs / "progress.json", run)
            raise
        entry.update(status="completed", finished_at=now())
        write_state(logs / "progress.json", run)
    run.update(status="completed", finished_at=now())
    write_state(logs / "progress.json", run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce and inspect the Abstract Interpretation with Error Bounds artifact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sanity", help="Run lightweight tests and validate reference results.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run fast or full experiments.")
    evaluate_parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    evaluate_parser.add_argument(
        "--stage",
        choices=["all", "prepare", "upper", "lower", "report", "check"],
        default="all",
    )
    evaluate_parser.add_argument("--jobs", type=int, default=None)
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.add_argument(
        "--background", action="store_true",
        help="Detach from the terminal; save status and logs under results/<mode>/logs/.",
    )
    evaluate_parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save large per-constraint traces; intended only for small debugging runs.",
    )

    check_parser = subparsers.add_parser("check", help="Check experimental claims.")
    check_parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    check_parser.add_argument(
        "--source", choices=["current", "reference"], default="current"
    )

    report_parser = subparsers.add_parser("report", help="Regenerate reports from current pair outputs.")
    report_parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    report_parser.add_argument(
        "--source", choices=["current", "reference"], default="current"
    )

    status_parser = subparsers.add_parser("status", help="Inspect a background evaluation.")
    status_parser.add_argument("--mode", choices=["fast", "full"], default="full")
    status_parser.add_argument("--json", action="store_true", help="Emit a machine-readable progress snapshot.")
    view = status_parser.add_mutually_exclusive_group()
    view.add_argument("--watch", action="store_true", help="Refresh until completion; Ctrl-C stops monitoring only.")
    view.add_argument("--serve", type=int, metavar="PORT", help="Serve read-only progress on 127.0.0.1:PORT.")
    status_parser.add_argument("--interval", type=float, default=10.0, help="Watch refresh interval in seconds (at least 1).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sanity":
        sanity()
    elif args.command == "evaluate":
        from scripts.background import evaluation_lock, launch

        if args.background:
            command = [sys.executable, "-u", str(ROOT / "artifact.py"),
                       *[arg for arg in sys.argv[1:] if arg != "--background"]]
            state = launch(command, ROOT, mode_paths(args.mode)["logs"], worker_environment())
            print(f"Background evaluation started (supervisor PID {state['pid']}).")
            print(f"Log: {state['log']}")
            print(f"Status: python artifact.py status --mode {args.mode}")
        else:
            with evaluation_lock(mode_paths(args.mode)["logs"]):
                evaluate(args)
    elif args.command == "status":
        from scripts.progress import display

        if args.interval < 1:
            parser.error("--interval must be at least 1 second")
        if args.serve is not None and not 0 <= args.serve <= 65535:
            parser.error("--serve must be a port between 0 and 65535")
        display(ROOT, args.mode, as_json=args.json, watch=args.watch,
                interval=args.interval, serve=args.serve)
    elif args.command == "check":
        check_claims(args.mode, args.source)
    elif args.command == "report":
        if args.source == "reference":
            report_reference()
        else:
            report_current(args.mode, load_config(args.mode))
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
