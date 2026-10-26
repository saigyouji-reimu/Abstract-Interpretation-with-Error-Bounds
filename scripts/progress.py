"""Read-only progress snapshots from per-run state and worker completion logs."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import time
import uuid

from scripts.background import lock_held, now, status as background_status, write_state

STAGES = ("prepare", "upper", "lower", "report", "check")
ALGORITHMS = ("centroid_triangulated", "grid_cover_square", "grid_cover_triangle")


def worker_specs(config: dict) -> list[tuple[str, int, int]]:
    start, end = int(config["sample_start"]), int(config["sample_end"])
    splits = max(1, int(config["splits_per_distribution"]))
    boundaries = [start + ((end - start) * i) // splits for i in range(splits + 1)]
    return [(distribution, left, right) for distribution in config["distributions"]
            for left, right in zip(boundaries, boundaries[1:]) if left < right]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def timestamp(value: str | None) -> float | None:
    return datetime.fromisoformat(value).timestamp() if value else None


def begin_run(logs: Path, config: dict, stages: list[str], jobs: int,
              force: bool, background_started_at: str | None) -> dict:
    started = now()
    run = {"status": "running", "run_id": uuid.uuid4().hex, "mode": config["mode"], "config": config,
           "jobs": jobs, "force": force, "started_at": started, "finished_at": None,
           "background_started_at": background_started_at,
           "stages": {stage: {"status": "pending" if stage in stages else "not_requested"}
                      for stage in STAGES}}
    logs.mkdir(parents=True, exist_ok=True)
    write_state(logs / "progress.json", run)
    return run


def matching_log(path: Path, since: float | None, run_id: str | None = None) -> bool:
    try:
        if since is None:
            return False
        if run_id:
            with path.open(encoding="utf-8", errors="replace") as handle:
                return handle.readline().strip() == f"# artifact-run-id: {run_id}"
        return since is not None and path.stat().st_mtime >= since
    except FileNotFoundError:
        return False


def worker_counts(path: Path, stage: str, left: int, right: int, config: dict) -> dict:
    """Count only complete done/skip lines, deduplicated by sample and unit."""
    units = {}
    errors = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    continue
                if any(word in line for word in ("Traceback", "FAILED", "Error:", "[missing points]")):
                    errors.append(line.strip())
                    errors = errors[-3:]
                if not line.startswith(("[done] ", "[skip] ")):
                    continue
                match = re.search(r"sample_(\d+)\s+(\w+)", line)
                if not match or not left <= int(match[1]) < right:
                    continue
                unit = match[2]
                if stage == "lower":
                    if not unit.startswith("program_") or not unit[8:].isdigit():
                        continue
                    if not config["program_start"] <= int(unit[8:]) < config["program_end"]:
                        continue
                elif unit not in ALGORITHMS:
                    continue
                units[(int(match[1]), unit)] = line.startswith("[skip] ")
    except FileNotFoundError:
        pass
    reused = sum(units.values())
    return {"completed": len(units), "reused": reused,
            "computed": len(units) - reused, "errors": errors}


def snapshot(root: Path, mode: str) -> dict:
    config = read_json(root / "configs" / f"{mode}.json")
    if not config:
        raise ValueError(f"Missing configuration for {mode}")
    results = root / "results" / mode
    logs = results / "logs"
    background = background_status(logs)
    run = read_json(logs / "progress.json")
    # A just-started background job must not inherit the previous run's ledger.
    if run and background.get("started_at"):
        related = run.get("background_started_at") == background["started_at"]
        newer_foreground = (run.get("background_started_at") is None
                            and timestamp(run["started_at"]) > timestamp(background["started_at"]))
        if not related and not newer_foreground:
            run = {}
    effective = {**background, **{k: v for k, v in run.items() if k != "stages"}}
    if run and run.get("background_started_at") is None:
        effective = {k: v for k, v in run.items() if k != "stages"}
    if run and run.get("background_started_at") and background["status"] in {"failed", "interrupted"}:
        effective["status"] = background["status"]
    if effective["status"] == "running" and not lock_held(logs):
        effective["status"] = "interrupted"
    if run:
        config = run["config"]
    specs = worker_specs(config)
    programs = config["program_end"] - config["program_start"]
    samples = len(config["distributions"]) * (config["sample_end"] - config["sample_start"])
    counts = {"prepare": (samples * len(ALGORITHMS), "selected sets"),
              "upper": (samples * len(ALGORITHMS) * programs * 3, "program/site/policy rows"),
              "lower": (samples * programs, "input/program pairs"),
              "report": (3, "report groups"), "check": (1, "claim-check report")}
    started = timestamp(effective.get("started_at"))
    finished = timestamp(effective.get("finished_at"))
    clock = time.time()
    requested = list(STAGES)
    command = background.get("command", [])
    for index, arg in enumerate(command):
        if arg == "--stage" and index + 1 < len(command) and command[index + 1] != "all":
            requested = [command[index + 1]]
        elif arg.startswith("--stage=") and arg != "--stage=all":
            requested = [arg.partition("=")[2]]
    stages = {}
    last_activity = started
    errors = []
    for stage in STAGES:
        entry = dict(run.get("stages", {}).get(stage, {}))
        since = timestamp(entry.get("started_at")) if run else started
        completed = computed = reused = 0
        distributions = {name: 0 for name in config["distributions"]}
        active_logs = 0
        factor = programs * 3 if stage == "upper" else 1
        if stage in {"prepare", "upper", "lower"}:
            for distribution, left, right in specs:
                log = logs / f"{stage}_{distribution}_{left:03d}_{right:03d}.log"
                if matching_log(log, since, run.get("run_id")):
                    active_logs += 1
                    unit = worker_counts(log, stage, left, right, config)
                    completed += unit["completed"] * factor
                    computed += unit["computed"] * factor
                    reused += unit["reused"] * factor
                    distributions[distribution] += unit["completed"] * factor
                    last_activity = max(last_activity or 0, log.stat().st_mtime)
                    errors.extend(f"{log.name}: {error}" for error in unit["errors"])
        else:
            markers = ([results / "notebooks/upper_bound_experiment.ipynb",
                        results / "notebooks/lower_bound_experiment.ipynb",
                        results / "paper/manifest.json"] if stage == "report"
                       else [results / "claim_check_current.json"])
            for marker in markers if not run else []:
                if matching_log(marker, since) and read_json(marker):
                    completed += 1
                    last_activity = max(last_activity or 0, marker.stat().st_mtime)
            for log in logs.glob("report_*.log") if stage == "report" else []:
                if matching_log(log, since, run.get("run_id")):
                    active_logs += 1
                    last_activity = max(last_activity or 0, log.stat().st_mtime)
            if run:
                completed = entry.get("completed_units", 0)
                if entry["status"] == "completed":
                    completed = counts[stage][0]
        total, unit_name = counts[stage]
        if not run:
            entry["status"] = "pending" if stage in requested else "not_requested"
            if stage in requested and effective["status"] == "completed":
                entry["status"] = "completed"
            elif stage in requested and (active_logs or completed):
                entry["status"] = "completed" if completed == total else "running"
        if entry.get("status") == "running" and effective["status"] in {"failed", "interrupted"}:
            entry["status"] = effective["status"]
        elapsed = None
        if entry.get("started_at"):
            end = timestamp(entry.get("finished_at")) or finished or clock
            elapsed = max(0.0, end - timestamp(entry["started_at"]))
        eta = None
        if entry.get("status") == "running" and elapsed and computed and completed < total:
            eta = elapsed * (total - completed) / computed
        entry.update(completed=completed, total=total, unit=unit_name,
                     percent=round(100 * completed / total, 2) if total else 100.0,
                     computed=computed, reused=reused, elapsed_seconds=elapsed,
                     eta_seconds=eta, by_distribution=distributions if stage in {"prepare", "upper", "lower"} else {})
        stages[stage] = entry
    if not run and effective["status"] == "running":
        remaining = [name for name in requested if stages[name]["status"] != "completed"]
        if remaining:
            stages[remaining[0]]["status"] = "running"
    current = next((name for name, entry in stages.items()
                    if entry["status"] in {"running", "failed", "interrupted"}), None)
    effective.pop("config", None)
    effective.update(mode=mode, observed_at=now(), current_stage=current,
                     elapsed_seconds=max(0.0, (finished or clock) - started) if started else None,
                     last_activity_at=datetime.fromtimestamp(last_activity, timezone.utc).isoformat() if last_activity else None,
                     seconds_since_activity=max(0.0, (finished or clock) - last_activity) if last_activity else None,
                     stages=stages, recent_errors=errors[-10:],
                     progress_source="current-run completion logs; done and validated skip entries",
                     eta_note="Rough current-stage estimate only; excludes later stages and is unavailable without stage timestamps.")
    return effective


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m {seconds % 60:02d}s"


def format_snapshot(data: dict) -> str:
    lines = [f"{data['mode']} evaluation: {data['status']}",
             f"Elapsed: {duration(data['elapsed_seconds'])} | Current stage: {data['current_stage'] or '-'}",
             "", f"{'Stage':<9} {'Status':<14} {'Completed / total':>23} {'Progress':>9}"]
    for stage, entry in data["stages"].items():
        counter = f"{entry['completed']:,} / {entry['total']:,}"
        lines.append(f"{stage:<9} {entry['status']:<14} {counter:>23} {entry['percent']:>8.1f}%  {entry['unit']}")
    current = data.get("current_stage")
    if current:
        entry = data["stages"][current]
        lines.extend(["", f"Current stage: computed {entry['computed']:,}; reused {entry['reused']:,}",
                      f"Approximate remaining time for this stage: {duration(entry['eta_seconds'])}"])
        for distribution, completed in entry["by_distribution"].items():
            lines.append(f"  {distribution}: {completed:,}")
    if data.get("exit_code") is not None:
        lines.append(f"Exit code: {data['exit_code']}")
    if data.get("last_activity_at"):
        lines.append(f"Last recorded activity (UTC): {data['last_activity_at']}")
    if current and data["stages"][current].get("error"):
        lines.append(f"Error: {data['stages'][current]['error']}")
    lines.extend(data.get("recent_errors", []))
    if data.get("log"):
        lines.append(f"Log: {data['log']}")
    lines.append("Counts are per stage, not a time-weighted overall percentage.")
    return "\n".join(lines)


def make_server(root: Path, mode: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/", "/status.json"}:
                self.send_error(404)
                return
            data = snapshot(root, mode)
            is_json = self.path == "/status.json"
            body = (json.dumps(data, indent=2) if is_json else format_snapshot(data)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8" if is_json else "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            if not is_json:
                self.send_header("Refresh", "10")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def display(root: Path, mode: str, *, as_json: bool = False, watch: bool = False,
            interval: float = 10.0, serve: int | None = None) -> None:
    try:
        if serve is not None:
            with make_server(root, mode, serve) as server:
                print(f"Read-only progress: http://127.0.0.1:{server.server_port}/", flush=True)
                print("JSON API: /status.json | Ctrl-C stops monitoring, not the experiment.", flush=True)
                server.serve_forever()
            return
        while True:
            data = snapshot(root, mode)
            print(json.dumps(data, indent=2) if as_json else format_snapshot(data), flush=True)
            if not watch or data["status"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(interval)
            print("\n---\n", flush=True)
    except KeyboardInterrupt:
        print("Monitoring stopped; the experiment was not stopped.")
