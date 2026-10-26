"""Exercise progress counts, resumes, force reruns, failures and read-only HTTP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import artifact
from scripts.background import evaluation_lock, write_state
from scripts.progress import (
    ALGORITHMS, STAGES, begin_run, format_snapshot, make_server, snapshot,
    worker_counts, worker_specs,
)


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="artifact-progress-test-")
        self.root = Path(self.temp.name)
        self.config = {"mode": "full", "distributions": ["triangle_uniform"],
                       "sample_start": 0, "sample_end": 2,
                       "program_start": 0, "program_end": 2,
                       "splits_per_distribution": 1}
        self.configs = self.root / "configs"
        self.configs.mkdir()
        self.write(self.configs / "full.json", self.config)
        self.logs = self.root / "results/full/logs"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_state(path, data)

    def background(self, state="running", age=120):
        payload = {"status": state, "started_at": iso(time.time() - age),
                   "finished_at": iso(time.time()) if state == "completed" else None,
                   "exit_code": 0 if state == "completed" else None,
                   "command": ["python", "artifact.py", "evaluate", "--mode", "full"]}
        self.write(self.logs / "background.json", payload)
        return payload

    def worker_log(self, stage, text):
        self.logs.mkdir(parents=True, exist_ok=True)
        path = self.logs / f"{stage}_triangle_uniform_000_002.log"
        path.write_text(text)
        return path

    def test_unstarted_is_read_only(self):
        data = snapshot(self.root, "full")
        self.assertEqual(data["status"], "not_started")
        self.assertTrue(all(entry["completed"] == 0 for entry in data["stages"].values()))
        self.assertFalse((self.root / "results").exists())

    def test_legacy_completed_counts(self):
        self.background("completed")
        lines = "".join(f"[done] triangle_uniform sample_{i:03d} {alg}: rows=6\n"
                        for i in range(2) for alg in ALGORITHMS)
        self.worker_log("prepare", lines)
        self.worker_log("upper", lines)
        self.worker_log("lower", "".join(
            f"[done] triangle_uniform sample_{i:03d} program_{j:03d}\n"
            for i in range(2) for j in range(2)))
        for name in ["notebooks/upper_bound_experiment.ipynb", "notebooks/lower_bound_experiment.ipynb",
                     "paper/manifest.json", "claim_check_current.json"]:
            self.write(self.root / "results/full" / name, {"passed": True})
        data = snapshot(self.root, "full")
        self.assertEqual(data["status"], "completed")
        self.assertEqual([data["stages"][s]["completed"] for s in STAGES], [6, 36, 4, 3, 1])
        self.assertTrue(all(v["percent"] == 100 for v in data["stages"].values()))
        self.assertIn("100.0%", format_snapshot(data))

    def test_partial_resumed_force_and_stale_logs(self):
        bg = self.background()
        run = begin_run(self.logs, self.config, ["lower"], 1, True, bg["started_at"])
        start = time.time() - 60
        run["stages"]["lower"].update(status="running", started_at=iso(start))
        self.write(self.logs / "progress.json", run)
        text = ("[done] triangle_uniform sample_000 program_000\n"
                              "[skip] triangle_uniform sample_000 program_001\n"
                              "[skip] triangle_uniform sample_000 program_001\n"
                              "[done] triangle_uniform sample_099 program_000\n"
                              "[done] triangle_uniform sample_001 program_000")
        log = self.worker_log("lower", "# artifact-run-id: old-run\n" + text)
        with evaluation_lock(self.logs):
            self.assertEqual(snapshot(self.root, "full")["stages"]["lower"]["completed"], 0)
            log.write_text(f"# artifact-run-id: {run['run_id']}\n" + text)
            data = snapshot(self.root, "full")
            entry = data["stages"]["lower"]
            self.assertEqual((entry["completed"], entry["computed"], entry["reused"]), (2, 1, 1))
            self.assertEqual(entry["percent"], 50)
            self.assertAlmostEqual(entry["eta_seconds"], 120, delta=5)
            self.assertEqual(data["stages"]["prepare"]["status"], "not_requested")
            self.assertEqual(data["current_stage"], "lower")
        self.assertEqual(snapshot(self.root, "full")["status"], "interrupted")

    def test_upper_skip_paths(self):
        path = self.worker_log("upper", "[skip] /work/points/triangle_uniform/sample_000 centroid_triangulated\n"
                               "[done] triangle_uniform sample_001 grid_cover_square: rows=6\n"
                               "[done] triangle_uniform sample_001 invalid_algorithm: rows=6\n")
        self.assertEqual(worker_counts(path, "upper", 0, 2, self.config)["completed"], 2)

    def test_new_background_ignores_old_ledger(self):
        bg = self.background("completed", age=3600)
        run = begin_run(self.logs, self.config, list(STAGES), 1, False, bg["started_at"])
        run.update(status="completed", started_at=iso(time.time() - 3500), finished_at=iso(time.time() - 100))
        for entry in run["stages"].values():
            entry.update(status="completed", started_at=run["started_at"])
        self.write(self.logs / "progress.json", run)
        log = self.worker_log("lower", "[done] triangle_uniform sample_000 program_000\n")
        os.utime(log, (time.time() - 100, time.time() - 100))
        self.background("running", age=1)
        with evaluation_lock(self.logs):
            data = snapshot(self.root, "full")
            self.assertEqual(data["status"], "running")
            self.assertEqual(data["stages"]["lower"]["completed"], 0)

    def test_orchestrator_success_and_failure(self):
        args = argparse.Namespace(mode="full", stage="check", jobs=1, force=False, save_trace=False)
        def succeed(mode, source):
            self.write(self.root / "results/full/claim_check_current.json", {"passed": True})
        with patch.object(artifact, "ROOT", self.root), patch.object(artifact, "check_claims", succeed):
            artifact.evaluate(args)
        data = snapshot(self.root, "full")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["stages"]["check"]["completed"], 1)
        self.assertEqual(data["stages"]["upper"]["status"], "not_requested")
        with patch.object(artifact, "ROOT", self.root), patch.object(artifact, "check_claims", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                artifact.evaluate(args)
        data = snapshot(self.root, "full")
        self.assertEqual(data["status"], "failed")
        self.assertIn("test failure", format_snapshot(data))
        self.assertEqual(data["stages"]["check"]["completed"], 0)

    def test_worker_partition_unchanged(self):
        config = {**self.config, "distributions": ["a", "b"], "sample_end": 5, "splits_per_distribution": 2}
        self.assertEqual(worker_specs(config), [("a", 0, 2), ("a", 2, 5), ("b", 0, 2), ("b", 2, 5)])

    def test_real_worker_log_run_id(self):
        args = argparse.Namespace(mode="full", stage="prepare", jobs=1, force=False, save_trace=False)
        lines = [f"[done] triangle_uniform sample_{i:03d} {alg}: selected=800"
                 for i in range(2) for alg in ALGORITHMS]
        def prepare(mode, config, jobs, force):
            artifact.run_logged([sys.executable, "-c", f"print({chr(10).join(lines)!r})"],
                                self.logs / "prepare_triangle_uniform_000_002.log")
        with patch.object(artifact, "ROOT", self.root), patch.object(artifact, "prepare", prepare):
            artifact.evaluate(args)
        data = snapshot(self.root, "full")
        self.assertEqual(data["stages"]["prepare"]["completed"], 6)
        self.assertEqual(data["stages"]["prepare"]["computed"], 6)
        text = (self.logs / "prepare_triangle_uniform_000_002.log").read_text()
        self.assertTrue(text.startswith(f"# artifact-run-id: {data['run_id']}\n"))

    def test_report_intermediate_progress(self):
        args = argparse.Namespace(mode="full", stage="report", jobs=1, force=False, save_trace=False)
        def report(mode, config, progress):
            progress(1)
            data = snapshot(self.root, "full")
            self.assertEqual(data["stages"]["report"]["completed"], 1)
            self.assertEqual(data["status"], "running")
        with evaluation_lock(self.logs), patch.object(artifact, "ROOT", self.root), patch.object(artifact, "report_current", report):
            artifact.evaluate(args)
        self.assertEqual(snapshot(self.root, "full")["stages"]["report"]["completed"], 3)

    def test_http_read_only(self):
        with make_server(self.root, "full", 0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/status.json", timeout=5) as response:
                    data = json.load(response)
                    self.assertEqual(data["status"], "not_started")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                with urlopen(url + "/", timeout=5) as response:
                    self.assertIn("not_started", response.read().decode())
                    self.assertEqual(response.headers["Refresh"], "10")
                with self.assertRaises(HTTPError) as error:
                    urlopen(url + "/README.md", timeout=5)
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        self.assertFalse((self.root / "results").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
