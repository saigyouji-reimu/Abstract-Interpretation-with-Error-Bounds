"""Check background execution and the paper's printed tables/curve endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from artifact import worker_environment
from scripts.background import evaluation_lock, launch, status, write_state
from scripts.package_release import include
from scripts.reproduce_paper import generate, validate_rows


def wait_for_completion(run_dir: Path) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = status(run_dir)
        if state["status"] in {"completed", "failed"}:
            try:
                os.waitpid(state["pid"], 0)
            except ChildProcessError:
                pass  # Popen may already have reaped a short-lived supervisor.
            return state
        time.sleep(0.05)
    raise AssertionError(f"Background job timed out: {run_dir}")


def test_background(temp: Path) -> None:
    assert status(temp / "missing")["status"] == "not_started"
    run = temp / "success"
    command = [sys.executable, "-u", "-c",
               "import os,time; print('started',flush=True); time.sleep(2); "
               "assert os.getenv('OMP_NUM_THREADS') == '1'; print('done')"]
    started = time.monotonic()
    launch(command, ROOT, run, worker_environment())
    assert time.monotonic() - started < 2
    assert status(run)["status"] == "running"
    try:
        launch(command, ROOT, run, worker_environment())
    except RuntimeError as exc:
        assert "already running" in str(exc)
    else:
        raise AssertionError("Duplicate evaluation was allowed")
    state = wait_for_completion(run)
    assert state["status"] == "completed" and state["exit_code"] == 0
    assert "done" in (run / "evaluation.log").read_text()
    failure = temp / "failure"
    launch([sys.executable, "-c", "raise SystemExit(7)"], ROOT, failure, worker_environment())
    state = wait_for_completion(failure)
    assert state["status"] == "failed" and state["exit_code"] == 7
    stale = temp / "stale"
    stale.mkdir()
    write_state(stale / "background.json", {"status": "running", "pid": -1})
    assert status(stale)["status"] == "interrupted"
    with evaluation_lock(temp / "foreground"):
        try:
            with evaluation_lock(temp / "foreground"):
                raise AssertionError("Foreground lock was not exclusive")
        except RuntimeError:
            pass
    print("background supervision: ok")


def test_paper(temp: Path) -> None:
    output = generate("reference", "full", temp / "paper")
    upper = pd.read_csv(output / "table1.csv")
    expected_upper = [
        [31.92, 32.27, 41.09, 93.2, 33.2, 0.35, 9.18, 96.2],
        [32.73, 33.04, 42.03, 91.3, 33.8, 0.32, 9.30, 96.6],
        [315.65, 318.20, 346.88, 98.0, 74.5, 2.56, 31.23, 91.8],
    ]
    for row, expected in zip(upper.itertuples(index=False, name=None), expected_upper):
        values = list(row[1:])
        for index in (3, 4, 7):
            values[index] *= 100
        rounded = [round(value, 1 if i in (3, 4, 7) else 2) for i, value in enumerate(values)]
        assert rounded == expected, (rounded, expected)
    lower = pd.read_csv(output / "table2.csv")
    expected_lower = [
        [73.04, 56.81, 44.28, 26.41, .463, .477, 53.69, 34.98, .644, .711, .528, .555, .531, .577],
        [315.65, 220.69, 0, 0, 0, 0, 93.70, 20.99, .228, .129, .145, .001, .159, .015],
    ]
    for (_, row), expected in zip(lower.iterrows(), expected_lower):
        rounded = [round(float(row[column]), 3 if "recovery" in column else 2)
                   for column in lower.columns if column != "location"]
        assert rounded == expected, (rounded, expected)
    checkpoints = pd.read_csv(output / "figure11_data.csv").set_index("checkpoint")
    indexed = lower.set_index("location")
    for location, metric in [("p_t", "combined_recovery_ratio"), ("q_3", "assert3_combined_recovery_ratio")]:
        count = checkpoints.loc[0, metric + "_valid_count"]
        total = checkpoints.loc[0, "num_rows"]
        # The historical curve omits undefined 0/0, while the initial table stores zero.
        np.testing.assert_allclose(checkpoints.loc[0, metric + "_mean"],
                                   indexed.loc[location, "initial_recovery_mean"] * total / count,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(checkpoints.loc[2000, metric + "_mean"],
                                   indexed.loc[location, "posterior_recovery_mean"], rtol=1e-12, atol=1e-12)
    for figure in ("figure9", "figure11"):
        pixels = np.asarray(Image.open(output / f"{figure}.png").convert("RGB"))
        assert pixels.shape[1] > 1000 and np.std(pixels) > 10
        assert (output / f"{figure}.pdf").read_bytes().startswith(b"%PDF")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["upper"]["rows"] == 540000
    assert manifest["inputs"]["lower"]["rows"] == 60000
    # Reject incomplete datasets, even when their aggregate means look plausible.
    config = json.loads((ROOT / "configs/fast.json").read_text())
    incomplete = pd.DataFrame(columns=["distribution", "sample_id", "program_id"])
    try:
        validate_rows(incomplete, incomplete, incomplete, config)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete data accepted")
    print("paper tables, figures and checkpoints: ok")


def test_packaging() -> None:
    for relative in ["work/fast/points.npz", "results/full/table1.csv", "dist/old.zip",
                     ".venv/bin/python", "notebooks/old.ipynb", "reference_results/figures/old.png",
                     "core/__pycache__/geometry.pyc", ".pytest_cache/data"]:
        assert not include(ROOT / relative), relative
    for relative in ["artifact.py", "scripts/reproduce_paper.py", "paper.pdf",
                     "reference_results/upper/three_assert_upper_bound_program_site_rows.csv.gz"]:
        assert include(ROOT / relative), relative
    print("release exclusions: ok")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="artifact-release-test-") as name:
        temp = Path(name)
        test_background(temp)
        test_paper(temp)
    test_packaging()
    print("smoke_release: ok")


if __name__ == "__main__":
    main()
