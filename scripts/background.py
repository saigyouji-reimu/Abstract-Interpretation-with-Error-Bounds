"""Detached evaluation supervision on Linux, with an exclusive per-mode lock."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LOCK_ENV = "ARTIFACT_EVALUATION_LOCK_FD"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def evaluation_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "evaluation.lock"
    inherited = os.environ.get(LOCK_ENV)
    if inherited is not None:
        fd = int(inherited)
        if not os.path.samestat(os.fstat(fd), lock_path.stat()):
            raise RuntimeError("Invalid inherited evaluation lock.")
        yield fd
        return
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"An evaluation is already running: {run_dir}") from exc
        yield handle.fileno()


def launch(command: list[str], root: Path, run_dir: Path, env: dict[str, str]) -> dict:
    run_dir = run_dir.resolve()
    state_path = run_dir / "background.json"
    log_path = run_dir / "evaluation.log"
    with evaluation_lock(run_dir) as fd:
        state = {
            "status": "starting", "started_at": now(), "finished_at": None,
            "pid": None, "worker_pid": None, "exit_code": None,
            "command": command, "log": str(log_path),
        }
        write_state(state_path, state)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{state['started_at']}] Starting: {command}\n")
            log.flush()
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--supervise", str(state_path), "--lock-fd", str(fd), "--", *command],
                cwd=root, env={**env, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(fd,),
            )
        # Wait only for the supervisor handshake, never for the experiment.
        for _ in range(100):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("pid") is not None:
                return state
            if process.poll() is not None:
                raise RuntimeError(f"Background supervisor failed; inspect {log_path}")
            time.sleep(0.05)
        raise RuntimeError(f"Supervisor did not acknowledge startup; inspect {log_path}")


def lock_held(run_dir: Path) -> bool:
    try:
        with (run_dir / "evaluation.lock").open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def status(run_dir: Path) -> dict:
    state_path = run_dir / "background.json"
    if not state_path.exists():
        return {"status": "not_started", "message": "No background evaluation recorded."}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] in {"starting", "running"}:
        if not lock_held(run_dir):
            state["status"] = "interrupted"
            state["message"] = "The process exited without recording completion; inspect the log."
    return state


def supervise(path: Path, fd: int, command: list[str]) -> int:
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(status="running", pid=os.getpid())
    write_state(path, state)
    exit_code = 1
    try:
        child = subprocess.Popen(
            command, env={**os.environ, LOCK_ENV: str(fd)}, pass_fds=(fd,),
        )
        state["worker_pid"] = child.pid
        write_state(path, state)
        exit_code = child.wait()
    except Exception as exc:
        state["error"] = repr(exc)
        print(f"Background evaluation failed: {exc}", file=sys.stderr, flush=True)
    finally:
        state.update(status="completed" if exit_code == 0 else "failed",
                     exit_code=exit_code, finished_at=now())
        write_state(path, state)
        print(f"[{state['finished_at']}] {state['status']}; exit code {exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervise", type=Path, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.exit(supervise(args.supervise, args.lock_fd, command))
