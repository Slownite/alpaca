import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from alpaca import process


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Point ALPACA_HOME at a tmp dir so process.py writes there, and return run/."""
    monkeypatch.setenv("ALPACA_HOME", str(tmp_path))
    return tmp_path / "run"


@pytest.fixture
def sleeper():
    """Spawn a short-lived child process and clean it up after the test."""
    procs: list[subprocess.Popen] = []

    def _spawn(seconds: float = 30.0) -> subprocess.Popen:
        p = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
        procs.append(p)
        return p

    yield _spawn

    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


# --- terminate_running_process ---


def test_terminate_running_process_stops_live_pid(sleeper):
    p = sleeper(30)
    process.terminate_running_process(p.pid, timeout=5.0)
    assert not psutil.pid_exists(p.pid) or psutil.Process(p.pid).status() == psutil.STATUS_ZOMBIE


def test_terminate_running_process_force_kills_on_timeout(monkeypatch):
    """If terminate() doesn't finish within timeout, kill() should be invoked."""
    calls = []

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            raise psutil.TimeoutExpired(timeout)

        def kill(self):
            calls.append("kill")

    monkeypatch.setattr(process.psutil, "Process", FakeProc)
    process.terminate_running_process(1234, timeout=0.1)
    assert "terminate" in calls
    assert "kill" in calls


# --- cleanup ---


def test_cleanup_removes_existing_file(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    process.cleanup(f)
    assert not f.exists()


def test_cleanup_missing_file_is_noop(tmp_path):
    f = tmp_path / "does-not-exist.json"
    process.cleanup(f)  # should not raise
    assert not f.exists()


# --- record ---


def test_record_uses_name_when_provided(run_dir):
    meta = {"name": "m1", "model": "mistral", "pid": os.getpid(), "port": 8000}
    process.record(meta)
    written = run_dir / "m1.json"
    assert written.exists()
    assert json.loads(written.read_text()) == meta


def test_record_falls_back_to_model_when_name_missing(run_dir):
    meta = {"model": "mistral", "pid": os.getpid()}
    process.record(meta)
    written = run_dir / "mistral.json"
    assert written.exists()
    data = json.loads(written.read_text())
    assert data["name"] == "mistral"
    assert data["model"] == "mistral"


def test_record_overwrites_existing_entry(run_dir):
    process.record({"name": "m1", "model": "a", "pid": os.getpid()})
    process.record({"name": "m1", "model": "b", "pid": os.getpid()})
    data = json.loads((run_dir / "m1.json").read_text())
    assert data["model"] == "b"


# --- list_running ---


def test_list_running_returns_alive_processes(run_dir, sleeper):
    p = sleeper(30)
    process.record({"name": "alive", "model": "m", "pid": p.pid})
    listed = process.list_running()
    assert any(m["name"] == "alive" and m["pid"] == p.pid for m in listed)


def test_list_running_cleans_up_dead_processes(run_dir, sleeper):
    p = sleeper(30)
    p.kill()
    p.wait(timeout=5)
    process.record({"name": "dead", "model": "m", "pid": p.pid})
    listed = process.list_running()
    assert all(m["name"] != "dead" for m in listed)
    assert not (run_dir / "dead.json").exists()


def test_list_running_empty_returns_empty_list(run_dir):
    assert process.list_running() == []


# --- find ---


def test_find_returns_meta_by_name(run_dir, sleeper):
    p = sleeper(30)
    process.record({"name": "m1", "model": "mistral", "pid": p.pid})
    meta = process.find("m1")
    assert meta is not None
    assert meta["name"] == "m1"
    assert meta["pid"] == p.pid


def test_find_returns_none_when_absent(run_dir):
    assert process.find("nope") is None


# --- stop ---


def test_stop_terminates_process_and_returns_true(run_dir, sleeper):
    p = sleeper(30)
    process.record({"name": "m1", "model": "m", "pid": p.pid})
    assert process.stop("m1", timeout=5.0) is True
    # Give the OS a moment to reap
    for _ in range(20):
        if not psutil.pid_exists(p.pid) or psutil.Process(p.pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(p.pid) or psutil.Process(p.pid).status() == psutil.STATUS_ZOMBIE


def test_stop_removes_meta_file(run_dir, sleeper):
    p = sleeper(30)
    process.record({"name": "m1", "model": "m", "pid": p.pid})
    process.stop("m1", timeout=5.0)
    assert not (run_dir / "m1.json").exists()


def test_stop_returns_false_for_dead_pid(run_dir, sleeper):
    p = sleeper(30)
    p.kill()
    p.wait(timeout=5)
    process.record({"name": "dead", "model": "m", "pid": p.pid})
    assert process.stop("dead") is False
