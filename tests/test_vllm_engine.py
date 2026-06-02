from __future__ import annotations

import importlib
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest


def test_module_imports_without_vllm_extra():
    """Importing the module must not require the optional `gpu` extra."""
    # Simulate `vllm` not being installed by blocking its import.
    blocker = {"vllm": None}
    saved = sys.modules.pop("alpaca.engine.vllm", None)
    with patch.dict(sys.modules, blocker):
        module = importlib.import_module("alpaca.engine.vllm")
        assert hasattr(module, "VLLMEngine")
    if saved is not None:
        sys.modules["alpaca.engine.vllm"] = saved


def _running_popen() -> MagicMock:
    """A Popen mock that reports the child is alive."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = None
    return proc


def _dead_popen(returncode: int = 0) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = returncode
    return proc


def test_endpoint_raises_before_start():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with pytest.raises(RuntimeError):
        engine.endpoint()


def test_health_false_before_start():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    assert engine.health() is False


def test_stop_is_noop_before_start():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    # Must not raise.
    engine.stop()
    engine.stop()


def test_start_spawns_subprocess_with_required_flags():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()) as popen:
        engine.start("meta-llama/Llama-3-8B", 8000)

    cmd = popen.call_args.args[0]
    assert cmd[:3] == ["vllm", "serve", "meta-llama/Llama-3-8B"]
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8000" in cmd


def test_start_maps_recognised_opts_to_vllm_flags():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()) as popen:
        engine.start(
            "model",
            8001,
            host="0.0.0.0",
            dtype="bfloat16",
            max_model_len=4096,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
        )

    cmd = popen.call_args.args[0]
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--dtype" in cmd and "bfloat16" in cmd
    assert "--max-model-len" in cmd and "4096" in cmd
    assert "--tensor-parallel-size" in cmd and "2" in cmd
    assert "--gpu-memory-utilization" in cmd and "0.9" in cmd


def test_start_ignores_unknown_opts():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()) as popen:
        engine.start("model", 8002, mystery_flag="ignored")

    cmd = popen.call_args.args[0]
    assert "mystery_flag" not in cmd
    assert "ignored" not in cmd


def test_start_twice_without_stop_raises():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8003)
        with pytest.raises(RuntimeError):
            engine.start("model", 8003)


def test_start_after_stop_is_allowed():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    proc = _running_popen()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=proc):
        engine.start("model", 8004)
        engine.stop()
        # After stop, start again should not raise.
        engine.start("model", 8004)


def test_start_after_crash_is_allowed():
    """A dead child should not block a fresh start."""
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    crashed = _dead_popen(returncode=1)
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=crashed):
        engine.start("model", 8005)
        # Child has crashed (poll returns 1) — restart should succeed.
        engine.start("model", 8005)


def test_endpoint_returns_v1_url_after_start():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8006, host="10.0.0.5")
    assert engine.endpoint() == "http://10.0.0.5:8006/v1"


def test_endpoint_raises_after_stop():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8007)
        engine.stop()
    with pytest.raises(RuntimeError):
        engine.endpoint()


def test_stop_terminates_subprocess():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    proc = _running_popen()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=proc):
        engine.start("model", 8008)
        engine.stop()

    proc.terminate.assert_called_once()
    proc.wait.assert_called()
    proc.kill.assert_not_called()


def test_stop_force_kills_after_grace_period():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    proc = _running_popen()
    # First wait() times out, then succeeds after kill().
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="vllm", timeout=10), 0]
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=proc):
        engine.start("model", 8009)
        engine.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2


def test_stop_is_idempotent_after_clean_exit():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    proc = _running_popen()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=proc):
        engine.start("model", 8010)
        engine.stop()
        # Second stop is a no-op; nothing further on the (already cleared) proc.
        engine.stop()
    assert proc.terminate.call_count == 1


def test_stop_handles_already_exited_child():
    """If the child has already exited, stop must not try to terminate it."""
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    proc = _running_popen()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=proc):
        engine.start("model", 8011)
    # Simulate the child exiting on its own before stop().
    proc.poll.return_value = 0
    engine.stop()
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_health_true_when_models_endpoint_returns_200():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    response = MagicMock()
    response.status_code = 200
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8012)
    with patch("alpaca.engine.vllm.httpx.get", return_value=response) as get:
        assert engine.health() is True
    # Confirm we probe the right URL.
    called_url = get.call_args.args[0]
    assert called_url == "http://127.0.0.1:8012/v1/models"


def test_health_false_on_non_200():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    response = MagicMock()
    response.status_code = 503
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8013)
    with patch("alpaca.engine.vllm.httpx.get", return_value=response):
        assert engine.health() is False


def test_health_false_on_connection_refused_and_does_not_raise():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8014)
    with patch(
        "alpaca.engine.vllm.httpx.get",
        side_effect=httpx.ConnectError("refused"),
    ):
        assert engine.health() is False


def test_health_false_on_timeout_and_does_not_raise():
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    with patch("alpaca.engine.vllm.subprocess.Popen", return_value=_running_popen()):
        engine.start("model", 8015)
    with patch(
        "alpaca.engine.vllm.httpx.get",
        side_effect=httpx.ReadTimeout("slow"),
    ):
        assert engine.health() is False


# ---------------------------------------------------------------------------
# GPU-only end-to-end test (skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_vllm_engine_lifecycle_gpu():
    """Real vLLM lifecycle: start -> health -> endpoint -> stop. Requires a GPU."""
    from alpaca.engine.vllm import VLLMEngine

    engine = VLLMEngine()
    try:
        engine.start("facebook/opt-125m", 8099)
        # Poll for readiness; vLLM startup can take a while.
        deadline = time.time() + 120
        while time.time() < deadline:
            if engine.health():
                break
            time.sleep(2)
        assert engine.health() is True
        assert engine.endpoint() == "http://127.0.0.1:8099/v1"
    finally:
        engine.stop()
    assert engine.health() is False
