import argparse

import pytest

from alpaca.cli import HANDLERS, build_parser, main, select_engine
from alpaca.engine.mock import MockEngine
from alpaca.engine.vllm import VLLMEngine


# --- AC2: each command parses its arguments into the expected attributes ---


def parse(argv):
    return build_parser().parse_args(argv)


def test_parser_lists_commands():
    assert parse(["ps"]).command == "ps"


def test_pull_args():
    args = parse(["pull", "mistral", "--revision", "abc123"])
    assert args.command == "pull"
    assert args.model == "mistral"
    assert args.revision == "abc123"


def test_serve_args():
    args = parse(
        ["serve", "mistral", "--name", "m1", "--port", "9000", "--host", "0.0.0.0",
         "--dtype", "float16", "--max-model-len", "4096"]
    )
    assert args.command == "serve"
    assert args.model == "mistral"
    assert args.name == "m1"
    assert args.port == 9000
    assert args.host == "0.0.0.0"
    assert args.dtype == "float16"
    assert args.max_model_len == 4096


def test_serve_defaults():
    args = parse(["serve", "mistral"])
    assert args.port == 8000
    assert args.host == "127.0.0.1"


def test_run_args():
    args = parse(["run", "m1", "hello world", "--max-tokens", "64", "--stream"])
    assert args.command == "run"
    assert args.name == "m1"
    assert args.prompt == "hello world"
    assert args.max_tokens == 64
    assert args.stream is True


def test_ps_args():
    assert parse(["ps"]).command == "ps"


def test_stop_args():
    args = parse(["stop", "m1"])
    assert args.command == "stop"
    assert args.name == "m1"


def test_cluster_up_args():
    args = parse(["cluster-up", "--port", "6379", "--dashboard-port", "8265"])
    assert args.command == "cluster-up"
    assert args.port == 6379
    assert args.dashboard_port == 8265


def test_ray_head_args():
    args = parse(["ray-head", "--port", "6379", "--advertise-ip", "10.0.0.1"])
    assert args.command == "ray-head"
    assert args.port == 6379
    assert args.advertise_ip == "10.0.0.1"


def test_ray_worker_args():
    args = parse(["ray-worker", "10.0.0.1:6379", "--advertise-ip", "10.0.0.2"])
    assert args.command == "ray-worker"
    assert args.head_address == "10.0.0.1:6379"
    assert args.advertise_ip == "10.0.0.2"


def test_serve_ray_args():
    args = parse(
        ["serve-ray", "mistral", "--name", "m1", "--port", "9000", "--tensor-parallel-size", "4"]
    )
    assert args.command == "serve-ray"
    assert args.model == "mistral"
    assert args.name == "m1"
    assert args.port == 9000
    assert args.tensor_parallel_size == 4


def test_bench_args():
    args = parse(
        ["bench", "http://localhost:8000", "--concurrency", "8", "--num-prompts", "100",
         "--max-tokens", "32", "--output", "csv", "--plot"]
    )
    assert args.command == "bench"
    assert args.endpoint == "http://localhost:8000"
    assert args.concurrency == 8
    assert args.num_prompts == 100
    assert args.max_tokens == 32
    assert args.output == "csv"
    assert args.plot is True


# --- AC3: main() dispatches by command name and returns the handler's exit code ---


def test_dispatch_calls_named_handler(monkeypatch):
    seen = {}

    def fake(args):
        seen["command"] = args.command
        return 42

    monkeypatch.setitem(HANDLERS, "ps", fake)
    assert main(["ps"]) == 42
    assert seen["command"] == "ps"


def test_every_command_has_a_handler():
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == set(HANDLERS)


# --- AC4: unknown/missing command prints help and returns non-zero ---


def test_missing_command_prints_help_returns_nonzero(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc != 0
    assert "usage: alpaca" in out


def test_unknown_command_errors_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code != 0


# --- AC5: --debug toggles full-traceback vs one-line error output ---


def _raising_handler(args):
    raise RuntimeError("kaboom")


def test_error_without_debug_is_one_line(monkeypatch, capsys):
    monkeypatch.setitem(HANDLERS, "ps", _raising_handler)
    rc = main(["ps"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "kaboom" in err
    assert "Traceback" not in err


def test_error_with_debug_prints_traceback(monkeypatch, capsys):
    monkeypatch.setitem(HANDLERS, "ps", _raising_handler)
    rc = main(["--debug", "ps"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" in err


# --- AC6: --engine mock selects MockEngine everywhere an engine is used ---


def test_engine_default_is_vllm():
    assert parse(["serve", "m"]).engine == "vllm"
    assert select_engine("vllm") is VLLMEngine


def test_engine_mock_selected_globally():
    args = parse(["--engine", "mock", "serve", "m"])
    assert args.engine == "mock"
    assert select_engine(args.engine) is MockEngine


def test_engine_mock_selected_per_command():
    args = parse(["serve", "m", "--engine", "mock"])
    assert args.engine == "mock"


def test_engine_available_on_every_command():
    for command in ["pull", "ps", "serve-ray", "bench"]:
        argv = {
            "pull": ["pull", "m"],
            "ps": ["ps"],
            "serve-ray": ["serve-ray", "m"],
            "bench": ["bench", "http://x"],
        }[command]
        assert parse(["--engine", "mock", *argv]).engine == "mock"
