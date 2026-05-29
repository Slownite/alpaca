"""argparse command parsing and dispatch for the alpaca CLI."""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable

from alpaca import __version__
from alpaca.engine.base import Engine


def select_engine(name: str) -> type[Engine]:
    """Return the Engine class for the resolved ``--engine`` value.

    The single seam every command uses to obtain a backend, so ``--engine mock``
    routes to MockEngine without any command importing vLLM.
    """
    if name == "mock":
        from alpaca.engine.mock import MockEngine

        return MockEngine
    from alpaca.engine.vllm import VLLMEngine

    return VLLMEngine


def _not_implemented(command: str) -> int:
    print(f"alpaca {command}: not implemented yet", file=sys.stderr)
    return 1


def cmd_pull(args: argparse.Namespace) -> int:
    return _not_implemented("pull")


def cmd_serve(args: argparse.Namespace) -> int:
    return _not_implemented("serve")


def cmd_run(args: argparse.Namespace) -> int:
    return _not_implemented("run")


def cmd_ps(args: argparse.Namespace) -> int:
    return _not_implemented("ps")


def cmd_stop(args: argparse.Namespace) -> int:
    return _not_implemented("stop")


def cmd_cluster_up(args: argparse.Namespace) -> int:
    return _not_implemented("cluster-up")


def cmd_ray_head(args: argparse.Namespace) -> int:
    return _not_implemented("ray-head")


def cmd_ray_worker(args: argparse.Namespace) -> int:
    return _not_implemented("ray-worker")


def cmd_serve_ray(args: argparse.Namespace) -> int:
    return _not_implemented("serve-ray")


def cmd_bench(args: argparse.Namespace) -> int:
    return _not_implemented("bench")


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "pull": cmd_pull,
    "serve": cmd_serve,
    "run": cmd_run,
    "ps": cmd_ps,
    "stop": cmd_stop,
    "cluster-up": cmd_cluster_up,
    "ray-head": cmd_ray_head,
    "ray-worker": cmd_ray_worker,
    "serve-ray": cmd_serve_ray,
    "bench": cmd_bench,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpaca",
        description="Ollama-style vLLM wrapper with Ray-distributed inference.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--engine",
        choices=["vllm", "mock"],
        default="vllm",
        help="Inference backend to use (default: vllm)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks on error instead of a one-line message",
    )

    # Commands that instantiate an engine accept --engine after the subcommand
    # too. SUPPRESS keeps the subparser from clobbering a value parsed globally
    # when the flag is omitted after the subcommand.
    engine_opt = argparse.ArgumentParser(add_help=False)
    engine_opt.add_argument(
        "--engine",
        choices=["vllm", "mock"],
        default=argparse.SUPPRESS,
        help="Inference backend to use (default: vllm)",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    pull_parser = sub.add_parser("pull", help="Pull a model")
    pull_parser.add_argument("model", type=str, help="Model to pull")
    pull_parser.add_argument("--revision", type=str, help="Revision to pin to")

    serve_parser = sub.add_parser("serve", help="Start a single-node server", parents=[engine_opt])
    serve_parser.add_argument("model", type=str, help="Model to serve")
    serve_parser.add_argument("--name", type=str, help="Name of the server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to serve on")
    serve_parser.add_argument("--dtype", type=str, help="Data type")
    serve_parser.add_argument("--max-model-len", type=int, help="Max model length")

    run_parser = sub.add_parser("run", help="Run a model")
    run_parser.add_argument("name", type=str, help="Name of the server")
    run_parser.add_argument("prompt", type=str, help="Prompt to run")
    run_parser.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    run_parser.add_argument("--stream", action="store_true", help="Stream the response")

    sub.add_parser("ps", help="List running servers")

    stop_parser = sub.add_parser("stop", help="Stop a server")
    stop_parser.add_argument("name", type=str, help="Name of the server to stop")

    cluster_up_parser = sub.add_parser("cluster-up", help="Start a local Ray head + register this node")
    cluster_up_parser.add_argument("--port", type=int, help="Ray head port")
    cluster_up_parser.add_argument("--dashboard-port", type=int, help="Ray dashboard port")

    ray_head_parser = sub.add_parser("ray-head", help="Start a Ray head node")
    ray_head_parser.add_argument("--port", type=int, help="Ray head port")
    ray_head_parser.add_argument("--advertise-ip", type=str, help="Address to advertise to workers")

    ray_worker_parser = sub.add_parser("ray-worker", help="Join an existing Ray head")
    ray_worker_parser.add_argument("head_address", type=str, metavar="head-address", help="Address of the Ray head")
    ray_worker_parser.add_argument("--advertise-ip", type=str, help="Address to advertise to the head")

    serve_ray_parser = sub.add_parser(
        "serve-ray", help="Distributed serve (tensor parallel, multi-node)", parents=[engine_opt]
    )
    serve_ray_parser.add_argument("model", type=str, help="Model to serve")
    serve_ray_parser.add_argument("--name", type=str, help="Name of the server")
    serve_ray_parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    serve_ray_parser.add_argument("--tensor-parallel-size", type=int, help="Number of tensor-parallel ranks")

    bench_parser = sub.add_parser("bench", help="Load-test a server and report throughput and latency")
    bench_parser.add_argument("endpoint", type=str, help="OpenAI-compatible endpoint to benchmark")
    bench_parser.add_argument("--concurrency", type=int, help="Number of requests in flight")
    bench_parser.add_argument("--num-prompts", type=int, help="Total number of prompts to send")
    bench_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate per request")
    bench_parser.add_argument("--output", type=str, choices=["table", "csv"], default="table", help="Output format")
    bench_parser.add_argument("--plot", action="store_true", help="Render a latency/throughput chart")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
