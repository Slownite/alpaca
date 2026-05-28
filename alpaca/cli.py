"""argparse command dispatch for the alpaca CLI."""
from __future__ import annotations

import argparse

from alpaca import __version__

COMMANDS = [
    "pull",
    "serve",
    "run",
    "ps",
    "stop",
    "cluster-up",
    "ray-head",
    "ray-worker",
    "serve-ray",
    "bench",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpaca",
        description="Ollama-style vLLM wrapper with Ray-distributed inference.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name in COMMANDS:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    raise NotImplementedError(f"command {args.command!r} is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
