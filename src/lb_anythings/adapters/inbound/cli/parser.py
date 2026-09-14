"""Command-line surface. Parsing only; the composition root decides what each command does."""

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lb-anythings", description="Detector-agnostic Label Studio ML backend"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve the Label Studio ML backend over HTTP")
    serve.add_argument("--host", default=None, help="bind address (default: LB_HOST)")
    serve.add_argument("--port", type=int, default=None, help="port (default: LB_PORT)")

    commands.add_parser(
        "train",
        help="run one Training Run on the Training Set in this process (the server spawns this)",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
