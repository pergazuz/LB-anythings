"""Command-line surface. Parsing only; the composition root decides what each command does."""

import argparse
from collections.abc import Sequence
from pathlib import Path


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

    mine = commands.add_parser(
        "mine", help="write the frames of a video the Detector is least sure about"
    )
    mine.add_argument(
        "--video", type=Path, default=None, help="video to mine (default: LB_MINE_VIDEO)"
    )
    mine.add_argument("--stride", type=int, default=None, help="sample every Nth frame")
    mine.add_argument("--top-n", type=int, default=None, help="how many frames to keep")
    mine.add_argument("--gap", type=int, default=None, help="fewest frames between two picks")
    mine.add_argument("--uncertain-lo", type=float, default=None, help="uncertainty band floor")
    mine.add_argument("--uncertain-hi", type=float, default=None, help="uncertainty band ceiling")
    mine.add_argument("--conf", type=float, default=None, help="detection confidence floor")
    mine.add_argument("--out", type=Path, default=None, help="where to write the frames")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
