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

    train = commands.add_parser(
        "train",
        help="run one Training Run on the Training Set in this process (the server spawns this)",
    )
    train.add_argument(
        "--tracked-as",
        default=None,
        help="the recorded run this continues; the server passes it so the launch and the "
        "metrics land on one row",
    )

    up = commands.add_parser(
        "up",
        help="run Label Studio, the MLflow UI and this backend together, wired to a project",
    )
    up.add_argument(
        "--project", type=int, default=None, help="wire this project id (default: LB_PROJECT)"
    )
    up.add_argument(
        "--title",
        default=None,
        help="the project title to find, or create (default: LB_PROJECT_TITLE)",
    )
    up.add_argument(
        "--new", action="store_true", help="create a project even when that title is taken"
    )
    up.add_argument(
        "--label",
        action="append",
        dest="labels",
        metavar="LABEL",
        help="a label a created project offers; repeat for more (default: the Training Set's)",
    )
    up.add_argument("--host", default=None, help="bind address for the backend")
    up.add_argument("--port", type=int, default=None, help="port for the backend")
    up.add_argument(
        "--no-label-studio",
        dest="label_studio",
        action="store_false",
        help="do not start Label Studio; one is expected at LABEL_STUDIO_URL",
    )
    up.add_argument(
        "--no-mlflow", dest="mlflow", action="store_false", help="do not start the MLflow UI"
    )
    up.add_argument(
        "--no-wiring",
        dest="wire",
        action="store_false",
        help="start the Services and change nothing in Label Studio",
    )
    up.add_argument(
        "--no-open", dest="open_browser", action="store_false", help="do not open the project"
    )
    up.add_argument(
        "--quiet",
        action="store_true",
        help="do not echo what the Services print; their logs still have all of it",
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
